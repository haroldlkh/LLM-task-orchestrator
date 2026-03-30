import time

import polars as pl

from .adapter_loader import build_adapter
from .batch_processor import process_batches
from .merge import merge_results_back
from .prompt_loader import load_prompt_context
from .runtime import (
    build_runtime_metadata,
    default_runtime,
    success_or_terminal_unit_ids,
    validate_llm_step,
)
from .state import (
    download_all_parquet_by_prefix,
    download_latest_parquet_if_exists,
    ensure_llm_state_layout,
    upload_versioned_json,
    upload_versioned_parquet,
)
from .llm_task_loader import build_task_handler
from .traces import debug_rows_to_df, progress_rows_to_df, result_rows_to_df
from .work_units import build_work_units


def _load_existing_progress(connector, state_folder: str, temp_dir: str) -> pl.DataFrame:
    progress_df = download_latest_parquet_if_exists(
        connector=connector,
        location=state_folder,
        prefix="progress",
        temp_dir=temp_dir,
    )
    if progress_df is None:
        return progress_rows_to_df([])
    return progress_df


def _load_existing_results(connector, results_folder: str, temp_dir: str) -> pl.DataFrame:
    chunks = download_all_parquet_by_prefix(
        connector=connector,
        location=results_folder,
        prefix="results",
        temp_dir=temp_dir,
    )
    if not chunks:
        return result_rows_to_df([])
    return pl.concat(chunks, how="vertical_relaxed")


def _merge_progress(base_progress: pl.DataFrame, new_progress: pl.DataFrame) -> pl.DataFrame:
    if base_progress.is_empty():
        return new_progress

    combined = pl.concat([base_progress, new_progress], how="vertical_relaxed")
    combined = combined.sort(["unit_id", "updated_at"])
    latest = combined.group_by("unit_id").tail(1)
    return latest


def _flush_state(
    connector,
    folders: dict,
    temp_dir: str,
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    result_rows,
    debug_rows,
    metadata: dict,
):
    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="manifest",
        df=work_units_df,
        temp_dir=temp_dir,
    )

    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="progress",
        df=progress_df,
        temp_dir=temp_dir,
    )

    if result_rows:
        upload_versioned_parquet(
            connector=connector,
            location=folders["results_folder"],
            prefix="results",
            df=result_rows_to_df(result_rows),
            temp_dir=temp_dir,
        )

    if debug_rows:
        debug_df = debug_rows_to_df(debug_rows)
        upload_versioned_parquet(
            connector=connector,
            location=folders["debug_folder"],
            prefix="traces",
            df=debug_df,
            temp_dir=temp_dir,
        )

        review_df = debug_df.filter(
            (pl.col("status") != "success") | (pl.col("review_flag") == True)
        )
        if not review_df.is_empty():
            upload_versioned_parquet(
                connector=connector,
                location=folders["debug_folder"],
                prefix="review",
                df=review_df,
                temp_dir=temp_dir,
            )

    upload_versioned_json(
        connector=connector,
        location=folders["state_folder"],
        prefix="metadata",
        payload=metadata,
        temp_dir=temp_dir,
    )


def execute_llm_step(data, step_config: dict, runtime_context: dict):
    validate_llm_step(step_config)

    source_df = data.collect(streaming=True) if isinstance(data, pl.LazyFrame) else data

    runtime = default_runtime(step_config)
    pipeline_name = runtime_context["pipeline_name"]
    dest_connector = runtime_context["dest_connector"]
    dest_location = runtime_context["dest_location"]
    temp_dir = runtime_context["temp_dir"]
    user_runtime_config = runtime_context.get("user_runtime_config", {})

    provider_config_key = step_config["provider_config_key"]
    if provider_config_key not in user_runtime_config:
        raise KeyError(
            f"LLM step provider_config_key '{provider_config_key}' not found "
            f"in USER_RUNTIME_CONFIG_JSON"
        )

    provider_config = user_runtime_config[provider_config_key]
    adapter = build_adapter(step_config["adapter"], provider_config)
    task_handler = build_task_handler(step_config["task_handler"])
    prompt_context = load_prompt_context(step_config)

    folders = ensure_llm_state_layout(
        dest_connector=dest_connector,
        dest_location=dest_location,
        pipeline_name=pipeline_name,
        step_name=step_config["name"],
    )

    work_units_df = build_work_units(source_df, step_config)

    progress_df = _load_existing_progress(
        connector=dest_connector,
        state_folder=folders["state_folder"],
        temp_dir=temp_dir,
    )

    existing_results_df = _load_existing_results(
        connector=dest_connector,
        results_folder=folders["results_folder"],
        temp_dir=temp_dir,
    )

    terminal_ids = success_or_terminal_unit_ids(progress_df)
    pending_units_df = work_units_df.filter(~pl.col("unit_id").is_in(list(terminal_ids)))

    if pending_units_df.is_empty():
        from .models import LLMRunOutcome
        outcome = LLMRunOutcome(
            status="complete",
            processed_units=0,
            remaining_units=0,
        )
        metadata = build_runtime_metadata(
            step_config=step_config,
            work_units_df=work_units_df,
            progress_df=progress_df,
            outcome=outcome,
        )
        _flush_state(
            connector=dest_connector,
            folders=folders,
            temp_dir=temp_dir,
            work_units_df=work_units_df,
            progress_df=progress_df,
            result_rows=[],
            debug_rows=[],
            metadata=metadata,
        )
        return merge_results_back(
            source_df=source_df,
            all_results_df=existing_results_df,
            row_id_column=step_config["row_id_column"],
            task_handler=task_handler,
            step_config=step_config,
        )

    start_time = time.time()
    batch_out = process_batches(
        pending_rows=pending_units_df.to_dicts(),
        step_config=step_config,
        runtime=runtime,
        adapter=adapter,
        task_handler=task_handler,
        prompt_context=prompt_context,
        progress_df=progress_df,
        start_time=start_time,
    )

    new_progress_df = progress_rows_to_df(batch_out["progress_rows"])
    if not new_progress_df.is_empty():
        progress_df = _merge_progress(progress_df, new_progress_df)

    metadata = build_runtime_metadata(
        step_config=step_config,
        work_units_df=work_units_df,
        progress_df=progress_df,
        outcome=batch_out["outcome"],
    )

    _flush_state(
        connector=dest_connector,
        folders=folders,
        temp_dir=temp_dir,
        work_units_df=work_units_df,
        progress_df=progress_df,
        result_rows=batch_out["result_rows"],
        debug_rows=batch_out["debug_rows"],
        metadata=metadata,
    )

    existing_results_df = _load_existing_results(
        connector=dest_connector,
        results_folder=folders["results_folder"],
        temp_dir=temp_dir,
    )

    return merge_results_back(
        source_df=source_df,
        all_results_df=existing_results_df,
        row_id_column=step_config["row_id_column"],
        task_handler=task_handler,
        step_config=step_config,
    )