import time

import polars as pl

from .adapter_loader import build_adapter
from .batch_flush import (
    build_pair_status_df,
    build_partial_output_df,
    released_row_ids_for_flush,
)
from .batch_processor import process_batches
from .dataframes import (
    debug_rows_to_df,
    empty_progress_df,
    empty_result_df,
    ensure_progress_df,
    ensure_result_df,
    progress_rows_to_df,
    result_rows_to_df,
)
from .llm_task_loader import build_task_handler
from .merge import merge_results_back
from .models import LLMRunOutcome
from .prompt_loader import load_prompt_context
from .runtime import (
    apply_input_row_window,
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
from .work_units import build_work_units


def _load_existing_progress(connector, state_folder: str, temp_dir: str) -> pl.DataFrame:
    progress_df = download_latest_parquet_if_exists(
        connector=connector,
        location=state_folder,
        prefix="progress",
        temp_dir=temp_dir,
    )
    if progress_df is None:
        return empty_progress_df()
    return ensure_progress_df(progress_df)


def _load_existing_results(connector, results_folder: str, temp_dir: str) -> pl.DataFrame:
    chunks = download_all_parquet_by_prefix(
        connector=connector,
        location=results_folder,
        prefix="results",
        temp_dir=temp_dir,
    )
    if not chunks:
        return empty_result_df()
    normalized = [ensure_result_df(chunk) for chunk in chunks]
    return pl.concat(normalized, how="vertical_relaxed")


def _merge_progress(base_progress: pl.DataFrame, new_progress: pl.DataFrame) -> pl.DataFrame:
    if base_progress.is_empty():
        return ensure_progress_df(new_progress)

    combined = pl.concat(
        [ensure_progress_df(base_progress), ensure_progress_df(new_progress)],
        how="vertical_relaxed",
    )
    combined = combined.sort(["unit_id", "updated_at"])
    latest = combined.group_by("unit_id").tail(1)
    return ensure_progress_df(latest)


def _merge_results(base_results: pl.DataFrame, new_results: pl.DataFrame) -> pl.DataFrame:
    if new_results.is_empty():
        return ensure_result_df(base_results)
    if base_results.is_empty():
        return ensure_result_df(new_results)

    combined = pl.concat(
        [ensure_result_df(base_results), ensure_result_df(new_results)],
        how="vertical_relaxed",
    )
    combined = combined.sort(["unit_id"])
    latest = combined.group_by("unit_id").tail(1)
    return ensure_result_df(latest)


def _upload_manifest(connector, folders: dict, temp_dir: str, work_units_df: pl.DataFrame):
    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="manifest",
        df=work_units_df,
        temp_dir=temp_dir,
    )


def _flush_incremental_state(
    connector,
    folders: dict,
    temp_dir: str,
    progress_df: pl.DataFrame,
    result_rows,
    debug_rows,
    pair_status_df: pl.DataFrame,
    partial_output_df: pl.DataFrame,
    metadata: dict,
    runtime: dict,
):
    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="progress",
        df=ensure_progress_df(progress_df),
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

    if runtime["write_pair_status"] and pair_status_df is not None and not pair_status_df.is_empty():
        upload_versioned_parquet(
            connector=connector,
            location=folders["debug_folder"],
            prefix="pair_status",
            df=pair_status_df,
            temp_dir=temp_dir,
        )

    if (
        runtime["write_partial_merged_output"]
        and partial_output_df is not None
        and not partial_output_df.is_empty()
    ):
        upload_versioned_parquet(
            connector=connector,
            location=folders["results_folder"],
            prefix="partial_output",
            df=partial_output_df,
            temp_dir=temp_dir,
        )

    upload_versioned_json(
        connector=connector,
        location=folders["state_folder"],
        prefix="metadata",
        payload=metadata,
        temp_dir=temp_dir,
    )


def _flush_final_state(
    connector,
    folders: dict,
    temp_dir: str,
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
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
        df=ensure_progress_df(progress_df),
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
    source_df = apply_input_row_window(source_df, runtime)

    pipeline_name = runtime_context["pipeline_name"]
    workflow_location = runtime_context["dest_location"]
    dest_connector = runtime_context["dest_connector"]
    temp_dir = runtime_context["temp_dir"]
    user_runtime_config = runtime_context.get("user_runtime_config", {})

    provider_config_key = step_config["provider_config_key"]
    if provider_config_key not in user_runtime_config:
        raise KeyError(
            f"LLM step provider_config_key '{provider_config_key}' not found in USER_RUNTIME_CONFIG_JSON"
        )

    provider_config = user_runtime_config[provider_config_key]
    adapter = build_adapter(step_config["adapter"], provider_config)
    task_handler = build_task_handler(step_config["task_handler"])
    prompt_context = load_prompt_context(step_config)

    folders = ensure_llm_state_layout(
        dest_connector=dest_connector,
        workflow_location=workflow_location,
    )

    work_units_df = build_work_units(source_df, step_config)
    progress_df = _load_existing_progress(dest_connector, folders["state_folder"], temp_dir)
    existing_results_df = _load_existing_results(dest_connector, folders["results_folder"], temp_dir)
    all_results_df = ensure_result_df(existing_results_df)

    terminal_ids = success_or_terminal_unit_ids(progress_df)
    pending_units_df = work_units_df.filter(~pl.col("unit_id").is_in(list(terminal_ids)))

    print(
        (
            f"[llm:{step_config['name']}] start total_units={work_units_df.height} "
            f"already_terminal={len(terminal_ids)} pending_units={pending_units_df.height} "
            f"initial_group_size={runtime['initial_group_size']} min_group_size={runtime['min_group_size']} "
            f"max_group_size={runtime['max_group_size']} flush_scope={runtime['flush_scope']} "
            f"max_flushes_per_run={runtime['max_flushes_per_run']} "
            f"max_concurrent_requests={runtime['max_concurrent_requests']} "
            f"workflow_folder={pipeline_name}"
        ),
        flush=True,
    )

    _upload_manifest(dest_connector, folders, temp_dir, work_units_df)

    if pending_units_df.is_empty():
        outcome = LLMRunOutcome(status="complete", processed_units=0, remaining_units=0)
        metadata = build_runtime_metadata(step_config, work_units_df, progress_df, outcome)
        metadata["workflow_folder"] = pipeline_name
        _flush_final_state(dest_connector, folders, temp_dir, work_units_df, progress_df, metadata)
        return merge_results_back(
            source_df,
            all_results_df,
            step_config["row_id_column"],
            task_handler,
            step_config,
        )

    start_time = time.time()

    def flush_callback(payload: dict):
        nonlocal progress_df
        nonlocal all_results_df

        new_progress_df = progress_rows_to_df(payload["progress_rows"])
        if not new_progress_df.is_empty():
            progress_df = _merge_progress(progress_df, new_progress_df)

        new_results_df = result_rows_to_df(payload["result_rows"])
        if not new_results_df.is_empty():
            all_results_df = _merge_results(all_results_df, new_results_df)

        released_row_ids = released_row_ids_for_flush(
            work_units_df=work_units_df,
            progress_df=progress_df,
            flush_scope=runtime["flush_scope"],
        )

        partial_output_df = build_partial_output_df(
            source_df=source_df,
            all_results_df=all_results_df,
            row_id_column=step_config["row_id_column"],
            task_handler=task_handler,
            step_config=step_config,
            released_row_ids=released_row_ids,
        )

        pair_status_df = build_pair_status_df(
            work_units_df=work_units_df,
            progress_df=progress_df,
            step_config=step_config,
        )

        partial_outcome = LLMRunOutcome(
            status="running",
            processed_units=payload["processed_units"],
            remaining_units=payload["remaining_units"],
        )
        metadata = build_runtime_metadata(
            step_config,
            work_units_df,
            progress_df,
            partial_outcome,
        )
        metadata["current_group_size"] = payload["current_group_size"]
        metadata["current_concurrency"] = payload.get("current_concurrency")
        metadata["current_load_budget"] = payload.get("current_load_budget")
        metadata["released_success_rows"] = len(released_row_ids)
        metadata["workflow_folder"] = pipeline_name

        _flush_incremental_state(
            connector=dest_connector,
            folders=folders,
            temp_dir=temp_dir,
            progress_df=progress_df,
            result_rows=payload["result_rows"],
            debug_rows=payload["debug_rows"],
            pair_status_df=pair_status_df,
            partial_output_df=partial_output_df,
            metadata=metadata,
            runtime=runtime,
        )

        print(
            (
                f"[llm:{step_config['name']}] checkpoint "
                f"processed_units={payload['processed_units']} "
                f"remaining_units={payload['remaining_units']} "
                f"current_group_size={payload['current_group_size']} "
                f"current_concurrency={payload.get('current_concurrency')} "
                f"current_load_budget={payload.get('current_load_budget')} "
                f"progress_rows={len(payload['progress_rows'])} "
                f"result_rows={len(payload['result_rows'])} "
                f"debug_rows={len(payload['debug_rows'])} "
                f"released_success_rows={len(released_row_ids)}"
            ),
            flush=True,
        )

        counted_flush = len(released_row_ids) > 0 or runtime["flush_scope"] == "unit"
        return {
            "released_success_rows": len(released_row_ids),
            "counted_flush": counted_flush,
        }

    batch_out = process_batches(
        pending_rows=pending_units_df.to_dicts(),
        step_config=step_config,
        runtime=runtime,
        adapter=adapter,
        task_handler=task_handler,
        prompt_context=prompt_context,
        progress_df=progress_df,
        start_time=start_time,
        flush_callback=flush_callback,
    )

    metadata = build_runtime_metadata(step_config, work_units_df, progress_df, batch_out["outcome"])
    metadata["workflow_folder"] = pipeline_name
    metadata["completed_flushes"] = batch_out.get("completed_flushes")
    metadata["stop_reason"] = batch_out.get("stop_reason")
    metadata["current_group_size"] = batch_out.get("current_group_size")
    metadata["current_concurrency"] = batch_out.get("current_concurrency")
    metadata["current_load_budget"] = batch_out.get("current_load_budget")

    _flush_final_state(dest_connector, folders, temp_dir, work_units_df, progress_df, metadata)

    print(
        (
            f"[llm:{step_config['name']}] final_flush "
            f"processed_units={batch_out['outcome'].processed_units} "
            f"remaining_units={batch_out['outcome'].remaining_units} "
            f"outcome={batch_out['outcome'].status}"
        ),
        flush=True,
    )

    return merge_results_back(
        source_df,
        all_results_df,
        step_config["row_id_column"],
        task_handler,
        step_config,
    )