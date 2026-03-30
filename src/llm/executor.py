import time

import polars as pl

from .adapter_loader import build_adapter
from .batch_flush import build_progress_status_map, build_row_to_unit_ids
from .batch_processor import process_batches
from .dataframes import debug_rows_to_df, empty_progress_df, empty_result_df, ensure_debug_df, ensure_progress_df, ensure_result_df, progress_rows_to_df, result_rows_to_df
from .llm_task_loader import build_task_handler
from .merge import merge_results_back
from .models import LLMRunOutcome
from .prompt_loader import load_prompt_context
from .runtime import build_runtime_metadata, default_runtime, success_or_terminal_unit_ids, validate_llm_step
from .state import download_all_parquet_by_prefix, download_latest_parquet_if_exists, ensure_llm_state_layout, upload_versioned_json, upload_versioned_parquet
from .work_units import build_work_units


def _load_existing_progress(connector, state_folder: str, temp_dir: str) -> pl.DataFrame:
    progress_df = download_latest_parquet_if_exists(connector=connector, location=state_folder, prefix="progress", temp_dir=temp_dir)
    if progress_df is None:
        return empty_progress_df()
    return ensure_progress_df(progress_df)


def _load_existing_results(connector, results_folder: str, temp_dir: str) -> pl.DataFrame:
    chunks = download_all_parquet_by_prefix(connector=connector, location=results_folder, prefix="results", temp_dir=temp_dir)
    if not chunks:
        return empty_result_df()
    normalized = [ensure_result_df(chunk) for chunk in chunks]
    return pl.concat(normalized, how="vertical_relaxed")


def _merge_progress(base_progress: pl.DataFrame, new_progress: pl.DataFrame) -> pl.DataFrame:
    if base_progress.is_empty():
        return ensure_progress_df(new_progress)
    combined = pl.concat([ensure_progress_df(base_progress), ensure_progress_df(new_progress)], how="vertical_relaxed")
    combined = combined.sort(["unit_id", "updated_at"])
    latest = combined.group_by("unit_id").tail(1)
    return ensure_progress_df(latest)


def _merge_results(base_results: pl.DataFrame, new_results: pl.DataFrame) -> pl.DataFrame:
    if base_results.is_empty():
        return ensure_result_df(new_results)
    combined = pl.concat([ensure_result_df(base_results), ensure_result_df(new_results)], how="vertical_relaxed")
    combined = combined.sort(["unit_id", "row_id", "output_column"])
    latest = combined.group_by(["unit_id", "row_id", "output_column"]).tail(1)
    return ensure_result_df(latest)


def _upload_manifest(connector, folders: dict, temp_dir: str, work_units_df: pl.DataFrame):
    upload_versioned_parquet(connector=connector, location=folders["state_folder"], prefix="manifest", df=work_units_df, temp_dir=temp_dir)


def _flush_incremental_state(connector, folders: dict, temp_dir: str, progress_df: pl.DataFrame, result_rows, debug_rows, metadata: dict, source_df: pl.DataFrame, all_results_df: pl.DataFrame, row_id_column: str, task_handler, step_config: dict, write_partial_merged_output: bool):
    upload_versioned_parquet(connector=connector, location=folders["state_folder"], prefix="progress", df=ensure_progress_df(progress_df), temp_dir=temp_dir)
    if result_rows:
        upload_versioned_parquet(connector=connector, location=folders["results_folder"], prefix="results", df=result_rows_to_df(result_rows), temp_dir=temp_dir)
    if debug_rows:
        debug_df = ensure_debug_df(debug_rows_to_df(debug_rows))
        upload_versioned_parquet(connector=connector, location=folders["debug_folder"], prefix="traces", df=debug_df, temp_dir=temp_dir)
        review_df = debug_df.filter((pl.col("status") != "success") | (pl.col("review_flag") == True))
        if not review_df.is_empty():
            upload_versioned_parquet(connector=connector, location=folders["debug_folder"], prefix="review", df=review_df, temp_dir=temp_dir)
    if write_partial_merged_output:
        merged_snapshot = merge_results_back(source_df=source_df, all_results_df=ensure_result_df(all_results_df), row_id_column=row_id_column, task_handler=task_handler, step_config=step_config)
        upload_versioned_parquet(connector=connector, location=folders["results_folder"], prefix="partial_output", df=merged_snapshot, temp_dir=temp_dir)
    upload_versioned_json(connector=connector, location=folders["state_folder"], prefix="metadata", payload=metadata, temp_dir=temp_dir)


def _flush_final_state(connector, folders: dict, temp_dir: str, work_units_df: pl.DataFrame, progress_df: pl.DataFrame, metadata: dict):
    upload_versioned_parquet(connector=connector, location=folders["state_folder"], prefix="manifest", df=work_units_df, temp_dir=temp_dir)
    upload_versioned_parquet(connector=connector, location=folders["state_folder"], prefix="progress", df=ensure_progress_df(progress_df), temp_dir=temp_dir)
    upload_versioned_json(connector=connector, location=folders["state_folder"], prefix="metadata", payload=metadata, temp_dir=temp_dir)


def _apply_input_subset(source_df: pl.DataFrame, runtime: dict, step_name: str) -> pl.DataFrame:
    offset = int(runtime["input_row_offset"] or 0)
    limit = runtime["input_row_limit"]
    if limit is None and offset == 0:
        return source_df
    original_height = source_df.height
    if limit is None:
        subset_df = source_df.slice(offset)
        print(f"[llm:{step_name}] input_subset offset={offset} limit=None selected_rows={subset_df.height}/{original_height}", flush=True)
        return subset_df
    subset_df = source_df.slice(offset, int(limit))
    print(f"[llm:{step_name}] input_subset offset={offset} limit={int(limit)} selected_rows={subset_df.height}/{original_height}", flush=True)
    return subset_df


def execute_llm_step(data, step_config: dict, runtime_context: dict):
    validate_llm_step(step_config)
    source_df = data.collect(streaming=True) if isinstance(data, pl.LazyFrame) else data
    runtime = default_runtime(step_config)
    source_df = _apply_input_subset(source_df, runtime, step_config["name"])
    pipeline_name = runtime_context["pipeline_name"]
    dest_connector = runtime_context["dest_connector"]
    dest_location = runtime_context["dest_location"]
    temp_dir = runtime_context["temp_dir"]
    user_runtime_config = runtime_context.get("user_runtime_config", {})

    provider_config_key = step_config["provider_config_key"]
    if provider_config_key not in user_runtime_config:
        raise KeyError(f"LLM step provider_config_key '{provider_config_key}' not found in USER_RUNTIME_CONFIG_JSON")

    provider_config = user_runtime_config[provider_config_key]
    adapter = build_adapter(step_config["adapter"], provider_config)
    task_handler = build_task_handler(step_config["task_handler"])
    prompt_context = load_prompt_context(step_config)

    folders = ensure_llm_state_layout(dest_connector=dest_connector, dest_location=dest_location, pipeline_name=pipeline_name, step_name=step_config["name"])

    work_units_df = build_work_units(source_df, step_config)
    work_units_rows = work_units_df.to_dicts()
    row_to_unit_ids = build_row_to_unit_ids(work_units_rows)

    progress_df = _load_existing_progress(connector=dest_connector, state_folder=folders["state_folder"], temp_dir=temp_dir)
    current_results_df = _load_existing_results(connector=dest_connector, results_folder=folders["results_folder"], temp_dir=temp_dir)

    terminal_ids = success_or_terminal_unit_ids(progress_df)
    pending_units_df = work_units_df.filter(~pl.col("unit_id").is_in(list(terminal_ids)))
    progress_status_by_unit = build_progress_status_map(progress_df)

    print((
        f"[llm:{step_config['name']}] start total_units={work_units_df.height} already_terminal={len(terminal_ids)} "
        f"pending_units={pending_units_df.height} initial_group_size={runtime['initial_group_size']} "
        f"min_group_size={runtime['min_group_size']} max_group_size={runtime['max_group_size']} "
        f"flush_scope={runtime['flush_scope']} max_flushes_per_run={runtime['max_flushes_per_run']}"
    ), flush=True)

    _upload_manifest(connector=dest_connector, folders=folders, temp_dir=temp_dir, work_units_df=work_units_df)

    if pending_units_df.is_empty():
        outcome = LLMRunOutcome(status="complete", processed_units=0, remaining_units=0)
        metadata = build_runtime_metadata(step_config=step_config, work_units_df=work_units_df, progress_df=progress_df, outcome=outcome)
        _flush_final_state(connector=dest_connector, folders=folders, temp_dir=temp_dir, work_units_df=work_units_df, progress_df=progress_df, metadata=metadata)
        return merge_results_back(source_df=source_df, all_results_df=current_results_df, row_id_column=step_config["row_id_column"], task_handler=task_handler, step_config=step_config)

    start_time = time.time()

    def flush_callback(payload: dict):
        nonlocal progress_df, current_results_df
        new_progress_df = progress_rows_to_df(payload["progress_rows"])
        if not new_progress_df.is_empty():
            progress_df = _merge_progress(progress_df, new_progress_df)
        new_results_df = result_rows_to_df(payload["result_rows"])
        if not new_results_df.is_empty():
            current_results_df = _merge_results(current_results_df, new_results_df)
        partial_outcome = LLMRunOutcome(status="running", processed_units=payload["processed_units"], remaining_units=payload["remaining_units"])
        metadata = build_runtime_metadata(step_config=step_config, work_units_df=work_units_df, progress_df=progress_df, outcome=partial_outcome)
        metadata["current_group_size"] = payload["current_group_size"]
        metadata["success_streak"] = payload["success_streak"]
        metadata["released_success_row_count"] = len(payload.get("released_success_row_ids", []))
        metadata["released_terminal_row_count"] = len(payload.get("released_terminal_row_ids", []))
        _flush_incremental_state(connector=dest_connector, folders=folders, temp_dir=temp_dir, progress_df=progress_df, result_rows=payload["result_rows"], debug_rows=payload["debug_rows"], metadata=metadata, source_df=source_df, all_results_df=current_results_df, row_id_column=step_config["row_id_column"], task_handler=task_handler, step_config=step_config, write_partial_merged_output=runtime["write_partial_merged_output"])
        print((
            f"[llm:{step_config['name']}] checkpoint processed_units={payload['processed_units']} remaining_units={payload['remaining_units']} "
            f"current_group_size={payload['current_group_size']} progress_rows={len(payload['progress_rows'])} "
            f"result_rows={len(payload['result_rows'])} debug_rows={len(payload['debug_rows'])} "
            f"released_success_rows={len(payload.get('released_success_row_ids', []))}"
        ), flush=True)
        return len(payload.get("released_success_row_ids", [])) > 0 if runtime["flush_scope"] == "row_complete" else True

    batch_out = process_batches(pending_rows=pending_units_df.to_dicts(), step_config=step_config, runtime=runtime, adapter=adapter, task_handler=task_handler, prompt_context=prompt_context, progress_df=progress_df, start_time=start_time, flush_callback=flush_callback, row_to_unit_ids=row_to_unit_ids, progress_status_by_unit=progress_status_by_unit)

    metadata = build_runtime_metadata(step_config=step_config, work_units_df=work_units_df, progress_df=progress_df, outcome=batch_out["outcome"])
    metadata["completed_flushes"] = batch_out.get("completed_flushes", 0)
    _flush_final_state(connector=dest_connector, folders=folders, temp_dir=temp_dir, work_units_df=work_units_df, progress_df=progress_df, metadata=metadata)
    print(f"[llm:{step_config['name']}] final_flush processed_units={batch_out['outcome'].processed_units} remaining_units={batch_out['outcome'].remaining_units} outcome={batch_out['outcome'].status}", flush=True)
    return merge_results_back(source_df=source_df, all_results_df=current_results_df, row_id_column=step_config["row_id_column"], task_handler=task_handler, step_config=step_config)
