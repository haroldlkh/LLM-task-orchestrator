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
    ensure_debug_df,
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
    success_unit_ids,
    validate_llm_step,
)
from .state import (
    cleanup_llm_artifacts,
    ensure_llm_state_layout,
    upload_versioned_json,
    upload_versioned_parquet,
    utc_now_run_id,
)
from .work_units import build_manifest_df, build_work_units


def _load_latest_parquet(connector, folder_id: str, prefix: str, temp_dir: str):
    candidates = [
        obj
        for obj in connector.list_objects(folder_id)
        if obj["name"].endswith(".parquet") and obj["name"].startswith(prefix)
    ]
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x["name"])
    latest = candidates[-1]
    local_path = f"{temp_dir}/{latest['name']}"
    connector.download_object(latest["id"], local_path)
    return pl.read_parquet(local_path)


def _load_existing_progress(connector, state_folder: str, temp_dir: str) -> pl.DataFrame:
    df = _load_latest_parquet(connector, state_folder, "progress_", temp_dir)
    if df is None:
        return ensure_progress_df(pl.DataFrame())
    return ensure_progress_df(df)


def _load_existing_results(connector, results_folder: str, temp_dir: str) -> pl.DataFrame:
    objects = sorted(connector.list_objects(results_folder), key=lambda x: x["name"])

    checkpoint_obj = None
    for obj in objects:
        if obj["name"].endswith('.parquet') and obj["name"].startswith('results_checkpoint'):
            checkpoint_obj = obj

    checkpoint_df = None
    checkpoint_name = None
    if checkpoint_obj is not None:
        checkpoint_name = checkpoint_obj['name']
        local_path = f"{temp_dir}/{checkpoint_name}"
        connector.download_object(checkpoint_obj['id'], local_path)
        checkpoint_df = ensure_result_df(pl.read_parquet(local_path))

    delta_parts = []
    for obj in objects:
        name = obj['name']
        if not name.endswith('.parquet'):
            continue
        if name.startswith('results_delta') and (checkpoint_name is None or name > checkpoint_name):
            local_path = f"{temp_dir}/{name}"
            connector.download_object(obj['id'], local_path)
            delta_parts.append(pl.read_parquet(local_path))

    if checkpoint_df is None and not delta_parts:
        legacy_snapshot_df = _load_latest_parquet(connector, results_folder, 'results_snapshot_', temp_dir)
        if legacy_snapshot_df is not None:
            return ensure_result_df(legacy_snapshot_df)

        parts = []
        for obj in objects:
            name = obj['name']
            if name.endswith('.parquet') and name.startswith('results_') and not name.startswith('results_delta') and not name.startswith('results_checkpoint'):
                local_path = f"{temp_dir}/{name}"
                connector.download_object(obj['id'], local_path)
                parts.append(pl.read_parquet(local_path))
        if not parts:
            return ensure_result_df(pl.DataFrame())
        return ensure_result_df(pl.concat(parts, how='vertical_relaxed'))

    merged = checkpoint_df if checkpoint_df is not None else ensure_result_df(pl.DataFrame())
    for delta_df in delta_parts:
        merged = _merge_results(merged, ensure_result_df(delta_df))
    return ensure_result_df(merged)


def _load_existing_debug_snapshot(connector, debug_folder: str, prefix: str, temp_dir: str) -> pl.DataFrame:
    df = _load_latest_parquet(connector, debug_folder, f"{prefix}_", temp_dir)
    if df is None:
        return ensure_debug_df(pl.DataFrame())
    return ensure_debug_df(df)

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
    ).with_row_index("_merge_order")
    latest = (
        combined
        .group_by(["unit_id", "output_column"])
        .agg(pl.all().sort_by("_merge_order").last())
        .drop("_merge_order")
    )
    return ensure_result_df(latest)


def _progress_snapshot_outcome(
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    all_results_df: pl.DataFrame,
    status: str,
) -> LLMRunOutcome:
    total_units = int(work_units_df.height)
    completed_ids = _success_unit_ids_from_results(all_results_df)
    processed_units = min(len(completed_ids), total_units)
    remaining_units = max(total_units - processed_units, 0)

    return LLMRunOutcome(
        status=status,
        processed_units=processed_units,
        remaining_units=remaining_units,
    )


def _success_unit_ids_from_results(results_df: pl.DataFrame) -> set[str]:
    if results_df is None or results_df.is_empty():
        return set()
    success_df = results_df.filter(pl.col("status") == "success")
    if success_df.is_empty():
        return set()
    return set(success_df["unit_id"].drop_nulls().to_list())


def _reconcile_progress_with_results(progress_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:
    progress_df = ensure_progress_df(progress_df)
    result_success_ids = _success_unit_ids_from_results(results_df)
    if not result_success_ids:
        return progress_df

    existing_success_ids = success_unit_ids(progress_df)
    missing_ids = sorted(result_success_ids - existing_success_ids)
    if not missing_ids:
        return progress_df

    synthetic_rows = [
        {
            "unit_id": unit_id,
            "status": "success",
            "retry_count": 0,
            "last_error_type": None,
            "last_error_message": None,
            "updated_at": utc_now_run_id(),
        }
        for unit_id in missing_ids
    ]
    return _merge_progress(progress_df, progress_rows_to_df(synthetic_rows))


def _merge_debug_latest(base_debug_df: pl.DataFrame, new_debug_df: pl.DataFrame) -> pl.DataFrame:
    if new_debug_df is None or new_debug_df.is_empty():
        return ensure_debug_df(base_debug_df)
    if base_debug_df is None or base_debug_df.is_empty():
        return ensure_debug_df(new_debug_df)

    combined = pl.concat(
        [ensure_debug_df(base_debug_df), ensure_debug_df(new_debug_df)],
        how="vertical_relaxed",
    ).with_row_index("_merge_order")
    latest = (
        combined
        .group_by("unit_id")
        .agg(pl.all().sort_by("_merge_order").last())
        .drop("_merge_order")
    )
    return ensure_debug_df(latest)


def _upload_manifest(connector, folders: dict, temp_dir: str, manifest_df: pl.DataFrame, run_id: str):
    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="manifest",
        df=manifest_df,
        temp_dir=temp_dir,
        run_id=run_id,
    )


def _flush_canonical_state(
    connector,
    folders: dict,
    temp_dir: str,
    progress_df: pl.DataFrame,
    new_results_df: pl.DataFrame,
    debug_rows,
    cumulative_permanent_review_df: pl.DataFrame,
    metadata: dict,
    run_id: str,
):
    written_prefixes: dict[str, set[str]] = {
        "state_folder": set(),
        "results_folder": set(),
        "debug_folder": set(),
    }

    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="progress",
        df=ensure_progress_df(progress_df),
        temp_dir=temp_dir,
        run_id=run_id,
    )
    written_prefixes["state_folder"].add("progress")

    if new_results_df is not None and not new_results_df.is_empty():
        upload_versioned_parquet(
            connector=connector,
            location=folders["results_folder"],
            prefix="results_delta",
            df=ensure_result_df(new_results_df),
            temp_dir=temp_dir,
            run_id=run_id,
        )
        written_prefixes["results_folder"].add("results_delta")

    if debug_rows:
        debug_df = debug_rows_to_df(debug_rows)
        upload_versioned_parquet(
            connector=connector,
            location=folders["debug_folder"],
            prefix="traces",
            df=debug_df,
            temp_dir=temp_dir,
            run_id=run_id,
        )
        written_prefixes["debug_folder"].add("traces")

        review_df = debug_df.filter((pl.col("status") != "success") | (pl.col("review_flag") == True))
        if not review_df.is_empty():
            upload_versioned_parquet(
                connector=connector,
                location=folders["debug_folder"],
                prefix="review",
                df=review_df,
                temp_dir=temp_dir,
                run_id=run_id,
            )
            written_prefixes["debug_folder"].add("review")

    if cumulative_permanent_review_df is not None and not cumulative_permanent_review_df.is_empty():
        upload_versioned_parquet(
            connector=connector,
            location=folders["debug_folder"],
            prefix="permanent_review",
            df=ensure_debug_df(cumulative_permanent_review_df),
            temp_dir=temp_dir,
            run_id=run_id,
        )
        written_prefixes["debug_folder"].add("permanent_review")

    upload_versioned_json(
        connector=connector,
        location=folders["state_folder"],
        prefix="metadata",
        payload=metadata,
        temp_dir=temp_dir,
        run_id=run_id,
    )
    written_prefixes["state_folder"].add("metadata")
    return written_prefixes


def _flush_results_checkpoint(
    connector,
    folders: dict,
    temp_dir: str,
    all_results_df: pl.DataFrame,
    run_id: str,
):
    written_prefixes: dict[str, set[str]] = {
        "state_folder": set(),
        "results_folder": set(),
        "debug_folder": set(),
    }
    upload_versioned_parquet(
        connector=connector,
        location=folders["results_folder"],
        prefix="results_checkpoint",
        df=ensure_result_df(all_results_df),
        temp_dir=temp_dir,
        run_id=run_id,
    )
    written_prefixes["results_folder"].add("results_checkpoint")
    return written_prefixes


def _flush_materialized_views(
    connector,
    folders: dict,
    temp_dir: str,
    pair_status_df: pl.DataFrame,
    partial_output_df: pl.DataFrame,
    runtime: dict,
    run_id: str,
):
    written_prefixes: dict[str, set[str]] = {
        "state_folder": set(),
        "results_folder": set(),
        "debug_folder": set(),
    }

    if runtime["write_pair_status"] and pair_status_df is not None and not pair_status_df.is_empty():
        upload_versioned_parquet(
            connector=connector,
            location=folders["debug_folder"],
            prefix="pair_status",
            df=pair_status_df,
            temp_dir=temp_dir,
            run_id=run_id,
        )
        written_prefixes["debug_folder"].add("pair_status")

    if runtime["write_partial_merged_output"] and partial_output_df is not None and not partial_output_df.is_empty():
        upload_versioned_parquet(
            connector=connector,
            location=folders["results_folder"],
            prefix="partial_output",
            df=partial_output_df,
            temp_dir=temp_dir,
            run_id=run_id,
        )
        written_prefixes["results_folder"].add("partial_output")

    return written_prefixes


def _flush_final_state(
    connector,
    folders: dict,
    temp_dir: str,
    manifest_df: pl.DataFrame,
    metadata: dict,
    run_id: str,
):
    written_prefixes = {"state_folder": set(), "results_folder": set(), "debug_folder": set()}
    upload_versioned_parquet(
        connector=connector,
        location=folders["state_folder"],
        prefix="manifest",
        df=manifest_df,
        temp_dir=temp_dir,
        run_id=run_id,
    )
    written_prefixes["state_folder"].add("manifest")
    upload_versioned_json(
        connector=connector,
        location=folders["state_folder"],
        prefix="metadata",
        payload=metadata,
        temp_dir=temp_dir,
        run_id=run_id,
    )
    written_prefixes["state_folder"].add("metadata")
    return written_prefixes


def _merge_written_prefixes(*prefix_maps):
    merged = {"state_folder": set(), "results_folder": set(), "debug_folder": set()}
    for prefix_map in prefix_maps:
        if not prefix_map:
            continue
        for folder_key, prefixes in prefix_map.items():
            merged.setdefault(folder_key, set()).update(prefixes)
    return merged


def _build_materialized_views(source_df, work_units_df, progress_df, all_results_df, step_config, task_handler, release_mode: str):
    released_row_ids = released_row_ids_for_flush(
        work_units_df=work_units_df,
        progress_df=progress_df,
        release_mode=release_mode,
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
        released_row_ids=released_row_ids,
    )
    return released_row_ids, partial_output_df, pair_status_df


def _should_materialize(flush_count: int, last_materialize_at: float | None, runtime: dict, force: bool = False) -> bool:
    if force:
        return True
    if flush_count <= 0:
        return False
    if flush_count % int(runtime.get("materialize_every_n_flushes", 5)) == 0:
        return True
    if last_materialize_at is None:
        return False
    return (time.time() - last_materialize_at) >= float(runtime.get("materialize_every_n_seconds", 60))


def _should_checkpoint(flush_count: int, last_checkpoint_at: float | None, runtime: dict, force: bool = False) -> bool:
    if force:
        return True
    if flush_count <= 0:
        return False
    if flush_count % int(runtime.get("checkpoint_every_n_flushes", 10)) == 0:
        return True
    if last_checkpoint_at is None:
        return False
    return (time.time() - last_checkpoint_at) >= float(runtime.get("checkpoint_every_n_seconds", 300))

def execute_llm_step(data, step_config: dict, runtime_context: dict):
    validate_llm_step(step_config)
    source_df = data.collect(streaming=True) if isinstance(data, pl.LazyFrame) else data
    runtime = default_runtime(step_config)
    source_df = apply_input_row_window(source_df, runtime)
    run_id = utc_now_run_id()

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

    pool_size = int(getattr(adapter, "lane_count", getattr(adapter, "pool_size", 1)) or 1)
    runtime["available_lane_count"] = max(pool_size, 1)

    task_handler = build_task_handler(step_config["task_handler"])
    prompt_context = load_prompt_context(step_config)
    folders = ensure_llm_state_layout(
        dest_connector=dest_connector,
        workflow_location=workflow_location,
    )

    work_units_df = build_work_units(source_df, step_config)
    manifest_df = build_manifest_df(work_units_df)
    progress_df = _load_existing_progress(dest_connector, folders["state_folder"], temp_dir)
    existing_results_df = _load_existing_results(dest_connector, folders["results_folder"], temp_dir)
    all_results_df = ensure_result_df(existing_results_df)
    progress_df = _reconcile_progress_with_results(progress_df, all_results_df)
    cumulative_permanent_review_df = _load_existing_debug_snapshot(
        dest_connector,
        folders["debug_folder"],
        "permanent_review",
        temp_dir,
    )

    completed_ids = _success_unit_ids_from_results(all_results_df)
    pending_units_df = work_units_df.filter(~pl.col("unit_id").is_in(list(completed_ids)))

    release_mode = "strict_row_success" if runtime["flush_scope"] == "row_complete" else "unit_partial"
    print(
        (
            f"[llm:{step_config['name']}] start total_units={work_units_df.height} already_success={len(completed_ids)} "
            f"pending_units={pending_units_df.height} existing_permanent_reviews={cumulative_permanent_review_df.height} initial_group_size={runtime['initial_group_size']} "
            f"min_group_size={runtime['min_group_size']} max_group_size={runtime['max_group_size']} "
            f"release_mode={release_mode} max_flushes_per_run={runtime['max_flushes_per_run']} "
            f"max_concurrent_requests={runtime['max_concurrent_requests']} workflow_folder={pipeline_name}"
        ),
        flush=True,
    )

    _upload_manifest(dest_connector, folders, temp_dir, manifest_df, run_id)

    if pending_units_df.is_empty():
        _, partial_output_df, pair_status_df = _build_materialized_views(
            source_df=source_df,
            work_units_df=work_units_df,
            progress_df=progress_df,
            all_results_df=all_results_df,
            step_config=step_config,
            task_handler=task_handler,
            release_mode=release_mode,
        )
        successful_unit_ids = _success_unit_ids_from_results(all_results_df)
        if successful_unit_ids and cumulative_permanent_review_df is not None and not cumulative_permanent_review_df.is_empty():
            cumulative_permanent_review_df = cumulative_permanent_review_df.filter(
                ~pl.col("unit_id").is_in(list(successful_unit_ids))
            )
        outcome = _progress_snapshot_outcome(
            work_units_df=work_units_df,
            progress_df=progress_df,
            all_results_df=all_results_df,
            status="complete",
        )
        metadata = build_runtime_metadata(step_config, work_units_df, progress_df, outcome, all_results_df=all_results_df)
        metadata["workflow_folder"] = pipeline_name
        print(f"[llm:{step_config['name']}] flush_canonical_start reason=no_pending", flush=True)
        canonical_written = _flush_canonical_state(
            dest_connector,
            folders,
            temp_dir,
            progress_df,
            ensure_result_df(pl.DataFrame()),
            [],
            cumulative_permanent_review_df,
            metadata,
            run_id,
        )
        checkpoint_written = _flush_results_checkpoint(
            dest_connector,
            folders,
            temp_dir,
            all_results_df,
            run_id,
        )
        print(f"[llm:{step_config['name']}] flush_canonical_done reason=no_pending", flush=True)
        print(f"[llm:{step_config['name']}] flush_materialized_start reason=no_pending", flush=True)
        materialized_written = _flush_materialized_views(
            dest_connector,
            folders,
            temp_dir,
            pair_status_df,
            partial_output_df,
            runtime,
            run_id,
        )
        print(f"[llm:{step_config['name']}] flush_materialized_done reason=no_pending", flush=True)
        final_written_prefixes = _flush_final_state(dest_connector, folders, temp_dir, manifest_df, metadata, run_id)
        written_prefixes = _merge_written_prefixes(canonical_written, checkpoint_written, materialized_written, final_written_prefixes)
        cleanup_summary = cleanup_llm_artifacts(dest_connector, folders, runtime, include_final_outputs=False, new_artifact_prefixes=written_prefixes)
        print(f"[llm:{step_config['name']}] cleanup_done reason=run_end final_outputs_deleted={cleanup_summary['final_outputs']} artifacts_deleted={cleanup_summary['artifacts']}", flush=True)
        return merge_results_back(
            source_df,
            all_results_df,
            step_config["row_id_column"],
            task_handler,
            step_config,
        )

    start_time = time.time()
    flush_count = 0
    last_materialize_at = time.time()
    last_checkpoint_at = time.time()

    def flush_callback(payload: dict):
        nonlocal flush_count
        nonlocal last_materialize_at
        nonlocal last_checkpoint_at
        nonlocal progress_df
        nonlocal all_results_df
        nonlocal cumulative_permanent_review_df

        new_progress_df = progress_rows_to_df(payload["progress_rows"])
        if not new_progress_df.is_empty():
            progress_df = _merge_progress(progress_df, new_progress_df)

        new_results_df = result_rows_to_df(payload["result_rows"])
        if not new_results_df.is_empty():
            all_results_df = _merge_results(all_results_df, new_results_df)


        debug_df = debug_rows_to_df(payload["debug_rows"])
        if not debug_df.is_empty():
            permanent_from_flush = debug_df.filter(pl.col("status") == "permanent_error")
            if not permanent_from_flush.is_empty():
                cumulative_permanent_review_df = _merge_debug_latest(
                    cumulative_permanent_review_df,
                    permanent_from_flush,
                )

        successful_unit_ids = _success_unit_ids_from_results(all_results_df)
        if successful_unit_ids and cumulative_permanent_review_df is not None and not cumulative_permanent_review_df.is_empty():
            cumulative_permanent_review_df = cumulative_permanent_review_df.filter(
                ~pl.col("unit_id").is_in(list(successful_unit_ids))
            )

        flush_count += 1
        running_outcome = _progress_snapshot_outcome(
            work_units_df=work_units_df,
            progress_df=progress_df,
            all_results_df=all_results_df,
            status="running",
        )
        metadata = build_runtime_metadata(step_config, work_units_df, progress_df, running_outcome, all_results_df=all_results_df)
        metadata["workflow_folder"] = pipeline_name
        print(f"[llm:{step_config['name']}] flush_canonical_start flush_count={flush_count}", flush=True)
        canonical_written = _flush_canonical_state(
            dest_connector,
            folders,
            temp_dir,
            progress_df,
            new_results_df,
            payload["debug_rows"],
            cumulative_permanent_review_df,
            metadata,
            run_id,
        )
        print(f"[llm:{step_config['name']}] flush_canonical_done flush_count={flush_count}", flush=True)

        written_prefixes = canonical_written
        if _should_checkpoint(flush_count, last_checkpoint_at, runtime):
            print(f"[llm:{step_config['name']}] flush_checkpoint_start flush_count={flush_count}", flush=True)
            checkpoint_written = _flush_results_checkpoint(
                dest_connector,
                folders,
                temp_dir,
                all_results_df,
                run_id,
            )
            last_checkpoint_at = time.time()
            print(f"[llm:{step_config['name']}] flush_checkpoint_done flush_count={flush_count}", flush=True)
            written_prefixes = _merge_written_prefixes(written_prefixes, checkpoint_written)
        if _should_materialize(flush_count, last_materialize_at, runtime):
            print(f"[llm:{step_config['name']}] flush_materialized_start flush_count={flush_count}", flush=True)
            _, partial_output_df, pair_status_df = _build_materialized_views(
                source_df=source_df,
                work_units_df=work_units_df,
                progress_df=progress_df,
                all_results_df=all_results_df,
                step_config=step_config,
                task_handler=task_handler,
                release_mode=release_mode,
            )
            materialized_written = _flush_materialized_views(
                dest_connector,
                folders,
                temp_dir,
                pair_status_df,
                partial_output_df,
                runtime,
                run_id,
            )
            last_materialize_at = time.time()
            print(f"[llm:{step_config['name']}] flush_materialized_done flush_count={flush_count}", flush=True)
            written_prefixes = _merge_written_prefixes(canonical_written, materialized_written)

        if runtime.get("artifact_retention_mode", "standard") == "standard":
            print(f"[llm:{step_config['name']}] cleanup_start flush_count={flush_count}", flush=True)
            cleanup_summary = cleanup_llm_artifacts(
                dest_connector,
                folders,
                runtime,
                include_final_outputs=False,
                new_artifact_prefixes=written_prefixes,
            )
            print(
                f"[llm:{step_config['name']}] cleanup_done flush_count={flush_count} artifacts_deleted={cleanup_summary['artifacts']}",
                flush=True,
            )
        return {"counted_flush": True}

    batch_out = process_batches(
        pending_rows=pending_units_df.iter_rows(named=True),
        step_config=step_config,
        runtime=runtime,
        adapter=adapter,
        task_handler=task_handler,
        prompt_context=prompt_context,
        progress_df=progress_df,
        start_time=start_time,
        flush_callback=flush_callback,
    )

    _, partial_output_df, pair_status_df = _build_materialized_views(
        source_df=source_df,
        work_units_df=work_units_df,
        progress_df=progress_df,
        all_results_df=all_results_df,
        step_config=step_config,
        task_handler=task_handler,
        release_mode=release_mode,
    )

    successful_unit_ids = _success_unit_ids_from_results(all_results_df)
    if successful_unit_ids and cumulative_permanent_review_df is not None and not cumulative_permanent_review_df.is_empty():
        cumulative_permanent_review_df = cumulative_permanent_review_df.filter(
            ~pl.col("unit_id").is_in(list(successful_unit_ids))
        )

    metadata = build_runtime_metadata(step_config, work_units_df, progress_df, batch_out["outcome"], all_results_df=all_results_df)
    metadata["workflow_folder"] = pipeline_name
    print(f"[llm:{step_config['name']}] flush_canonical_start reason=run_end", flush=True)
    canonical_written = _flush_canonical_state(
        dest_connector,
        folders,
        temp_dir,
        progress_df,
        ensure_result_df(pl.DataFrame()),
        [],
        cumulative_permanent_review_df,
        metadata,
        run_id,
    )
    print(f"[llm:{step_config['name']}] flush_canonical_done reason=run_end", flush=True)
    print(f"[llm:{step_config['name']}] flush_checkpoint_start reason=run_end", flush=True)
    checkpoint_written = _flush_results_checkpoint(
        dest_connector,
        folders,
        temp_dir,
        all_results_df,
        run_id,
    )
    print(f"[llm:{step_config['name']}] flush_checkpoint_done reason=run_end", flush=True)
    print(f"[llm:{step_config['name']}] flush_materialized_start reason=run_end", flush=True)
    materialized_written = _flush_materialized_views(
        dest_connector,
        folders,
        temp_dir,
        pair_status_df,
        partial_output_df,
        runtime,
        run_id,
    )
    print(f"[llm:{step_config['name']}] flush_materialized_done reason=run_end", flush=True)
    final_written_prefixes = _flush_final_state(dest_connector, folders, temp_dir, manifest_df, metadata, run_id)
    written_prefixes = _merge_written_prefixes(canonical_written, checkpoint_written, materialized_written, final_written_prefixes)

    print(f"[llm:{step_config['name']}] cleanup_start reason=run_end", flush=True)
    cleanup_summary = cleanup_llm_artifacts(dest_connector, folders, runtime, include_final_outputs=False, new_artifact_prefixes=written_prefixes)
    print(f"[llm:{step_config['name']}] cleanup final_outputs_deleted={cleanup_summary['final_outputs']} artifacts_deleted={cleanup_summary['artifacts']}", flush=True)

    final_merged = merge_results_back(
        source_df,
        all_results_df,
        step_config["row_id_column"],
        task_handler,
        step_config,
    )
    return final_merged