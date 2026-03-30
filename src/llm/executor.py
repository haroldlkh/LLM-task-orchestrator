import hashlib
import json
import time
from typing import List

import polars as pl

from .adapter_loader import build_adapter
from .models import LLMProgressRecord, LLMResultRecord, LLMRunOutcome, LLMWorkUnit
from .prompt_loader import load_prompt_context
from .state import (
    download_all_parquet_by_prefix,
    download_latest_parquet_if_exists,
    ensure_llm_state_layout,
    upload_versioned_json,
    upload_versioned_parquet,
    utc_now_iso,
)
from .llm_task_loader import build_task_handler
from .validators import (
    validate_adapter_batch_results,
    validate_task_parse_result,
)


def _hash_unit_id(parts: List[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _default_runtime(step_config: dict) -> dict:
    runtime = step_config.get("runtime", {})
    return {
        "max_units_per_batch": runtime.get("max_units_per_batch", 10),
        "flush_every_n_units": runtime.get("flush_every_n_units", 50),
        "soft_time_limit_minutes": runtime.get("soft_time_limit_minutes", 40),
        "max_request_retries": runtime.get("max_request_retries", 3),
        "retry_backoff_seconds": runtime.get("retry_backoff_seconds", 5),
    }


def _validate_llm_step(step_config: dict):
    required = [
        "name",
        "adapter",
        "task_handler",
        "provider_config_key",
        "model",
        "row_id_column",
        "input_columns",
        "output_columns",
    ]
    missing = [k for k in required if k not in step_config]
    if missing:
        raise KeyError(f"LLM step missing required keys: {missing}")

    if not isinstance(step_config["input_columns"], list) or not step_config["input_columns"]:
        raise ValueError("LLM step 'input_columns' must be a non-empty list")

    if not isinstance(step_config["output_columns"], dict) or not step_config["output_columns"]:
        raise ValueError("LLM step 'output_columns' must be a non-empty dict")

    for col in step_config["input_columns"]:
        if col not in step_config["output_columns"]:
            raise ValueError(
                f"LLM step input column '{col}' missing from output_columns mapping"
            )


def _build_work_units(data: pl.DataFrame, step_config: dict) -> pl.DataFrame:
    row_id_column = step_config["row_id_column"]
    model = step_config["model"]
    step_name = step_config["name"]
    prompt_version = step_config.get("kwargs", {}).get("prompt_version", "v1")

    units = []

    for row in data.iter_rows(named=True):
        row_id = str(row[row_id_column])

        for input_col in step_config["input_columns"]:
            input_text = row.get(input_col)
            if input_text is None:
                continue

            input_text = str(input_text).strip()
            if not input_text:
                continue

            output_column = step_config["output_columns"][input_col]
            unit_id = _hash_unit_id([
                step_name,
                row_id,
                input_col,
                output_column,
                model,
                prompt_version,
            ])

            unit = LLMWorkUnit(
                unit_id=unit_id,
                row_id=row_id,
                field_name=input_col,
                input_text=input_text,
                output_column=output_column,
                model=model,
                step_name=step_name,
                prompt_version=prompt_version,
                step_kwargs=step_config.get("kwargs", {}),
            )
            units.append(unit.to_dict())

    if not units:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "field_name": pl.Utf8,
                "input_text": pl.Utf8,
                "output_column": pl.Utf8,
                "model": pl.Utf8,
                "step_name": pl.Utf8,
                "prompt_version": pl.Utf8,
            }
        )

    return pl.DataFrame(units)


def _load_existing_progress(connector, state_folder: str, temp_dir: str) -> pl.DataFrame:
    progress_df = download_latest_parquet_if_exists(
        connector=connector,
        location=state_folder,
        prefix="progress",
        temp_dir=temp_dir,
    )
    if progress_df is None:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "status": pl.Utf8,
                "retry_count": pl.Int64,
                "last_error_type": pl.Utf8,
                "last_error_message": pl.Utf8,
                "updated_at": pl.Utf8,
            }
        )
    return progress_df


def _load_existing_results(connector, results_folder: str, temp_dir: str) -> pl.DataFrame:
    chunks = download_all_parquet_by_prefix(
        connector=connector,
        location=results_folder,
        prefix="results",
        temp_dir=temp_dir,
    )
    if not chunks:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "output_column": pl.Utf8,
                "status": pl.Utf8,
                "parsed_output": pl.Utf8,
                "output_value": pl.Utf8,
                "raw_output": pl.Utf8,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
            }
        )
    return pl.concat(chunks, how="vertical_relaxed")


def _merge_progress(base_progress: pl.DataFrame, new_progress: pl.DataFrame) -> pl.DataFrame:
    if base_progress.is_empty():
        return new_progress

    combined = pl.concat([base_progress, new_progress], how="vertical_relaxed")
    combined = combined.sort(["unit_id", "updated_at"])
    latest = combined.group_by("unit_id").tail(1)
    return latest


def _success_or_terminal_unit_ids(progress_df: pl.DataFrame) -> set:
    if progress_df.is_empty():
        return set()

    terminal = progress_df.filter(
        pl.col("status").is_in(["success", "permanent_error"])
    )
    return set(terminal["unit_id"].to_list())


def _current_retry_count(progress_df: pl.DataFrame, unit_id: str) -> int:
    if progress_df.is_empty():
        return 0
    rows = progress_df.filter(pl.col("unit_id") == unit_id)
    if rows.is_empty():
        return 0
    return int(rows["retry_count"].to_list()[-1])


def _build_runtime_metadata(
    step_config: dict,
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    outcome: LLMRunOutcome,
) -> dict:
    return {
        "step_name": step_config["name"],
        "model": step_config["model"],
        "provider_config_key": step_config["provider_config_key"],
        "prompt_version": step_config.get("kwargs", {}).get("prompt_version", "v1"),
        "total_units": work_units_df.height,
        "completed_or_terminal_units": len(_success_or_terminal_unit_ids(progress_df)),
        "processed_units_this_run": outcome.processed_units,
        "remaining_units": outcome.remaining_units,
        "outcome": outcome.status,
        "updated_at": utc_now_iso(),
    }


def _result_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "output_column": pl.Utf8,
                "status": pl.Utf8,
                "parsed_output": pl.Utf8,
                "output_value": pl.Utf8,
                "raw_output": pl.Utf8,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def _progress_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "status": pl.Utf8,
                "retry_count": pl.Int64,
                "last_error_type": pl.Utf8,
                "last_error_message": pl.Utf8,
                "updated_at": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def _flush_state(
    connector,
    folders: dict,
    temp_dir: str,
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    result_rows: List[dict],
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
        result_df = _result_rows_to_df(result_rows)
        upload_versioned_parquet(
            connector=connector,
            location=folders["results_folder"],
            prefix="results",
            df=result_df,
            temp_dir=temp_dir,
        )

    upload_versioned_json(
        connector=connector,
        location=folders["state_folder"],
        prefix="metadata",
        payload=metadata,
        temp_dir=temp_dir,
    )


def _merge_results_back(source_df: pl.DataFrame, all_results_df: pl.DataFrame, row_id_column: str) -> pl.DataFrame:
    if all_results_df.is_empty():
        return source_df

    success_df = all_results_df.filter(pl.col("status") == "success")
    if success_df.is_empty():
        return source_df

    wide = (
        success_df
        .select(["row_id", "output_column", "output_value"])
        .pivot(
            index="row_id",
            on="output_column",
            values="output_value",
            aggregate_function="first",
        )
        .rename({"row_id": row_id_column})
    )

    return source_df.with_columns(pl.col(row_id_column).cast(pl.Utf8)).join(
        wide.with_columns(pl.col(row_id_column).cast(pl.Utf8)),
        on=row_id_column,
        how="left",
    )


def execute_llm_step(
    data,
    step_config: dict,
    runtime_context: dict,
):
    _validate_llm_step(step_config)

    if isinstance(data, pl.LazyFrame):
        source_df = data.collect(streaming=True)
    else:
        source_df = data

    runtime = _default_runtime(step_config)
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

    work_units_df = _build_work_units(source_df, step_config)

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

    terminal_ids = _success_or_terminal_unit_ids(progress_df)
    pending_units_df = work_units_df.filter(~pl.col("unit_id").is_in(list(terminal_ids)))

    if pending_units_df.is_empty():
        outcome = LLMRunOutcome(
            status="complete",
            processed_units=0,
            remaining_units=0,
        )
        metadata = _build_runtime_metadata(
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
            metadata=metadata,
        )
        return _merge_results_back(
            source_df=source_df,
            all_results_df=existing_results_df,
            row_id_column=step_config["row_id_column"],
        )

    start_time = time.time()
    soft_time_limit_seconds = runtime["soft_time_limit_minutes"] * 60
    batch_size = runtime["max_units_per_batch"]
    flush_every_n_units = runtime["flush_every_n_units"]
    max_request_retries = runtime["max_request_retries"]
    retry_backoff_seconds = runtime["retry_backoff_seconds"]

    processed_count = 0
    batch_result_rows = []
    batch_progress_rows = []

    pending_rows = pending_units_df.to_dicts()

    for batch_start in range(0, len(pending_rows), batch_size):
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            break

        batch = pending_rows[batch_start: batch_start + batch_size]
        expected_unit_ids = [unit["unit_id"] for unit in batch]

        requests = task_handler.build_requests(batch, step_config, prompt_context)

        last_exception = None
        adapter_results = None

        for attempt in range(max_request_retries + 1):
            try:
                adapter_results = adapter.execute_batch(requests, step_config)
                validate_adapter_batch_results(adapter_results, expected_unit_ids)
                last_exception = None
                break
            except Exception as e:
                last_exception = e
                if attempt >= max_request_retries:
                    break
                time.sleep(retry_backoff_seconds * (attempt + 1))

        if last_exception is not None:
            now = utc_now_iso()
            for unit in batch:
                retry_count = _current_retry_count(progress_df, unit["unit_id"]) + 1
                batch_progress_rows.append(
                    LLMProgressRecord(
                        unit_id=unit["unit_id"],
                        status="retryable_error",
                        retry_count=retry_count,
                        last_error_type="batch_exception",
                        last_error_message=str(last_exception),
                        updated_at=now,
                    ).to_dict()
                )
            processed_count += len(batch)
        else:
            now = utc_now_iso()

            for adapter_result in adapter_results:
                unit = next(u for u in batch if u["unit_id"] == adapter_result["unit_id"])
                transport_status = adapter_result["status"]

                if transport_status != "success":
                    retry_count = _current_retry_count(progress_df, adapter_result["unit_id"])
                    if transport_status == "retryable_error":
                        retry_count += 1

                    batch_progress_rows.append(
                        LLMProgressRecord(
                            unit_id=adapter_result["unit_id"],
                            status=transport_status,
                            retry_count=retry_count,
                            last_error_type=adapter_result.get("error_type"),
                            last_error_message=adapter_result.get("error_message"),
                            updated_at=now,
                        ).to_dict()
                    )
                    continue

                parsed = task_handler.parse_result(
                    raw_output=adapter_result["raw_output"],
                    unit=unit,
                    step_config=step_config,
                    prompt_context=prompt_context,
                )
                validate_task_parse_result(parsed)

                retry_count = _current_retry_count(progress_df, adapter_result["unit_id"])
                if parsed["status"] == "retryable_error":
                    retry_count += 1

                batch_progress_rows.append(
                    LLMProgressRecord(
                        unit_id=adapter_result["unit_id"],
                        status=parsed["status"],
                        retry_count=retry_count,
                        last_error_type=parsed.get("error_type"),
                        last_error_message=parsed.get("error_message"),
                        updated_at=now,
                    ).to_dict()
                )

                if parsed["status"] == "success":
                    batch_result_rows.append(
                        LLMResultRecord(
                            unit_id=adapter_result["unit_id"],
                            row_id=unit["row_id"],
                            output_column=unit["output_column"],
                            status=parsed["status"],
                            parsed_output=(
                                parsed["parsed_output"]
                                if isinstance(parsed["parsed_output"], str)
                                else json.dumps(parsed["parsed_output"], ensure_ascii=False)
                            ) if parsed.get("parsed_output") is not None else None,
                            output_value=parsed.get("output_value"),
                            raw_output=adapter_result.get("raw_output"),
                            error_type=parsed.get("error_type"),
                            error_message=parsed.get("error_message"),
                        ).to_dict()
                    )

            processed_count += len(batch)

        if processed_count > 0 and processed_count % flush_every_n_units == 0:
            new_progress_df = _progress_rows_to_df(batch_progress_rows)
            progress_df = _merge_progress(progress_df, new_progress_df)

            outcome = LLMRunOutcome(
                status="retryable_incomplete",
                processed_units=processed_count,
                remaining_units=max(len(pending_rows) - processed_count, 0),
            )

            metadata = _build_runtime_metadata(
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
                result_rows=batch_result_rows,
                metadata=metadata,
            )

            batch_result_rows = []
            batch_progress_rows = []

    if batch_progress_rows:
        new_progress_df = _progress_rows_to_df(batch_progress_rows)
        progress_df = _merge_progress(progress_df, new_progress_df)

    terminal_ids_after = _success_or_terminal_unit_ids(progress_df)
    remaining = work_units_df.filter(~pl.col("unit_id").is_in(list(terminal_ids_after))).height

    outcome_status = "complete" if remaining == 0 else "retryable_incomplete"
    outcome = LLMRunOutcome(
        status=outcome_status,
        processed_units=processed_count,
        remaining_units=remaining,
    )

    metadata = _build_runtime_metadata(
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
        result_rows=batch_result_rows,
        metadata=metadata,
    )

    existing_results_df = _load_existing_results(
        connector=dest_connector,
        results_folder=folders["results_folder"],
        temp_dir=temp_dir,
    )

    return _merge_results_back(
        source_df=source_df,
        all_results_df=existing_results_df,
        row_id_column=step_config["row_id_column"],
    )