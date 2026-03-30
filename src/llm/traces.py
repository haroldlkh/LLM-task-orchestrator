import json
from typing import List

import polars as pl

from .models import LLMProgressRecord, LLMResultRecord
from .state import utc_now_iso


def result_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "output_column": pl.Utf8,
                "status": pl.Utf8,
                "parsed_output": pl.Utf8,
                "output_value": pl.Int64,
                "raw_output": pl.Utf8,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
                "review_flag": pl.Boolean,
                "review_reason": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def progress_rows_to_df(rows: List[dict]) -> pl.DataFrame:
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


def debug_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "field_name": pl.Utf8,
                "output_column": pl.Utf8,
                "request_group_id": pl.Utf8,
                "request_group_size": pl.Int64,
                "status": pl.Utf8,
                "review_flag": pl.Boolean,
                "review_reason": pl.Utf8,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
                "input_text": pl.Utf8,
                "rendered_prompt": pl.Utf8,
                "raw_output": pl.Utf8,
                "parsed_output": pl.Utf8,
                "output_value": pl.Int64,
                "updated_at": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def build_batch_exception_rows(batch, request_lookup, progress_df, current_retry_count_fn, exception):
    now = utc_now_iso()
    progress_rows = []
    debug_rows = []

    for unit in batch:
        retry_count = current_retry_count_fn(progress_df, unit["unit_id"]) + 1
        progress_rows.append(
            LLMProgressRecord(
                unit_id=unit["unit_id"],
                status="retryable_error",
                retry_count=retry_count,
                last_error_type="batch_exception",
                last_error_message=str(exception),
                updated_at=now,
            ).to_dict()
        )
        debug_rows.append({
            "unit_id": unit["unit_id"],
            "row_id": unit["row_id"],
            "field_name": unit["field_name"],
            "output_column": unit["output_column"],
            "status": "retryable_error",
            "review_flag": True,
            "review_reason": "batch_exception",
            "error_type": "batch_exception",
            "error_message": str(exception),
            "input_text": unit["input_text"],
            "rendered_prompt": request_lookup.get(unit["unit_id"], {}).get("prompt"),
            "raw_output": None,
            "parsed_output": None,
            "output_value": None,
            "updated_at": now,
        })

    return progress_rows, debug_rows


def build_non_success_adapter_rows(unit, adapter_result, progress_df, current_retry_count_fn):
    now = utc_now_iso()
    status = adapter_result["status"]
    retry_count = current_retry_count_fn(progress_df, adapter_result["unit_id"])
    if status == "retryable_error":
        retry_count += 1

    progress_row = LLMProgressRecord(
        unit_id=adapter_result["unit_id"],
        status=status,
        retry_count=retry_count,
        last_error_type=adapter_result.get("error_type"),
        last_error_message=adapter_result.get("error_message"),
        updated_at=now,
    ).to_dict()

    return progress_row, now


def build_success_or_parse_rows(
    unit,
    adapter_result,
    parsed,
    progress_df,
    current_retry_count_fn,
):
    now = utc_now_iso()
    retry_count = current_retry_count_fn(progress_df, adapter_result["unit_id"])
    if parsed["status"] == "retryable_error":
        retry_count += 1

    progress_row = LLMProgressRecord(
        unit_id=adapter_result["unit_id"],
        status=parsed["status"],
        retry_count=retry_count,
        last_error_type=parsed.get("error_type"),
        last_error_message=parsed.get("error_message"),
        updated_at=now,
    ).to_dict()

    review_flag = parsed.get("review_flag", False)
    review_reason = parsed.get("review_reason")

    result_row = None
    if parsed["status"] == "success":
        result_row = LLMResultRecord(
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
            review_flag=review_flag,
            review_reason=review_reason,
        ).to_dict()

    debug_row = {
        "unit_id": adapter_result["unit_id"],
        "row_id": unit["row_id"],
        "field_name": unit["field_name"],
        "output_column": unit["output_column"],
        "status": parsed["status"],
        "review_flag": review_flag or (parsed["status"] != "success"),
        "review_reason": review_reason if review_reason else (
            "task_non_success" if parsed["status"] != "success" else None
        ),
        "error_type": parsed.get("error_type"),
        "error_message": parsed.get("error_message"),
        "input_text": unit["input_text"],
        "rendered_prompt": None,
        "raw_output": adapter_result.get("raw_output"),
        "parsed_output": (
            parsed["parsed_output"]
            if isinstance(parsed.get("parsed_output"), str)
            else json.dumps(parsed.get("parsed_output"), ensure_ascii=False)
        ) if parsed.get("parsed_output") is not None else None,
        "output_value": parsed.get("output_value"),
        "updated_at": now,
    }

    return progress_row, result_row, debug_row, now