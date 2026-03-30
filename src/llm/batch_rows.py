import json

from .models import LLMProgressRecord, LLMResultRecord
from .runtime import current_retry_count
from .state import utc_now_iso


def _serialize_jsonish(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _serialize_output_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def mark_group_transport_failure(
    group_units,
    request,
    request_result,
    progress_df,
    request_attempt_count: int,
):
    now = utc_now_iso()
    progress_rows = []
    debug_rows = []

    request_group_id = request.get("request_id")
    request_group_size = len(group_units)

    for unit in group_units:
        retry_count = current_retry_count(progress_df, unit["unit_id"])
        if request_result["status"] == "retryable_error":
            retry_count += 1

        progress_rows.append(
            LLMProgressRecord(
                unit_id=unit["unit_id"],
                status=request_result["status"],
                retry_count=retry_count,
                last_error_type=request_result.get("error_type"),
                last_error_message=request_result.get("error_message"),
                updated_at=now,
            ).to_dict()
        )

        debug_rows.append(
            {
                "unit_id": unit["unit_id"],
                "row_id": unit["row_id"],
                "field_name": unit["field_name"],
                "input_text": unit["input_text"],
                "output_column": unit["output_column"],
                "request_group_id": request_group_id,
                "request_group_size": request_group_size,
                "request_attempt_count": request_attempt_count,
                "status": request_result["status"],
                "review_flag": True,
                "review_reason": "group_transport_failure",
                "error_type": request_result.get("error_type"),
                "error_message": request_result.get("error_message"),
                "rendered_prompt": request["prompt"],
                "raw_output": request_result.get("raw_output"),
                "parsed_output": None,
                "output_value": None,
                "updated_at": now,
            }
        )

    return progress_rows, debug_rows


def mark_group_parse_failure(
    group_units,
    request,
    raw_output,
    error_type,
    error_message,
    progress_df,
    request_attempt_count: int,
):
    now = utc_now_iso()
    progress_rows = []
    debug_rows = []

    request_group_id = request.get("request_id")
    request_group_size = len(group_units)

    for unit in group_units:
        retry_count = current_retry_count(progress_df, unit["unit_id"]) + 1

        progress_rows.append(
            LLMProgressRecord(
                unit_id=unit["unit_id"],
                status="retryable_error",
                retry_count=retry_count,
                last_error_type=error_type,
                last_error_message=error_message,
                updated_at=now,
            ).to_dict()
        )

        debug_rows.append(
            {
                "unit_id": unit["unit_id"],
                "row_id": unit["row_id"],
                "field_name": unit["field_name"],
                "input_text": unit["input_text"],
                "output_column": unit["output_column"],
                "request_group_id": request_group_id,
                "request_group_size": request_group_size,
                "request_attempt_count": request_attempt_count,
                "status": "retryable_error",
                "review_flag": True,
                "review_reason": "group_parse_failure",
                "error_type": error_type,
                "error_message": error_message,
                "rendered_prompt": request["prompt"],
                "raw_output": raw_output,
                "parsed_output": None,
                "output_value": None,
                "updated_at": now,
            }
        )

    return progress_rows, debug_rows


def build_rows_from_group_parse(
    group_units,
    request,
    request_result,
    parse_results,
    progress_df,
    request_attempt_count: int,
):
    now = utc_now_iso()
    progress_rows = []
    result_rows = []
    debug_rows = []

    parse_lookup = {row["unit_id"]: row for row in parse_results}
    request_group_id = request.get("request_id")
    request_group_size = len(group_units)

    for unit in group_units:
        parsed = parse_lookup[unit["unit_id"]]

        retry_count = current_retry_count(progress_df, unit["unit_id"])
        if parsed["status"] == "retryable_error":
            retry_count += 1

        progress_rows.append(
            LLMProgressRecord(
                unit_id=unit["unit_id"],
                status=parsed["status"],
                retry_count=retry_count,
                last_error_type=parsed.get("error_type"),
                last_error_message=parsed.get("error_message"),
                updated_at=now,
            ).to_dict()
        )

        review_flag = parsed.get("review_flag", False)
        review_reason = parsed.get("review_reason")
        if parsed["status"] != "success":
            review_flag = True
            if review_reason is None:
                review_reason = "group_unit_non_success"

        debug_rows.append(
            {
                "unit_id": unit["unit_id"],
                "row_id": unit["row_id"],
                "field_name": unit["field_name"],
                "input_text": unit["input_text"],
                "output_column": unit["output_column"],
                "request_group_id": request_group_id,
                "request_group_size": request_group_size,
                "request_attempt_count": request_attempt_count,
                "status": parsed["status"],
                "review_flag": review_flag,
                "review_reason": review_reason,
                "error_type": parsed.get("error_type"),
                "error_message": parsed.get("error_message"),
                "rendered_prompt": request["prompt"],
                "raw_output": request_result.get("raw_output"),
                "parsed_output": _serialize_jsonish(parsed.get("parsed_output")),
                "output_value": _serialize_output_value(parsed.get("output_value")),
                "updated_at": now,
            }
        )

        if parsed["status"] == "success":
            result_rows.append(
                LLMResultRecord(
                    unit_id=unit["unit_id"],
                    row_id=unit["row_id"],
                    output_column=unit["output_column"],
                    status="success",
                    parsed_output=_serialize_jsonish(parsed.get("parsed_output")),
                    output_value=_serialize_output_value(parsed.get("output_value")),
                    raw_output=_serialize_jsonish(request_result.get("raw_output")),
                    error_type=parsed.get("error_type"),
                    error_message=parsed.get("error_message"),
                    review_flag=review_flag,
                    review_reason=review_reason,
                ).to_dict()
            )

    return progress_rows, result_rows, debug_rows
