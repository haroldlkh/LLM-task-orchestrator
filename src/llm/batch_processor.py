import json
import math
import time

from .models import LLMProgressRecord, LLMResultRecord, LLMRunOutcome
from .runtime import current_retry_count
from .state import utc_now_iso
from .validators import (
    validate_grouped_adapter_batch_results,
    validate_grouped_parse_results,
)


def _next_smaller_group_size(current_size: int, min_size: int, shrink_factor: float) -> int:
    shrunk = max(min_size, int(math.floor(current_size * shrink_factor)))
    if shrunk == current_size and current_size > min_size:
        shrunk = current_size - 1
    return max(min_size, shrunk)


def _next_larger_group_size(current_size: int, max_size: int, grow_step: int) -> int:
    return min(max_size, current_size + grow_step)


def _mark_group_transport_failure(
    group_units,
    request,
    request_result,
    progress_df,
):
    now = utc_now_iso()
    progress_rows = []
    debug_rows = []

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

        debug_rows.append({
            "unit_id": unit["unit_id"],
            "row_id": unit["row_id"],
            "field_name": unit["field_name"],
            "output_column": unit["output_column"],
            "status": request_result["status"],
            "review_flag": True,
            "review_reason": "group_transport_failure",
            "error_type": request_result.get("error_type"),
            "error_message": request_result.get("error_message"),
            "input_text": unit["input_text"],
            "rendered_prompt": request["prompt"],
            "raw_output": request_result.get("raw_output"),
            "parsed_output": None,
            "output_value": None,
            "updated_at": now,
        })

    return progress_rows, debug_rows


def _mark_group_parse_failure(
    group_units,
    request,
    raw_output,
    error_type,
    error_message,
    progress_df,
):
    now = utc_now_iso()
    progress_rows = []
    debug_rows = []

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

        debug_rows.append({
            "unit_id": unit["unit_id"],
            "row_id": unit["row_id"],
            "field_name": unit["field_name"],
            "output_column": unit["output_column"],
            "status": "retryable_error",
            "review_flag": True,
            "review_reason": "group_parse_failure",
            "error_type": error_type,
            "error_message": error_message,
            "input_text": unit["input_text"],
            "rendered_prompt": request["prompt"],
            "raw_output": raw_output,
            "parsed_output": None,
            "output_value": None,
            "updated_at": now,
        })

    return progress_rows, debug_rows


def _build_rows_from_group_parse(
    group_units,
    request,
    request_result,
    parse_results,
    progress_df,
):
    now = utc_now_iso()
    progress_rows = []
    result_rows = []
    debug_rows = []

    parse_lookup = {row["unit_id"]: row for row in parse_results}

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

        debug_rows.append({
            "unit_id": unit["unit_id"],
            "row_id": unit["row_id"],
            "field_name": unit["field_name"],
            "output_column": unit["output_column"],
            "status": parsed["status"],
            "review_flag": review_flag,
            "review_reason": review_reason,
            "error_type": parsed.get("error_type"),
            "error_message": parsed.get("error_message"),
            "input_text": unit["input_text"],
            "rendered_prompt": request["prompt"],
            "raw_output": request_result.get("raw_output"),
            "parsed_output": (
                parsed["parsed_output"]
                if isinstance(parsed.get("parsed_output"), str)
                else json.dumps(parsed.get("parsed_output"), ensure_ascii=False)
            ) if parsed.get("parsed_output") is not None else None,
            "output_value": parsed.get("output_value"),
            "updated_at": now,
        })

        if parsed["status"] == "success":
            result_rows.append(
                LLMResultRecord(
                    unit_id=unit["unit_id"],
                    row_id=unit["row_id"],
                    output_column=unit["output_column"],
                    status="success",
                    parsed_output=(
                        parsed["parsed_output"]
                        if isinstance(parsed["parsed_output"], str)
                        else json.dumps(parsed["parsed_output"], ensure_ascii=False)
                    ) if parsed.get("parsed_output") is not None else None,
                    output_value=parsed.get("output_value"),
                    raw_output=request_result.get("raw_output"),
                    error_type=parsed.get("error_type"),
                    error_message=parsed.get("error_message"),
                    review_flag=review_flag,
                    review_reason=review_reason,
                ).to_dict()
            )

    return progress_rows, result_rows, debug_rows


def process_batches(
    pending_rows,
    step_config,
    runtime,
    adapter,
    task_handler,
    prompt_context,
    progress_df,
    start_time,
):
    soft_time_limit_seconds = runtime["soft_time_limit_minutes"] * 60
    max_request_retries = runtime["max_request_retries"]
    retry_backoff_seconds = runtime["retry_backoff_seconds"]

    current_group_size = runtime["initial_group_size"]
    min_group_size = runtime["min_group_size"]
    max_group_size = runtime["max_group_size"]
    grow_after_successes = runtime["grow_after_successes"]
    grow_step = runtime["grow_step"]
    shrink_factor = runtime["shrink_factor"]

    processed_count = 0
    batch_result_rows = []
    batch_progress_rows = []
    batch_debug_rows = []

    cursor = 0
    success_streak = 0

    while cursor < len(pending_rows):
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            break

        group_units = pending_rows[cursor: cursor + current_group_size]

        requests = task_handler.build_requests(group_units, step_config, prompt_context)
        if not requests:
            raise ValueError("Task handler returned no grouped requests")

        # For this adaptive implementation, we expect one grouped request per group slice.
        if len(requests) != 1:
            raise ValueError(
                "Adaptive grouped batching expects task_handler.build_requests(...) "
                "to return exactly one grouped request"
            )

        request = requests[0]
        expected_request_ids = [request["request_id"]]

        last_exception = None
        adapter_results = None

        for attempt in range(max_request_retries + 1):
            try:
                adapter_results = adapter.execute_batch([request], step_config)
                validate_grouped_adapter_batch_results(adapter_results, expected_request_ids)
                last_exception = None
                break
            except Exception as e:
                last_exception = e
                if attempt >= max_request_retries:
                    break
                time.sleep(retry_backoff_seconds * (attempt + 1))

        if last_exception is not None:
            # If grouped request totally failed, shrink and retry same units if possible
            if current_group_size > min_group_size:
                current_group_size = _next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                success_streak = 0
                continue

            # Already at minimum size; mark these units as retryable_error and move on
            progress_rows, debug_rows = _mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=None,
                error_type="group_request_exception",
                error_message=str(last_exception),
                progress_df=progress_df,
            )
            batch_progress_rows.extend(progress_rows)
            batch_debug_rows.extend(debug_rows)
            processed_count += len(group_units)
            cursor += len(group_units)
            success_streak = 0
            continue

        request_result = adapter_results[0]

        if request_result["status"] != "success":
            # Retryable provider failure: shrink and retry same slice if possible
            if request_result["status"] == "retryable_error" and current_group_size > min_group_size:
                current_group_size = _next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                success_streak = 0
                continue

            # Otherwise mark each unit in group with same transport failure and move on
            progress_rows, debug_rows = _mark_group_transport_failure(
                group_units=group_units,
                request=request,
                request_result=request_result,
                progress_df=progress_df,
            )
            batch_progress_rows.extend(progress_rows)
            batch_debug_rows.extend(debug_rows)
            processed_count += len(group_units)
            cursor += len(group_units)
            success_streak = 0
            continue

        # Parse grouped response
        try:
            parse_results = task_handler.parse_grouped_result(
                raw_output=request_result["raw_output"],
                request=request,
                step_config=step_config,
                prompt_context=prompt_context,
            )

            expected_unit_ids = [unit["unit_id"] for unit in group_units]
            validate_grouped_parse_results(parse_results, expected_unit_ids)

        except Exception as e:
            # If grouped parse fails, shrink and retry same slice if possible
            if current_group_size > min_group_size:
                current_group_size = _next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                success_streak = 0
                continue

            progress_rows, debug_rows = _mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=request_result.get("raw_output"),
                error_type="group_parse_exception",
                error_message=str(e),
                progress_df=progress_df,
            )
            batch_progress_rows.extend(progress_rows)
            batch_debug_rows.extend(debug_rows)
            processed_count += len(group_units)
            cursor += len(group_units)
            success_streak = 0
            continue

        progress_rows, result_rows, debug_rows = _build_rows_from_group_parse(
            group_units=group_units,
            request=request,
            request_result=request_result,
            parse_results=parse_results,
            progress_df=progress_df,
        )

        batch_progress_rows.extend(progress_rows)
        batch_result_rows.extend(result_rows)
        batch_debug_rows.extend(debug_rows)

        processed_count += len(group_units)
        cursor += len(group_units)

        all_success = all(row["status"] == "success" for row in parse_results)

        if all_success:
            success_streak += 1
            if success_streak >= grow_after_successes and current_group_size < max_group_size:
                current_group_size = _next_larger_group_size(
                    current_group_size, max_group_size, grow_step
                )
                success_streak = 0
        else:
            # any non-success means back off aggressively
            if current_group_size > min_group_size:
                current_group_size = _next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
            success_streak = 0

    remaining_units = max(len(pending_rows) - processed_count, 0)
    outcome = LLMRunOutcome(
        status="complete" if remaining_units == 0 else "retryable_incomplete",
        processed_units=processed_count,
        remaining_units=remaining_units,
    )

    return {
        "outcome": outcome,
        "progress_rows": batch_progress_rows,
        "result_rows": batch_result_rows,
        "debug_rows": batch_debug_rows,
    }