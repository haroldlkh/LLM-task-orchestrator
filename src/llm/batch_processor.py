import time

from .models import LLMRunOutcome
from .runtime import current_retry_count
from .traces import (
    build_batch_exception_rows,
    build_non_success_adapter_rows,
    build_success_or_parse_rows,
)
from .validators import validate_adapter_batch_results, validate_task_parse_result


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
    batch_size = runtime["max_units_per_batch"]
    max_request_retries = runtime["max_request_retries"]
    retry_backoff_seconds = runtime["retry_backoff_seconds"]

    processed_count = 0
    batch_result_rows = []
    batch_progress_rows = []
    batch_debug_rows = []

    for batch_start in range(0, len(pending_rows), batch_size):
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            break

        batch = pending_rows[batch_start: batch_start + batch_size]
        expected_unit_ids = [unit["unit_id"] for unit in batch]

        requests = task_handler.build_requests(batch, step_config, prompt_context)
        request_lookup = {req["unit_id"]: req for req in requests}

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
            progress_rows, debug_rows = build_batch_exception_rows(
                batch=batch,
                request_lookup=request_lookup,
                progress_df=progress_df,
                current_retry_count_fn=current_retry_count,
                exception=last_exception,
            )
            batch_progress_rows.extend(progress_rows)
            batch_debug_rows.extend(debug_rows)
            processed_count += len(batch)
            continue

        for adapter_result in adapter_results:
            unit = next(u for u in batch if u["unit_id"] == adapter_result["unit_id"])

            if adapter_result["status"] != "success":
                progress_row, now = build_non_success_adapter_rows(
                    unit=unit,
                    adapter_result=adapter_result,
                    progress_df=progress_df,
                    current_retry_count_fn=current_retry_count,
                )
                batch_progress_rows.append(progress_row)
                batch_debug_rows.append({
                    "unit_id": adapter_result["unit_id"],
                    "row_id": unit["row_id"],
                    "field_name": unit["field_name"],
                    "output_column": unit["output_column"],
                    "status": adapter_result["status"],
                    "review_flag": True,
                    "review_reason": "adapter_non_success",
                    "error_type": adapter_result.get("error_type"),
                    "error_message": adapter_result.get("error_message"),
                    "input_text": unit["input_text"],
                    "rendered_prompt": request_lookup.get(unit["unit_id"], {}).get("prompt"),
                    "raw_output": adapter_result.get("raw_output"),
                    "parsed_output": None,
                    "output_value": None,
                    "updated_at": now,
                })
                continue

            parsed = task_handler.parse_result(
                raw_output=adapter_result["raw_output"],
                unit=unit,
                step_config=step_config,
                prompt_context=prompt_context,
            )
            validate_task_parse_result(parsed)

            progress_row, result_row, debug_row, _ = build_success_or_parse_rows(
                unit=unit,
                adapter_result=adapter_result,
                parsed=parsed,
                progress_df=progress_df,
                current_retry_count_fn=current_retry_count,
            )
            debug_row["rendered_prompt"] = request_lookup.get(unit["unit_id"], {}).get("prompt")

            batch_progress_rows.append(progress_row)
            batch_debug_rows.append(debug_row)
            if result_row is not None:
                batch_result_rows.append(result_row)

        processed_count += len(batch)

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