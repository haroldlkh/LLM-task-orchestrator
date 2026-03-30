import time

from .batch_rows import (
    build_rows_from_group_parse,
    mark_group_parse_failure,
    mark_group_transport_failure,
)
from .batch_runtime import (
    flush_buffers,
    log_progress,
    merge_progress_rows_in_memory,
    next_larger_group_size,
    next_smaller_group_size,
    should_flush,
)
from .models import LLMRunOutcome
from .validators import (
    validate_grouped_adapter_batch_results,
    validate_grouped_parse_results,
)


def process_batches(
    pending_rows,
    step_config,
    runtime,
    adapter,
    task_handler,
    prompt_context,
    progress_df,
    start_time,
    flush_callback,
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
    log_every_n_groups = runtime["log_every_n_groups"]

    total_units = len(pending_rows)
    processed_count = 0

    pending_result_rows = []
    pending_progress_rows = []
    pending_debug_rows = []

    progress_by_unit = {}
    cursor = 0
    success_streak = 0
    group_index = 0

    groups_since_flush = 0
    units_since_flush = 0
    last_flush_time = time.time()

    while cursor < total_units:
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note="soft_time_limit_reached",
                start_time=start_time,
            )
            break

        group_units = pending_rows[cursor: cursor + current_group_size]
        group_index += 1

        if group_index % log_every_n_groups == 0:
            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note=f"group_start size={len(group_units)}",
                start_time=start_time,
            )

        requests = task_handler.build_requests(group_units, step_config, prompt_context)
        if not requests:
            raise ValueError("Task handler returned no grouped requests")

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

                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"request_retry={attempt + 1}/{max_request_retries} "
                        f"group_size={current_group_size} "
                        f"error={str(e)}"
                    ),
                    flush=True,
                )
                time.sleep(retry_backoff_seconds * (attempt + 1))

        if last_exception is not None:
            if current_group_size > min_group_size:
                new_group_size = next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"action=shrink_on_request_exception "
                        f"old_group_size={current_group_size} "
                        f"new_group_size={new_group_size} "
                        f"error={str(last_exception)}"
                    ),
                    flush=True,
                )
                current_group_size = new_group_size
                success_streak = 0
                continue

            progress_rows, debug_rows = mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=None,
                error_type="group_request_exception",
                error_message=str(last_exception),
                progress_df=progress_df,
            )

            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)
            success_streak = 0

            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note="marked_group_request_exception_at_min_group_size",
                start_time=start_time,
            )

            if should_flush(
                pending_progress_rows=pending_progress_rows,
                pending_result_rows=pending_result_rows,
                pending_debug_rows=pending_debug_rows,
                groups_since_flush=groups_since_flush,
                units_since_flush=units_since_flush,
                last_flush_time=last_flush_time,
                runtime=runtime,
            ):
                flush_buffers(
                    flush_callback=flush_callback,
                    pending_progress_rows=pending_progress_rows,
                    pending_result_rows=pending_result_rows,
                    pending_debug_rows=pending_debug_rows,
                    processed_count=processed_count,
                    remaining_units=max(total_units - cursor, 0),
                    current_group_size=current_group_size,
                    success_streak=success_streak,
                )
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()

            continue

        request_result = adapter_results[0]

        if request_result["status"] != "success":
            if request_result["status"] == "retryable_error" and current_group_size > min_group_size:
                new_group_size = next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"action=shrink_on_transport_retryable "
                        f"old_group_size={current_group_size} "
                        f"new_group_size={new_group_size} "
                        f"error_type={request_result.get('error_type')} "
                        f"error_message={request_result.get('error_message')}"
                    ),
                    flush=True,
                )
                current_group_size = new_group_size
                success_streak = 0
                continue

            progress_rows, debug_rows = mark_group_transport_failure(
                group_units=group_units,
                request=request,
                request_result=request_result,
                progress_df=progress_df,
            )

            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)
            success_streak = 0

            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note=f"transport_non_success status={request_result['status']}",
                start_time=start_time,
            )

            if should_flush(
                pending_progress_rows=pending_progress_rows,
                pending_result_rows=pending_result_rows,
                pending_debug_rows=pending_debug_rows,
                groups_since_flush=groups_since_flush,
                units_since_flush=units_since_flush,
                last_flush_time=last_flush_time,
                runtime=runtime,
            ):
                flush_buffers(
                    flush_callback=flush_callback,
                    pending_progress_rows=pending_progress_rows,
                    pending_result_rows=pending_result_rows,
                    pending_debug_rows=pending_debug_rows,
                    processed_count=processed_count,
                    remaining_units=max(total_units - cursor, 0),
                    current_group_size=current_group_size,
                    success_streak=success_streak,
                )
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()

            continue

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
            if current_group_size > min_group_size:
                new_group_size = next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"action=shrink_on_parse_exception "
                        f"old_group_size={current_group_size} "
                        f"new_group_size={new_group_size} "
                        f"error={str(e)}"
                    ),
                    flush=True,
                )
                current_group_size = new_group_size
                success_streak = 0
                continue

            progress_rows, debug_rows = mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=request_result.get("raw_output"),
                error_type="group_parse_exception",
                error_message=str(e),
                progress_df=progress_df,
            )

            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)
            success_streak = 0

            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note="marked_group_parse_exception_at_min_group_size",
                start_time=start_time,
            )

            if should_flush(
                pending_progress_rows=pending_progress_rows,
                pending_result_rows=pending_result_rows,
                pending_debug_rows=pending_debug_rows,
                groups_since_flush=groups_since_flush,
                units_since_flush=units_since_flush,
                last_flush_time=last_flush_time,
                runtime=runtime,
            ):
                flush_buffers(
                    flush_callback=flush_callback,
                    pending_progress_rows=pending_progress_rows,
                    pending_result_rows=pending_result_rows,
                    pending_debug_rows=pending_debug_rows,
                    processed_count=processed_count,
                    remaining_units=max(total_units - cursor, 0),
                    current_group_size=current_group_size,
                    success_streak=success_streak,
                )
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()

            continue

        progress_rows, result_rows, debug_rows = build_rows_from_group_parse(
            group_units=group_units,
            request=request,
            request_result=request_result,
            parse_results=parse_results,
            progress_df=progress_df,
        )

        pending_progress_rows.extend(progress_rows)
        pending_result_rows.extend(result_rows)
        pending_debug_rows.extend(debug_rows)
        merge_progress_rows_in_memory(progress_by_unit, progress_rows)

        processed_count += len(group_units)
        cursor += len(group_units)
        groups_since_flush += 1
        units_since_flush += len(group_units)

        all_success = all(row["status"] == "success" for row in parse_results)

        if all_success:
            success_streak += 1
            if success_streak >= grow_after_successes and current_group_size < max_group_size:
                new_group_size = next_larger_group_size(
                    current_group_size, max_group_size, grow_step
                )
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"action=grow "
                        f"old_group_size={current_group_size} "
                        f"new_group_size={new_group_size}"
                    ),
                    flush=True,
                )
                current_group_size = new_group_size
                success_streak = 0
        else:
            if current_group_size > min_group_size:
                new_group_size = next_smaller_group_size(
                    current_group_size, min_group_size, shrink_factor
                )
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} "
                        f"action=shrink_on_partial_non_success "
                        f"old_group_size={current_group_size} "
                        f"new_group_size={new_group_size}"
                    ),
                    flush=True,
                )
                current_group_size = new_group_size
            success_streak = 0

        log_progress(
            step_name=step_config["name"],
            group_index=group_index,
            total_units=total_units,
            processed_count=processed_count,
            cursor=cursor,
            current_group_size=current_group_size,
            success_streak=success_streak,
            progress_by_unit=progress_by_unit,
            note="group_complete",
            start_time=start_time,
        )

        if should_flush(
            pending_progress_rows=pending_progress_rows,
            pending_result_rows=pending_result_rows,
            pending_debug_rows=pending_debug_rows,
            groups_since_flush=groups_since_flush,
            units_since_flush=units_since_flush,
            last_flush_time=last_flush_time,
            runtime=runtime,
        ):
            flush_buffers(
                flush_callback=flush_callback,
                pending_progress_rows=pending_progress_rows,
                pending_result_rows=pending_result_rows,
                pending_debug_rows=pending_debug_rows,
                processed_count=processed_count,
                remaining_units=max(total_units - cursor, 0),
                current_group_size=current_group_size,
                success_streak=success_streak,
            )
            groups_since_flush = 0
            units_since_flush = 0
            last_flush_time = time.time()

    flush_buffers(
        flush_callback=flush_callback,
        pending_progress_rows=pending_progress_rows,
        pending_result_rows=pending_result_rows,
        pending_debug_rows=pending_debug_rows,
        processed_count=processed_count,
        remaining_units=max(total_units - cursor, 0),
        current_group_size=current_group_size,
        success_streak=success_streak,
    )

    remaining_units = max(total_units - processed_count, 0)
    outcome = LLMRunOutcome(
        status="complete" if remaining_units == 0 else "retryable_incomplete",
        processed_units=processed_count,
        remaining_units=remaining_units,
    )

    print(
        (
            f"[llm:{step_config['name']}] "
            f"run_end "
            f"processed={processed_count}/{total_units} "
            f"remaining={remaining_units} "
            f"outcome={outcome.status}"
        ),
        flush=True,
    )

    return {"outcome": outcome}