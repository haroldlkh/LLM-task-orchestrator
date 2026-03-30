import time

from .batch_rows import (
    build_rows_from_group_parse,
    mark_group_parse_failure,
    mark_group_transport_failure,
)
from .batch_runtime import (
    decide_next_group_size,
    flush_buffers,
    init_controller_state,
    log_progress,
    merge_progress_rows_in_memory,
    should_flush,
    useful_work_count_for_group,
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
    max_flushes_per_run = runtime["max_flushes_per_run"]

    current_group_size = runtime["initial_group_size"]
    min_group_size = runtime["min_group_size"]
    max_group_size = runtime["max_group_size"]
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
    completed_flushes = 0

    groups_since_flush = 0
    units_since_flush = 0
    last_flush_time = time.time()
    stop_reason = None
    controller_state = init_controller_state()

    while cursor < total_units:
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            stop_reason = "soft_time_limit_reached"
            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note=stop_reason,
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
        request_attempt_count = 0
        request_started_at = time.time()

        for attempt in range(max_request_retries + 1):
            request_attempt_count = attempt + 1
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

        request_seconds = max(time.time() - request_started_at, 1e-6)

        if last_exception is not None:
            progress_rows, debug_rows = mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=None,
                error_type="group_request_exception",
                error_message=str(last_exception),
                progress_df=progress_df,
                request_attempt_count=request_attempt_count,
            )
            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)

            decision = decide_next_group_size(
                current_group_size=current_group_size,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                runtime=runtime,
                controller_state=controller_state,
                useful_work_count=0,
                total_units=len(group_units),
                failed_units=len(group_units),
                request_seconds=request_seconds,
                transport_failure=False,
                whole_group_parse_failure=True,
                success_streak=success_streak,
            )
            new_group_size = decision["new_group_size"]
            if new_group_size != current_group_size:
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} action={decision['action']} "
                        f"old_group_size={current_group_size} new_group_size={new_group_size} "
                        f"reason={decision['note']}"
                    ),
                    flush=True,
                )
            current_group_size = new_group_size
            success_streak = decision["new_success_streak"]

            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note="marked_group_request_exception",
                start_time=start_time,
                request_seconds=request_seconds,
                useful_work_count=0,
                failure_rate=1.0,
                score=decision["score"],
            )

            if should_flush(
                pending_progress_rows, pending_result_rows, pending_debug_rows,
                groups_since_flush, units_since_flush, last_flush_time, runtime,
            ):
                flush_out = flush_buffers(
                    flush_callback,
                    pending_progress_rows,
                    pending_result_rows,
                    pending_debug_rows,
                    processed_count,
                    max(total_units - cursor, 0),
                    current_group_size,
                    success_streak,
                )
                if flush_out.get("counted_flush"):
                    completed_flushes += 1
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()
                if max_flushes_per_run is not None and completed_flushes >= max_flushes_per_run:
                    stop_reason = "max_flushes_per_run_reached"
                    log_progress(
                        step_name=step_config["name"], group_index=group_index, total_units=total_units,
                        processed_count=processed_count, cursor=cursor, current_group_size=current_group_size,
                        success_streak=success_streak, progress_by_unit=progress_by_unit, note=stop_reason,
                        start_time=start_time,
                    )
                    break
            continue

        request_result = adapter_results[0]

        if request_result["status"] != "success":
            progress_rows, debug_rows = mark_group_transport_failure(
                group_units=group_units,
                request=request,
                request_result=request_result,
                progress_df=progress_df,
                request_attempt_count=request_attempt_count,
            )
            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)

            decision = decide_next_group_size(
                current_group_size=current_group_size,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                runtime=runtime,
                controller_state=controller_state,
                useful_work_count=0,
                total_units=len(group_units),
                failed_units=len(group_units),
                request_seconds=request_seconds,
                transport_failure=True,
                whole_group_parse_failure=False,
                success_streak=success_streak,
            )
            new_group_size = decision["new_group_size"]
            if new_group_size != current_group_size:
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} action={decision['action']} "
                        f"old_group_size={current_group_size} new_group_size={new_group_size} "
                        f"reason={decision['note']}"
                    ),
                    flush=True,
                )
            current_group_size = new_group_size
            success_streak = decision["new_success_streak"]

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
                request_seconds=request_seconds,
                useful_work_count=0,
                failure_rate=1.0,
                score=decision["score"],
            )

            if should_flush(
                pending_progress_rows, pending_result_rows, pending_debug_rows,
                groups_since_flush, units_since_flush, last_flush_time, runtime,
            ):
                flush_out = flush_buffers(
                    flush_callback,
                    pending_progress_rows,
                    pending_result_rows,
                    pending_debug_rows,
                    processed_count,
                    max(total_units - cursor, 0),
                    current_group_size,
                    success_streak,
                )
                if flush_out.get("counted_flush"):
                    completed_flushes += 1
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()
                if max_flushes_per_run is not None and completed_flushes >= max_flushes_per_run:
                    stop_reason = "max_flushes_per_run_reached"
                    log_progress(
                        step_name=step_config["name"], group_index=group_index, total_units=total_units,
                        processed_count=processed_count, cursor=cursor, current_group_size=current_group_size,
                        success_streak=success_streak, progress_by_unit=progress_by_unit, note=stop_reason,
                        start_time=start_time,
                    )
                    break
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
            progress_rows, debug_rows = mark_group_parse_failure(
                group_units=group_units,
                request=request,
                raw_output=request_result.get("raw_output"),
                error_type="group_parse_exception",
                error_message=str(e),
                progress_df=progress_df,
                request_attempt_count=request_attempt_count,
            )
            pending_progress_rows.extend(progress_rows)
            pending_debug_rows.extend(debug_rows)
            merge_progress_rows_in_memory(progress_by_unit, progress_rows)

            processed_count += len(group_units)
            cursor += len(group_units)
            groups_since_flush += 1
            units_since_flush += len(group_units)

            decision = decide_next_group_size(
                current_group_size=current_group_size,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                runtime=runtime,
                controller_state=controller_state,
                useful_work_count=0,
                total_units=len(group_units),
                failed_units=len(group_units),
                request_seconds=request_seconds,
                transport_failure=False,
                whole_group_parse_failure=True,
                success_streak=success_streak,
            )
            new_group_size = decision["new_group_size"]
            if new_group_size != current_group_size:
                print(
                    (
                        f"[llm:{step_config['name']}] "
                        f"group={group_index} action={decision['action']} "
                        f"old_group_size={current_group_size} new_group_size={new_group_size} "
                        f"reason={decision['note']}"
                    ),
                    flush=True,
                )
            current_group_size = new_group_size
            success_streak = decision["new_success_streak"]

            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=current_group_size,
                success_streak=success_streak,
                progress_by_unit=progress_by_unit,
                note="marked_group_parse_exception",
                start_time=start_time,
                request_seconds=request_seconds,
                useful_work_count=0,
                failure_rate=1.0,
                score=decision["score"],
            )

            if should_flush(
                pending_progress_rows, pending_result_rows, pending_debug_rows,
                groups_since_flush, units_since_flush, last_flush_time, runtime,
            ):
                flush_out = flush_buffers(
                    flush_callback,
                    pending_progress_rows,
                    pending_result_rows,
                    pending_debug_rows,
                    processed_count,
                    max(total_units - cursor, 0),
                    current_group_size,
                    success_streak,
                )
                if flush_out.get("counted_flush"):
                    completed_flushes += 1
                groups_since_flush = 0
                units_since_flush = 0
                last_flush_time = time.time()
                if max_flushes_per_run is not None and completed_flushes >= max_flushes_per_run:
                    stop_reason = "max_flushes_per_run_reached"
                    log_progress(
                        step_name=step_config["name"], group_index=group_index, total_units=total_units,
                        processed_count=processed_count, cursor=cursor, current_group_size=current_group_size,
                        success_streak=success_streak, progress_by_unit=progress_by_unit, note=stop_reason,
                        start_time=start_time,
                    )
                    break
            continue

        progress_rows, result_rows, debug_rows = build_rows_from_group_parse(
            group_units=group_units,
            request=request,
            request_result=request_result,
            parse_results=parse_results,
            progress_df=progress_df,
            request_attempt_count=request_attempt_count,
        )
        pending_progress_rows.extend(progress_rows)
        pending_result_rows.extend(result_rows)
        pending_debug_rows.extend(debug_rows)
        merge_progress_rows_in_memory(progress_by_unit, progress_rows)

        processed_count += len(group_units)
        cursor += len(group_units)
        groups_since_flush += 1
        units_since_flush += len(group_units)

        failed_units = sum(1 for row in parse_results if row["status"] != "success")
        useful_work_count = useful_work_count_for_group(group_units, parse_results, runtime["flush_scope"])
        decision = decide_next_group_size(
            current_group_size=current_group_size,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            runtime=runtime,
            controller_state=controller_state,
            useful_work_count=useful_work_count,
            total_units=len(group_units),
            failed_units=failed_units,
            request_seconds=request_seconds,
            transport_failure=False,
            whole_group_parse_failure=False,
            success_streak=success_streak,
        )
        new_group_size = decision["new_group_size"]
        if new_group_size != current_group_size:
            print(
                (
                    f"[llm:{step_config['name']}] "
                    f"group={group_index} action={decision['action']} "
                    f"old_group_size={current_group_size} new_group_size={new_group_size} "
                    f"reason={decision['note']}"
                ),
                flush=True,
            )
        current_group_size = new_group_size
        success_streak = decision["new_success_streak"]

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
            request_seconds=request_seconds,
            useful_work_count=useful_work_count,
            failure_rate=decision["failure_rate"],
            score=decision["score"],
        )

        if should_flush(
            pending_progress_rows, pending_result_rows, pending_debug_rows,
            groups_since_flush, units_since_flush, last_flush_time, runtime,
        ):
            flush_out = flush_buffers(
                flush_callback,
                pending_progress_rows,
                pending_result_rows,
                pending_debug_rows,
                processed_count,
                max(total_units - cursor, 0),
                current_group_size,
                success_streak,
            )
            if flush_out.get("counted_flush"):
                completed_flushes += 1
            groups_since_flush = 0
            units_since_flush = 0
            last_flush_time = time.time()
            if max_flushes_per_run is not None and completed_flushes >= max_flushes_per_run:
                stop_reason = "max_flushes_per_run_reached"
                log_progress(
                    step_name=step_config["name"], group_index=group_index, total_units=total_units,
                    processed_count=processed_count, cursor=cursor, current_group_size=current_group_size,
                    success_streak=success_streak, progress_by_unit=progress_by_unit, note=stop_reason,
                    start_time=start_time,
                )
                break

    flush_buffers(
        flush_callback,
        pending_progress_rows,
        pending_result_rows,
        pending_debug_rows,
        processed_count,
        max(total_units - cursor, 0),
        current_group_size,
        success_streak,
    )

    remaining_units = max(total_units - processed_count, 0)
    outcome = LLMRunOutcome(
        status="complete" if remaining_units == 0 else "retryable_incomplete",
        processed_units=processed_count,
        remaining_units=remaining_units,
    )

    print(
        (
            f"[llm:{step_config['name']}] run_end "
            f"processed={processed_count}/{total_units} remaining={remaining_units} "
            f"outcome={outcome.status} completed_flushes={completed_flushes}"
        ),
        flush=True,
    )

    return {"outcome": outcome, "completed_flushes": completed_flushes, "stop_reason": stop_reason}
