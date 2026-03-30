import time
from typing import Any, Dict, List

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


def _build_wave(
    pending_rows: List[Dict[str, Any]],
    cursor: int,
    current_group_size: int,
    max_concurrent_requests: int,
    step_config: dict,
    task_handler,
    prompt_context: dict,
    total_units: int,
    processed_count: int,
    progress_by_unit: dict,
    success_streak: int,
    start_time: float,
    next_group_index: int,
    log_every_n_groups: int,
) -> tuple[list[dict], int, int]:
    wave_items = []
    build_cursor = cursor
    group_index = next_group_index

    for _ in range(max_concurrent_requests):
        if build_cursor >= total_units:
            break

        group_units = pending_rows[build_cursor: build_cursor + current_group_size]
        group_index += 1

        if group_index % log_every_n_groups == 0:
            log_progress(
                step_name=step_config["name"],
                group_index=group_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=build_cursor,
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
        wave_items.append(
            {
                "group_index": group_index,
                "group_units": group_units,
                "request": request,
            }
        )
        build_cursor += len(group_units)

    return wave_items, build_cursor, group_index


def _execute_wave_with_retries(
    adapter,
    step_config: dict,
    wave_requests: List[dict],
    max_request_retries: int,
    retry_backoff_seconds: int,
) -> tuple[List[dict] | None, Exception | None, int, float]:
    expected_request_ids = [request["request_id"] for request in wave_requests]
    last_exception = None
    adapter_results = None
    attempt_count = 0
    wave_started_at = time.time()

    for attempt in range(max_request_retries + 1):
        attempt_count = attempt + 1
        try:
            adapter_results = adapter.execute_batch(wave_requests, step_config)
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
                    f"wave_retry={attempt + 1}/{max_request_retries} "
                    f"wave_requests={len(wave_requests)} "
                    f"error={str(e)}"
                ),
                flush=True,
            )
            time.sleep(retry_backoff_seconds * (attempt + 1))

    wave_seconds = max(time.time() - wave_started_at, 1e-6)
    return adapter_results, last_exception, attempt_count, wave_seconds


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
    max_concurrent_requests = runtime["max_concurrent_requests"]

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

        wave_items, _, group_index = _build_wave(
            pending_rows=pending_rows,
            cursor=cursor,
            current_group_size=current_group_size,
            max_concurrent_requests=max_concurrent_requests,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
            total_units=total_units,
            processed_count=processed_count,
            progress_by_unit=progress_by_unit,
            success_streak=success_streak,
            start_time=start_time,
            next_group_index=group_index,
            log_every_n_groups=log_every_n_groups,
        )
        if not wave_items:
            break

        wave_requests = [item["request"] for item in wave_items]
        adapter_results, last_exception, wave_attempt_count, wave_seconds = _execute_wave_with_retries(
            adapter=adapter,
            step_config=step_config,
            wave_requests=wave_requests,
            max_request_retries=max_request_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        result_map = {}
        if adapter_results is not None:
            result_map = {result["request_id"]: result for result in adapter_results}

        for item in wave_items:
            group_units = item["group_units"]
            request = item["request"]
            this_group_index = item["group_index"]

            if last_exception is not None:
                request_seconds = wave_seconds
                request_attempt_count = wave_attempt_count
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
                            f"group={this_group_index} action={decision['action']} "
                            f"old_group_size={current_group_size} new_group_size={new_group_size} "
                            f"reason={decision['note']}"
                        ),
                        flush=True,
                    )
                current_group_size = new_group_size
                success_streak = decision["new_success_streak"]

                log_progress(
                    step_name=step_config["name"],
                    group_index=this_group_index,
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
                continue

            request_result = result_map[request["request_id"]]
            request_seconds = float(request_result.get("request_seconds") or wave_seconds)
            request_attempt_count = int(request_result.get("request_attempt_count") or 1)

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
                            f"group={this_group_index} action={decision['action']} "
                            f"old_group_size={current_group_size} new_group_size={new_group_size} "
                            f"reason={decision['note']}"
                        ),
                        flush=True,
                    )
                current_group_size = new_group_size
                success_streak = decision["new_success_streak"]

                log_progress(
                    step_name=step_config["name"],
                    group_index=this_group_index,
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
                            f"group={this_group_index} action={decision['action']} "
                            f"old_group_size={current_group_size} new_group_size={new_group_size} "
                            f"reason={decision['note']}"
                        ),
                        flush=True,
                    )
                current_group_size = new_group_size
                success_streak = decision["new_success_streak"]

                log_progress(
                    step_name=step_config["name"],
                    group_index=this_group_index,
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
                        f"group={this_group_index} action={decision['action']} "
                        f"old_group_size={current_group_size} new_group_size={new_group_size} "
                        f"reason={decision['note']}"
                    ),
                    flush=True,
                )
            current_group_size = new_group_size
            success_streak = decision["new_success_streak"]

            log_progress(
                step_name=step_config["name"],
                group_index=this_group_index,
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
            pending_progress_rows,
            pending_result_rows,
            pending_debug_rows,
            groups_since_flush,
            units_since_flush,
            last_flush_time,
            runtime,
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
