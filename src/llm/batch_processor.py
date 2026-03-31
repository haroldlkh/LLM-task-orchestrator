import time

from .batch_rows import (
    build_rows_from_group_parse,
    mark_group_parse_failure,
    mark_group_transport_failure,
)
from .batch_runtime import (
    ControllerState,
    choose_next_controller_state,
    compute_wave_sleep_seconds,
    derive_group_size,
    log_progress,
    merge_progress_rows_in_memory,
    request_score,
    should_flush,
)
from .models import LLMRunOutcome
from .validators import (
    validate_grouped_adapter_batch_results,
    validate_grouped_parse_results,
)


def _build_wave_requests(
    pending_rows,
    cursor,
    group_size,
    concurrency,
    step_config,
    task_handler,
    prompt_context,
):
    wave = []
    next_cursor = cursor

    for _ in range(concurrency):
        if next_cursor >= len(pending_rows):
            break

        group_units = pending_rows[next_cursor: next_cursor + group_size]
        next_cursor += len(group_units)

        requests = task_handler.build_requests(group_units, step_config, prompt_context)
        if not requests:
            raise ValueError("Task handler returned no grouped requests")
        if len(requests) != 1:
            raise ValueError(
                "Grouped batching expects task_handler.build_requests(...) "
                "to return exactly one grouped request per group"
            )

        wave.append(
            {
                "group_units": group_units,
                "request": requests[0],
            }
        )

    return wave, next_cursor


def _execute_wave_requests(adapter, wave, step_config, runtime):
    request_ids = [item["request"]["request_id"] for item in wave]
    max_request_retries = runtime["max_request_retries"]
    retry_backoff_seconds = runtime["retry_backoff_seconds"]

    last_exception = None
    adapter_results = None
    final_attempt_count = 0

    for attempt in range(max_request_retries + 1):
        final_attempt_count = attempt + 1
        try:
            adapter_results = adapter.execute_batch(
                [item["request"] for item in wave],
                step_config,
            )
            validate_grouped_adapter_batch_results(adapter_results, request_ids)
            last_exception = None
            break
        except Exception as e:
            last_exception = e
            if attempt >= max_request_retries:
                break
            print(
                (
                    f"[llm:{step_config['name']}] "
                    f"wave_request_retry={attempt + 1}/{max_request_retries} "
                    f"wave_size={len(wave)} "
                    f"error={str(e)}"
                ),
                flush=True,
            )
            time.sleep(retry_backoff_seconds * (attempt + 1))

    return adapter_results, last_exception, final_attempt_count


def _sleep_between_waves(step_name: str, runtime: dict, wave_metrics: dict) -> None:
    sleep_seconds, sleep_reason = compute_wave_sleep_seconds(runtime, wave_metrics)
    if sleep_seconds <= 0:
        return

    print(
        (
            f"[llm:{step_name}] "
            f"sleep_s={sleep_seconds:.2f} "
            f"note={sleep_reason}"
        ),
        flush=True,
    )
    time.sleep(sleep_seconds)


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

    total_units = len(pending_rows)
    processed_count = 0

    pending_result_rows = []
    pending_progress_rows = []
    pending_debug_rows = []

    progress_by_unit = {}
    cursor = 0
    wave_index = 0
    groups_since_flush = 0
    units_since_flush = 0
    completed_flushes = 0
    last_flush_time = time.time()

    controller = ControllerState(
        load_budget=float(runtime["initial_load_budget"]),
        concurrency=int(runtime["initial_concurrency"]),
    )

    while cursor < total_units:
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            group_size = derive_group_size(
                controller.load_budget,
                controller.concurrency,
                runtime["min_group_size"],
                runtime["max_group_size"],
            )
            log_progress(
                step_name=step_config["name"],
                wave_index=wave_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=group_size,
                current_concurrency=controller.concurrency,
                current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak,
                progress_by_unit=progress_by_unit,
                note="soft_time_limit_reached",
                start_time=start_time,
            )
            break

        group_size = derive_group_size(
            controller.load_budget,
            controller.concurrency,
            runtime["min_group_size"],
            runtime["max_group_size"],
        )

        wave, reserved_cursor = _build_wave_requests(
            pending_rows=pending_rows,
            cursor=cursor,
            group_size=group_size,
            concurrency=controller.concurrency,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
        )
        if not wave:
            break

        wave_index += 1
        log_progress(
            step_name=step_config["name"],
            wave_index=wave_index,
            total_units=total_units,
            processed_count=processed_count,
            cursor=cursor,
            current_group_size=group_size,
            current_concurrency=controller.concurrency,
            current_load_budget=controller.load_budget,
            success_streak=controller.good_wave_streak,
            progress_by_unit=progress_by_unit,
            note=f"wave_start groups={len(wave)} size={group_size}",
            start_time=start_time,
        )

        wave_start = time.time()
        adapter_results, last_exception, wave_attempt_count = _execute_wave_requests(
            adapter=adapter,
            wave=wave,
            step_config=step_config,
            runtime=runtime,
        )

        wave_metrics = {
            "processed_units": 0,
            "useful_work": 0,
            "failure_units": 0,
            "transport_failures": 0,
            "successful_requests": 0,
            "total_requests": len(wave),
            "elapsed_request_seconds": 0.0,
        }

        if last_exception is not None:
            for item in wave:
                progress_rows, debug_rows = mark_group_parse_failure(
                    group_units=item["group_units"],
                    request=item["request"],
                    raw_output=None,
                    error_type="wave_request_exception",
                    error_message=str(last_exception),
                    progress_df=progress_df,
                    request_attempt_count=wave_attempt_count,
                )
                pending_progress_rows.extend(progress_rows)
                pending_debug_rows.extend(debug_rows)
                merge_progress_rows_in_memory(progress_by_unit, progress_rows)

                group_len = len(item["group_units"])
                processed_count += group_len
                cursor += group_len
                groups_since_flush += 1
                units_since_flush += group_len

                wave_metrics["processed_units"] += group_len
                wave_metrics["failure_units"] += group_len
                wave_metrics["transport_failures"] += 1

            wave_metrics["elapsed_request_seconds"] = max(time.time() - wave_start, 1e-9)
            controller, control_notes = choose_next_controller_state(
                controller=controller,
                runtime=runtime,
                group_size_used=group_size,
                wave_metrics=wave_metrics,
            )

            log_progress(
                step_name=step_config["name"],
                wave_index=wave_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=derive_group_size(
                    controller.load_budget,
                    controller.concurrency,
                    runtime["min_group_size"],
                    runtime["max_group_size"],
                ),
                current_concurrency=controller.concurrency,
                current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak,
                progress_by_unit=progress_by_unit,
                note=f"wave_request_exception {'|'.join(control_notes)}",
                start_time=start_time,
            )

            _sleep_between_waves(
                step_name=step_config["name"],
                runtime=runtime,
                wave_metrics=wave_metrics,
            )
        else:
            results_by_request_id = {row["request_id"]: row for row in adapter_results}

            for item in wave:
                request = item["request"]
                request_result = results_by_request_id[request["request_id"]]
                group_units = item["group_units"]
                group_len = len(group_units)

                request_seconds = float(request_result.get("request_seconds", 0.0) or 0.0)
                if request_seconds <= 0:
                    request_seconds = max(time.time() - wave_start, 1e-9)

                request_attempt_count = int(
                    request_result.get("request_attempt_count", wave_attempt_count) or wave_attempt_count
                )

                wave_metrics["elapsed_request_seconds"] += request_seconds
                wave_metrics["processed_units"] += group_len

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

                    processed_count += group_len
                    cursor += group_len
                    groups_since_flush += 1
                    units_since_flush += group_len

                    wave_metrics["failure_units"] += group_len
                    wave_metrics["transport_failures"] += 1

                    print(
                        (
                            f"[llm:{step_config['name']}] "
                            f"request_group={request.get('request_id')} "
                            f"request_s={request_seconds:.2f} "
                            f"useful_work=0 failure_rate=1.0000 score=0.0000 "
                            f"note=transport_non_success status={request_result['status']}"
                        ),
                        flush=True,
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

                    processed_count += group_len
                    cursor += group_len
                    groups_since_flush += 1
                    units_since_flush += group_len

                    useful_work = sum(1 for row in parse_results if row["status"] == "success")
                    failure_units = group_len - useful_work
                    failure_rate = failure_units / max(group_len, 1)
                    score = request_score(useful_work, request_seconds)

                    wave_metrics["useful_work"] += useful_work
                    wave_metrics["failure_units"] += failure_units
                    if useful_work > 0:
                        wave_metrics["successful_requests"] += 1
                    if failure_rate >= 1.0:
                        wave_metrics["transport_failures"] += 1

                    print(
                        (
                            f"[llm:{step_config['name']}] "
                            f"request_group={request.get('request_id')} "
                            f"request_s={request_seconds:.2f} "
                            f"useful_work={useful_work} "
                            f"failure_rate={failure_rate:.4f} "
                            f"score={score:.4f} "
                            f"note=group_complete"
                        ),
                        flush=True,
                    )

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

                    processed_count += group_len
                    cursor += group_len
                    groups_since_flush += 1
                    units_since_flush += group_len

                    wave_metrics["failure_units"] += group_len
                    wave_metrics["transport_failures"] += 1

                    print(
                        (
                            f"[llm:{step_config['name']}] "
                            f"request_group={request.get('request_id')} "
                            f"request_s={request_seconds:.2f} "
                            f"useful_work=0 failure_rate=1.0000 score=0.0000 "
                            f"note=group_parse_exception"
                        ),
                        flush=True,
                    )

            controller, control_notes = choose_next_controller_state(
                controller=controller,
                runtime=runtime,
                group_size_used=group_size,
                wave_metrics=wave_metrics,
            )

            log_progress(
                step_name=step_config["name"],
                wave_index=wave_index,
                total_units=total_units,
                processed_count=processed_count,
                cursor=cursor,
                current_group_size=derive_group_size(
                    controller.load_budget,
                    controller.concurrency,
                    runtime["min_group_size"],
                    runtime["max_group_size"],
                ),
                current_concurrency=controller.concurrency,
                current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak,
                progress_by_unit=progress_by_unit,
                note="wave_complete " + "|".join(control_notes),
                start_time=start_time,
            )

            _sleep_between_waves(
                step_name=step_config["name"],
                runtime=runtime,
                wave_metrics=wave_metrics,
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
            flush_payload = flush_callback(
                {
                    "progress_rows": list(pending_progress_rows),
                    "result_rows": list(pending_result_rows),
                    "debug_rows": list(pending_debug_rows),
                    "processed_units": processed_count,
                    "remaining_units": max(total_units - cursor, 0),
                    "current_group_size": derive_group_size(
                        controller.load_budget,
                        controller.concurrency,
                        runtime["min_group_size"],
                        runtime["max_group_size"],
                    ),
                    "current_concurrency": controller.concurrency,
                    "current_load_budget": int(controller.load_budget),
                }
            )
            pending_progress_rows.clear()
            pending_result_rows.clear()
            pending_debug_rows.clear()
            groups_since_flush = 0
            units_since_flush = 0
            last_flush_time = time.time()

            if flush_payload and flush_payload.get("counted_flush"):
                completed_flushes += 1
                if (
                    runtime.get("max_flushes_per_run") is not None
                    and completed_flushes >= runtime["max_flushes_per_run"]
                ):
                    log_progress(
                        step_name=step_config["name"],
                        wave_index=wave_index,
                        total_units=total_units,
                        processed_count=processed_count,
                        cursor=cursor,
                        current_group_size=derive_group_size(
                            controller.load_budget,
                            controller.concurrency,
                            runtime["min_group_size"],
                            runtime["max_group_size"],
                        ),
                        current_concurrency=controller.concurrency,
                        current_load_budget=controller.load_budget,
                        success_streak=controller.good_wave_streak,
                        progress_by_unit=progress_by_unit,
                        note="max_flushes_per_run_reached",
                        start_time=start_time,
                    )
                    break

    if pending_progress_rows or pending_result_rows or pending_debug_rows:
        flush_payload = flush_callback(
            {
                "progress_rows": list(pending_progress_rows),
                "result_rows": list(pending_result_rows),
                "debug_rows": list(pending_debug_rows),
                "processed_units": processed_count,
                "remaining_units": max(total_units - cursor, 0),
                "current_group_size": derive_group_size(
                    controller.load_budget,
                    controller.concurrency,
                    runtime["min_group_size"],
                    runtime["max_group_size"],
                ),
                "current_concurrency": controller.concurrency,
                "current_load_budget": int(controller.load_budget),
            }
        )
        if flush_payload and flush_payload.get("counted_flush"):
            completed_flushes += 1

    remaining_units = max(total_units - processed_count, 0)
    outcome = LLMRunOutcome(
        status="complete" if remaining_units == 0 else "retryable_incomplete",
        processed_units=processed_count,
        remaining_units=remaining_units,
    )

    print(
        (
            f"[llm:{step_config['name']}] "
            f"run_end processed={processed_count}/{total_units} "
            f"remaining={remaining_units} "
            f"outcome={outcome.status} "
            f"completed_flushes={completed_flushes}"
        ),
        flush=True,
    )

    return {
        "outcome": outcome,
        "current_group_size": derive_group_size(
            controller.load_budget,
            controller.concurrency,
            runtime["min_group_size"],
            runtime["max_group_size"],
        ),
        "current_concurrency": controller.concurrency,
        "current_load_budget": int(controller.load_budget),
    }