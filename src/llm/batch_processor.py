import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .batch_rows import (
    build_rows_from_group_parse,
    mark_group_parse_failure,
    mark_group_transport_failure,
)
from .batch_runtime import (
    ControllerState,
    choose_next_controller_state,
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

TERMINAL_STATUSES = {"success", "permanent_error"}


def _remaining_units_from_progress(progress_by_unit: dict, total_units: int) -> int:
    terminal = sum(1 for row in progress_by_unit.values() if row.get("status") in TERMINAL_STATUSES)
    return max(total_units - terminal, 0)


def _requeue_retryable_units(pending_rows: list, group_units: list, progress_rows: list) -> None:
    status_by_unit = {row["unit_id"]: row.get("status") for row in progress_rows}
    for unit in group_units:
        if status_by_unit.get(unit["unit_id"]) == "retryable_error":
            pending_rows.append(unit)


def _estimate_request_tokens(request: dict, group_units, runtime: dict) -> int:
    chars_per_token = float(runtime.get("token_estimation_chars_per_token", 4.0) or 4.0)
    response_tokens_per_unit = float(runtime.get("estimated_response_tokens_per_unit", 24) or 24)
    overhead_tokens = float(runtime.get("estimated_request_overhead_tokens", 250) or 250)

    prompt_text = request.get("prompt", "") or ""
    prompt_tokens = math.ceil(len(prompt_text) / max(chars_per_token, 0.1))
    response_tokens = math.ceil(len(group_units) * response_tokens_per_unit)
    total = int(prompt_tokens + response_tokens + overhead_tokens)
    return max(total, 1)


def _prune_token_window(token_window: list[dict], now: float) -> list[dict]:
    cutoff = now - 60.0
    return [entry for entry in token_window if entry["sent_at"] > cutoff]


def _rolling_window_tokens(token_window: list[dict], now: float) -> int:
    token_window[:] = _prune_token_window(token_window, now)
    return int(sum(entry["tokens"] for entry in token_window))


def _wait_for_tpm_budget(step_name: str, runtime: dict, token_window: list[dict], next_wave_tokens: int) -> None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if target_tpm is None:
        return

    safe_budget = float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)
    max_sleep = float(runtime.get("max_sleep_to_respect_tpm_seconds", 120) or 120)

    while True:
        now = time.time()
        rolling_tokens = _rolling_window_tokens(token_window, now)
        if rolling_tokens + next_wave_tokens <= safe_budget:
            print(
                f"[llm:{step_name}] tpm_window rolling_tokens={rolling_tokens} next_wave_tokens={next_wave_tokens} safe_budget={int(safe_budget)} note=within_budget",
                flush=True,
            )
            return

        if not token_window:
            return

        oldest_expiry = min(entry["sent_at"] + 60.0 for entry in token_window)
        sleep_seconds = max(0.0, oldest_expiry - now)
        if sleep_seconds <= 0:
            token_window[:] = _prune_token_window(token_window, time.time())
            continue

        sleep_seconds = min(sleep_seconds, max_sleep)
        print(
            f"[llm:{step_name}] tpm_window rolling_tokens={rolling_tokens} next_wave_tokens={next_wave_tokens} safe_budget={int(safe_budget)} sleep_s={sleep_seconds:.2f} reason=tpm_budget_wait",
            flush=True,
        )
        time.sleep(sleep_seconds)


def _fit_wave_to_tpm_budget(
    pending_rows,
    cursor,
    desired_group_size,
    concurrency,
    step_config,
    task_handler,
    prompt_context,
    runtime,
):
    target_tpm = runtime.get("target_tokens_per_minute")
    min_group_size = int(runtime["min_group_size"])
    current_group_size = int(desired_group_size)

    while True:
        wave, _ = _build_wave_requests(
            pending_rows=pending_rows,
            cursor=cursor,
            group_size=current_group_size,
            concurrency=concurrency,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
        )
        if not wave:
            return wave, current_group_size, []

        request_token_estimates = [
            _estimate_request_tokens(item["request"], item["group_units"], runtime)
            for item in wave
        ]
        next_wave_tokens = int(sum(request_token_estimates))

        if target_tpm is None:
            return wave, current_group_size, request_token_estimates

        safe_budget = float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)
        if next_wave_tokens <= safe_budget or current_group_size <= min_group_size:
            if next_wave_tokens > safe_budget and current_group_size <= min_group_size:
                print(
                    f"[llm:{step_config['name']}] tpm_fit_group_size group_size={current_group_size} estimated_tokens={next_wave_tokens} safe_budget={int(safe_budget)} note=min_group_size_floor",
                    flush=True,
                )
            return wave, current_group_size, request_token_estimates

        next_group_size = max(min_group_size, int(math.floor(current_group_size * 0.85)))
        if next_group_size >= current_group_size:
            next_group_size = current_group_size - 1
        next_group_size = max(next_group_size, min_group_size)

        print(
            f"[llm:{step_config['name']}] tpm_fit_group_size group_size={current_group_size} next_group_size={next_group_size} estimated_tokens={next_wave_tokens} safe_budget={int(safe_budget)} note=shrink_wave_to_fit_tpm",
            flush=True,
        )
        current_group_size = next_group_size


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
    request_timeout_seconds = runtime.get("max_request_wall_time_seconds")

    last_exception = None
    adapter_results = None
    final_attempt_count = 0

    for attempt in range(max_request_retries + 1):
        final_attempt_count = attempt + 1
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            adapter.execute_batch,
            [item["request"] for item in wave],
            step_config,
        )
        try:
            if request_timeout_seconds is None:
                adapter_results = future.result()
            else:
                adapter_results = future.result(timeout=float(request_timeout_seconds))
            validate_grouped_adapter_batch_results(adapter_results, request_ids)
            last_exception = None
            executor.shutdown(wait=True, cancel_futures=False)
            break
        except FutureTimeoutError:
            last_exception = TimeoutError(
                f"wave_request_timeout after {float(request_timeout_seconds):.1f}s"
            )
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            last_exception = e
            executor.shutdown(wait=False, cancel_futures=True)

        if attempt >= max_request_retries:
            break
        print(
            (
                f"[llm:{step_config['name']}] "
                f"wave_request_retry={attempt + 1}/{max_request_retries} "
                f"wave_size={len(wave)} "
                f"error={str(last_exception)}"
            ),
            flush=True,
        )
        time.sleep(retry_backoff_seconds * (attempt + 1))

    return adapter_results, last_exception, final_attempt_count


def _wave_sleep_seconds(runtime: dict, wave_metrics: dict, last_exception) -> tuple[float, str]:
    all_transport_failure = (
        wave_metrics["total_requests"] > 0
        and wave_metrics["transport_failures"] == wave_metrics["total_requests"]
    )
    any_transport_failure = wave_metrics["transport_failures"] > 0

    if last_exception is not None or all_transport_failure:
        return float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0), "all_transport_failure_cooldown"
    if any_transport_failure:
        return float(runtime.get("transport_failure_cooldown_seconds", 0) or 0), "transport_failure_cooldown"
    return float(runtime.get("min_inter_wave_sleep_seconds", 0) or 0), "inter_wave_sleep"


def _sleep_between_waves(step_name: str, seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    print(
        f"[llm:{step_name}] sleeping sleep_s={seconds:.2f} reason={reason}",
        flush=True,
    )
    time.sleep(seconds)


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
    pending_rows = [] if pending_rows is None else list(pending_rows)

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
    token_window = []

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

        desired_group_size = derive_group_size(
            controller.load_budget,
            controller.concurrency,
            runtime["min_group_size"],
            runtime["max_group_size"],
        )

        wave, group_size, request_token_estimates = _fit_wave_to_tpm_budget(
            pending_rows=pending_rows,
            cursor=cursor,
            desired_group_size=desired_group_size,
            concurrency=controller.concurrency,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
            runtime=runtime,
        )
        if not wave:
            break

        next_wave_tokens = int(sum(request_token_estimates))
        _wait_for_tpm_budget(
            step_name=step_config["name"],
            runtime=runtime,
            token_window=token_window,
            next_wave_tokens=next_wave_tokens,
        )

        wave_index += 1
        rolling_before_send = _rolling_window_tokens(token_window, time.time())
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
            note=f"wave_start groups={len(wave)} size={group_size} estimated_next_wave_tokens={next_wave_tokens} rolling_tokens={rolling_before_send}",
            start_time=start_time,
        )

        send_time = time.time()
        for est in request_token_estimates:
            token_window.append({"sent_at": send_time, "tokens": int(est)})

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
            error_type = "wave_request_timeout" if isinstance(last_exception, TimeoutError) else "wave_request_exception"
            for item in wave:
                progress_rows, debug_rows = mark_group_transport_failure(
                    group_units=item["group_units"],
                    request=item["request"],
                    request_result={
                        "status": "retryable_error",
                        "raw_output": None,
                        "error_type": error_type,
                        "error_message": str(last_exception),
                    },
                    progress_df=progress_df,
                    request_attempt_count=wave_attempt_count,
                )
                pending_progress_rows.extend(progress_rows)
                pending_debug_rows.extend(debug_rows)
                merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                _requeue_retryable_units(pending_rows, item["group_units"], progress_rows)

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
                    _requeue_retryable_units(pending_rows, group_units, progress_rows)

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
                    _requeue_retryable_units(pending_rows, group_units, progress_rows)

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
                    _requeue_retryable_units(pending_rows, group_units, progress_rows)

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
                    "remaining_units": _remaining_units_from_progress(progress_by_unit, total_units),
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
                if runtime.get("max_flushes_per_run") is not None and completed_flushes >= runtime["max_flushes_per_run"]:
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

        if cursor < total_units:
            sleep_seconds, sleep_reason = _wave_sleep_seconds(
                runtime=runtime,
                wave_metrics=wave_metrics,
                last_exception=last_exception,
            )
            _sleep_between_waves(
                step_name=step_config["name"],
                seconds=sleep_seconds,
                reason=sleep_reason,
            )

    if pending_progress_rows or pending_result_rows or pending_debug_rows:
        flush_payload = flush_callback(
            {
                "progress_rows": list(pending_progress_rows),
                "result_rows": list(pending_result_rows),
                "debug_rows": list(pending_debug_rows),
                "processed_units": processed_count,
                "remaining_units": _remaining_units_from_progress(progress_by_unit, total_units),
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

    remaining_units = _remaining_units_from_progress(progress_by_unit, total_units)
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
    }
