import math
import time
from collections import deque

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


def _estimate_request_tokens(request: dict, runtime: dict) -> int:
    chars_per_token = float(runtime.get("token_estimation_chars_per_token", 4.0) or 4.0)
    response_tokens_per_unit = float(runtime.get("estimated_response_tokens_per_unit", 24) or 24)
    overhead_tokens = float(runtime.get("estimated_request_overhead_tokens", 250) or 250)
    prompt = request.get("prompt", "") or ""
    units = request.get("units", []) or []
    prompt_tokens = math.ceil(len(prompt) / chars_per_token)
    response_tokens = math.ceil(len(units) * response_tokens_per_unit)
    return int(overhead_tokens + prompt_tokens + response_tokens)


def _trim_token_ledger(token_ledger: deque, now_ts: float) -> None:
    while token_ledger and now_ts - token_ledger[0][0] >= 60.0:
        token_ledger.popleft()


def _rolling_tokens(token_ledger: deque, now_ts: float) -> int:
    _trim_token_ledger(token_ledger, now_ts)
    return int(sum(tokens for _, tokens in token_ledger))


def _sleep_for_tpm_budget(step_name: str, runtime: dict, estimated_wave_tokens: int, token_ledger: deque) -> None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if not target_tpm:
        return

    budget = float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)
    max_sleep = float(runtime.get("max_sleep_to_respect_tpm_seconds", 120) or 0)
    now_ts = time.time()
    rolling_tokens = _rolling_tokens(token_ledger, now_ts)

    if rolling_tokens + estimated_wave_tokens <= budget:
        print(
            f"[llm:{step_name}] tpm_window rolling_tokens={rolling_tokens} estimated_next_wave_tokens={estimated_wave_tokens} budget={int(budget)}",
            flush=True,
        )
        return

    sleep_seconds = 0.0
    for ts, _ in token_ledger:
        candidate = max(0.0, 60.0 - (now_ts - ts))
        future_tokens = sum(tokens for entry_ts, tokens in token_ledger if now_ts + candidate - entry_ts < 60.0)
        if future_tokens + estimated_wave_tokens <= budget:
            sleep_seconds = candidate
            break
    else:
        if token_ledger:
            sleep_seconds = max(0.0, 60.0 - (now_ts - token_ledger[0][0]))

    if max_sleep > 0:
        sleep_seconds = min(sleep_seconds, max_sleep)

    if sleep_seconds > 0:
        print(
            f"[llm:{step_name}] sleeping sleep_s={sleep_seconds:.2f} reason=tpm_budget_wait rolling_tokens={rolling_tokens} estimated_next_wave_tokens={estimated_wave_tokens} budget={int(budget)}",
            flush=True,
        )
        time.sleep(sleep_seconds)


def _build_wave_requests(pending_rows, cursor, group_size, concurrency, step_config, task_handler, prompt_context):
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
            raise ValueError("Grouped batching expects exactly one grouped request per group")
        wave.append({"group_units": group_units, "request": requests[0]})
    return wave, next_cursor


def _fit_group_size_to_tpm_budget(pending_rows, cursor, group_size, concurrency, step_config, task_handler, prompt_context, runtime):
    target_tpm = runtime.get("target_tokens_per_minute")
    if not target_tpm:
        return group_size, None

    budget = float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)
    max_per_request = max(1, int(budget / max(concurrency, 1)))
    candidate = group_size
    fitted_wave = None

    while candidate >= runtime["min_group_size"]:
        wave, _ = _build_wave_requests(
            pending_rows=pending_rows,
            cursor=cursor,
            group_size=candidate,
            concurrency=concurrency,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
        )
        if not wave:
            return candidate, wave
        max_request_tokens = max(_estimate_request_tokens(item["request"], runtime) for item in wave)
        if max_request_tokens <= max_per_request or candidate == runtime["min_group_size"]:
            fitted_wave = wave
            break
        next_candidate = max(runtime["min_group_size"], int(candidate * 0.8))
        if next_candidate == candidate:
            next_candidate = candidate - 1
        candidate = next_candidate

    if candidate != group_size:
        print(
            f"[llm:{step_config['name']}] tpm_fit_group_size old_group_size={group_size} new_group_size={candidate} budget_per_request={max_per_request}",
            flush=True,
        )
    return candidate, fitted_wave


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
            adapter_results = adapter.execute_batch([item["request"] for item in wave], step_config)
            validate_grouped_adapter_batch_results(adapter_results, request_ids)
            last_exception = None
            break
        except Exception as exc:
            last_exception = exc
            if attempt >= max_request_retries:
                break
            print(
                f"[llm:{step_config['name']}] wave_request_retry={attempt + 1}/{max_request_retries} wave_size={len(wave)} error={str(exc)}",
                flush=True,
            )
            time.sleep(retry_backoff_seconds * (attempt + 1))
    return adapter_results, last_exception, final_attempt_count


def _wave_sleep_seconds(runtime: dict, wave_metrics: dict, last_exception) -> tuple[float, str]:
    all_transport_failure = wave_metrics["total_requests"] > 0 and wave_metrics["transport_failures"] == wave_metrics["total_requests"]
    any_transport_failure = wave_metrics["transport_failures"] > 0
    if last_exception is not None or all_transport_failure:
        return float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0), "all_transport_failure_cooldown"
    if any_transport_failure:
        return float(runtime.get("transport_failure_cooldown_seconds", 0) or 0), "transport_failure_cooldown"
    return float(runtime.get("min_inter_wave_sleep_seconds", 0) or 0), "inter_wave_sleep"


def _sleep_between_waves(step_name: str, seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    print(f"[llm:{step_name}] sleeping sleep_s={seconds:.2f} reason={reason}", flush=True)
    time.sleep(seconds)


def process_batches(pending_rows, step_config, runtime, adapter, task_handler, prompt_context, progress_df, start_time, flush_callback):
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
    token_ledger = deque()

    controller = ControllerState(
        load_budget=float(runtime["initial_load_budget"]),
        concurrency=int(runtime["initial_concurrency"]),
    )

    while cursor < total_units:
        elapsed = time.time() - start_time
        if elapsed >= soft_time_limit_seconds:
            group_size = derive_group_size(controller.load_budget, controller.concurrency, runtime["min_group_size"], runtime["max_group_size"])
            log_progress(
                step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                processed_count=processed_count, cursor=cursor, current_group_size=group_size,
                current_concurrency=controller.concurrency, current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak, progress_by_unit=progress_by_unit,
                note="soft_time_limit_reached", start_time=start_time,
            )
            break

        group_size = derive_group_size(controller.load_budget, controller.concurrency, runtime["min_group_size"], runtime["max_group_size"])
        group_size, prebuilt_wave = _fit_group_size_to_tpm_budget(
            pending_rows, cursor, group_size, controller.concurrency,
            step_config, task_handler, prompt_context, runtime,
        )
        if prebuilt_wave is not None:
            wave = prebuilt_wave
            reserved_cursor = cursor + sum(len(item["group_units"]) for item in wave)
        else:
            wave, reserved_cursor = _build_wave_requests(
                pending_rows=pending_rows, cursor=cursor, group_size=group_size,
                concurrency=controller.concurrency, step_config=step_config,
                task_handler=task_handler, prompt_context=prompt_context,
            )
        if not wave:
            break

        estimated_wave_tokens = sum(_estimate_request_tokens(item["request"], runtime) for item in wave)
        _sleep_for_tpm_budget(step_config["name"], runtime, estimated_wave_tokens, token_ledger)

        wave_index += 1
        log_progress(
            step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
            processed_count=processed_count, cursor=cursor, current_group_size=group_size,
            current_concurrency=controller.concurrency, current_load_budget=controller.load_budget,
            success_streak=controller.good_wave_streak, progress_by_unit=progress_by_unit,
            note=f"wave_start groups={len(wave)} size={group_size} estimated_tokens={estimated_wave_tokens}",
            start_time=start_time,
        )

        now_ts = time.time()
        _trim_token_ledger(token_ledger, now_ts)
        token_ledger.append((now_ts, estimated_wave_tokens))

        wave_start = time.time()
        adapter_results, last_exception, wave_attempt_count = _execute_wave_requests(adapter, wave, step_config, runtime)
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
                    group_units=item["group_units"], request=item["request"], raw_output=None,
                    error_type="wave_request_exception", error_message=str(last_exception),
                    progress_df=progress_df, request_attempt_count=wave_attempt_count,
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
            controller, control_notes = choose_next_controller_state(controller, wave_metrics, runtime)
            log_progress(
                step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                processed_count=processed_count, cursor=cursor,
                current_group_size=derive_group_size(controller.load_budget, controller.concurrency, runtime["min_group_size"], runtime["max_group_size"]),
                current_concurrency=controller.concurrency, current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak, progress_by_unit=progress_by_unit,
                note=f"wave_end transport_exception notes={control_notes}", start_time=start_time,
            )
        else:
            results_by_request_id = {result["request_id"]: result for result in adapter_results}
            for item in wave:
                group_units = item["group_units"]
                request = item["request"]
                request_result = results_by_request_id[request["request_id"]]
                group_len = len(group_units)

                if request_result["status"] != "success":
                    progress_rows, debug_rows = mark_group_transport_failure(
                        group_units=group_units, request=request, request_result=request_result,
                        progress_df=progress_df, request_attempt_count=wave_attempt_count,
                    )
                    pending_progress_rows.extend(progress_rows)
                    pending_debug_rows.extend(debug_rows)
                    merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                    wave_metrics["transport_failures"] += 1
                    wave_metrics["failure_units"] += group_len
                else:
                    wave_metrics["successful_requests"] += 1
                    try:
                        parse_results = task_handler.parse_grouped_result(
                            request_result["raw_output"], request, step_config, prompt_context
                        )
                        validate_grouped_parse_results(
                            parse_results,
                            expected_unit_ids=[unit["unit_id"] for unit in group_units],
                        )
                        progress_rows, result_rows, debug_rows = build_rows_from_group_parse(
                            group_units=group_units, request=request, request_result=request_result,
                            parse_results=parse_results, progress_df=progress_df,
                            request_attempt_count=wave_attempt_count,
                        )
                        pending_progress_rows.extend(progress_rows)
                        pending_result_rows.extend(result_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        success_units = sum(1 for row in parse_results if row["status"] == "success")
                        failure_units = group_len - success_units
                        wave_metrics["useful_work"] += success_units
                        wave_metrics["failure_units"] += failure_units
                    except Exception as exc:
                        progress_rows, debug_rows = mark_group_parse_failure(
                            group_units=group_units, request=request,
                            raw_output=request_result.get("raw_output"),
                            error_type="group_parse_failure", error_message=str(exc),
                            progress_df=progress_df, request_attempt_count=wave_attempt_count,
                        )
                        pending_progress_rows.extend(progress_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        wave_metrics["failure_units"] += group_len

                processed_count += group_len
                cursor += group_len
                groups_since_flush += 1
                units_since_flush += group_len
                wave_metrics["processed_units"] += group_len

            wave_metrics["elapsed_request_seconds"] = max(time.time() - wave_start, 1e-9)
            controller, control_notes = choose_next_controller_state(controller, wave_metrics, runtime)
            log_progress(
                step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                processed_count=processed_count, cursor=cursor,
                current_group_size=derive_group_size(controller.load_budget, controller.concurrency, runtime["min_group_size"], runtime["max_group_size"]),
                current_concurrency=controller.concurrency, current_load_budget=controller.load_budget,
                success_streak=controller.good_wave_streak, progress_by_unit=progress_by_unit,
                note=(
                    f"wave_end useful={wave_metrics['useful_work']} failures={wave_metrics['failure_units']} "
                    f"wave_score={request_score(wave_metrics):.4f} notes={control_notes}"
                ),
                start_time=start_time,
            )

        if should_flush(runtime, units_since_flush, groups_since_flush, last_flush_time, completed_flushes):
            flush_callback({
                "progress_rows": pending_progress_rows,
                "result_rows": pending_result_rows,
                "debug_rows": pending_debug_rows,
            })
            completed_flushes += 1
            pending_result_rows = []
            pending_progress_rows = []
            pending_debug_rows = []
            groups_since_flush = 0
            units_since_flush = 0
            last_flush_time = time.time()

            if runtime.get("max_flushes_per_run") is not None and completed_flushes >= runtime["max_flushes_per_run"]:
                remaining = max(total_units - cursor, 0)
                return {
                    "outcome": LLMRunOutcome(status="retryable_incomplete", processed_units=processed_count, remaining_units=remaining),
                    "progress_rows": pending_progress_rows,
                    "result_rows": pending_result_rows,
                    "debug_rows": pending_debug_rows,
                }

        sleep_seconds, sleep_reason = _wave_sleep_seconds(runtime, wave_metrics, last_exception)
        _sleep_between_waves(step_config["name"], sleep_seconds, sleep_reason)

    remaining_units = max(total_units - cursor, 0)
    outcome_status = "complete" if remaining_units == 0 else "retryable_incomplete"
    return {
        "outcome": LLMRunOutcome(status=outcome_status, processed_units=processed_count, remaining_units=remaining_units),
        "progress_rows": pending_progress_rows,
        "result_rows": pending_result_rows,
        "debug_rows": pending_debug_rows,
    }
