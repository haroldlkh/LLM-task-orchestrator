import math
import time

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


def _estimate_request_tokens(request: dict, runtime: dict) -> int:
    prompt_text = str(request.get("prompt") or "")
    chars_per_token = float(runtime.get("token_estimation_chars_per_token", 4.0) or 4.0)
    response_tokens_per_unit = int(runtime.get("estimated_response_tokens_per_unit", 0) or 0)
    request_overhead_tokens = int(runtime.get("estimated_request_overhead_tokens", 0) or 0)
    units = request.get("units") or []

    prompt_tokens = math.ceil(len(prompt_text) / max(chars_per_token, 1e-9))
    response_tokens = len(units) * response_tokens_per_unit
    total_tokens = prompt_tokens + response_tokens + request_overhead_tokens
    return max(int(total_tokens), 1)


def _estimate_wave_tokens(wave, runtime: dict) -> tuple[int, list[dict]]:
    per_request = []
    total = 0
    for item in wave:
        estimated_tokens = _estimate_request_tokens(item["request"], runtime)
        per_request.append(
            {
                "request_id": item["request"].get("request_id"),
                "estimated_tokens": estimated_tokens,
            }
        )
        total += estimated_tokens
    return total, per_request


def _prune_token_ledger(token_ledger: list[dict], now: float) -> None:
    cutoff = now - 60.0
    token_ledger[:] = [row for row in token_ledger if row["sent_at"] > cutoff]


def _rolling_window_tokens(token_ledger: list[dict]) -> int:
    return int(sum(int(row.get("estimated_tokens", 0) or 0) for row in token_ledger))


def _budget_limit(runtime: dict) -> int:
    target_tpm = int(runtime.get("target_tokens_per_minute", 0) or 0)
    if target_tpm <= 0:
        return 0
    safety_margin = float(runtime.get("tpm_safety_margin", 1.0) or 1.0)
    return max(int(target_tpm * safety_margin), 1)


def _fit_wave_to_tpm_budget(
    pending_rows,
    cursor,
    controller,
    step_config,
    task_handler,
    prompt_context,
    runtime,
):
    requested_group_size = derive_group_size(
        controller.load_budget,
        controller.concurrency,
        runtime["min_group_size"],
        runtime["max_group_size"],
    )
    group_size = requested_group_size
    budget_limit = _budget_limit(runtime)

    while True:
        wave, _ = _build_wave_requests(
            pending_rows=pending_rows,
            cursor=cursor,
            group_size=group_size,
            concurrency=controller.concurrency,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
        )
        if not wave:
            return wave, group_size, None

        if budget_limit <= 0:
            return wave, group_size, None

        estimated_wave_tokens, _ = _estimate_wave_tokens(wave, runtime)
        if estimated_wave_tokens <= budget_limit or group_size <= runtime["min_group_size"]:
            note = None
            if group_size != requested_group_size:
                note = (
                    f"tpm_fit_group_size {requested_group_size}->{group_size} "
                    f"estimated_wave_tokens={estimated_wave_tokens} budget_limit={budget_limit}"
                )
            return wave, group_size, note

        next_group_size = max(runtime["min_group_size"], int(math.floor(group_size * 0.80)))
        if next_group_size >= group_size:
            next_group_size = group_size - 1
        if next_group_size < runtime["min_group_size"]:
            next_group_size = runtime["min_group_size"]
        if next_group_size == group_size:
            return wave, group_size, None
        group_size = next_group_size


def _wait_for_tpm_budget(step_name: str, runtime: dict, token_ledger: list[dict], wave) -> tuple[int, int]:
    budget_limit = _budget_limit(runtime)
    if budget_limit <= 0 or not wave:
        return 0, 0

    estimated_wave_tokens, _ = _estimate_wave_tokens(wave, runtime)
    max_sleep_chunk = float(runtime.get("max_sleep_to_respect_tpm_seconds", 180) or 0)

    while True:
        now = time.time()
        _prune_token_ledger(token_ledger, now)
        rolling_tokens = _rolling_window_tokens(token_ledger)

        if rolling_tokens + estimated_wave_tokens <= budget_limit:
            if rolling_tokens > 0 or estimated_wave_tokens > 0:
                print(
                    (
                        f"[llm:{step_name}] tpm_window rolling_tokens={rolling_tokens} "
                        f"estimated_next_wave_tokens={estimated_wave_tokens} "
                        f"budget_limit={budget_limit}"
                    ),
                    flush=True,
                )
            return rolling_tokens, estimated_wave_tokens

        if not token_ledger:
            print(
                (
                    f"[llm:{step_name}] tpm_warning estimated_next_wave_tokens={estimated_wave_tokens} "
                    f"budget_limit={budget_limit} note=single_wave_estimate_exceeds_budget"
                ),
                flush=True,
            )
            return rolling_tokens, estimated_wave_tokens

        needed = (rolling_tokens + estimated_wave_tokens) - budget_limit
        sorted_entries = sorted(token_ledger, key=lambda row: row["sent_at"])
        releasable = 0
        wait_until = None
        for row in sorted_entries:
            releasable += int(row.get("estimated_tokens", 0) or 0)
            wait_until = row["sent_at"] + 60.0
            if releasable >= needed:
                break

        if wait_until is None:
            return rolling_tokens, estimated_wave_tokens

        wait_seconds = max(wait_until - now, 0.0)
        if max_sleep_chunk > 0:
            wait_seconds = min(wait_seconds, max_sleep_chunk)

        if wait_seconds <= 0:
            _prune_token_ledger(token_ledger, time.time())
            continue

        print(
            (
                f"[llm:{step_name}] sleeping sleep_s={wait_seconds:.2f} "
                f"reason=tpm_budget_wait rolling_tokens={rolling_tokens} "
                f"estimated_next_wave_tokens={estimated_wave_tokens} budget_limit={budget_limit}"
            ),
            flush=True,
        )
        time.sleep(wait_seconds)


# noinspection PyShadowingNames

def process_llm_batches(
    *,
    adapter,
    task_handler,
    prompt_context,
    pending_rows,
    progress_df,
    progress_by_unit,
    step_config,
    runtime,
    flush_callback,
):
    total_units = len(pending_rows)
    if total_units == 0:
        return {
            "outcome": LLMRunOutcome(status="complete", processed_units=0, remaining_units=0),
            "current_group_size": runtime["initial_group_size"],
            "current_concurrency": runtime["initial_concurrency"],
            "current_load_budget": int(runtime["initial_load_budget"]),
        }

    controller = ControllerState(
        load_budget=float(runtime["initial_load_budget"]),
        concurrency=int(runtime["initial_concurrency"]),
        throughput_ema=None,
        good_wave_streak=0,
        bad_wave_streak=0,
        waves_since_concurrency_change=0,
    )

    cursor = 0
    processed_count = 0
    start_time = time.time()
    last_flush_time = start_time
    completed_flushes = 0
    wave_index = 0
    groups_since_flush = 0
    units_since_flush = 0
    token_ledger: list[dict] = []

    pending_progress_rows = []
    pending_result_rows = []
    pending_debug_rows = []

    print(
        (
            f"[llm:{step_config['name']}] run_start total_units={total_units} "
            f"initial_group_size={runtime['initial_group_size']} "
            f"initial_concurrency={runtime['initial_concurrency']} "
            f"initial_load_budget={int(runtime['initial_load_budget'])} "
            f"target_tokens_per_minute={int(runtime.get('target_tokens_per_minute', 0) or 0)} "
            f"tpm_safety_margin={float(runtime.get('tpm_safety_margin', 1.0) or 1.0):.2f}"
        ),
        flush=True,
    )

    while cursor < total_units:
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes >= runtime["soft_time_limit_minutes"]:
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
                note="soft_time_limit_reached",
                start_time=start_time,
            )
            break

        wave_index += 1
        wave, group_size, tpm_fit_note = _fit_wave_to_tpm_budget(
            pending_rows=pending_rows,
            cursor=cursor,
            controller=controller,
            step_config=step_config,
            task_handler=task_handler,
            prompt_context=prompt_context,
            runtime=runtime,
        )
        if not wave:
            break

        rolling_tokens_before, estimated_wave_tokens = _wait_for_tpm_budget(
            step_name=step_config["name"],
            runtime=runtime,
            token_ledger=token_ledger,
            wave=wave,
        )

        for item in wave:
            item["estimated_tokens"] = _estimate_request_tokens(item["request"], runtime)

        wave_start = time.time()
        for item in wave:
            token_ledger.append(
                {
                    "sent_at": wave_start,
                    "estimated_tokens": item["estimated_tokens"],
                    "request_id": item["request"].get("request_id"),
                }
            )

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

        log_note_suffix = []
        if tpm_fit_note:
            log_note_suffix.append(tpm_fit_note)
        if estimated_wave_tokens:
            log_note_suffix.append(
                f"tpm_estimated_wave_tokens={estimated_wave_tokens} rolling_tokens_before={rolling_tokens_before}"
            )

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
                note=f"wave_request_exception {'|'.join(control_notes + log_note_suffix)}",
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
                            f"estimated_tokens={item.get('estimated_tokens', 0)} "
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
                            f"estimated_tokens={item.get('estimated_tokens', 0)} "
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
                            f"estimated_tokens={item.get('estimated_tokens', 0)} "
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
                note="wave_complete " + "|".join(control_notes + log_note_suffix),
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



def process_batches(*, adapter, task_handler, prompt_context, pending_rows, progress_df, step_config, runtime, flush_callback, start_time=None):
    progress_by_unit = {}
    if progress_df is not None and not progress_df.is_empty():
        for row in progress_df.to_dicts():
            progress_by_unit[row["unit_id"]] = row
    return process_llm_batches(
        adapter=adapter,
        task_handler=task_handler,
        prompt_context=prompt_context,
        pending_rows=pending_rows,
        progress_df=progress_df,
        progress_by_unit=progress_by_unit,
        step_config=step_config,
        runtime=runtime,
        flush_callback=flush_callback,
    )
