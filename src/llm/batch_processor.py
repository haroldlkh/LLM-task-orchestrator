import math
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

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
from .validators import validate_grouped_parse_results


TERMINAL_STATUSES = {"success", "permanent_error"}


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


def _lane_safe_budget(runtime: dict) -> float | None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if target_tpm is None:
        return None
    return float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)


def _lane_wait_seconds_for_tpm(runtime: dict, token_window: list[dict], next_wave_tokens: int, now: float) -> float:
    safe_budget = _lane_safe_budget(runtime)
    if safe_budget is None:
        return 0.0

    rolling_tokens = _rolling_window_tokens(token_window, now)
    if rolling_tokens + next_wave_tokens <= safe_budget:
        return 0.0
    if not token_window:
        return 0.0

    oldest_expiry = min(entry["sent_at"] + 60.0 for entry in token_window)
    sleep_seconds = max(0.0, oldest_expiry - now)
    return min(sleep_seconds, float(runtime.get("max_sleep_to_respect_tpm_seconds", 120) or 120))


def _build_group_request(queue_snapshot, group_size, step_config, task_handler, prompt_context):
    group_units = list(queue_snapshot)[:group_size]
    if not group_units:
        return None, None
    requests = task_handler.build_requests(group_units, step_config, prompt_context)
    if not requests:
        raise ValueError("Task handler returned no grouped requests")
    if len(requests) != 1:
        raise ValueError(
            "Grouped batching expects task_handler.build_requests(...) to return exactly one grouped request per group"
        )
    return group_units, requests[0]


def _latest_statuses(progress_by_unit: dict) -> dict[str, str]:
    return {
        unit_id: (row.get("status") if isinstance(row, dict) else None)
        for unit_id, row in progress_by_unit.items()
    }


def _remaining_units_from_statuses(progress_by_unit: dict, total_units: int) -> int:
    statuses = _latest_statuses(progress_by_unit)
    terminal_count = sum(1 for status in statuses.values() if status in TERMINAL_STATUSES)
    return max(total_units - terminal_count, 0)


def _terminal_units_from_statuses(progress_by_unit: dict) -> int:
    statuses = _latest_statuses(progress_by_unit)
    return sum(1 for status in statuses.values() if status in TERMINAL_STATUSES)


def _seed_progress_map(progress_df) -> dict:
    if progress_df is None or progress_df.is_empty():
        return {}
    progress_df = progress_df.sort(["unit_id", "updated_at"]).group_by("unit_id").tail(1)
    return {row["unit_id"]: row for row in progress_df.iter_rows(named=True)}


def _requeue_retryable_units(pending_queue: deque, group_units, statuses_by_unit: dict):
    for unit in group_units:
        if statuses_by_unit.get(unit["unit_id"]) == "retryable_error":
            pending_queue.append(unit)


def _lane_health_key(lane: dict) -> tuple:
    return (lane["health_score"], -lane["waves_started"], -lane["cooldown_until"])


def _lane_next_available_delay(lane: dict, runtime: dict, queue_snapshot, step_config, task_handler, prompt_context, now: float):
    if lane["in_flight"]:
        return None
    group_size = derive_group_size(
        lane["controller"].load_budget,
        lane["controller"].concurrency,
        runtime["min_group_size"],
        runtime["max_group_size"],
    )
    group_units, request = _build_group_request(queue_snapshot, group_size, step_config, task_handler, prompt_context)
    if not group_units:
        return None
    estimated_tokens = _estimate_request_tokens(request, group_units, runtime)
    delay = max(0.0, lane["cooldown_until"] - now)
    delay = max(delay, _lane_wait_seconds_for_tpm(runtime, lane["token_window"], estimated_tokens, now))
    return delay


def _safe_primary_candidate(lanes, primary_key_alias):
    if primary_key_alias is None:
        return None
    for lane in lanes:
        if lane["key_alias"] == primary_key_alias:
            return lane
    return None


def _count_recent_shared_failures(shared_failure_events: deque, runtime: dict, now: float) -> int:
    window = float(runtime.get("shared_failure_window_seconds", 90) or 90)
    while shared_failure_events and (now - shared_failure_events[0][0]) > window:
        shared_failure_events.popleft()
    return len({lane_alias for _, lane_alias in shared_failure_events})


def _eligible_candidate_for_lane(lane, runtime, queue_snapshot, step_config, task_handler, prompt_context, now: float):
    if lane["in_flight"]:
        return None
    group_size = derive_group_size(
        lane["controller"].load_budget,
        lane["controller"].concurrency,
        runtime["min_group_size"],
        runtime["max_group_size"],
    )
    group_units, request = _build_group_request(queue_snapshot, group_size, step_config, task_handler, prompt_context)
    if not group_units:
        return None
    estimated_tokens = _estimate_request_tokens(request, group_units, runtime)
    tpm_delay = _lane_wait_seconds_for_tpm(runtime, lane["token_window"], estimated_tokens, now)
    cooldown_delay = max(0.0, lane["cooldown_until"] - now)
    return {
        "lane": lane,
        "group_units": group_units,
        "request": request,
        "estimated_tokens": estimated_tokens,
        "tpm_delay": tpm_delay,
        "cooldown_delay": cooldown_delay,
    }


def _select_dispatch_lane(
    lanes,
    runtime,
    pending_queue,
    step_config,
    task_handler,
    prompt_context,
    active_lane_limit: int,
    current_primary_lane: str | None = None,
):
    now = time.time()
    queue_snapshot = list(pending_queue)
    strategy = runtime.get("lane_strategy", "hybrid")

    if sum(1 for lane in lanes if lane["in_flight"]) >= active_lane_limit:
        return None

    candidates = []
    blocked_candidates = []
    for lane in lanes:
        candidate = _eligible_candidate_for_lane(lane, runtime, queue_snapshot, step_config, task_handler, prompt_context, now)
        if candidate is None:
            continue
        if candidate["tpm_delay"] > 0 or candidate["cooldown_delay"] > 0:
            blocked_candidates.append(candidate)
            continue
        candidates.append(candidate)

    if strategy == "safe_single_active":
        primary_lane = _safe_primary_candidate(lanes, current_primary_lane)
        if primary_lane is not None:
            primary_candidates = [item for item in candidates if item["lane"]["key_alias"] == primary_lane["key_alias"]]
            if primary_candidates:
                return primary_candidates[0]
            if not runtime.get("allow_spillover_when_tpm_blocked", True):
                return None
            primary_blocked = [item for item in blocked_candidates if item["lane"]["key_alias"] == primary_lane["key_alias"]]
            if primary_blocked:
                return None

        if not candidates:
            return None
        candidates.sort(key=lambda item: _lane_health_key(item["lane"]), reverse=True)
        return candidates[0]

    if not candidates:
        return None
    candidates.sort(key=lambda item: _lane_health_key(item["lane"]), reverse=True)
    return candidates[0]


def _dispatch_to_lane(executor, adapter, lane, group_units, request, estimated_tokens):
    now = time.time()
    lane["token_window"].append({"sent_at": now, "tokens": int(estimated_tokens)})
    lane["in_flight"] = True
    lane["waves_started"] += 1
    future = executor.submit(adapter.execute_on_lane, lane["key_alias"], request, lane["step_config"])
    lane["last_started_at"] = now
    return future


def _update_lane_after_wave(lane: dict, runtime: dict, group_size: int, wave_metrics: dict):
    controller, control_notes = choose_next_controller_state(
        controller=lane["controller"],
        runtime=runtime,
        group_size_used=group_size,
        wave_metrics=wave_metrics,
    )
    lane["controller"] = controller

    if wave_metrics["transport_failures"] >= wave_metrics["total_requests"] and wave_metrics["total_requests"] > 0:
        cooldown = float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0)
        lane["health_score"] -= 8.0
    elif wave_metrics["transport_failures"] > 0:
        cooldown = float(runtime.get("transport_failure_cooldown_seconds", 0) or 0)
        lane["health_score"] -= 3.0
    else:
        cooldown = float(runtime.get("min_inter_wave_sleep_seconds", 0) or 0)
        lane["health_score"] += float(wave_metrics.get("useful_work", 0)) / max(float(wave_metrics.get("elapsed_request_seconds", 1.0)), 1.0)
        lane["health_score"] -= float(wave_metrics.get("failure_units", 0)) / max(float(wave_metrics.get("processed_units", 1.0)), 1.0)

    lane["cooldown_until"] = max(lane["cooldown_until"], time.time() + cooldown)
    return control_notes


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
    pending_queue = deque(pending_rows)
    total_units = len({row["unit_id"] for row in pending_rows})
    soft_time_limit_seconds = runtime["soft_time_limit_minutes"] * 60

    pending_result_rows = []
    pending_progress_rows = []
    pending_debug_rows = []

    progress_by_unit = _seed_progress_map(progress_df)
    attempted_dispatch_units = 0
    wave_index = 0
    groups_since_flush = 0
    units_since_flush = 0
    completed_flushes = 0
    last_flush_time = time.time()

    lanes = []
    for lane in adapter.lanes:
        lanes.append(
            {
                "key_alias": lane.key_alias,
                "provider": lane.provider,
                "model": lane.model,
                "controller": ControllerState(
                    load_budget=float(runtime["initial_load_budget"]),
                    concurrency=1,
                ),
                "token_window": [],
                "cooldown_until": 0.0,
                "in_flight": False,
                "health_score": 0.0,
                "waves_started": 0,
                "last_started_at": 0.0,
                "step_config": step_config,
            }
        )

    active = {}
    lane_strategy = runtime.get("lane_strategy", "hybrid")
    current_primary_lane = lanes[0]["key_alias"] if lanes else None
    current_active_lane_limit = 1 if lane_strategy == "safe_single_active" else min(max(1, int(runtime.get("initial_active_lanes", 1))), max(1, min(len(lanes), int(runtime.get("max_active_lanes", len(lanes) or 1)))))
    max_active_lanes = max(1, min(len(lanes), int(runtime.get("max_active_lanes", len(lanes) or 1))))
    shared_failure_events = deque()
    last_lane_limit_reduction_wave = 0

    with ThreadPoolExecutor(max_workers=max(1, len(lanes))) as executor:
        while pending_queue or active:
            elapsed = time.time() - start_time
            dispatch_allowed = elapsed < soft_time_limit_seconds

            while dispatch_allowed and pending_queue:
                selected = _select_dispatch_lane(
                    lanes, runtime, pending_queue, step_config, task_handler, prompt_context,
                    active_lane_limit=current_active_lane_limit,
                    current_primary_lane=current_primary_lane,
                )
                if selected is None:
                    break

                lane = selected["lane"]
                group_units = selected["group_units"]
                request = selected["request"]
                estimated_tokens = selected["estimated_tokens"]
                for _ in range(len(group_units)):
                    pending_queue.popleft()
                attempted_dispatch_units += len(group_units)
                wave_index += 1
                rolling_tokens = _rolling_window_tokens(lane["token_window"], time.time())
                log_progress(
                    step_name=step_config["name"],
                    wave_index=wave_index,
                    total_units=total_units,
                    processed_count=_terminal_units_from_statuses(progress_by_unit),
                    cursor=attempted_dispatch_units,
                    current_group_size=len(group_units),
                    current_concurrency=1,
                    current_load_budget=lane["controller"].load_budget,
                    success_streak=lane["controller"].good_wave_streak,
                    progress_by_unit=progress_by_unit,
                    note=(
                        f"lane={lane['key_alias']} wave_start groups=1 size={len(group_units)} "
                        f"estimated_next_wave_tokens={estimated_tokens} rolling_tokens={rolling_tokens} "
                        f"lane_strategy={lane_strategy} active_lane_limit={current_active_lane_limit}"
                    ),
                    start_time=start_time,
                    remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
                )
                future = _dispatch_to_lane(executor, adapter, lane, group_units, request, estimated_tokens)
                active[future] = {
                    "lane": lane,
                    "group_units": group_units,
                    "request": request,
                    "estimated_tokens": estimated_tokens,
                    "wave_index": wave_index,
                }

            if not active:
                if not dispatch_allowed:
                    log_progress(
                        step_name=step_config["name"],
                        wave_index=wave_index,
                        total_units=total_units,
                        processed_count=_terminal_units_from_statuses(progress_by_unit),
                        cursor=attempted_dispatch_units,
                        current_group_size=0,
                        current_concurrency=current_active_lane_limit,
                        current_load_budget=sum(l["controller"].load_budget for l in lanes),
                        success_streak=0,
                        progress_by_unit=progress_by_unit,
                        note="soft_time_limit_reached",
                        start_time=start_time,
                        remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
                    )
                    break
                delays = [
                    _lane_next_available_delay(lane, runtime, list(pending_queue), step_config, task_handler, prompt_context, time.time())
                    for lane in lanes
                ]
                delays = [d for d in delays if d is not None]
                if delays:
                    time.sleep(max(min(delays), 0.05))
                    continue
                break

            timeout = None
            delays = [
                _lane_next_available_delay(lane, runtime, list(pending_queue), step_config, task_handler, prompt_context, time.time())
                for lane in lanes
                if not lane["in_flight"]
            ]
            delays = [d for d in delays if d is not None and d > 0]
            if delays:
                timeout = max(min(delays), 0.05)

            done, _ = wait(active.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for future in done:
                context = active.pop(future)
                lane = context["lane"]
                lane["in_flight"] = False
                group_units = context["group_units"]
                request = context["request"]
                group_size = len(group_units)
                request_result = future.result()

                wave_metrics = {
                    "processed_units": group_size,
                    "useful_work": 0,
                    "failure_units": 0,
                    "transport_failures": 0,
                    "successful_requests": 0,
                    "total_requests": 1,
                    "elapsed_request_seconds": float(request_result.get("request_seconds", 0.0) or 0.0),
                }

                if request_result["status"] != "success":
                    progress_rows, debug_rows = mark_group_transport_failure(
                        group_units=group_units,
                        request=request,
                        request_result=request_result,
                        progress_df=progress_df,
                        request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1),
                    )
                    pending_progress_rows.extend(progress_rows)
                    pending_debug_rows.extend(debug_rows)
                    merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                    _requeue_retryable_units(
                        pending_queue,
                        group_units,
                        {row["unit_id"]: row["status"] for row in progress_rows},
                    )
                    groups_since_flush += 1
                    units_since_flush += group_size
                    wave_metrics["failure_units"] = group_size
                    wave_metrics["transport_failures"] = 1
                    control_notes = _update_lane_after_wave(lane, runtime, group_size, wave_metrics)
                    shared_failure_events.append((time.time(), lane["key_alias"]))
                    if lane_strategy == "safe_single_active" and current_primary_lane == lane["key_alias"]:
                        current_primary_lane = None
                    recent_shared_failures = _count_recent_shared_failures(shared_failure_events, runtime, time.time())
                    if lane_strategy == "hybrid" and recent_shared_failures >= int(runtime.get("shared_failure_lane_threshold", 2)):
                        if (context["wave_index"] - last_lane_limit_reduction_wave) >= int(runtime.get("lane_reduction_cooldown_waves", 2)):
                            current_active_lane_limit = max(1, current_active_lane_limit - 1)
                            last_lane_limit_reduction_wave = context["wave_index"]
                            control_notes.append(f"active_lanes_down_shared_failure={current_active_lane_limit}")
                    log_progress(
                        step_name=step_config["name"],
                        wave_index=context["wave_index"],
                        total_units=total_units,
                        processed_count=_terminal_units_from_statuses(progress_by_unit),
                        cursor=attempted_dispatch_units,
                        current_group_size=derive_group_size(lane["controller"].load_budget, 1, runtime["min_group_size"], runtime["max_group_size"]),
                        current_concurrency=1,
                        current_load_budget=lane["controller"].load_budget,
                        success_streak=lane["controller"].good_wave_streak,
                        progress_by_unit=progress_by_unit,
                        note=f"lane={lane['key_alias']} wave_complete {'|'.join(control_notes)} status={request_result['status']}",
                        start_time=start_time,
                        remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
                    )
                else:
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
                            request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1),
                        )
                        pending_progress_rows.extend(progress_rows)
                        pending_result_rows.extend(result_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        _requeue_retryable_units(
                            pending_queue,
                            group_units,
                            {row["unit_id"]: row["status"] for row in progress_rows},
                        )
                        groups_since_flush += 1
                        units_since_flush += group_size

                        useful_work = sum(1 for row in parse_results if row["status"] == "success")
                        failure_units = group_size - useful_work
                        wave_metrics["useful_work"] = useful_work
                        wave_metrics["failure_units"] = failure_units
                        wave_metrics["successful_requests"] = 1
                        control_notes = _update_lane_after_wave(lane, runtime, group_size, wave_metrics)
                        if lane_strategy == "safe_single_active":
                            current_primary_lane = lane["key_alias"]
                        elif lane_strategy == "hybrid":
                            current_primary_lane = lane["key_alias"]
                            if lane["controller"].good_wave_streak >= int(runtime.get("lane_exploration_success_waves", 2)) and current_active_lane_limit < max_active_lanes:
                                current_active_lane_limit += 1
                                control_notes.append(f"active_lanes_up_healthy={current_active_lane_limit}")
                        _count_recent_shared_failures(shared_failure_events, runtime, time.time())
                        log_progress(
                            step_name=step_config["name"],
                            wave_index=context["wave_index"],
                            total_units=total_units,
                            processed_count=_terminal_units_from_statuses(progress_by_unit),
                            cursor=attempted_dispatch_units,
                            current_group_size=derive_group_size(lane["controller"].load_budget, 1, runtime["min_group_size"], runtime["max_group_size"]),
                            current_concurrency=1,
                            current_load_budget=lane["controller"].load_budget,
                            success_streak=lane["controller"].good_wave_streak,
                            progress_by_unit=progress_by_unit,
                            note=(
                                f"lane={lane['key_alias']} wave_complete useful_work={useful_work} "
                                f"failure_rate={failure_units / max(group_size,1):.4f} score={request_score(useful_work, max(wave_metrics['elapsed_request_seconds'],1e-9)):.4f} {'|'.join(control_notes)}"
                            ),
                            start_time=start_time,
                            remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
                        )
                    except Exception as e:
                        progress_rows, debug_rows = mark_group_parse_failure(
                            group_units=group_units,
                            request=request,
                            raw_output=request_result.get("raw_output"),
                            error_type="group_parse_exception",
                            error_message=str(e),
                            progress_df=progress_df,
                            request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1),
                            request_result=request_result,
                        )
                        pending_progress_rows.extend(progress_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        _requeue_retryable_units(
                            pending_queue,
                            group_units,
                            {row["unit_id"]: row["status"] for row in progress_rows},
                        )
                        groups_since_flush += 1
                        units_since_flush += group_size
                        wave_metrics["failure_units"] = group_size
                        control_notes = _update_lane_after_wave(lane, runtime, group_size, wave_metrics)
                        log_progress(
                            step_name=step_config["name"],
                            wave_index=context["wave_index"],
                            total_units=total_units,
                            processed_count=_terminal_units_from_statuses(progress_by_unit),
                            cursor=attempted_dispatch_units,
                            current_group_size=derive_group_size(lane["controller"].load_budget, 1, runtime["min_group_size"], runtime["max_group_size"]),
                            current_concurrency=1,
                            current_load_budget=lane["controller"].load_budget,
                            success_streak=lane["controller"].good_wave_streak,
                            progress_by_unit=progress_by_unit,
                            note=f"lane={lane['key_alias']} wave_complete group_parse_exception {'|'.join(control_notes)}",
                            start_time=start_time,
                            remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
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
                            "processed_units": _terminal_units_from_statuses(progress_by_unit),
                            "remaining_units": _remaining_units_from_statuses(progress_by_unit, total_units),
                            "current_group_size": max(
                                derive_group_size(l["controller"].load_budget, 1, runtime["min_group_size"], runtime["max_group_size"])
                                for l in lanes
                            ),
                            "current_concurrency": sum(1 for l in lanes if l["in_flight"]),
                            "current_load_budget": int(sum(l["controller"].load_budget for l in lanes)),
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
                            pending_queue.clear()
                            break

    if pending_progress_rows or pending_result_rows or pending_debug_rows:
        flush_payload = flush_callback(
            {
                "progress_rows": list(pending_progress_rows),
                "result_rows": list(pending_result_rows),
                "debug_rows": list(pending_debug_rows),
                "processed_units": _terminal_units_from_statuses(progress_by_unit),
                "remaining_units": _remaining_units_from_statuses(progress_by_unit, total_units),
                "current_group_size": 0,
                "current_concurrency": 0,
                "current_load_budget": int(sum(l["controller"].load_budget for l in lanes)),
            }
        )
        if flush_payload and flush_payload.get("counted_flush"):
            completed_flushes += 1

    remaining_units = _remaining_units_from_statuses(progress_by_unit, total_units)
    outcome = LLMRunOutcome(
        status="complete" if remaining_units == 0 else "retryable_incomplete",
        processed_units=_terminal_units_from_statuses(progress_by_unit),
        remaining_units=remaining_units,
    )

    print(
        (
            f"[llm:{step_config['name']}] "
            f"run_end processed={outcome.processed_units}/{total_units} "
            f"remaining={remaining_units} "
            f"outcome={outcome.status} "
            f"completed_flushes={completed_flushes}"
        ),
        flush=True,
    )

    return {"outcome": outcome}
