import math
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .bandits import UCBPolicy, build_payload_arms, compute_lane_reward, compute_payload_reward
from .bandits.state import utc_now_iso
from .batch_rows import (
    build_rows_from_group_parse,
    mark_group_parse_failure,
    mark_group_transport_failure,
)
from .batch_runtime import (
    log_progress,
    merge_progress_rows_in_memory,
    request_score,
    should_flush,
)
from .models import LLMRunOutcome
from .validators import validate_grouped_parse_results

TERMINAL_STATUSES = {"success", "permanent_error"}
PERMANENT_ERROR_ELIGIBLE_TYPES = {
    "invalid_score_output",
    "invalid_cue_output",
    "missing_prompt_unit_result",
    "semantic_validation_failed",
    "invalid_unit_output",
    "missing_required_field",
    "unknown_prompt_unit_id",
}


def _eligible_for_permanent_error(row: dict) -> bool:
    if row.get("status") != "retryable_error":
        return False
    error_type = row.get("last_error_type")
    if not error_type:
        return False
    primary_error_type = str(error_type).split("|")[0]
    return primary_error_type in PERMANENT_ERROR_ELIGIBLE_TYPES


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


def _available_lane_count(runtime: dict, lanes) -> int:
    configured = runtime.get("available_lane_count")
    if configured is not None:
        return max(int(configured), 1)
    if lanes is not None:
        return max(len(lanes), 1)
    return 1


def _per_key_target_tpm(runtime: dict) -> float | None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if target_tpm is None:
        return None
    return float(target_tpm)


def _pool_target_tpm(runtime: dict, lanes=None) -> float | None:
    per_key = _per_key_target_tpm(runtime)
    if per_key is None:
        return None
    return per_key * _available_lane_count(runtime, lanes)


def _lane_safe_budget(runtime: dict) -> float | None:
    per_key = _per_key_target_tpm(runtime)
    if per_key is None:
        return None
    return per_key * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)


def _pool_safe_budget(runtime: dict, lanes=None) -> float | None:
    pool_target = _pool_target_tpm(runtime, lanes)
    if pool_target is None:
        return None
    return pool_target * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)


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


def _build_group_request(group_units, step_config, task_handler, prompt_context):
    if not group_units:
        return None, None
    requests = task_handler.build_requests(group_units, step_config, prompt_context)
    if not requests:
        raise ValueError("Task handler returned no grouped requests")
    if len(requests) != 1:
        raise ValueError("Grouped batching expects one grouped request per group")
    return group_units, requests[0]


def _build_group_request_for_payload_arm(queue_snapshot, payload_arm, runtime, step_config, task_handler, prompt_context):
    if not queue_snapshot:
        return None
    min_group = int(runtime.get("min_group_size", 1) or 1)
    max_group = int(runtime.get("max_group_size", max(min_group, 1)) or max(min_group, 1))
    target_tokens = int(payload_arm["target_request_tokens"])
    selected = []
    selected_request = None
    selected_tokens = 0
    for idx in range(1, min(len(queue_snapshot), max_group) + 1):
        maybe_units = list(queue_snapshot)[:idx]
        _, request = _build_group_request(maybe_units, step_config, task_handler, prompt_context)
        est = _estimate_request_tokens(request, maybe_units, runtime)
        if idx < min_group:
            selected, selected_request, selected_tokens = maybe_units, request, est
            continue
        if est <= target_tokens or not selected:
            selected, selected_request, selected_tokens = maybe_units, request, est
            if est <= target_tokens:
                continue
        if est > target_tokens:
            break
    if not selected:
        return None
    return {
        "group_units": selected,
        "request": selected_request,
        "estimated_tokens": selected_tokens,
        "payload_arm_id": payload_arm["arm_id"],
        "payload_target_request_tokens": target_tokens,
    }


def _latest_statuses(progress_by_unit: dict) -> dict[str, str]:
    return {unit_id: (row.get("status") if isinstance(row, dict) else None) for unit_id, row in progress_by_unit.items()}


def _remaining_units_from_statuses(progress_by_unit: dict, total_units: int) -> int:
    statuses = _latest_statuses(progress_by_unit)
    terminal_count = sum(1 for status in statuses.values() if status in TERMINAL_STATUSES)
    return max(total_units - terminal_count, 0)


def _terminal_units_from_statuses(progress_by_unit: dict) -> int:
    statuses = _latest_statuses(progress_by_unit)
    return sum(1 for status in statuses.values() if status in TERMINAL_STATUSES)


def _apply_retry_limits(progress_rows, runtime: dict):
    max_retries = int(runtime.get("max_request_retries", 0) or 0)
    if max_retries < 0:
        return progress_rows
    adjusted = []
    for row in progress_rows:
        new_row = dict(row)
        if _eligible_for_permanent_error(new_row) and int(new_row.get("retry_count", 0) or 0) >= max_retries:
            new_row["status"] = "permanent_error"
            prior_error_type = new_row.get("last_error_type")
            if prior_error_type and "max_request_retries_exhausted" not in str(prior_error_type):
                new_row["last_error_type"] = f"{prior_error_type}|max_request_retries_exhausted"
            elif not prior_error_type:
                new_row["last_error_type"] = "max_request_retries_exhausted"
            prior_error_message = new_row.get("last_error_message") or ""
            if "engine converted to permanent_error" not in prior_error_message:
                suffix = (
                    f" [engine converted to permanent_error after repeated unusable unit output reached retry_count={int(new_row.get('retry_count', 0) or 0)} and max_request_retries={max_retries}]"
                )
                new_row["last_error_message"] = f"{prior_error_message}{suffix}".strip()
        adjusted.append(new_row)
    return adjusted


def _run_scope_status_counts(progress_by_unit: dict, run_unit_ids: set[str]) -> dict[str, int]:
    counts = {"success": 0, "retryable_error": 0, "permanent_error": 0}
    if not run_unit_ids:
        return counts
    for unit_id in run_unit_ids:
        row = progress_by_unit.get(unit_id)
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _seed_progress_map(progress_df) -> dict:
    if progress_df is None or progress_df.is_empty():
        return {}
    progress_df = progress_df.sort(["unit_id", "updated_at"]).group_by("unit_id").tail(1)
    progress_df = progress_df.filter(progress_df["status"] == "success") if not progress_df.is_empty() else progress_df
    return {row["unit_id"]: row for row in progress_df.iter_rows(named=True)}


def _requeue_retryable_units(pending_queue: deque, group_units, statuses_by_unit: dict):
    for unit in group_units:
        if statuses_by_unit.get(unit["unit_id"]) == "retryable_error":
            pending_queue.append(unit)


def _lane_arm_id(lane: dict) -> str:
    return f"{lane['provider']}:{lane['model']}:{lane['key_alias']}"


def _provider_group_key(lane: dict) -> str:
    return str(lane.get("provider") or lane.get("key_alias"))


def _count_recent_shared_failures(shared_failure_events: deque, runtime: dict, now: float) -> int:
    window = float(runtime.get("shared_failure_window_seconds", 90) or 90)
    while shared_failure_events and (now - shared_failure_events[0][0]) > window:
        shared_failure_events.popleft()
    return len({lane_alias for _, lane_alias in shared_failure_events})


def _lane_next_available_delay(lane: dict, runtime: dict, queue_snapshot, step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, now: float):
    if lane["in_flight_count"] >= int(runtime.get("lane_max_in_flight_per_key", 2) or 2):
        return None
    allowed_payload_arm_ids = sorted(payload_arms_by_id.keys())
    if not allowed_payload_arm_ids:
        return None
    payload_arm_id = payload_bandit.select_arm(allowed_payload_arm_ids)
    candidate = _build_group_request_for_payload_arm(queue_snapshot, payload_arms_by_id[payload_arm_id], runtime, step_config, task_handler, prompt_context)
    if not candidate:
        return None
    delay = max(0.0, lane["cooldown_until"] - now)
    delay = max(delay, _lane_wait_seconds_for_tpm(runtime, lane["token_window"], candidate["estimated_tokens"], now))
    return delay


def _eligible_candidate_for_lane(lane, runtime, queue_snapshot, step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, provider_group_state, now: float):
    if lane["in_flight_count"] >= int(runtime.get("lane_max_in_flight_per_key", 2) or 2):
        return None
    group_key = _provider_group_key(lane)
    provider_pressure = provider_group_state.get(group_key, {})
    if int(provider_pressure.get("recent_throttles", 0) or 0) >= int(runtime.get("provider_group_throttle_threshold", 2) or 2):
        lane["cooldown_until"] = max(lane["cooldown_until"], now + float(runtime.get("transport_failure_cooldown_seconds", 0) or 0) * float(runtime.get("provider_group_cooldown_multiplier", 1.5) or 1.5))
    allowed_payload_arm_ids = sorted(payload_arms_by_id.keys())
    if not allowed_payload_arm_ids:
        return None
    payload_arm_id = payload_bandit.select_arm(allowed_payload_arm_ids)
    candidate = _build_group_request_for_payload_arm(queue_snapshot, payload_arms_by_id[payload_arm_id], runtime, step_config, task_handler, prompt_context)
    if not candidate:
        return None
    estimated_tokens = candidate["estimated_tokens"]
    tpm_delay = _lane_wait_seconds_for_tpm(runtime, lane["token_window"], estimated_tokens, now)
    cooldown_delay = max(0.0, lane["cooldown_until"] - now)
    return {
        "lane": lane,
        "group_units": candidate["group_units"],
        "request": candidate["request"],
        "estimated_tokens": estimated_tokens,
        "tpm_delay": tpm_delay,
        "cooldown_delay": cooldown_delay,
        "payload_arm_id": candidate["payload_arm_id"],
        "payload_target_request_tokens": candidate["payload_target_request_tokens"],
        "lane_arm_id": lane["lane_arm_id"],
        "provider_group": group_key,
    }


def _select_dispatch_candidate(lanes, runtime, pending_queue, step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, lane_bandit, provider_group_state, active_request_count: int):
    now = time.time()
    queue_snapshot = list(pending_queue)
    if active_request_count >= int(runtime.get("max_total_in_flight_requests", max(1, len(lanes))) or max(1, len(lanes))):
        return None
    candidates = []
    for lane in lanes:
        candidate = _eligible_candidate_for_lane(lane, runtime, queue_snapshot, step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, provider_group_state, now)
        if candidate is None:
            continue
        if candidate["tpm_delay"] > 0 or candidate["cooldown_delay"] > 0:
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    chosen_lane_arm_id = lane_bandit.select_arm(sorted({item["lane_arm_id"] for item in candidates}))
    for item in candidates:
        if item["lane_arm_id"] == chosen_lane_arm_id:
            return item
    return candidates[0]


def _dispatch_to_lane(executor, adapter, lane, request, estimated_tokens):
    now = time.time()
    lane["token_window"].append({"sent_at": now, "tokens": int(estimated_tokens)})
    lane["in_flight_count"] += 1
    lane["waves_started"] += 1
    future = executor.submit(adapter.execute_on_lane, lane["key_alias"], request, lane["step_config"], lane)
    lane["last_started_at"] = now
    return future


def _record_provider_group_event(provider_group_state: dict, group_key: str, throttled: bool, latency_seconds: float):
    state = provider_group_state.setdefault(group_key, {"recent_throttles": 0, "latency_ema": None, "updated_at": utc_now_iso()})
    if throttled:
        state["recent_throttles"] = int(state.get("recent_throttles", 0) or 0) + 1
    else:
        state["recent_throttles"] = max(int(state.get("recent_throttles", 0) or 0) - 1, 0)
    prev = state.get("latency_ema")
    alpha = 0.3
    state["latency_ema"] = latency_seconds if prev is None else (alpha * latency_seconds + (1 - alpha) * prev)
    state["updated_at"] = utc_now_iso()


def _export_bandit_state(payload_bandit, lane_bandit, provider_group_state: dict) -> dict:
    return {
        "updated_at": utc_now_iso(),
        "payload_bandit": payload_bandit.export_state(),
        "lane_bandit": lane_bandit.export_state(),
        "provider_group_state": provider_group_state,
    }


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
    initial_bandit_state=None,
):
    pending_rows = [] if pending_rows is None else list(pending_rows)
    pending_queue = deque(pending_rows)
    initial_pending_unit_ids = {row["unit_id"] for row in pending_rows}
    initial_pending_units = len(initial_pending_unit_ids)
    already_success_units = _terminal_units_from_statuses(_seed_progress_map(progress_df))
    total_units = initial_pending_units + already_success_units
    soft_time_limit_seconds = runtime["soft_time_limit_minutes"] * 60

    pending_result_rows = []
    pending_progress_rows = []
    pending_debug_rows = []
    progress_by_unit = _seed_progress_map(progress_df)
    already_terminal_units = _terminal_units_from_statuses(progress_by_unit)
    total_units = initial_pending_units + already_terminal_units
    attempted_dispatch_units = 0
    wave_index = 0
    groups_since_flush = 0
    units_since_flush = 0
    completed_flushes = 0
    last_flush_time = time.time()
    scheduler_delay_seconds = 0.0
    inflight_wait_seconds = 0.0
    api_send_to_receive_seconds = 0.0

    payload_arms = build_payload_arms([int(x) for x in runtime.get("payload_arm_targets", [8000, 16000, 24000, 32000])])
    payload_arms_by_id = {arm["arm_id"]: arm for arm in payload_arms}
    payload_bandit_state = (initial_bandit_state or {}).get("payload_bandit") if runtime.get("bandit_state_mode") == "cross_run" else None
    lane_bandit_state = (initial_bandit_state or {}).get("lane_bandit") if runtime.get("bandit_state_mode") == "cross_run" else None
    payload_bandit = UCBPolicy(arm_ids=sorted(payload_arms_by_id.keys()), exploration=float(runtime.get("payload_bandit_ucb_c", 1.2) or 1.2), state=payload_bandit_state)
    provider_group_state = dict((initial_bandit_state or {}).get("provider_group_state") or {})

    lanes = []
    lane_arm_ids = []
    for lane in adapter.lanes:
        lane_dict = {
            "key_alias": lane.key_alias,
            "provider": lane.provider,
            "model": lane.model,
            "token_window": [],
            "cooldown_until": 0.0,
            "in_flight_count": 0,
            "waves_started": 0,
            "last_started_at": 0.0,
            "success_request_seconds_window": [],
            "step_config": step_config,
        }
        lane_dict["lane_arm_id"] = _lane_arm_id(lane_dict)
        lanes.append(lane_dict)
        lane_arm_ids.append(lane_dict["lane_arm_id"])
    lane_bandit = UCBPolicy(arm_ids=sorted(lane_arm_ids), exploration=float(runtime.get("lane_bandit_ucb_c", 1.2) or 1.2), state=lane_bandit_state)

    active = {}
    shared_failure_events = deque()
    per_key_target_tpm = _per_key_target_tpm(runtime)
    per_key_safe_budget = _lane_safe_budget(runtime)
    pool_target_tpm = _pool_target_tpm(runtime, lanes)
    pool_safe_budget = _pool_safe_budget(runtime, lanes)
    if per_key_target_tpm is not None:
        print((
            f"[llm:{step_config['name']}] tpm_budget "
            f"per_key_target_tpm={int(per_key_target_tpm)} per_key_safe_budget={int(per_key_safe_budget or 0)} "
            f"pool_target_tpm={int(pool_target_tpm or 0)} pool_safe_budget={int(pool_safe_budget or 0)} available_lanes={len(lanes)}"
        ), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, int(runtime.get("max_total_in_flight_requests", len(lanes) or 1)))) as executor:
        while pending_queue or active:
            elapsed = time.time() - start_time
            dispatch_allowed = elapsed < soft_time_limit_seconds

            while dispatch_allowed and pending_queue:
                selected = _select_dispatch_candidate(
                    lanes, runtime, pending_queue, step_config, task_handler, prompt_context,
                    payload_bandit, payload_arms_by_id, lane_bandit, provider_group_state, len(active)
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
                    step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                    processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                    current_group_size=len(group_units), current_concurrency=len(active) + 1,
                    current_load_budget=selected["payload_target_request_tokens"], success_streak=0,
                    progress_by_unit=progress_by_unit,
                    note=(
                        f"event=request_sent lane={lane['key_alias']} batch_units={len(group_units)} "
                        f"payload_arm_id={selected['payload_arm_id']} payload_target_tokens={selected['payload_target_request_tokens']} "
                        f"estimated_request_tokens={estimated_tokens} lane_rolling_tokens={rolling_tokens} lane_arm_id={selected['lane_arm_id']}"
                    ),
                    start_time=start_time,
                    remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units),
                    run_total_units=initial_pending_units,
                    run_status_counts=_run_scope_status_counts(progress_by_unit, initial_pending_unit_ids),
                    queue_pending=len(pending_queue),
                )
                future = _dispatch_to_lane(executor, adapter, lane, request, estimated_tokens)
                active[future] = {
                    "lane": lane,
                    "group_units": group_units,
                    "request": request,
                    "estimated_tokens": estimated_tokens,
                    "wave_index": wave_index,
                    "payload_arm_id": selected["payload_arm_id"],
                    "lane_arm_id": selected["lane_arm_id"],
                    "provider_group": selected["provider_group"],
                    "payload_target_request_tokens": selected["payload_target_request_tokens"],
                }

            if not active:
                if not dispatch_allowed:
                    log_progress(step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                                 processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                                 current_group_size=0, current_concurrency=0, current_load_budget=0, success_streak=0,
                                 progress_by_unit=progress_by_unit, note="soft_time_limit_reached", start_time=start_time,
                                 remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units))
                    break
                delays = [
                    _lane_next_available_delay(lane, runtime, list(pending_queue), step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, time.time())
                    for lane in lanes
                ]
                delays = [d for d in delays if d is not None]
                if delays:
                    imposed_wait = max(min(delays), 0.05)
                    scheduler_delay_seconds += imposed_wait
                    log_progress(step_name=step_config["name"], wave_index=wave_index, total_units=total_units,
                                 processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                                 current_group_size=0, current_concurrency=0, current_load_budget=0, success_streak=0,
                                 progress_by_unit=progress_by_unit,
                                 note=f"event=scheduler_wait imposed_wait_s={imposed_wait:.2f}", start_time=start_time,
                                 remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units))
                    time.sleep(imposed_wait)
                    continue
                break

            timeout = float(runtime.get("inflight_heartbeat_seconds", 15) or 15)
            delays = [
                _lane_next_available_delay(lane, runtime, list(pending_queue), step_config, task_handler, prompt_context, payload_bandit, payload_arms_by_id, time.time())
                for lane in lanes if lane["in_flight_count"] < int(runtime.get("lane_max_in_flight_per_key", 2) or 2)
            ]
            delays = [d for d in delays if d is not None and d > 0]
            if delays:
                timeout = max(min(min(delays), timeout), 0.05)
            else:
                timeout = max(timeout, 0.05)
            wait_started = time.time()
            done, _ = wait(active.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
            waited_for = max(time.time() - wait_started, 0.0)
            inflight_wait_seconds += waited_for
            if not done:
                continue

            for future in list(done):
                context = active.pop(future)
                lane = context["lane"]
                lane["in_flight_count"] = max(lane["in_flight_count"] - 1, 0)
                request_result = future.result()
                api_send_to_receive_seconds += max(float(request_result.get("elapsed_request_seconds", 0.0) or 0.0), 0.0)
                group_units = context["group_units"]
                request = context["request"]
                group_size = len(group_units)
                useful_work = 0
                failure_units = 0
                retryable_units = 0
                throttled_flag = False
                wave_metrics = {
                    "processed_units": group_size,
                    "failure_units": 0,
                    "useful_work": 0,
                    "transport_failures": 0,
                    "successful_requests": 0,
                    "total_requests": 1,
                    "elapsed_request_seconds": float(request_result.get("elapsed_request_seconds", 0.0) or 0.0),
                    "retryable_units": 0,
                    "all_transport_failure": False,
                    "throttled_flag": False,
                }
                if request_result["status"] != "success":
                    progress_rows, debug_rows = mark_group_transport_failure(
                        group_units=group_units, request=request, request_result=request_result,
                        progress_df=progress_df, request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1),
                    )
                    progress_rows = _apply_retry_limits(progress_rows, runtime)
                    pending_progress_rows.extend(progress_rows)
                    pending_debug_rows.extend(debug_rows)
                    merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                    _requeue_retryable_units(pending_queue, group_units, {row["unit_id"]: row["status"] for row in progress_rows})
                    groups_since_flush += 1
                    units_since_flush += group_size
                    wave_metrics["failure_units"] = group_size
                    wave_metrics["transport_failures"] = 1
                    wave_metrics["retryable_units"] = sum(1 for row in progress_rows if row["status"] == "retryable_error")
                    wave_metrics["all_transport_failure"] = True
                    failure_units = group_size
                    retryable_units = wave_metrics["retryable_units"]
                    throttled_flag = "throttle" in str(request_result.get("error_type") or "").lower() or "rate" in str(request_result.get("error_message") or "").lower()
                    wave_metrics["throttled_flag"] = throttled_flag
                    if request_result["status"] == "retryable_error":
                        shared_failure_events.append((time.time(), lane["key_alias"]))
                    cooldown = float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0)
                    lane["cooldown_until"] = max(lane["cooldown_until"], time.time() + cooldown)
                    _record_provider_group_event(provider_group_state, context["provider_group"], throttled_flag, wave_metrics["elapsed_request_seconds"])
                    log_progress(step_name=step_config["name"], wave_index=context["wave_index"], total_units=total_units,
                                 processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                                 current_group_size=group_size, current_concurrency=len(active), current_load_budget=context["payload_target_request_tokens"], success_streak=0,
                                 progress_by_unit=progress_by_unit,
                                 note=f"event=response_done lane={lane['key_alias']} status={request_result['status']} payload_arm_id={context['payload_arm_id']}",
                                 start_time=start_time, remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units))
                else:
                    try:
                        parse_results = validate_grouped_parse_results(
                            group_units=group_units,
                            raw_output=request_result.get("raw_output"),
                            parsed_output=request_result.get("parsed_output"),
                            output_schema=request.get("output_schema"),
                            step_config=step_config,
                        )
                        progress_rows, result_rows, debug_rows = build_rows_from_group_parse(
                            group_units=group_units, request=request, request_result=request_result,
                            parse_results=parse_results, progress_df=progress_df,
                            request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1),
                        )
                        progress_rows = _apply_retry_limits(progress_rows, runtime)
                        pending_progress_rows.extend(progress_rows)
                        pending_result_rows.extend(result_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        _requeue_retryable_units(pending_queue, group_units, {row["unit_id"]: row["status"] for row in progress_rows})
                        groups_since_flush += 1
                        units_since_flush += group_size
                        useful_work = sum(1 for row in progress_rows if row["status"] == "success")
                        failure_units = sum(1 for row in progress_rows if row["status"] != "success")
                        retryable_units = sum(1 for row in progress_rows if row["status"] == "retryable_error")
                        wave_metrics["failure_units"] = failure_units
                        wave_metrics["useful_work"] = useful_work
                        wave_metrics["successful_requests"] = 1
                        wave_metrics["retryable_units"] = retryable_units
                        _record_provider_group_event(provider_group_state, context["provider_group"], False, wave_metrics["elapsed_request_seconds"])
                        log_progress(step_name=step_config["name"], wave_index=context["wave_index"], total_units=total_units,
                                 processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                                 current_group_size=group_size, current_concurrency=len(active), current_load_budget=context["payload_target_request_tokens"], success_streak=0,
                                 progress_by_unit=progress_by_unit,
                                 note=(f"event=response_done lane={lane['key_alias']} status=success useful_work={useful_work} failure_units={failure_units} "
                                       f"api_request_s={wave_metrics['elapsed_request_seconds']:.2f} score={request_score(useful_work, max(wave_metrics['elapsed_request_seconds'],1e-9)):.4f} payload_arm_id={context['payload_arm_id']}"),
                                 start_time=start_time, remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units))
                    except Exception as e:
                        progress_rows, debug_rows = mark_group_parse_failure(
                            group_units=group_units, request=request, raw_output=request_result.get("raw_output"),
                            error_type="group_parse_exception", error_message=str(e), progress_df=progress_df,
                            request_attempt_count=int(request_result.get("request_attempt_count", 1) or 1), request_result=request_result,
                        )
                        progress_rows = _apply_retry_limits(progress_rows, runtime)
                        pending_progress_rows.extend(progress_rows)
                        pending_debug_rows.extend(debug_rows)
                        merge_progress_rows_in_memory(progress_by_unit, progress_rows)
                        _requeue_retryable_units(pending_queue, group_units, {row["unit_id"]: row["status"] for row in progress_rows})
                        groups_since_flush += 1
                        units_since_flush += group_size
                        failure_units = group_size
                        retryable_units = sum(1 for row in progress_rows if row["status"] == "retryable_error")
                        wave_metrics["failure_units"] = failure_units
                        wave_metrics["retryable_units"] = retryable_units
                        _record_provider_group_event(provider_group_state, context["provider_group"], False, wave_metrics["elapsed_request_seconds"])
                        log_progress(step_name=step_config["name"], wave_index=context["wave_index"], total_units=total_units,
                                 processed_count=_terminal_units_from_statuses(progress_by_unit), cursor=attempted_dispatch_units,
                                 current_group_size=group_size, current_concurrency=len(active), current_load_budget=context["payload_target_request_tokens"], success_streak=0,
                                 progress_by_unit=progress_by_unit,
                                 note=f"event=response_done lane={lane['key_alias']} status=group_parse_exception payload_arm_id={context['payload_arm_id']}",
                                 start_time=start_time, remaining_override=_remaining_units_from_statuses(progress_by_unit, total_units))
                payload_bandit.update(context["payload_arm_id"], compute_payload_reward(wave_metrics, runtime), meta=wave_metrics)
                lane_bandit.update(context["lane_arm_id"], compute_lane_reward(wave_metrics, runtime), meta=wave_metrics)
                _count_recent_shared_failures(shared_failure_events, runtime, time.time())

                if should_flush(
                    pending_progress_rows=pending_progress_rows,
                    pending_result_rows=pending_result_rows,
                    pending_debug_rows=pending_debug_rows,
                    groups_since_flush=groups_since_flush,
                    units_since_flush=units_since_flush,
                    last_flush_time=last_flush_time,
                    runtime=runtime,
                ):
                    flush_payload = flush_callback({
                        "progress_rows": list(pending_progress_rows),
                        "result_rows": list(pending_result_rows),
                        "debug_rows": list(pending_debug_rows),
                        "processed_units": _terminal_units_from_statuses(progress_by_unit),
                        "remaining_units": _remaining_units_from_statuses(progress_by_unit, total_units),
                        "current_group_size": group_size,
                        "current_concurrency": len(active),
                        "current_load_budget": int(context["payload_target_request_tokens"]),
                        "bandit_state": _export_bandit_state(payload_bandit, lane_bandit, provider_group_state),
                    })
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
        flush_payload = flush_callback({
            "progress_rows": list(pending_progress_rows),
            "result_rows": list(pending_result_rows),
            "debug_rows": list(pending_debug_rows),
            "processed_units": _terminal_units_from_statuses(progress_by_unit),
            "remaining_units": _remaining_units_from_statuses(progress_by_unit, total_units),
            "current_group_size": 0,
            "current_concurrency": len(active),
            "current_load_budget": 0,
            "bandit_state": _export_bandit_state(payload_bandit, lane_bandit, provider_group_state),
        })
        if flush_payload and flush_payload.get("counted_flush"):
            completed_flushes += 1

    remaining_units = _remaining_units_from_statuses(progress_by_unit, total_units)
    run_counts = _run_scope_status_counts(progress_by_unit, initial_pending_unit_ids)
    outcome = LLMRunOutcome(status="complete" if remaining_units == 0 else "retryable_incomplete", processed_units=_terminal_units_from_statuses(progress_by_unit), remaining_units=remaining_units)
    total_wall_seconds = max(time.time() - start_time, 0.0)
    engine_processing_seconds = max(total_wall_seconds - api_send_to_receive_seconds - scheduler_delay_seconds, 0.0)
    print((
        f"[llm:{step_config['name']}] event=run_end processed={outcome.processed_units}/{total_units} remaining={remaining_units} "
        f"run_success={run_counts['success']} run_retryable_error={run_counts['retryable_error']} run_permanent_error={run_counts['permanent_error']} "
        f"run_processed={run_counts['success'] + run_counts['permanent_error']}/{initial_pending_units} "
        f"run_remaining={max(initial_pending_units - (run_counts['success'] + run_counts['permanent_error']), 0)} outcome={outcome.status} "
        f"completed_flushes={completed_flushes} total_wall_s={total_wall_seconds:.2f} api_send_to_receive_s={api_send_to_receive_seconds:.2f} "
        f"engine_processing_s={engine_processing_seconds:.2f} scheduler_delay_s={scheduler_delay_seconds:.2f} inflight_wait_s={inflight_wait_seconds:.2f}"
    ), flush=True)
    return {"outcome": outcome, "bandit_state": _export_bandit_state(payload_bandit, lane_bandit, provider_group_state)}
