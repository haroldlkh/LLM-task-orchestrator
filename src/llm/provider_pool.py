import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List

RETRYABLE_STATUSES = {"retryable_error"}
TRANSPORT_ERROR_TYPES = {
    "rate_limit",
    "server_error",
    "network_error",
    "transient_api_error",
    "empty_response",
    "wave_request_exception",
    "wave_request_timeout",
    "missing_lane_result",
}


def _estimate_request_tokens(request: Dict[str, Any], runtime: dict) -> int:
    chars_per_token = float(runtime.get("token_estimation_chars_per_token", 4.0) or 4.0)
    response_tokens_per_unit = float(runtime.get("estimated_response_tokens_per_unit", 24) or 24)
    overhead_tokens = float(runtime.get("estimated_request_overhead_tokens", 250) or 250)
    prompt_text = request.get("prompt", "") or ""
    unit_count = len(request.get("units") or [])
    prompt_tokens = math.ceil(len(prompt_text) / max(chars_per_token, 0.1))
    response_tokens = math.ceil(unit_count * response_tokens_per_unit)
    return max(int(prompt_tokens + response_tokens + overhead_tokens), 1)


def _prune_token_window(token_window: List[dict], now: float) -> List[dict]:
    cutoff = now - 60.0
    return [entry for entry in token_window if entry["sent_at"] > cutoff]


def _rolling_window_tokens(token_window: List[dict], now: float) -> int:
    token_window[:] = _prune_token_window(token_window, now)
    return int(sum(entry["tokens"] for entry in token_window))


@dataclass
class ProviderLane:
    key_alias: str
    provider: str | None
    adapter: Any
    model_override: str | None = None
    weight: float = 1.0
    cooldown_until: float = 0.0
    recent_successes: int = 0
    recent_failures: int = 0
    last_used_at: float = 0.0
    token_window: List[dict] = field(default_factory=list)

    def is_available(self, now: float) -> bool:
        return now >= self.cooldown_until

    def record_success(self, now: float) -> None:
        self.recent_successes += 1
        self.recent_failures = max(self.recent_failures - 1, 0)
        self.last_used_at = now

    def record_failure(self, now: float, cooldown_seconds: float) -> None:
        self.recent_failures += 1
        self.last_used_at = now
        if cooldown_seconds > 0:
            self.cooldown_until = max(self.cooldown_until, now + cooldown_seconds)


class ProviderPoolAdapter:
    def __init__(self, lanes: List[ProviderLane]):
        if not lanes:
            raise ValueError("ProviderPoolAdapter requires at least one lane")
        self.lanes = lanes
        self.is_lane_pool = True
        self._lane_cursor = 0
        self._preferred_lane_alias: str | None = None
        self._active_lane_limit = 1
        self._healthy_streak = 0
        self._failure_events: List[tuple[float, str]] = []

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def pool_size(self) -> int:
        return len(self.lanes)

    def get_lane(self, key_alias: str) -> ProviderLane:
        for lane in self.lanes:
            if lane.key_alias == key_alias:
                return lane
        raise KeyError(f"Unknown provider lane '{key_alias}'")

    def execute_on_lane(self, key_alias: str, request: Dict[str, Any], step_config: dict) -> Dict[str, Any]:
        lane = self.get_lane(key_alias)
        lane_request = copy.deepcopy(request)
        if lane.model_override:
            lane_request["model"] = lane.model_override
        result = lane.adapter.execute_batch([lane_request], step_config)[0]
        enriched = dict(result)
        enriched["key_alias"] = lane.key_alias
        enriched["provider"] = lane.provider
        enriched["model"] = enriched.get("model") or lane_request.get("model")
        return enriched

    def _runtime(self, step_config: dict) -> dict:
        return step_config.get("runtime", {}) or {}

    def _safe_budget(self, runtime: dict) -> float | None:
        target_tpm = runtime.get("target_tokens_per_minute")
        if target_tpm is None:
            return None
        return float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)

    def _ordered_lanes(self) -> List[ProviderLane]:
        now = time.time()
        eligible = [lane for lane in self.lanes if lane.is_available(now)]
        if not eligible:
            return []
        ordered = sorted(
            eligible,
            key=lambda lane: (
                lane.recent_failures,
                -lane.recent_successes,
                lane.last_used_at,
                self.lanes.index(lane),
            ),
        )
        if len(ordered) > 1:
            offset = self._lane_cursor % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
        if self._preferred_lane_alias:
            ordered.sort(key=lambda lane: 0 if lane.key_alias == self._preferred_lane_alias else 1)
        return ordered

    def _fit_lane_budget(self, ordered: List[ProviderLane], requests: List[Dict[str, Any]], runtime: dict) -> List[ProviderLane]:
        safe_budget = self._safe_budget(runtime)
        if safe_budget is None:
            return ordered
        now = time.time()
        request_estimates = [_estimate_request_tokens(req, runtime) for req in requests]
        fit = []
        for lane in ordered:
            rolling = _rolling_window_tokens(lane.token_window, now)
            smallest = min(request_estimates) if request_estimates else 0
            if rolling + smallest <= safe_budget:
                fit.append(lane)
        return fit

    def _wait_for_any_lane_budget(self, requests: List[Dict[str, Any]], runtime: dict) -> None:
        safe_budget = self._safe_budget(runtime)
        if safe_budget is None:
            return
        smallest = min((_estimate_request_tokens(req, runtime) for req in requests), default=0)
        while True:
            now = time.time()
            ordered = self._ordered_lanes()
            if not ordered:
                next_ready = min(lane.cooldown_until for lane in self.lanes)
                time.sleep(max(next_ready - now, 0.0))
                continue
            any_fit = False
            next_budget_ready = None
            for lane in ordered:
                rolling = _rolling_window_tokens(lane.token_window, now)
                if rolling + smallest <= safe_budget:
                    any_fit = True
                    break
                if lane.token_window:
                    candidate = min(entry["sent_at"] + 60.0 for entry in lane.token_window)
                    next_budget_ready = candidate if next_budget_ready is None else min(next_budget_ready, candidate)
            if any_fit:
                return
            if next_budget_ready is None:
                return
            time.sleep(max(next_budget_ready - now, 0.0))

    def _active_lane_limit_for_strategy(self, runtime: dict) -> int:
        strategy = runtime.get("lane_strategy", "hybrid")
        if strategy == "safe_single_active":
            return 1
        configured_max = int(runtime.get("max_active_lanes", len(self.lanes)) or len(self.lanes))
        configured_max = max(1, min(configured_max, len(self.lanes)))
        if strategy == "parallel":
            return configured_max
        initial = int(runtime.get("initial_active_lanes", 1) or 1)
        if self._active_lane_limit < 1:
            self._active_lane_limit = initial
        return max(1, min(self._active_lane_limit, configured_max))

    def _choose_lanes(self, requests: List[Dict[str, Any]], runtime: dict) -> List[ProviderLane]:
        self._wait_for_any_lane_budget(requests, runtime)
        ordered = self._ordered_lanes()
        if not ordered:
            return []
        fit = self._fit_lane_budget(ordered, requests, runtime) or ordered[:1]
        limit = self._active_lane_limit_for_strategy(runtime)
        if runtime.get("lane_strategy", "hybrid") == "safe_single_active":
            return fit[:1]
        return fit[:limit]

    def _assign_requests(self, requests: List[Dict[str, Any]], runtime: dict) -> Dict[str, List[Dict[str, Any]]]:
        assignments: Dict[str, List[Dict[str, Any]]] = {lane.key_alias: [] for lane in self.lanes}
        selected = self._choose_lanes(requests, runtime)
        if not selected:
            raise RuntimeError("No provider lanes available for request assignment")
        for idx, request in enumerate(requests):
            lane = selected[idx % len(selected)]
            request_copy = copy.deepcopy(request)
            if lane.model_override:
                request_copy["model"] = lane.model_override
            assignments[lane.key_alias].append(request_copy)
        self._lane_cursor = (self._lane_cursor + len(requests)) % max(len(selected), 1)
        return assignments

    def _record_lane_tokens(self, lane: ProviderLane, requests: List[Dict[str, Any]], runtime: dict) -> None:
        now = time.time()
        for request in requests:
            lane.token_window.append({"sent_at": now, "tokens": _estimate_request_tokens(request, runtime)})

    def _transport_failure_count(self, lane_results: List[Dict[str, Any]]) -> int:
        count = 0
        for enriched in lane_results:
            if enriched.get("status") in RETRYABLE_STATUSES and (
                enriched.get("error_type") in TRANSPORT_ERROR_TYPES or enriched.get("raw_output") is None
            ):
                count += 1
        return count

    def _update_strategy_state(self, runtime: dict, active_lanes: List[ProviderLane], failed_lanes: List[ProviderLane], succeeded_lanes: List[ProviderLane]) -> None:
        strategy = runtime.get("lane_strategy", "hybrid")
        if strategy == "safe_single_active":
            if failed_lanes:
                self._preferred_lane_alias = None
            elif succeeded_lanes:
                self._preferred_lane_alias = succeeded_lanes[0].key_alias
            return
        if strategy != "hybrid":
            return
        now = time.time()
        window_seconds = float(runtime.get("shared_failure_window_seconds", 90) or 90)
        threshold = int(runtime.get("shared_failure_lane_threshold", 2) or 2)
        self._failure_events.extend((now, lane.key_alias) for lane in failed_lanes)
        self._failure_events = [(ts, alias) for ts, alias in self._failure_events if ts >= now - window_seconds]
        failed_aliases = {alias for _, alias in self._failure_events}
        if len(failed_aliases) >= threshold:
            self._active_lane_limit = max(1, self._active_lane_limit - 1)
            self._healthy_streak = 0
            self._failure_events.clear()
            return
        if failed_lanes:
            self._healthy_streak = 0
            return
        if succeeded_lanes:
            self._healthy_streak += 1
            needed = int(runtime.get("lane_exploration_success_waves", 2) or 2)
            configured_max = int(runtime.get("max_active_lanes", len(self.lanes)) or len(self.lanes))
            configured_max = max(1, min(configured_max, len(self.lanes)))
            if self._healthy_streak >= needed and self._active_lane_limit < configured_max:
                self._active_lane_limit += 1
                self._healthy_streak = 0

    def execute_batch(self, requests: List[Dict[str, Any]], step_config: dict) -> List[Dict[str, Any]]:
        if not requests:
            return []
        runtime = self._runtime(step_config)
        assignments = self._assign_requests(requests, runtime)
        active = [lane for lane in self.lanes if assignments.get(lane.key_alias)]
        for lane in active:
            self._record_lane_tokens(lane, assignments[lane.key_alias], runtime)
        results_by_request_id: Dict[str, Dict[str, Any]] = {}
        succeeded_lanes: List[ProviderLane] = []
        failed_lanes: List[ProviderLane] = []
        with ThreadPoolExecutor(max_workers=max(len(active), 1)) as executor:
            future_to_lane = {
                executor.submit(lane.adapter.execute_batch, assignments[lane.key_alias], step_config): lane
                for lane in active
            }
            for future in as_completed(future_to_lane):
                lane = future_to_lane[future]
                now = time.time()
                lane_requests = assignments[lane.key_alias]
                try:
                    lane_results = future.result()
                except Exception as exc:
                    cooldown = float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                    failed_lanes.append(lane)
                    for request in lane_requests:
                        results_by_request_id[request["request_id"]] = {
                            "request_id": request["request_id"],
                            "status": "retryable_error",
                            "raw_output": None,
                            "error_type": "wave_request_exception",
                            "error_message": str(exc),
                            "request_seconds": None,
                            "request_attempt_count": 1,
                            "key_alias": lane.key_alias,
                            "provider": lane.provider,
                            "model": request.get("model"),
                        }
                    continue
                transport_failures = self._transport_failure_count(lane_results)
                total = len(lane_results)
                for result in lane_results:
                    enriched = dict(result)
                    enriched["key_alias"] = lane.key_alias
                    enriched["provider"] = lane.provider
                    enriched["model"] = enriched.get("model") or lane.model_override or next((req.get("model") for req in lane_requests if req["request_id"] == enriched.get("request_id")), None)
                    results_by_request_id[enriched["request_id"]] = enriched
                if total > 0 and transport_failures >= total:
                    cooldown = float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                    failed_lanes.append(lane)
                elif transport_failures > 0:
                    cooldown = float(runtime.get("transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                    failed_lanes.append(lane)
                else:
                    lane.record_success(now)
                    succeeded_lanes.append(lane)
        for request in requests:
            if request["request_id"] not in results_by_request_id:
                results_by_request_id[request["request_id"]] = {
                    "request_id": request["request_id"],
                    "status": "retryable_error",
                    "raw_output": None,
                    "error_type": "missing_lane_result",
                    "error_message": f"No result returned for request_id {request['request_id']}",
                    "request_seconds": None,
                    "request_attempt_count": 1,
                    "key_alias": None,
                    "provider": None,
                    "model": request.get("model"),
                }
        self._update_strategy_state(runtime, active, failed_lanes, succeeded_lanes)
        return [results_by_request_id[request["request_id"]] for request in requests]
