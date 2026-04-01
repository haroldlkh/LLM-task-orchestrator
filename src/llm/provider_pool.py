import copy
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
}


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
        self._lane_cursor = 0

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def capacity_multiplier(self) -> int:
        return max(len(self.lanes), 1)

    def _sleep_until_lane_available(self) -> None:
        now = time.time()
        next_ready = min(lane.cooldown_until for lane in self.lanes)
        sleep_seconds = max(next_ready - now, 0.0)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _eligible_lanes(self) -> List[ProviderLane]:
        while True:
            now = time.time()
            eligible = [lane for lane in self.lanes if lane.is_available(now)]
            if eligible:
                return eligible
            self._sleep_until_lane_available()

    def _ordered_lanes(self) -> List[ProviderLane]:
        eligible = self._eligible_lanes()
        if not eligible:
            return []
        ordered = sorted(
            eligible,
            key=lambda lane: (
                lane.recent_failures,
                lane.last_used_at,
                self.lanes.index(lane),
            ),
        )
        if len(ordered) <= 1:
            return ordered
        offset = self._lane_cursor % len(ordered)
        return ordered[offset:] + ordered[:offset]

    def _assign_requests(self, requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        assignments: Dict[str, List[Dict[str, Any]]] = {lane.key_alias: [] for lane in self.lanes}
        ordered = self._ordered_lanes()
        if not ordered:
            raise RuntimeError("No provider lanes available for request assignment")

        for idx, request in enumerate(requests):
            lane = ordered[idx % len(ordered)]
            request_copy = copy.deepcopy(request)
            if lane.model_override:
                request_copy["model"] = lane.model_override
            assignments[lane.key_alias].append(request_copy)

        self._lane_cursor = (self._lane_cursor + len(requests)) % max(len(ordered), 1)
        return assignments

    def execute_batch(self, requests: List[Dict[str, Any]], step_config: dict) -> List[Dict[str, Any]]:
        if not requests:
            return []

        assignments = self._assign_requests(requests)
        active = [lane for lane in self.lanes if assignments.get(lane.key_alias)]
        results_by_request_id: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(len(active), 1)) as executor:
            future_to_lane = {
                executor.submit(lane.adapter.execute_batch, assignments[lane.key_alias], step_config): lane
                for lane in active
            }
            for future in as_completed(future_to_lane):
                lane = future_to_lane[future]
                now = time.time()
                try:
                    lane_results = future.result()
                except Exception as exc:
                    cooldown = float(step_config.get("runtime", {}).get("all_transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                    for request in assignments[lane.key_alias]:
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

                transport_failures = 0
                total = len(lane_results)
                for result in lane_results:
                    enriched = dict(result)
                    enriched["key_alias"] = lane.key_alias
                    enriched["provider"] = lane.provider
                    enriched["model"] = enriched.get("model") or assignments[lane.key_alias][0].get("model")
                    results_by_request_id[enriched["request_id"]] = enriched

                    if enriched.get("status") in RETRYABLE_STATUSES and (
                        enriched.get("error_type") in TRANSPORT_ERROR_TYPES or enriched.get("raw_output") is None
                    ):
                        transport_failures += 1

                if total > 0 and transport_failures >= total:
                    cooldown = float(step_config.get("runtime", {}).get("all_transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                elif transport_failures > 0:
                    cooldown = float(step_config.get("runtime", {}).get("transport_failure_cooldown_seconds", 0) or 0)
                    lane.record_failure(now, cooldown)
                else:
                    lane.record_success(now)

        return [results_by_request_id[request["request_id"]] for request in requests]
