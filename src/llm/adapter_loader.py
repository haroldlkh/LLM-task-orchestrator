import copy
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ProviderLane:
    key_alias: str
    provider: str | None
    model: str | None
    adapter: Any
    provider_config: Dict[str, Any]


class ProviderPoolAdapter:
    def __init__(self, lanes: List[ProviderLane]):
        if not lanes:
            raise ValueError("ProviderPoolAdapter requires at least one lane")
        self.lanes = lanes
        self.is_lane_pool = True

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
        lane_request = dict(request)
        if lane.model:
            lane_request["model"] = lane.model

        result = lane.adapter.execute_batch([lane_request], step_config)[0]
        result["key_alias"] = lane.key_alias
        result["provider"] = lane.provider
        result["model"] = lane_request.get("model")
        return result

    def execute_batch(self, requests: List[Dict[str, Any]], step_config: dict) -> List[Dict[str, Any]]:
        if not requests:
            return []

        assignments: Dict[str, List[Dict[str, Any]]] = {lane.key_alias: [] for lane in self.lanes}
        for idx, request in enumerate(requests):
            lane = self.lanes[idx % len(self.lanes)]
            lane_request = dict(request)
            if lane.model:
                lane_request["model"] = lane.model
            assignments[lane.key_alias].append(lane_request)

        results_by_request_id: Dict[str, Dict[str, Any]] = {}
        active_lanes = [lane for lane in self.lanes if assignments[lane.key_alias]]

        with ThreadPoolExecutor(max_workers=max(len(active_lanes), 1)) as executor:
            future_to_lane = {
                executor.submit(lane.adapter.execute_batch, assignments[lane.key_alias], step_config): lane
                for lane in active_lanes
            }
            for future in as_completed(future_to_lane):
                lane = future_to_lane[future]
                lane_requests = assignments[lane.key_alias]
                try:
                    lane_results = future.result()
                except Exception as exc:
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

                for result in lane_results:
                    enriched = dict(result)
                    enriched["key_alias"] = lane.key_alias
                    enriched["provider"] = lane.provider
                    enriched["model"] = enriched.get("model") or lane.model or next(
                        (req.get("model") for req in lane_requests if req["request_id"] == enriched.get("request_id")),
                        None,
                    )
                    results_by_request_id[enriched["request_id"]] = enriched

        missing_request_ids = [request["request_id"] for request in requests if request["request_id"] not in results_by_request_id]
        for request in requests:
            if request["request_id"] in results_by_request_id:
                continue
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

        return [results_by_request_id[request["request_id"]] for request in requests]


def load_adapter_class(adapter_config: dict):
    if "module" not in adapter_config or "class" not in adapter_config:
        raise KeyError(
            "LLM step adapter config must include 'module' and 'class'"
        )

    module_name = adapter_config["module"]
    class_name = adapter_config["class"]

    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise AttributeError(
            f"LLM adapter module '{module_name}' does not have class '{class_name}'"
        )

    cls = getattr(module, class_name)
    return cls


def _is_single_provider_config(config: dict) -> bool:
    return isinstance(config, dict) and "api_key" in config


def _is_pool_provider_config(config: dict) -> bool:
    if not isinstance(config, dict) or not config:
        return False
    if "api_key" in config:
        return False
    return all(isinstance(value, dict) and "api_key" in value for value in config.values())


def _build_lane(default_adapter_config: dict, key_alias: str, lane_config: dict) -> ProviderLane:
    lane_adapter_config = copy.deepcopy(lane_config.get("adapter") or default_adapter_config)
    adapter_cls = load_adapter_class(lane_adapter_config)
    adapter = adapter_cls(lane_config)
    return ProviderLane(
        key_alias=key_alias,
        provider=lane_config.get("provider"),
        model=lane_config.get("model"),
        adapter=adapter,
        provider_config=lane_config,
    )


def build_adapter(adapter_config: dict, provider_config: dict):
    if _is_single_provider_config(provider_config):
        lane = _build_lane(adapter_config, provider_config.get("key_alias", "default"), provider_config)
        return ProviderPoolAdapter([lane])

    if _is_pool_provider_config(provider_config):
        lanes = [
            _build_lane(adapter_config, key_alias, lane_config)
            for key_alias, lane_config in provider_config.items()
        ]
        return ProviderPoolAdapter(lanes)

    raise ValueError(
        "Provider config must either be a single provider object with 'api_key' or a flat pool of top-level provider entries"
    )
