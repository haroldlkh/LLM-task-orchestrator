import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple


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


def _is_single_provider_config(provider_config: dict) -> bool:
    return isinstance(provider_config, dict) and "api_key" in provider_config


def _normalize_provider_pool(provider_config: dict) -> List[Tuple[str, dict]]:
    if _is_single_provider_config(provider_config):
        return [("default", dict(provider_config))]

    if not isinstance(provider_config, dict) or not provider_config:
        raise ValueError("Provider config must be a non-empty dict")

    lanes = []
    for key_alias, lane_config in provider_config.items():
        if not isinstance(lane_config, dict) or "api_key" not in lane_config:
            raise ValueError(
                "Multi-key provider config must be an object whose values are provider configs containing 'api_key'"
            )
        lanes.append((str(key_alias), dict(lane_config)))
    return lanes


class PooledAdapter:
    def __init__(self, default_adapter_config: dict, provider_config: dict):
        self.default_adapter_config = dict(default_adapter_config)
        self.lanes = []
        for index, (key_alias, lane_provider_config) in enumerate(_normalize_provider_pool(provider_config)):
            lane_provider_config = dict(lane_provider_config)
            lane_adapter_config = lane_provider_config.pop("adapter", None) or self.default_adapter_config
            cls = load_adapter_class(lane_adapter_config)
            adapter = cls(lane_provider_config)
            self.lanes.append(
                {
                    "index": index,
                    "key_alias": key_alias,
                    "provider": lane_provider_config.get("provider"),
                    "model": lane_provider_config.get("model"),
                    "adapter": adapter,
                    "config": lane_provider_config,
                }
            )

        self.pool_size = len(self.lanes)
        if self.pool_size == 0:
            raise ValueError("Provider pool must contain at least one lane")
        self._rr_counter = 0

    def _assign_lanes(self, requests: List[Dict[str, Any]]) -> List[Tuple[dict, dict]]:
        assignments = []
        for offset, request in enumerate(requests):
            lane = self.lanes[(self._rr_counter + offset) % self.pool_size]
            lane_request = dict(request)
            if lane.get("model"):
                lane_request["model"] = lane["model"]
            assignments.append((lane, lane_request))
        self._rr_counter = (self._rr_counter + len(requests)) % self.pool_size
        return assignments

    def _execute_one(self, lane: dict, request: dict, step_config: dict) -> Dict[str, Any]:
        result = lane["adapter"].execute_batch([request], step_config)[0]
        result = dict(result)
        result["key_alias"] = lane.get("key_alias")
        result["provider"] = lane.get("provider")
        result["model"] = request.get("model")
        return result

    def execute_batch(self, requests: List[Dict[str, Any]], step_config: dict) -> List[Dict[str, Any]]:
        if not requests:
            return []

        assignments = self._assign_lanes(requests)
        if len(assignments) == 1:
            lane, request = assignments[0]
            return [self._execute_one(lane, request, step_config)]

        results_by_request_id = {}
        max_workers = max(1, min(len(assignments), self.pool_size))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_request_id = {
                executor.submit(self._execute_one, lane, request, step_config): request["request_id"]
                for lane, request in assignments
            }
            for future in as_completed(future_to_request_id):
                request_id = future_to_request_id[future]
                results_by_request_id[request_id] = future.result()

        return [results_by_request_id[request["request_id"]] for request in requests]


def build_adapter(adapter_config: dict, provider_config: dict):
    normalized_lanes = _normalize_provider_pool(provider_config)
    if len(normalized_lanes) == 1:
        cls = load_adapter_class(adapter_config)
        _, lane_provider_config = normalized_lanes[0]
        adapter = cls(lane_provider_config)
        if not hasattr(adapter, "pool_size"):
            adapter.pool_size = 1
        return adapter
    return PooledAdapter(adapter_config, provider_config)
