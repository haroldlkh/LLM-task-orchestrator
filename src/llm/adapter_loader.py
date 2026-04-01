import copy
import importlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

from .request_timeout import EngineRequestTimeoutError, run_with_timeout, timeout_window_summary


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

    def execute_on_lane(self, key_alias: str, request: Dict[str, Any], step_config: dict, lane_state: dict | None = None) -> Dict[str, Any]:
        lane = self.get_lane(key_alias)
        lane_request = dict(request)
        if lane.model:
            lane_request["model"] = lane.model

        timeout_summary = timeout_window_summary(step_config, lane_state)
        timeout_seconds = None if not step_config.get("runtime", {}).get("request_timeout_enabled", True) else float(timeout_summary["timeout_seconds"])
        lane_request["engine_timeout_seconds"] = timeout_seconds
        started_at = perf_counter()
        try:
            result = run_with_timeout(
                lane.adapter.execute_batch,
                timeout_seconds,
                [lane_request],
                step_config,
            )[0]
        except EngineRequestTimeoutError as exc:
            elapsed = perf_counter() - started_at
            result = {
                "request_id": lane_request.get("request_id"),
                "status": "retryable_error",
                "raw_output": None,
                "error_type": "engine_request_timeout",
                "error_message": str(exc),
                "request_seconds": elapsed,
                "request_attempt_count": 1,
            }

        result["key_alias"] = lane.key_alias
        result["provider"] = lane.provider
        result["model"] = lane_request.get("model")
        result["engine_timeout_seconds"] = timeout_seconds
        result["engine_timeout_source"] = timeout_summary["source"] if timeout_seconds is not None else None
        result["engine_timeout_window_sample_count"] = timeout_summary["window_sample_count"] if timeout_seconds is not None else 0
        return result


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
