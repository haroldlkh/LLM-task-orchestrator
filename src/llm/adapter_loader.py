import copy
import importlib
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
