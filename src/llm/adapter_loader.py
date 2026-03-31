import importlib
from typing import Any, Dict, List

from .provider_pool import ProviderLane, ProviderPoolAdapter


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


def _is_single_provider_config(provider_config: Dict[str, Any]) -> bool:
    return isinstance(provider_config, dict) and "api_key" in provider_config


def _normalize_provider_lanes(provider_config_key: str, provider_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(provider_config, dict) or not provider_config:
        raise ValueError(
            f"Provider config '{provider_config_key}' must be a non-empty JSON object"
        )

    if _is_single_provider_config(provider_config):
        lane = dict(provider_config)
        lane.setdefault("key_alias", provider_config_key)
        return [lane]

    lanes: List[Dict[str, Any]] = []
    for key_alias, lane_config in provider_config.items():
        if not isinstance(lane_config, dict):
            raise ValueError(
                f"Provider pool '{provider_config_key}' entry '{key_alias}' must be an object"
            )
        if "api_key" not in lane_config:
            raise ValueError(
                f"Provider pool '{provider_config_key}' entry '{key_alias}' must include 'api_key'"
            )
        lane = dict(lane_config)
        lane.setdefault("key_alias", key_alias)
        lanes.append(lane)

    if not lanes:
        raise ValueError(f"Provider pool '{provider_config_key}' contains no usable lanes")

    return lanes


def _build_lane(default_adapter_config: dict, lane_config: Dict[str, Any]):
    lane_adapter_config = lane_config.get("adapter") or default_adapter_config
    cls = load_adapter_class(lane_adapter_config)
    lane_adapter = cls(lane_config)
    return ProviderLane(
        key_alias=lane_config["key_alias"],
        provider=lane_config.get("provider"),
        adapter=lane_adapter,
        model_override=lane_config.get("model"),
        weight=float(lane_config.get("weight", 1.0) or 1.0),
    )


def build_adapter(adapter_config: dict, provider_config_key: str, provider_config: dict):
    lanes = _normalize_provider_lanes(provider_config_key, provider_config)
    if len(lanes) == 1:
        cls = load_adapter_class(lanes[0].get("adapter") or adapter_config)
        return cls(lanes[0])

    provider_lanes = [_build_lane(adapter_config, lane_config) for lane_config in lanes]
    return ProviderPoolAdapter(provider_lanes)
