import copy
import importlib
from typing import Any, Dict

from .provider_pool import ProviderLane, ProviderPoolAdapter


def load_adapter_class(adapter_config: dict):
    if "module" not in adapter_config or "class" not in adapter_config:
        raise KeyError("LLM step adapter config must include 'module' and 'class'")
    module_name = adapter_config["module"]
    class_name = adapter_config["class"]
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"LLM adapter module '{module_name}' does not have class '{class_name}'")
    return getattr(module, class_name)


def _is_single_provider_config(config: dict) -> bool:
    return isinstance(config, dict) and "api_key" in config


def _is_pool_provider_config(config: dict) -> bool:
    if not isinstance(config, dict) or not config or "api_key" in config:
        return False
    return all(isinstance(value, dict) and "api_key" in value for value in config.values())


def _build_lane(default_adapter_config: dict, key_alias: str, lane_config: Dict[str, Any]) -> ProviderLane:
    lane_adapter_config = copy.deepcopy(lane_config.get("adapter") or default_adapter_config)
    adapter_cls = load_adapter_class(lane_adapter_config)
    adapter = adapter_cls(lane_config)
    return ProviderLane(
        key_alias=key_alias,
        provider=lane_config.get("provider"),
        adapter=adapter,
        model_override=lane_config.get("model"),
        weight=float(lane_config.get("weight", 1.0) or 1.0),
    )


def build_adapter(adapter_config: dict, provider_config: dict):
    if _is_single_provider_config(provider_config):
        alias = provider_config.get("key_alias") or provider_config.get("provider") or "default"
        return ProviderPoolAdapter([_build_lane(adapter_config, alias, provider_config)])
    if _is_pool_provider_config(provider_config):
        lanes = [_build_lane(adapter_config, key_alias, lane_config) for key_alias, lane_config in provider_config.items()]
        return ProviderPoolAdapter(lanes)
    raise ValueError("Provider config must either be a single provider object with 'api_key' or a flat pool of top-level provider entries")
