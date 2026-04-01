from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from statistics import median
from typing import Any, Callable, Iterable


class EngineRequestTimeoutError(TimeoutError):
    pass


def _recent_success_seconds(lane_state: dict | None) -> list[float]:
    if lane_state is None:
        return []
    values = lane_state.get("success_request_seconds_window") or []
    cleaned = []
    for value in values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if value_f > 0:
            cleaned.append(value_f)
    return cleaned


def timeout_window_summary(step_config: dict, lane_state: dict | None = None) -> dict:
    runtime = step_config.get("runtime", {})
    base_timeout = float(runtime.get("request_timeout_seconds", 300) or 300)
    max_timeout = float(runtime.get("request_timeout_max_seconds", 1800) or 1800)
    multiplier = float(runtime.get("request_timeout_margin_multiplier", 2.0) or 2.0)
    min_successes = int(runtime.get("request_timeout_min_success_samples", 1) or 1)
    statistic = str(runtime.get("request_timeout_window_statistic", "median") or "median").lower()

    success_window = _recent_success_seconds(lane_state)
    sample_count = len(success_window)
    window_metric = None
    source = "base"
    adaptive_timeout = base_timeout

    if sample_count >= min_successes:
        if statistic == "mean":
            window_metric = sum(success_window) / sample_count
        else:
            window_metric = median(success_window)
        adaptive_timeout = max(base_timeout, float(window_metric) * multiplier)
        source = f"window_{statistic}"

    timeout_seconds = min(max(adaptive_timeout, base_timeout), max_timeout)
    return {
        "timeout_seconds": timeout_seconds,
        "source": source,
        "window_statistic": statistic,
        "window_metric_seconds": window_metric,
        "window_sample_count": sample_count,
        "window_size": int(runtime.get("request_timeout_window_size", 8) or 8),
        "margin_multiplier": multiplier,
        "base_timeout_seconds": base_timeout,
        "max_timeout_seconds": max_timeout,
    }


def effective_request_timeout_seconds(step_config: dict, lane_state: dict | None = None) -> float | None:
    runtime = step_config.get("runtime", {})
    if not runtime.get("request_timeout_enabled", True):
        return None
    return float(timeout_window_summary(step_config, lane_state)["timeout_seconds"])


def run_with_timeout(func: Callable[..., Any], timeout_seconds: float | None, *args, **kwargs):
    if timeout_seconds is None:
        return func(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise EngineRequestTimeoutError(
                f"Engine request timeout after {timeout_seconds:.2f}s"
            ) from exc
