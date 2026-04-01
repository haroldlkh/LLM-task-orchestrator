from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from statistics import mean, median, pstdev
from typing import Any, Callable


class EngineRequestTimeoutError(TimeoutError):
    pass


def _recent_success_seconds(lane_state: dict | None) -> list[float]:
    if lane_state is None:
        return []
    values = lane_state.get("success_request_seconds_window") or []
    cleaned: list[float] = []
    for value in values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if value_f > 0:
            cleaned.append(value_f)
    return cleaned


def _center_statistic(samples: list[float], statistic: str) -> float | None:
    if not samples:
        return None
    statistic = (statistic or "median").lower()
    if statistic == "mean":
        return mean(samples)
    return median(samples)


def _spread_statistic(samples: list[float], statistic: str, center: float | None = None) -> float | None:
    if not samples:
        return None
    statistic = (statistic or "stdev").lower()
    if statistic == "none":
        return 0.0
    if len(samples) < 2:
        return 0.0
    if statistic == "mad":
        c = center if center is not None else median(samples)
        deviations = [abs(value - c) for value in samples]
        return median(deviations)
    return pstdev(samples)


def timeout_window_summary(step_config: dict, lane_state: dict | None = None) -> dict:
    runtime = step_config.get("runtime", {})
    base_timeout = float(runtime.get("request_timeout_seconds", 300) or 300)
    max_timeout = float(runtime.get("request_timeout_max_seconds", 1800) or 1800)
    min_successes = int(runtime.get("request_timeout_min_success_samples", 1) or 1)
    center_statistic = str(runtime.get("request_timeout_window_statistic", "median") or "median").lower()
    spread_statistic = str(runtime.get("request_timeout_spread_statistic", "stdev") or "stdev").lower()
    spread_multiplier = float(runtime.get("request_timeout_spread_multiplier", 2.0) or 2.0)
    min_margin_seconds = float(runtime.get("request_timeout_min_margin_seconds", 15.0) or 15.0)

    success_window = _recent_success_seconds(lane_state)
    sample_count = len(success_window)
    center_seconds = None
    spread_seconds = None
    margin_seconds = None
    source = "base"
    adaptive_timeout = base_timeout

    if sample_count >= min_successes:
        center_seconds = _center_statistic(success_window, center_statistic)
        spread_seconds = _spread_statistic(success_window, spread_statistic, center_seconds)
        computed_margin = max(min_margin_seconds, float(spread_seconds or 0.0) * spread_multiplier)
        margin_seconds = computed_margin
        adaptive_timeout = max(base_timeout, float(center_seconds or 0.0) + computed_margin)
        source = f"window_{center_statistic}_plus_{spread_statistic}"

    timeout_seconds = min(max(adaptive_timeout, base_timeout), max_timeout)
    return {
        "timeout_seconds": timeout_seconds,
        "source": source,
        "window_center_statistic": center_statistic,
        "window_center_seconds": center_seconds,
        "window_spread_statistic": spread_statistic,
        "window_spread_seconds": spread_seconds,
        "window_margin_seconds": margin_seconds,
        "window_sample_count": sample_count,
        "window_size": int(runtime.get("request_timeout_window_size", 8) or 8),
        "spread_multiplier": spread_multiplier,
        "min_margin_seconds": min_margin_seconds,
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
