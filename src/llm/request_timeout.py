from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable


class EngineRequestTimeoutError(TimeoutError):
    pass


def effective_request_timeout_seconds(step_config: dict, lane_state: dict | None = None) -> float | None:
    runtime = step_config.get("runtime", {})
    if not runtime.get("request_timeout_enabled", True):
        return None

    base_timeout = float(runtime.get("request_timeout_seconds", 300) or 300)
    max_timeout = float(runtime.get("request_timeout_max_seconds", 1800) or 1800)
    multiplier = float(runtime.get("request_timeout_margin_multiplier", 2.0) or 2.0)
    min_successes = int(runtime.get("request_timeout_min_success_samples", 1) or 1)

    adaptive_timeout = base_timeout
    if lane_state is not None:
        success_ema = lane_state.get("success_request_seconds_ema")
        success_samples = int(lane_state.get("success_request_samples", 0) or 0)
        if success_ema is not None and success_samples >= min_successes:
            adaptive_timeout = max(base_timeout, float(success_ema) * multiplier)

    return min(max(adaptive_timeout, base_timeout), max_timeout)


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
