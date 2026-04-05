def safe_div(num: float, den: float) -> float:
    den = float(den)
    if den <= 0:
        return 0.0
    return float(num) / den


def compute_payload_reward(metrics: dict, runtime: dict) -> float:
    processed = max(int(metrics.get("processed_units", 0) or 0), 1)
    elapsed = max(float(metrics.get("elapsed_request_seconds", 0.0) or 0.0), 1e-9)
    useful_work = float(metrics.get("useful_work", 0) or 0)
    failure_units = float(metrics.get("failure_units", 0) or 0)
    retryable_units = float(metrics.get("retryable_units", 0) or 0)
    timeout_flag = 1.0 if bool(metrics.get("all_transport_failure", False)) else 0.0
    return (
        safe_div(useful_work, elapsed)
        - float(runtime.get("payload_reward_unusable_penalty", 2.0) or 2.0) * safe_div(failure_units, processed)
        - float(runtime.get("payload_reward_retryable_penalty", 1.0) or 1.0) * safe_div(retryable_units, processed)
        - float(runtime.get("payload_reward_timeout_penalty", 4.0) or 4.0) * timeout_flag
    )


def compute_lane_reward(metrics: dict, runtime: dict) -> float:
    processed = max(int(metrics.get("processed_units", 0) or 0), 1)
    elapsed = max(float(metrics.get("elapsed_request_seconds", 0.0) or 0.0), 1e-9)
    useful_work = float(metrics.get("useful_work", 0) or 0)
    retryable_units = float(metrics.get("retryable_units", 0) or 0)
    transport_failures = float(metrics.get("transport_failures", 0) or 0)
    throttled_flag = 1.0 if bool(metrics.get("throttled_flag", False)) else 0.0
    return (
        safe_div(useful_work, elapsed)
        - float(runtime.get("lane_reward_retryable_penalty", 1.0) or 1.0) * safe_div(retryable_units, processed)
        - float(runtime.get("lane_reward_transport_penalty", 3.0) or 3.0) * transport_failures
        - float(runtime.get("lane_reward_throttle_penalty", 4.0) or 4.0) * throttled_flag
    )
