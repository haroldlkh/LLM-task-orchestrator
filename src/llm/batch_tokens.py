import math


def estimate_request_tokens(request: dict, group_units, runtime: dict) -> int:
    chars_per_token = float(runtime.get("token_estimation_chars_per_token", 4.0) or 4.0)
    response_tokens_per_unit = float(runtime.get("estimated_response_tokens_per_unit", 24) or 24)
    overhead_tokens = float(runtime.get("estimated_request_overhead_tokens", 250) or 250)

    prompt_text = request.get("prompt", "") or ""
    prompt_tokens = math.ceil(len(prompt_text) / max(chars_per_token, 0.1))
    response_tokens = math.ceil(len(group_units) * response_tokens_per_unit)
    total = int(prompt_tokens + response_tokens + overhead_tokens)
    return max(total, 1)


def prune_token_window(token_window: list[dict], now: float) -> list[dict]:
    cutoff = now - 60.0
    return [entry for entry in token_window if entry["sent_at"] > cutoff]


def rolling_window_tokens(token_window: list[dict], now: float) -> int:
    token_window[:] = prune_token_window(token_window, now)
    return int(sum(entry["tokens"] for entry in token_window))


def configured_lane_target_tpm(runtime: dict) -> float | None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if target_tpm is None:
        return None
    return float(target_tpm)


def lane_safe_budget(runtime: dict) -> float | None:
    target_tpm = configured_lane_target_tpm(runtime)
    if target_tpm is None:
        return None
    return target_tpm * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)


def pool_safe_budget(runtime: dict) -> float | None:
    per_lane_safe = lane_safe_budget(runtime)
    if per_lane_safe is None:
        return None
    lane_count = max(1, int(runtime.get("available_lane_count", 1) or 1))
    return per_lane_safe * lane_count


def tpm_budget_snapshot(runtime: dict) -> dict:
    per_key_target_tpm = configured_lane_target_tpm(runtime)
    if per_key_target_tpm is None:
        return {
            "enabled": False,
            "lane_count": max(1, int(runtime.get("available_lane_count", 1) or 1)),
        }
    safety_margin = float(runtime.get("tpm_safety_margin", 0.80) or 0.80)
    per_key_safe_budget = lane_safe_budget(runtime)
    pool_safe = pool_safe_budget(runtime)
    lane_count = max(1, int(runtime.get("available_lane_count", 1) or 1))
    return {
        "enabled": True,
        "lane_count": lane_count,
        "safety_margin": safety_margin,
        "per_key_target_tpm": per_key_target_tpm,
        "per_key_safe_budget": per_key_safe_budget,
        "pool_target_tpm": per_key_target_tpm * lane_count,
        "pool_safe_budget": pool_safe,
    }


def lane_wait_seconds_for_tpm(runtime: dict, token_window: list[dict], next_wave_tokens: int, now: float) -> float:
    safe_budget = lane_safe_budget(runtime)
    if safe_budget is None:
        return 0.0

    rolling_tokens = rolling_window_tokens(token_window, now)
    if rolling_tokens + next_wave_tokens <= safe_budget:
        return 0.0
    if not token_window:
        return 0.0

    oldest_expiry = min(entry["sent_at"] + 60.0 for entry in token_window)
    sleep_seconds = max(0.0, oldest_expiry - now)
    return min(sleep_seconds, float(runtime.get("max_sleep_to_respect_tpm_seconds", 120) or 120))


def describe_tpm_state(runtime: dict, token_window: list[dict], next_wave_tokens: int, now: float) -> str:
    budget = tpm_budget_snapshot(runtime)
    if not budget.get("enabled"):
        return "tpm=disabled"
    rolling_tokens = rolling_window_tokens(token_window, now)
    wait_seconds = lane_wait_seconds_for_tpm(runtime, token_window, next_wave_tokens, now)
    return (
        f"tpm=per_key_safe_budget:{int(budget['per_key_safe_budget'])} "
        f"pool_safe_budget:{int(budget['pool_safe_budget'])} "
        f"per_key_target_tpm:{int(budget['per_key_target_tpm'])} "
        f"pool_target_tpm:{int(budget['pool_target_tpm'])} "
        f"lane_count:{budget['lane_count']} "
        f"lane_rolling:{rolling_tokens} "
        f"next_wave:{int(next_wave_tokens)} "
        f"wait_s:{wait_seconds:.2f}"
    )
