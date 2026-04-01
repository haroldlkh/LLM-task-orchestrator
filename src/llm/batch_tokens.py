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


def lane_safe_budget(runtime: dict) -> float | None:
    target_tpm = runtime.get("target_tokens_per_minute")
    if target_tpm is None:
        return None
    return float(target_tpm) * float(runtime.get("tpm_safety_margin", 0.80) or 0.80)


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
    safe_budget = lane_safe_budget(runtime)
    if safe_budget is None:
        return "tpm=disabled"
    rolling_tokens = rolling_window_tokens(token_window, now)
    wait_seconds = lane_wait_seconds_for_tpm(runtime, token_window, next_wave_tokens, now)
    return (
        f"tpm=safe_budget:{int(safe_budget)} rolling:{rolling_tokens} "
        f"next_wave:{int(next_wave_tokens)} wait_s:{wait_seconds:.2f}"
    )
