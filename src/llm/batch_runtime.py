from dataclasses import dataclass
import math
import time


@dataclass
class ControllerState:
    load_budget: float
    concurrency: int
    throughput_ema: float | None = None
    good_wave_streak: int = 0
    bad_wave_streak: int = 0
    waves_since_concurrency_change: int = 0


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))


def derive_group_size(load_budget: float, concurrency: int, min_group_size: int, max_group_size: int) -> int:
    raw = int(round(load_budget / max(concurrency, 1)))
    return int(clamp(raw, min_group_size, max_group_size))


def effective_load(group_size: int, concurrency: int) -> int:
    return group_size * concurrency


def update_ema(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return alpha * value + (1 - alpha) * previous


def merge_progress_rows_in_memory(progress_by_unit: dict, new_rows) -> None:
    for row in new_rows:
        progress_by_unit[row["unit_id"]] = row


def status_counts_from_progress_map(progress_by_unit: dict) -> dict:
    counts = {"success": 0, "retryable_error": 0, "permanent_error": 0}
    for row in progress_by_unit.values():
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def elapsed_seconds(start_time: float) -> int:
    return int(time.time() - start_time)


def log_progress(
    step_name: str,
    wave_index: int,
    total_units: int,
    processed_count: int,
    cursor: int,
    current_group_size: int,
    current_concurrency: int,
    current_load_budget: float,
    success_streak: int,
    progress_by_unit: dict,
    note: str,
    start_time: float,
    remaining_override: int | None = None,
    run_total_units: int | None = None,
    run_status_counts: dict | None = None,
    queue_pending: int | None = None,
) -> None:
    counts = status_counts_from_progress_map(progress_by_unit)
    elapsed = elapsed_seconds(start_time)
    remaining = remaining_override if remaining_override is not None else max(total_units - cursor, 0)
    run_counts = run_status_counts or {}
    run_total = int(run_total_units or 0)
    run_success = int(run_counts.get("success", 0) or 0)
    run_retryable = int(run_counts.get("retryable_error", 0) or 0)
    run_permanent = int(run_counts.get("permanent_error", 0) or 0)
    run_done = run_success + run_permanent
    run_remaining = max(run_total - run_done, 0) if run_total > 0 else 0

    print(
        (
            f"[llm:{step_name}] "
            f"waves={wave_index} "
            f"processed={processed_count}/{total_units} "
            f"cursor={cursor} "
            f"remaining={remaining} "
            f"group_size={current_group_size} "
            f"concurrency={current_concurrency} "
            f"load_budget={int(current_load_budget)} "
            f"success_streak={success_streak} "
            f"success={counts['success']} "
            f"retryable_error={counts['retryable_error']} "
            f"permanent_error={counts['permanent_error']} "
            f"run_success={run_success} "
            f"run_retryable_error={run_retryable} "
            f"run_permanent_error={run_permanent} "
            f"run_processed={run_done}/{run_total} "
            f"run_remaining={run_remaining} "
            f"queue_pending={int(queue_pending or 0)} "
            f"elapsed_s={elapsed} "
            f"note={note}"
        ),
        flush=True,
    )


def should_flush(
    pending_progress_rows,
    pending_result_rows,
    pending_debug_rows,
    groups_since_flush: int,
    units_since_flush: int,
    last_flush_time: float,
    runtime: dict,
) -> bool:
    has_pending = bool(pending_progress_rows or pending_result_rows or pending_debug_rows)
    if not has_pending:
        return False

    if units_since_flush >= runtime["flush_every_n_units"]:
        return True
    if groups_since_flush >= runtime["flush_every_n_groups"]:
        return True
    if (time.time() - last_flush_time) >= runtime["flush_every_n_seconds"]:
        return True
    return False


def flush_buffers(
    flush_callback,
    pending_progress_rows,
    pending_result_rows,
    pending_debug_rows,
    processed_count: int,
    remaining_units: int,
    current_group_size: int,
    current_concurrency: int,
    current_load_budget: float,
) -> None:
    if not pending_progress_rows and not pending_result_rows and not pending_debug_rows:
        return

    flush_callback(
        {
            "progress_rows": list(pending_progress_rows),
            "result_rows": list(pending_result_rows),
            "debug_rows": list(pending_debug_rows),
            "processed_units": processed_count,
            "remaining_units": remaining_units,
            "current_group_size": current_group_size,
            "current_concurrency": current_concurrency,
            "current_load_budget": int(current_load_budget),
        }
    )

    pending_progress_rows.clear()
    pending_result_rows.clear()
    pending_debug_rows.clear()


def request_score(useful_work: int, request_seconds: float) -> float:
    if request_seconds <= 0:
        return 0.0
    return useful_work / request_seconds


def compute_wave_sleep_seconds(runtime: dict, wave_metrics: dict) -> tuple[float, str | None]:
    transport_failures = int(wave_metrics.get("transport_failures", 0) or 0)
    total_requests = int(wave_metrics.get("total_requests", 0) or 0)

    if total_requests <= 0:
        return 0.0, None

    if transport_failures >= total_requests:
        seconds = float(runtime.get("all_transport_failure_cooldown_seconds", 0) or 0)
        if seconds > 0:
            return seconds, "all_transport_failure_cooldown"
        return 0.0, None

    if transport_failures > 0:
        seconds = float(runtime.get("transport_failure_cooldown_seconds", 0) or 0)
        if seconds > 0:
            return seconds, "transport_failure_cooldown"
        return 0.0, None

    seconds = float(runtime.get("min_inter_wave_sleep_seconds", 0) or 0)
    if seconds > 0:
        return seconds, "min_inter_wave_sleep"

    return 0.0, None


def choose_next_controller_state(
    controller: ControllerState,
    runtime: dict,
    group_size_used: int,
    wave_metrics: dict,
) -> tuple[ControllerState, list[str]]:
    notes = []

    load_budget = float(controller.load_budget)
    concurrency = int(controller.concurrency)
    throughput_ema = controller.throughput_ema
    good_wave_streak = controller.good_wave_streak
    bad_wave_streak = controller.bad_wave_streak
    waves_since_concurrency_change = controller.waves_since_concurrency_change + 1

    processed_units = max(wave_metrics["processed_units"], 1)
    transport_failures = wave_metrics["transport_failures"]
    successful_requests = wave_metrics["successful_requests"]
    total_requests = max(wave_metrics["total_requests"], 1)
    failure_rate = wave_metrics["failure_units"] / processed_units
    wave_score = wave_metrics["useful_work"] / max(wave_metrics["elapsed_request_seconds"], 1e-9)
    throughput_ema = update_ema(throughput_ema, wave_score, runtime["throughput_ema_alpha"])

    mixed_transport_pressure = transport_failures > 0 and successful_requests > 0
    all_transport_failure = transport_failures == total_requests

    min_load_budget = runtime["min_group_size"] * runtime["min_concurrency"]
    max_load_budget = runtime["max_group_size"] * runtime["max_concurrent_requests"]

    if all_transport_failure:
        load_budget *= runtime["load_shrink_factor"]
        if concurrency > runtime["min_concurrency"] and waves_since_concurrency_change >= runtime["concurrency_shrink_cooldown_waves"]:
            concurrency -= 1
            waves_since_concurrency_change = 0
            notes.append("concurrency_down_all_transport_failure")
        notes.append("load_budget_down_all_transport_failure")
        good_wave_streak = 0
        bad_wave_streak += 1
    elif mixed_transport_pressure:
        if concurrency > runtime["min_concurrency"] and waves_since_concurrency_change >= runtime["concurrency_shrink_cooldown_waves"]:
            concurrency -= 1
            waves_since_concurrency_change = 0
            notes.append("concurrency_down_mixed_transport_pressure")
        else:
            load_budget *= runtime["mild_load_shrink_factor"]
            notes.append("load_budget_mild_down_mixed_transport_pressure")
        good_wave_streak = 0
        bad_wave_streak += 1
    elif failure_rate >= runtime["hard_failure_rate"]:
        load_budget *= runtime["load_shrink_factor"]
        notes.append("load_budget_down_hard_failure_rate")
        good_wave_streak = 0
        bad_wave_streak += 1
    elif failure_rate >= runtime["soft_failure_rate"]:
        load_budget *= runtime["mild_load_shrink_factor"]
        notes.append("load_budget_mild_down_soft_failure_rate")
        good_wave_streak = 0
        bad_wave_streak += 1
    else:
        good_wave_streak += 1
        bad_wave_streak = 0

        improving = throughput_ema is None or wave_score >= throughput_ema * (1 - runtime["throughput_tolerance"])
        if improving:
            load_budget *= runtime["load_growth_factor"]
            notes.append("load_budget_up_improving_throughput")

            if (
                good_wave_streak >= runtime["concurrency_growth_cooldown_waves"]
                and concurrency < runtime["max_concurrent_requests"]
                and waves_since_concurrency_change >= runtime["concurrency_growth_cooldown_waves"]
            ):
                concurrency += 1
                waves_since_concurrency_change = 0
                notes.append("concurrency_up_sustained_good_waves")
        else:
            load_budget *= runtime["mild_load_shrink_factor"]
            notes.append("load_budget_mild_down_non_improving_throughput")

    load_budget = clamp(load_budget, min_load_budget, max_load_budget)

    next_group_size = derive_group_size(
        load_budget=load_budget,
        concurrency=concurrency,
        min_group_size=runtime["min_group_size"],
        max_group_size=runtime["max_group_size"],
    )

    # If clamped group size would force too much idle concurrency, compress concurrency down.
    max_useful_concurrency = max(int(math.floor(load_budget / runtime["min_group_size"])), 1)
    if concurrency > max_useful_concurrency:
        concurrency = max(runtime["min_concurrency"], max_useful_concurrency)
        next_group_size = derive_group_size(
            load_budget=load_budget,
            concurrency=concurrency,
            min_group_size=runtime["min_group_size"],
            max_group_size=runtime["max_group_size"],
        )
        notes.append("concurrency_down_to_match_load_budget")

    return (
        ControllerState(
            load_budget=load_budget,
            concurrency=concurrency,
            throughput_ema=throughput_ema,
            good_wave_streak=good_wave_streak,
            bad_wave_streak=bad_wave_streak,
            waves_since_concurrency_change=waves_since_concurrency_change,
        ),
        notes,
    )
