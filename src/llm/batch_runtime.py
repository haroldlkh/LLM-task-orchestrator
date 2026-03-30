import math
import time
from typing import Dict, List, Optional


def next_smaller_group_size(current_size: int, min_size: int, shrink_factor: float) -> int:
    shrunk = max(min_size, int(math.floor(current_size * shrink_factor)))
    if shrunk == current_size and current_size > min_size:
        shrunk = current_size - 1
    return max(min_size, shrunk)


def next_larger_group_size(current_size: int, max_size: int, grow_step: int) -> int:
    return min(max_size, current_size + grow_step)


def merge_progress_rows_in_memory(progress_by_unit: dict, new_rows) -> None:
    for row in new_rows:
        progress_by_unit[row["unit_id"]] = row


def status_counts_from_progress_map(progress_by_unit: dict) -> dict:
    counts = {
        "success": 0,
        "retryable_error": 0,
        "permanent_error": 0,
    }
    for row in progress_by_unit.values():
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def elapsed_seconds(start_time: float) -> int:
    return int(time.time() - start_time)


def log_progress(
    step_name: str,
    group_index: int,
    total_units: int,
    processed_count: int,
    cursor: int,
    current_group_size: int,
    success_streak: int,
    progress_by_unit: dict,
    note: str,
    start_time: float,
    request_seconds: Optional[float] = None,
    useful_work_count: Optional[int] = None,
    failure_rate: Optional[float] = None,
    score: Optional[float] = None,
) -> None:
    counts = status_counts_from_progress_map(progress_by_unit)
    elapsed = elapsed_seconds(start_time)
    remaining = max(total_units - cursor, 0)

    extra = []
    if request_seconds is not None:
        extra.append(f"request_s={request_seconds:.2f}")
    if useful_work_count is not None:
        extra.append(f"useful_work={useful_work_count}")
    if failure_rate is not None:
        extra.append(f"failure_rate={failure_rate:.4f}")
    if score is not None:
        extra.append(f"score={score:.4f}")
    extra_str = (" " + " ".join(extra)) if extra else ""

    print(
        (
            f"[llm:{step_name}] "
            f"groups={group_index} "
            f"processed={processed_count}/{total_units} "
            f"cursor={cursor} "
            f"remaining={remaining} "
            f"group_size={current_group_size} "
            f"success_streak={success_streak} "
            f"success={counts['success']} "
            f"retryable_error={counts['retryable_error']} "
            f"permanent_error={counts['permanent_error']} "
            f"elapsed_s={elapsed}{extra_str} "
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
    success_streak: int,
):
    if not pending_progress_rows and not pending_result_rows and not pending_debug_rows:
        return {"released_success_rows": 0, "counted_flush": False}

    out = flush_callback(
        {
            "progress_rows": list(pending_progress_rows),
            "result_rows": list(pending_result_rows),
            "debug_rows": list(pending_debug_rows),
            "processed_units": processed_count,
            "remaining_units": remaining_units,
            "current_group_size": current_group_size,
            "success_streak": success_streak,
        }
    )

    pending_progress_rows.clear()
    pending_result_rows.clear()
    pending_debug_rows.clear()
    return out or {"released_success_rows": 0, "counted_flush": False}


def useful_work_count_for_group(
    group_units: List[Dict],
    parse_results: List[Dict],
    flush_scope: str,
) -> int:
    if not parse_results:
        return 0

    success_by_unit = {
        row["unit_id"]: row["status"] == "success"
        for row in parse_results
    }

    if flush_scope == "unit":
        return sum(1 for row in parse_results if row["status"] == "success")

    expected_per_row: Dict[str, int] = {}
    success_per_row: Dict[str, int] = {}

    for unit in group_units:
        row_id = unit["row_id"]
        expected_per_row[row_id] = expected_per_row.get(row_id, 0) + 1
        if success_by_unit.get(unit["unit_id"], False):
            success_per_row[row_id] = success_per_row.get(row_id, 0) + 1

    complete_row_ids = {
        row_id
        for row_id, expected in expected_per_row.items()
        if success_per_row.get(row_id, 0) == expected
    }

    return sum(1 for unit in group_units if unit["row_id"] in complete_row_ids and success_by_unit.get(unit["unit_id"], False))


def init_controller_state() -> dict:
    return {
        "best_score": 0.0,
        "last_growth_attempt_size": None,
        "size_stats": {},
    }


def _update_ema(current: Optional[float], new_value: float, alpha: float) -> float:
    if current is None:
        return new_value
    return (alpha * new_value) + ((1 - alpha) * current)


def decide_next_group_size(
    *,
    current_group_size: int,
    min_group_size: int,
    max_group_size: int,
    runtime: dict,
    controller_state: dict,
    useful_work_count: int,
    total_units: int,
    failed_units: int,
    request_seconds: float,
    transport_failure: bool,
    whole_group_parse_failure: bool,
    success_streak: int,
) -> dict:
    soft_failure_rate = runtime["soft_failure_rate"]
    hard_failure_rate = runtime["hard_failure_rate"]
    tolerance = runtime["throughput_tolerance"]
    alpha = runtime["throughput_ema_alpha"]

    failure_rate = (failed_units / total_units) if total_units else 0.0
    score = (useful_work_count / request_seconds) if request_seconds > 0 else 0.0

    stats = controller_state["size_stats"].setdefault(
        current_group_size,
        {"score_ema": None, "failure_rate_ema": None, "samples": 0},
    )
    stats["score_ema"] = _update_ema(stats["score_ema"], score, alpha)
    stats["failure_rate_ema"] = _update_ema(stats["failure_rate_ema"], failure_rate, alpha)
    stats["samples"] += 1

    current_score_ema = stats["score_ema"]
    controller_state["best_score"] = max(controller_state["best_score"], current_score_ema)
    best_score = controller_state["best_score"]

    action = "hold"
    note = "throughput_hold"
    new_group_size = current_group_size
    new_success_streak = 0 if failed_units > 0 else success_streak

    if transport_failure or whole_group_parse_failure:
        action = "shrink_aggressive"
        note = "transport_or_group_parse_failure"
        new_group_size = next_smaller_group_size(current_group_size, min_group_size, runtime["shrink_factor"])
        controller_state["last_growth_attempt_size"] = None
        return {
            "action": action,
            "note": note,
            "new_group_size": new_group_size,
            "new_success_streak": 0,
            "score": score,
            "failure_rate": failure_rate,
            "score_ema": current_score_ema,
        }

    if failure_rate >= hard_failure_rate:
        action = "shrink_aggressive"
        note = "hard_failure_rate_exceeded"
        new_group_size = next_smaller_group_size(current_group_size, min_group_size, runtime["shrink_factor"])
        controller_state["last_growth_attempt_size"] = None
        return {
            "action": action,
            "note": note,
            "new_group_size": new_group_size,
            "new_success_streak": 0,
            "score": score,
            "failure_rate": failure_rate,
            "score_ema": current_score_ema,
        }

    if failure_rate > soft_failure_rate:
        note = "soft_failure_rate_tolerated"
        return {
            "action": action,
            "note": note,
            "new_group_size": new_group_size,
            "new_success_streak": 0,
            "score": score,
            "failure_rate": failure_rate,
            "score_ema": current_score_ema,
        }

    new_success_streak = success_streak + 1

    near_best = best_score == 0 or current_score_ema >= (best_score * (1 - tolerance))
    if near_best and new_success_streak >= runtime["grow_after_successes"] and current_group_size < max_group_size:
        action = "grow"
        note = "throughput_near_best_grow"
        new_group_size = next_larger_group_size(current_group_size, max_group_size, runtime["grow_step"])
        controller_state["last_growth_attempt_size"] = new_group_size
        new_success_streak = 0
    elif (
        controller_state.get("last_growth_attempt_size") == current_group_size
        and current_score_ema < (best_score * (1 - tolerance))
        and current_group_size > min_group_size
    ):
        action = "shrink_mild"
        note = "growth_overshot_best_throughput"
        new_group_size = next_smaller_group_size(current_group_size, min_group_size, runtime["mild_shrink_factor"])
        controller_state["last_growth_attempt_size"] = None
        new_success_streak = 0

    return {
        "action": action,
        "note": note,
        "new_group_size": new_group_size,
        "new_success_streak": new_success_streak,
        "score": score,
        "failure_rate": failure_rate,
        "score_ema": current_score_ema,
    }
