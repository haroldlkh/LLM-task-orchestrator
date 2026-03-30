import math
import time


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
) -> None:
    counts = status_counts_from_progress_map(progress_by_unit)
    elapsed = elapsed_seconds(start_time)
    remaining = max(total_units - cursor, 0)

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
