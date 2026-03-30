from typing import Dict, List, Set

from .dataframes import TERMINAL_STATUSES


def build_row_to_unit_ids(work_units_rows: List[dict]) -> Dict[str, Set[str]]:
    row_to_unit_ids: Dict[str, Set[str]] = {}
    for row in work_units_rows:
        row_to_unit_ids.setdefault(str(row["row_id"]), set()).add(str(row["unit_id"]))
    return row_to_unit_ids


def build_progress_status_map(progress_df) -> Dict[str, str]:
    if progress_df is None or progress_df.is_empty():
        return {}
    return {
        str(row["unit_id"]): row["status"]
        for row in progress_df.select(["unit_id", "status"]).iter_rows(named=True)
    }


def apply_progress_rows_to_status_map(progress_status_by_unit: Dict[str, str], progress_rows: List[dict]) -> None:
    for row in progress_rows:
        progress_status_by_unit[str(row["unit_id"])] = row["status"]


def _row_all_success(row_id: str, row_to_unit_ids: Dict[str, Set[str]], progress_status_by_unit: Dict[str, str]) -> bool:
    unit_ids = row_to_unit_ids.get(str(row_id), set())
    if not unit_ids:
        return False
    return all(progress_status_by_unit.get(unit_id) == "success" for unit_id in unit_ids)


def _row_all_terminal(row_id: str, row_to_unit_ids: Dict[str, Set[str]], progress_status_by_unit: Dict[str, str]) -> bool:
    unit_ids = row_to_unit_ids.get(str(row_id), set())
    if not unit_ids:
        return False
    return all(progress_status_by_unit.get(unit_id) in TERMINAL_STATUSES for unit_id in unit_ids)


def release_flushable_rows(pending_result_rows: List[dict], pending_debug_rows: List[dict], row_to_unit_ids: Dict[str, Set[str]], progress_status_by_unit: Dict[str, str], flush_scope: str):
    if flush_scope == "unit":
        result_rows = list(pending_result_rows)
        debug_rows = list(pending_debug_rows)
        pending_result_rows.clear()
        pending_debug_rows.clear()
        released_success_row_ids = sorted({str(row["row_id"]) for row in result_rows})
        released_terminal_row_ids = sorted({str(row["row_id"]) for row in debug_rows})
        return {
            "result_rows": result_rows,
            "debug_rows": debug_rows,
            "released_success_row_ids": released_success_row_ids,
            "released_terminal_row_ids": released_terminal_row_ids,
        }

    if flush_scope != "row_complete":
        raise ValueError(f"Unsupported flush_scope '{flush_scope}'")

    success_ready_row_ids = {
        str(row["row_id"])
        for row in pending_result_rows
        if _row_all_success(str(row["row_id"]), row_to_unit_ids, progress_status_by_unit)
    }
    terminal_ready_row_ids = {
        str(row["row_id"])
        for row in pending_debug_rows
        if _row_all_terminal(str(row["row_id"]), row_to_unit_ids, progress_status_by_unit)
    }

    releasable_result_rows = [row for row in pending_result_rows if str(row["row_id"]) in success_ready_row_ids]
    deferred_result_rows = [row for row in pending_result_rows if str(row["row_id"]) not in success_ready_row_ids]

    releasable_debug_rows = [row for row in pending_debug_rows if str(row["row_id"]) in terminal_ready_row_ids]
    deferred_debug_rows = [row for row in pending_debug_rows if str(row["row_id"]) not in terminal_ready_row_ids]

    pending_result_rows[:] = deferred_result_rows
    pending_debug_rows[:] = deferred_debug_rows

    return {
        "result_rows": releasable_result_rows,
        "debug_rows": releasable_debug_rows,
        "released_success_row_ids": sorted(success_ready_row_ids),
        "released_terminal_row_ids": sorted(terminal_ready_row_ids),
    }
