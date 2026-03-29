from typing import Dict, List

from .models import VALID_UNIT_STATUSES


REQUIRED_RESULT_KEYS = {
    "unit_id",
    "status",
    "parsed_output",
    "raw_output",
    "error_type",
    "error_message",
}


def validate_adapter_result(result: Dict) -> None:
    missing = REQUIRED_RESULT_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"Adapter result missing keys: {sorted(missing)}")

    status = result["status"]
    if status not in VALID_UNIT_STATUSES:
        raise ValueError(
            f"Invalid adapter result status '{status}'. "
            f"Expected one of {sorted(VALID_UNIT_STATUSES)}"
        )


def validate_adapter_batch_results(
    batch_results: List[Dict],
    expected_unit_ids: List[str],
) -> None:
    if len(batch_results) != len(expected_unit_ids):
        raise ValueError(
            f"Adapter returned {len(batch_results)} results for "
            f"{len(expected_unit_ids)} work units"
        )

    seen = set()
    for result in batch_results:
        validate_adapter_result(result)
        unit_id = result["unit_id"]
        if unit_id not in expected_unit_ids:
            raise ValueError(f"Adapter returned unknown unit_id '{unit_id}'")
        if unit_id in seen:
            raise ValueError(f"Adapter returned duplicate unit_id '{unit_id}'")
        seen.add(unit_id)