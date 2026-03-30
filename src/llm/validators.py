from typing import Dict, List

from .models import VALID_UNIT_STATUSES


GROUPED_ADAPTER_RESULT_KEYS = {
    "request_id",
    "status",
    "raw_output",
    "error_type",
    "error_message",
}

GROUPED_PARSE_RESULT_KEYS = {
    "unit_id",
    "status",
    "parsed_output",
    "output_value",
    "error_type",
    "error_message",
}


def validate_grouped_adapter_result(result: Dict) -> None:
    missing = GROUPED_ADAPTER_RESULT_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"Grouped adapter result missing keys: {sorted(missing)}")

    status = result["status"]
    if status not in VALID_UNIT_STATUSES:
        raise ValueError(
            f"Invalid grouped adapter result status '{status}'. "
            f"Expected one of {sorted(VALID_UNIT_STATUSES)}"
        )


def validate_grouped_adapter_batch_results(batch_results: List[Dict], expected_request_ids: List[str]) -> None:
    if len(batch_results) != len(expected_request_ids):
        raise ValueError(
            f"Adapter returned {len(batch_results)} grouped results for {len(expected_request_ids)} grouped requests"
        )

    seen = set()
    for result in batch_results:
        validate_grouped_adapter_result(result)
        request_id = result["request_id"]
        if request_id not in expected_request_ids:
            raise ValueError(f"Adapter returned unknown request_id '{request_id}'")
        if request_id in seen:
            raise ValueError(f"Adapter returned duplicate request_id '{request_id}'")
        seen.add(request_id)


def validate_grouped_parse_results(parse_results: List[Dict], expected_unit_ids: List[str]) -> None:
    seen = set()
    expected_set = set(expected_unit_ids)
    observed_ids = []

    for result in parse_results:
        missing = GROUPED_PARSE_RESULT_KEYS - set(result.keys())
        if missing:
            raise ValueError(f"Grouped parse result missing keys: {sorted(missing)}")

        status = result["status"]
        if status not in VALID_UNIT_STATUSES:
            raise ValueError(
                f"Invalid grouped parse result status '{status}'. Expected one of {sorted(VALID_UNIT_STATUSES)}"
            )

        unit_id = result["unit_id"]
        observed_ids.append(unit_id)
        if unit_id not in expected_set:
            raise ValueError(f"Grouped parse returned unexpected unit_id '{unit_id}'")
        if unit_id in seen:
            raise ValueError(f"Grouped parse returned duplicate unit_id '{unit_id}'")
        seen.add(unit_id)

    missing_ids = [unit_id for unit_id in expected_unit_ids if unit_id not in seen]
    if missing_ids:
        preview = missing_ids[:5]
        suffix = " ..." if len(missing_ids) > 5 else ""
        raise ValueError(
            f"Grouped parse missing {len(missing_ids)} expected unit_id(s): {preview}{suffix}"
        )
