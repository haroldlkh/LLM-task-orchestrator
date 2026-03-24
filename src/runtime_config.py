import json
import os
from typing import Dict


def require_env(name: str, allow_empty: bool = False) -> str:
    value = os.environ.get(name, "")
    if not allow_empty and not value.strip():
        raise ValueError(f"Required environment variable {name} is missing or empty.")
    return value.strip()


def parse_json_env(name: str, allow_empty: bool = True) -> dict:
    raw = os.environ.get(name, "").strip()

    if not raw:
        if allow_empty:
            return {}
        raise ValueError(f"Required JSON environment variable {name} is missing or empty.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} is not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object/dict.")

    return parsed


def parse_entry_inputs() -> Dict[str, str]:
    task_script = os.environ.get("TASK_SCRIPT", "").strip()
    pipeline_script = os.environ.get("PIPELINE_SCRIPT", "").strip()

    provided = [x for x in [task_script, pipeline_script] if x]
    if len(provided) == 0:
        raise ValueError("You must provide exactly one of TASK_SCRIPT or PIPELINE_SCRIPT.")
    if len(provided) > 1:
        raise ValueError("Provide only one of TASK_SCRIPT or PIPELINE_SCRIPT, not both.")

    if task_script:
        return {"entry_type": "task", "path": task_script}
    return {"entry_type": "pipeline", "path": pipeline_script}