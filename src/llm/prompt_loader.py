import json
import os
from typing import Dict, List


def _repo_root_from_anchor(anchor_path: str) -> str:
    abs_path = os.path.abspath(anchor_path)
    marker = os.sep + "user_repo" + os.sep
    if marker not in abs_path:
        raise ValueError(f"Could not determine user_repo root from anchor_path: {anchor_path}")
    return abs_path.split(marker)[0] + marker.rstrip(os.sep)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_optional_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(repo_root: str, paths: List[str]) -> List[str]:
    return [os.path.join(repo_root, p) for p in paths]


def load_prompt_context(step_config: Dict) -> Dict:
    prompt_cfg = step_config.get("prompt", {})
    if not prompt_cfg:
        return {
            "template_text": "",
            "schema_json": None,
            "schema_text": None,
            "notes_texts": [],
        }

    anchor_path = step_config.get("_anchor_path")
    if not anchor_path:
        raise ValueError("LLM step missing internal '_anchor_path' needed for prompt file resolution")

    repo_root = _repo_root_from_anchor(anchor_path)

    template_text = ""
    schema_json = None
    schema_text = None
    notes_texts = []

    template_file = prompt_cfg.get("template_file")
    if template_file:
        template_path = os.path.join(repo_root, template_file)
        template_text = _read_text_file(template_path)

    schema_file = prompt_cfg.get("schema_file")
    if schema_file:
        schema_path = os.path.join(repo_root, schema_file)
        schema_text = _read_text_file(schema_path)
        try:
            schema_json = _read_optional_json_file(schema_path)
        except Exception:
            schema_json = None

    notes_files = prompt_cfg.get("notes_files", [])
    for path in _resolve_paths(repo_root, notes_files):
        notes_texts.append(_read_text_file(path))

    return {
        "template_text": template_text,
        "schema_json": schema_json,
        "schema_text": schema_text,
        "notes_texts": notes_texts,
    }