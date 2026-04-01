import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import polars as pl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def ensure_llm_state_layout(
    dest_connector,
    workflow_location: str,
) -> Dict[str, str]:
    state_folder = dest_connector.ensure_subdir(workflow_location, "state")
    results_folder = dest_connector.ensure_subdir(workflow_location, "results")
    debug_folder = dest_connector.ensure_subdir(workflow_location, "debug")

    return {
        "workflow_folder": workflow_location,
        "state_folder": state_folder,
        "results_folder": results_folder,
        "debug_folder": debug_folder,
    }


def latest_object_by_prefix(objects: List[Dict], prefix: str) -> Optional[Dict]:
    matches = [obj for obj in objects if obj["name"].startswith(prefix)]
    if not matches:
        return None
    matches.sort(key=lambda x: x["name"])
    return matches[-1]


def download_latest_parquet_if_exists(
    connector,
    location: str,
    prefix: str,
    temp_dir: str,
) -> Optional[pl.DataFrame]:
    objects = connector.list_objects(location)
    latest = latest_object_by_prefix(objects, prefix)
    if latest is None:
        return None

    local_path = os.path.join(temp_dir, latest["name"])
    connector.download_object(latest["id"], local_path)
    return pl.read_parquet(local_path)


def download_all_parquet_by_prefix(
    connector,
    location: str,
    prefix: str,
    temp_dir: str,
) -> List[pl.DataFrame]:
    objects = connector.list_objects(location)
    matches = [obj for obj in objects if obj["name"].startswith(prefix)]
    matches.sort(key=lambda x: x["name"])

    dfs = []
    for obj in matches:
        local_path = os.path.join(temp_dir, obj["name"])
        connector.download_object(obj["id"], local_path)
        dfs.append(pl.read_parquet(local_path))
    return dfs


def _versioned_filename(prefix: str, extension: str, run_id: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if run_id:
        return f"{prefix}__run_{run_id}__{ts}.{extension}"
    return f"{prefix}_{ts}.{extension}"


def upload_versioned_parquet(
    connector,
    location: str,
    prefix: str,
    df: pl.DataFrame,
    temp_dir: str,
    run_id: str | None = None,
) -> str:
    filename = _versioned_filename(prefix, "parquet", run_id=run_id)
    local_path = os.path.join(temp_dir, filename)
    df.write_parquet(local_path)
    connector.upload_object(local_path, location, filename)
    os.remove(local_path)
    return filename


def upload_versioned_json(
    connector,
    location: str,
    prefix: str,
    payload: dict,
    temp_dir: str,
    run_id: str | None = None,
) -> str:
    filename = _versioned_filename(prefix, "json", run_id=run_id)
    local_path = os.path.join(temp_dir, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    connector.upload_object(local_path, location, filename)
    os.remove(local_path)
    return filename


def _run_id_from_name(name: str) -> str | None:
    match = re.search(r"__run_(\d{8}_\d{6})__", name)
    if match:
        return match.group(1)
    return None


def _prune_keep_last_n_by_prefix(connector, location: str, prefix: str, keep_last_n: int) -> int:
    objects = [obj for obj in connector.list_objects(location) if obj["name"].startswith(prefix)]
    objects.sort(key=lambda x: x["name"])
    to_delete = objects[:-keep_last_n] if len(objects) > keep_last_n else []
    for obj in to_delete:
        connector.delete_object(obj["id"])
    return len(to_delete)


def _prune_final_outputs(connector, workflow_location: str, keep_last_n: int, expected_prefixes: list[str] | None = None) -> int:
    expected_prefixes = [prefix for prefix in (expected_prefixes or []) if prefix]
    objects = []
    for obj in connector.list_objects(workflow_location):
        name = obj["name"]
        if not name.endswith('.parquet'):
            continue
        if expected_prefixes and not any(name.startswith(prefix) for prefix in expected_prefixes):
            continue
        objects.append(obj)
    objects.sort(key=lambda x: x["name"])
    to_delete = objects[:-keep_last_n] if len(objects) > keep_last_n else []
    for obj in to_delete:
        connector.delete_object(obj["id"])
    return len(to_delete)


def _group_objects_by_run_id(objects: List[Dict], prefixes: List[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for obj in objects:
        if not any(obj["name"].startswith(prefix) for prefix in prefixes):
            continue
        run_id = _run_id_from_name(obj["name"])
        if run_id is None:
            continue
        grouped.setdefault(run_id, []).append(obj)
    return grouped


def _prune_debug_runs(connector, folders: Dict[str, str], keep_last_runs: int) -> int:
    folder_prefixes = {
        "state_folder": ["manifest", "progress", "metadata"],
        "results_folder": ["results", "partial_output"],
        "debug_folder": ["traces", "review", "pair_status"],
    }
    run_ids = set()
    grouped_by_folder = {}
    for folder_key, prefixes in folder_prefixes.items():
        objs = connector.list_objects(folders[folder_key])
        grouped = _group_objects_by_run_id(objs, prefixes)
        grouped_by_folder[folder_key] = grouped
        run_ids.update(grouped.keys())

    keep_ids = set(sorted(run_ids)[-keep_last_runs:])
    deleted = 0
    for folder_key, grouped in grouped_by_folder.items():
        for run_id, objs in grouped.items():
            if run_id in keep_ids:
                continue
            for obj in objs:
                connector.delete_object(obj["id"])
                deleted += 1
    return deleted


def cleanup_llm_artifacts(
    connector,
    folders: Dict[str, str],
    runtime: dict,
    include_final_outputs: bool = True,
    expected_final_output_prefixes: list[str] | None = None,
) -> Dict[str, int]:
    mode = runtime.get("artifact_retention_mode", "standard")
    deleted = {"final_outputs": 0, "artifacts": 0}

    if include_final_outputs:
        deleted["final_outputs"] = _prune_final_outputs(
            connector=connector,
            workflow_location=folders["workflow_folder"],
            keep_last_n=int(runtime.get("keep_last_final_outputs", 3)),
            expected_prefixes=expected_final_output_prefixes,
        )

    if mode == "standard":
        keep_last_n = int(runtime.get("keep_last_flushes", 3))
        for folder_key, prefixes in {
            "state_folder": ["manifest", "progress", "metadata"],
            "results_folder": ["results", "partial_output"],
            "debug_folder": ["traces", "review", "pair_status"],
        }.items():
            for prefix in prefixes:
                deleted["artifacts"] += _prune_keep_last_n_by_prefix(
                    connector=connector,
                    location=folders[folder_key],
                    prefix=prefix,
                    keep_last_n=keep_last_n,
                )
    else:
        deleted["artifacts"] = _prune_debug_runs(
            connector=connector,
            folders=folders,
            keep_last_runs=int(runtime.get("keep_last_runs", 10)),
        )

    return deleted
