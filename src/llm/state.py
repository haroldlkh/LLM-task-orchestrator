import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import polars as pl


ARTIFACT_FAMILIES = [
    "results_checkpoint",
    "results_delta",
    "results_snapshot",
    "partial_output",
    "pair_status",
    "permanent_review",
    "manifest",
    "progress",
    "metadata",
    "traces",
    "review",
    "results",
]


def _matches_artifact_family(name: str, family: str) -> bool:
    if not name.startswith(family):
        return False
    if len(name) == len(family):
        return True
    remainder = name[len(family):]
    return remainder.startswith("_") or remainder.startswith("__") or remainder.startswith(".")


def _artifact_family_from_name(name: str) -> str | None:
    for family in sorted(ARTIFACT_FAMILIES, key=len, reverse=True):
        if _matches_artifact_family(name, family):
            return family
    return None


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
    matches = [obj for obj in objects if _matches_artifact_family(obj["name"], prefix)]
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
    result = download_latest_parquet_with_meta_if_exists(connector, location, prefix, temp_dir)
    if result is None:
        return None
    return result["df"]


def download_latest_parquet_with_meta_if_exists(
    connector,
    location: str,
    prefix: str,
    temp_dir: str,
) -> Optional[Dict]:
    objects = connector.list_objects(location)
    latest = latest_object_by_prefix(objects, prefix)
    if latest is None:
        return None

    local_path = os.path.join(temp_dir, latest["name"])
    connector.download_object(latest["id"], local_path)
    return {"df": pl.read_parquet(local_path), "object": latest}


def download_all_parquet_by_prefix(
    connector,
    location: str,
    prefix: str,
    temp_dir: str,
) -> List[pl.DataFrame]:
    objects = connector.list_objects(location)
    matches = [obj for obj in objects if _matches_artifact_family(obj["name"], prefix)]
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
    objects = [obj for obj in connector.list_objects(location) if _matches_artifact_family(obj["name"], prefix)]
    objects.sort(key=lambda x: x["name"])
    to_delete = objects[:-keep_last_n] if len(objects) > keep_last_n else []
    for obj in to_delete:
        connector.delete_object(obj["id"])
    return len(to_delete)


def _family_keep_count(runtime: dict, prefix: str) -> int:
    default_keep = int(runtime.get("keep_last_flushes", 3))
    family_keep_counts = {
        "manifest": int(runtime.get("keep_last_manifests", default_keep)),
        "progress": int(runtime.get("keep_last_progress", default_keep)),
        "metadata": int(runtime.get("keep_last_metadata", default_keep)),
        "results_snapshot": int(runtime.get("keep_last_results_snapshots", default_keep)),
        "results_checkpoint": int(runtime.get("keep_last_results_checkpoints", default_keep)),
        "results_delta": int(runtime.get("keep_last_results_deltas", default_keep)),
        "partial_output": int(runtime.get("keep_last_partial_outputs", default_keep)),
        "traces": int(runtime.get("keep_last_traces", default_keep)),
        "review": int(runtime.get("keep_last_reviews", default_keep)),
        "pair_status": int(runtime.get("keep_last_pair_status", default_keep)),
        "permanent_review": int(runtime.get("keep_last_permanent_reviews", default_keep)),
    }
    return family_keep_counts.get(prefix, default_keep)


def _prune_final_outputs(
    connector,
    workflow_location: str,
    keep_last_n: int,
    expected_prefixes: Optional[List[str]] = None,
) -> int:
    """Prune workflow-level final outputs by filename prefix.

    Final outputs are not artifact families like state/debug/result artifacts; they are
    timestamped workflow result files such as ``perspective_taking_llm_20260403_023709.parquet``.
    Those names should be matched by plain filename prefix, not artifact-family boundary logic.
    """
    objects = [obj for obj in connector.list_objects(workflow_location) if obj["name"].endswith('.parquet')]
    if expected_prefixes:
        objects = [
            obj for obj in objects
            if any(obj["name"].startswith(prefix) for prefix in expected_prefixes)
        ]
    objects.sort(key=lambda x: x["name"])
    to_delete = objects[:-keep_last_n] if len(objects) > keep_last_n else []
    for obj in to_delete:
        connector.delete_object(obj["id"])
    return len(to_delete)


def _group_objects_by_run_id(objects: List[Dict], prefixes: List[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for obj in objects:
        if _artifact_family_from_name(obj["name"]) not in set(prefixes):
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
        "debug_folder": ["traces", "review", "pair_status", "permanent_review"],
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


def _prune_results_deltas_by_kept_checkpoint_sets(connector, results_folder: str, keep_checkpoint_sets: int) -> int:
    objects = sorted(connector.list_objects(results_folder), key=lambda x: x["name"])
    checkpoints = [obj for obj in objects if _matches_artifact_family(obj["name"], "results_checkpoint")]
    deltas = [obj for obj in objects if _matches_artifact_family(obj["name"], "results_delta")]
    if not deltas:
        return 0
    if not checkpoints:
        keep_last_n = max(int(keep_checkpoint_sets), 1)
        to_delete = deltas[:-keep_last_n] if len(deltas) > keep_last_n else []
        for obj in to_delete:
            connector.delete_object(obj["id"])
        return len(to_delete)

    keep_checkpoint_sets = max(int(keep_checkpoint_sets), 1)
    kept_checkpoints = checkpoints[-keep_checkpoint_sets:] if len(checkpoints) > keep_checkpoint_sets else checkpoints
    oldest_kept_checkpoint_name = kept_checkpoints[0]["name"]

    deleted = 0
    for obj in deltas:
        if obj["name"] < oldest_kept_checkpoint_name:
            connector.delete_object(obj["id"])
            deleted += 1
    return deleted




def prune_final_outputs_only(connector, workflow_location: str, keep_last_n: int, expected_prefixes: Optional[List[str]] = None) -> int:
    return _prune_final_outputs(
        connector=connector,
        workflow_location=workflow_location,
        keep_last_n=keep_last_n,
        expected_prefixes=expected_prefixes,
    )

def cleanup_llm_artifacts(
    connector,
    folders: Dict[str, str],
    runtime: dict,
    include_final_outputs: bool = True,
    expected_final_output_prefixes: Optional[List[str]] = None,
    new_artifact_prefixes: Optional[Dict[str, set[str]]] = None,
) -> Dict[str, int]:
    mode = runtime.get("artifact_retention_mode", "standard")
    deleted = {"final_outputs": 0, "artifacts": 0}

    if include_final_outputs:
        deleted["final_outputs"] = _prune_final_outputs(
            connector=connector,
            workflow_location=folders["workflow_folder"],
            keep_last_n=int(runtime.get("keep_last_final_outputs", 1)),
            expected_prefixes=expected_final_output_prefixes,
        )

    if mode == "standard":
        prune_plan = {
            "state_folder": ["manifest", "progress", "metadata"],
            "results_folder": ["results", "partial_output", "results_snapshot", "results_checkpoint", "results_delta"],
            "debug_folder": ["traces", "review", "pair_status", "permanent_review"],
        }
        for folder_key, prefixes in prune_plan.items():
            allowed_prefixes = None if new_artifact_prefixes is None else set(new_artifact_prefixes.get(folder_key, set()))
            for prefix in prefixes:
                if allowed_prefixes is not None and prefix not in allowed_prefixes:
                    continue
                if folder_key == "results_folder" and prefix == "results_delta":
                    deleted["artifacts"] += _prune_results_deltas_by_kept_checkpoint_sets(
                        connector,
                        folders["results_folder"],
                        keep_checkpoint_sets=_family_keep_count(runtime, "results_delta"),
                    )
                    continue
                deleted["artifacts"] += _prune_keep_last_n_by_prefix(
                    connector=connector,
                    location=folders[folder_key],
                    prefix=prefix,
                    keep_last_n=_family_keep_count(runtime, prefix),
                )
    else:
        deleted["artifacts"] = _prune_debug_runs(
            connector=connector,
            folders=folders,
            keep_last_runs=int(runtime.get("keep_last_runs", 10)),
        )

    return deleted
