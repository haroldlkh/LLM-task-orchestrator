import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import polars as pl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    dest_location: str,
    pipeline_name: str,
    step_name: str,
) -> Dict[str, str]:
    pipeline_folder = dest_connector.ensure_subdir(dest_location, sanitize_name(pipeline_name))
    step_folder = dest_connector.ensure_subdir(pipeline_folder, sanitize_name(step_name))
    state_folder = dest_connector.ensure_subdir(step_folder, "state")
    results_folder = dest_connector.ensure_subdir(step_folder, "results")
    debug_folder = dest_connector.ensure_subdir(step_folder, "debug")

    return {
        "pipeline_folder": pipeline_folder,
        "step_folder": step_folder,
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


def upload_versioned_parquet(
    connector,
    location: str,
    prefix: str,
    df: pl.DataFrame,
    temp_dir: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.parquet"
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
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.json"
    local_path = os.path.join(temp_dir, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    connector.upload_object(local_path, location, filename)
    os.remove(local_path)
    return filename