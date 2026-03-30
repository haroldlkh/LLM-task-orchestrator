import polars as pl

from .models import LLMRunOutcome
from .state import utc_now_iso


ALLOWED_FLUSH_SCOPES = {"unit", "row_complete"}


def default_runtime(step_config: dict) -> dict:
    runtime = step_config.get("runtime", {})
    return {
        "initial_group_size": runtime.get("initial_group_size", 4),
        "min_group_size": runtime.get("min_group_size", 1),
        "max_group_size": runtime.get("max_group_size", 16),
        "grow_after_successes": runtime.get("grow_after_successes", 2),
        "grow_step": runtime.get("grow_step", 1),
        "shrink_factor": runtime.get("shrink_factor", 0.5),
        "flush_every_n_units": runtime.get("flush_every_n_units", 50),
        "flush_every_n_groups": runtime.get("flush_every_n_groups", 1),
        "flush_every_n_seconds": runtime.get("flush_every_n_seconds", 60),
        "max_flushes_per_run": runtime.get("max_flushes_per_run"),
        "log_every_n_groups": runtime.get("log_every_n_groups", 1),
        "soft_time_limit_minutes": runtime.get("soft_time_limit_minutes", 40),
        "max_request_retries": runtime.get("max_request_retries", 3),
        "retry_backoff_seconds": runtime.get("retry_backoff_seconds", 10),
        "flush_scope": runtime.get("flush_scope", "unit"),
        "input_row_limit": runtime.get("input_row_limit"),
        "input_row_offset": runtime.get("input_row_offset", 0),
        "write_partial_merged_output": runtime.get("write_partial_merged_output", True),
    }


def validate_llm_step(step_config: dict):
    required = ["name", "adapter", "task_handler", "provider_config_key", "model", "row_id_column", "input_columns", "output_columns"]
    missing = [k for k in required if k not in step_config]
    if missing:
        raise KeyError(f"LLM step missing required keys: {missing}")
    if not isinstance(step_config["input_columns"], list) or not step_config["input_columns"]:
        raise ValueError("LLM step 'input_columns' must be a non-empty list")
    if not isinstance(step_config["output_columns"], dict) or not step_config["output_columns"]:
        raise ValueError("LLM step 'output_columns' must be a non-empty dict")
    for col in step_config["input_columns"]:
        if col not in step_config["output_columns"]:
            raise ValueError(f"LLM step input column '{col}' missing from output_columns mapping")
    runtime = default_runtime(step_config)
    if runtime["initial_group_size"] < 1:
        raise ValueError("LLM runtime 'initial_group_size' must be >= 1")
    if runtime["min_group_size"] < 1:
        raise ValueError("LLM runtime 'min_group_size' must be >= 1")
    if runtime["max_group_size"] < runtime["min_group_size"]:
        raise ValueError("LLM runtime 'max_group_size' must be >= 'min_group_size'")
    if runtime["initial_group_size"] < runtime["min_group_size"]:
        raise ValueError("LLM runtime 'initial_group_size' must be >= 'min_group_size'")
    if runtime["initial_group_size"] > runtime["max_group_size"]:
        raise ValueError("LLM runtime 'initial_group_size' must be <= 'max_group_size'")
    if runtime["grow_after_successes"] < 1:
        raise ValueError("LLM runtime 'grow_after_successes' must be >= 1")
    if runtime["grow_step"] < 1:
        raise ValueError("LLM runtime 'grow_step' must be >= 1")
    if runtime["shrink_factor"] <= 0 or runtime["shrink_factor"] >= 1:
        raise ValueError("LLM runtime 'shrink_factor' must be > 0 and < 1")
    if runtime["flush_every_n_units"] < 1:
        raise ValueError("LLM runtime 'flush_every_n_units' must be >= 1")
    if runtime["flush_every_n_groups"] < 1:
        raise ValueError("LLM runtime 'flush_every_n_groups' must be >= 1")
    if runtime["flush_every_n_seconds"] < 1:
        raise ValueError("LLM runtime 'flush_every_n_seconds' must be >= 1")
    if runtime["log_every_n_groups"] < 1:
        raise ValueError("LLM runtime 'log_every_n_groups' must be >= 1")
    if runtime["soft_time_limit_minutes"] <= 0:
        raise ValueError("LLM runtime 'soft_time_limit_minutes' must be > 0")
    if runtime["max_request_retries"] < 0:
        raise ValueError("LLM runtime 'max_request_retries' must be >= 0")
    if runtime["retry_backoff_seconds"] < 0:
        raise ValueError("LLM runtime 'retry_backoff_seconds' must be >= 0")
    if runtime["flush_scope"] not in ALLOWED_FLUSH_SCOPES:
        raise ValueError(f"LLM runtime 'flush_scope' must be one of {sorted(ALLOWED_FLUSH_SCOPES)}")
    if runtime["input_row_offset"] is None or int(runtime["input_row_offset"]) < 0:
        raise ValueError("LLM runtime 'input_row_offset' must be >= 0")
    if runtime["input_row_limit"] is not None and int(runtime["input_row_limit"]) < 1:
        raise ValueError("LLM runtime 'input_row_limit' must be >= 1 when provided")
    if runtime["max_flushes_per_run"] is not None and int(runtime["max_flushes_per_run"]) < 1:
        raise ValueError("LLM runtime 'max_flushes_per_run' must be >= 1 when provided")


def success_or_terminal_unit_ids(progress_df: pl.DataFrame) -> set:
    if progress_df.is_empty():
        return set()
    terminal = progress_df.filter(pl.col("status").is_in(["success", "permanent_error"]))
    return set(terminal["unit_id"].to_list())


def current_retry_count(progress_df: pl.DataFrame, unit_id: str) -> int:
    if progress_df.is_empty():
        return 0
    rows = progress_df.filter(pl.col("unit_id") == unit_id)
    if rows.is_empty():
        return 0
    return int(rows["retry_count"].to_list()[-1])


def build_runtime_metadata(step_config: dict, work_units_df: pl.DataFrame, progress_df: pl.DataFrame, outcome: LLMRunOutcome) -> dict:
    return {
        "step_name": step_config["name"],
        "model": step_config["model"],
        "provider_config_key": step_config["provider_config_key"],
        "prompt_version": step_config.get("kwargs", {}).get("prompt_version", "v1"),
        "total_units": work_units_df.height,
        "completed_or_terminal_units": len(success_or_terminal_unit_ids(progress_df)),
        "processed_units_this_run": outcome.processed_units,
        "remaining_units": outcome.remaining_units,
        "outcome": outcome.status,
        "updated_at": utc_now_iso(),
    }
