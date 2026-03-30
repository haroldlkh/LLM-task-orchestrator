from .models import LLMRunOutcome
from .state import utc_now_iso


def default_runtime(step_config: dict) -> dict:
    runtime = step_config.get("runtime", {})
    return {
        "max_units_per_batch": runtime.get("max_units_per_batch", 10),
        "flush_every_n_units": runtime.get("flush_every_n_units", 50),
        "soft_time_limit_minutes": runtime.get("soft_time_limit_minutes", 40),
        "max_request_retries": runtime.get("max_request_retries", 3),
        "retry_backoff_seconds": runtime.get("retry_backoff_seconds", 5),
    }


def validate_llm_step(step_config: dict):
    required = [
        "name",
        "adapter",
        "task_handler",
        "provider_config_key",
        "model",
        "row_id_column",
        "input_columns",
        "output_columns",
    ]
    missing = [k for k in required if k not in step_config]
    if missing:
        raise KeyError(f"LLM step missing required keys: {missing}")

    if not isinstance(step_config["input_columns"], list) or not step_config["input_columns"]:
        raise ValueError("LLM step 'input_columns' must be a non-empty list")

    if not isinstance(step_config["output_columns"], dict) or not step_config["output_columns"]:
        raise ValueError("LLM step 'output_columns' must be a non-empty dict")

    for col in step_config["input_columns"]:
        if col not in step_config["output_columns"]:
            raise ValueError(
                f"LLM step input column '{col}' missing from output_columns mapping"
            )


def success_or_terminal_unit_ids(progress_df) -> set:
    if progress_df.is_empty():
        return set()

    terminal = progress_df.filter(
        progress_df["status"].is_in(["success", "permanent_error"])
    )
    return set(terminal["unit_id"].to_list())


def current_retry_count(progress_df, unit_id: str) -> int:
    if progress_df.is_empty():
        return 0
    rows = progress_df.filter(progress_df["unit_id"] == unit_id)
    if rows.is_empty():
        return 0
    return int(rows["retry_count"].to_list()[-1])


def build_runtime_metadata(
    step_config: dict,
    work_units_df,
    progress_df,
    outcome: LLMRunOutcome,
) -> dict:
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