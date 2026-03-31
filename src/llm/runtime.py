import polars as pl

from .models import LLMRunOutcome
from .state import utc_now_iso


def default_runtime(step_config: dict) -> dict:
    runtime = step_config.get("runtime", {})
    initial_concurrency = runtime.get("initial_concurrency", 1)
    max_concurrent_requests = runtime.get("max_concurrent_requests", 4)

    if initial_concurrency > max_concurrent_requests:
        initial_concurrency = max_concurrent_requests

    return {
        "initial_group_size": runtime.get("initial_group_size", 4),
        "min_group_size": runtime.get("min_group_size", 1),
        "max_group_size": runtime.get("max_group_size", 16),
        "grow_after_successes": runtime.get("grow_after_successes", 2),
        "grow_step": runtime.get("grow_step", 1),

        # kept for compatibility with earlier controller versions
        "shrink_factor": runtime.get("shrink_factor", 0.75),
        "mild_shrink_factor": runtime.get("mild_shrink_factor", 0.90),

        # throughput / failure-rate controller
        "soft_failure_rate": runtime.get("soft_failure_rate", 0.02),
        "hard_failure_rate": runtime.get("hard_failure_rate", 0.10),
        "throughput_tolerance": runtime.get("throughput_tolerance", 0.05),
        "throughput_ema_alpha": runtime.get("throughput_ema_alpha", 0.30),

        # tandem load controller
        "initial_concurrency": initial_concurrency,
        "min_concurrency": runtime.get("min_concurrency", 1),
        "max_concurrent_requests": max_concurrent_requests,
        "initial_load_budget": runtime.get(
            "initial_load_budget",
            runtime.get("initial_group_size", 4) * initial_concurrency,
        ),
        "load_growth_factor": runtime.get("load_growth_factor", 1.35),
        "load_shrink_factor": runtime.get("load_shrink_factor", 0.60),
        "mild_load_shrink_factor": runtime.get("mild_load_shrink_factor", 0.85),
        "concurrency_growth_cooldown_waves": runtime.get(
            "concurrency_growth_cooldown_waves", 2
        ),
        "concurrency_shrink_cooldown_waves": runtime.get(
            "concurrency_shrink_cooldown_waves", 1
        ),

        # flushing / logging / timing
        "flush_every_n_units": runtime.get("flush_every_n_units", 50),
        "flush_every_n_groups": runtime.get("flush_every_n_groups", 1),
        "flush_every_n_seconds": runtime.get("flush_every_n_seconds", 60),
        "max_flushes_per_run": runtime.get("max_flushes_per_run"),
        "log_every_n_groups": runtime.get("log_every_n_groups", 1),
        "soft_time_limit_minutes": runtime.get("soft_time_limit_minutes", 40),
        "max_request_retries": runtime.get("max_request_retries", 3),
        "retry_backoff_seconds": runtime.get("retry_backoff_seconds", 10),
        "min_inter_wave_sleep_seconds": runtime.get("min_inter_wave_sleep_seconds", 0),
        "transport_failure_cooldown_seconds": runtime.get(
            "transport_failure_cooldown_seconds", 0
        ),
        "all_transport_failure_cooldown_seconds": runtime.get(
            "all_transport_failure_cooldown_seconds", 0
        ),

        # proactive TPM governor
        "target_tokens_per_minute": runtime.get("target_tokens_per_minute", 0),
        "token_estimation_chars_per_token": runtime.get(
            "token_estimation_chars_per_token", 4.0
        ),
        "estimated_response_tokens_per_unit": runtime.get(
            "estimated_response_tokens_per_unit", 14
        ),
        "estimated_request_overhead_tokens": runtime.get(
            "estimated_request_overhead_tokens", 48
        ),
        "tpm_safety_margin": runtime.get("tpm_safety_margin", 0.85),
        "max_sleep_to_respect_tpm_seconds": runtime.get(
            "max_sleep_to_respect_tpm_seconds", 180
        ),

        # release / artifacts
        "flush_scope": runtime.get("flush_scope", "unit"),
        "write_partial_merged_output": runtime.get("write_partial_merged_output", False),
        "write_pair_status": runtime.get("write_pair_status", False),

        # subset-window testing
        "input_row_limit": runtime.get("input_row_limit"),
        "input_row_offset": runtime.get("input_row_offset", 0),
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
    if runtime["mild_shrink_factor"] <= 0 or runtime["mild_shrink_factor"] >= 1:
        raise ValueError("LLM runtime 'mild_shrink_factor' must be > 0 and < 1")

    if runtime["soft_failure_rate"] < 0 or runtime["soft_failure_rate"] > 1:
        raise ValueError("LLM runtime 'soft_failure_rate' must be between 0 and 1")
    if runtime["hard_failure_rate"] < 0 or runtime["hard_failure_rate"] > 1:
        raise ValueError("LLM runtime 'hard_failure_rate' must be between 0 and 1")
    if runtime["soft_failure_rate"] > runtime["hard_failure_rate"]:
        raise ValueError("LLM runtime 'soft_failure_rate' must be <= 'hard_failure_rate'")

    if runtime["throughput_tolerance"] < 0:
        raise ValueError("LLM runtime 'throughput_tolerance' must be >= 0")
    if runtime["throughput_ema_alpha"] <= 0 or runtime["throughput_ema_alpha"] > 1:
        raise ValueError("LLM runtime 'throughput_ema_alpha' must be > 0 and <= 1")

    if runtime["initial_concurrency"] < 1:
        raise ValueError("LLM runtime 'initial_concurrency' must be >= 1")
    if runtime["min_concurrency"] < 1:
        raise ValueError("LLM runtime 'min_concurrency' must be >= 1")
    if runtime["max_concurrent_requests"] < 1:
        raise ValueError("LLM runtime 'max_concurrent_requests' must be >= 1")

    if runtime["max_concurrent_requests"] > 7:
        raise ValueError("LLM runtime 'max_concurrent_requests' must be <= 7")

    if runtime["initial_concurrency"] > runtime["max_concurrent_requests"]:
        raise ValueError(
            "LLM runtime 'initial_concurrency' must be <= 'max_concurrent_requests'"
        )
    if runtime["min_concurrency"] > runtime["max_concurrent_requests"]:
        raise ValueError(
            "LLM runtime 'min_concurrency' must be <= 'max_concurrent_requests'"
        )

    if runtime["initial_load_budget"] < 1:
        raise ValueError("LLM runtime 'initial_load_budget' must be >= 1")
    if runtime["load_growth_factor"] <= 1:
        raise ValueError("LLM runtime 'load_growth_factor' must be > 1")
    if runtime["load_shrink_factor"] <= 0 or runtime["load_shrink_factor"] >= 1:
        raise ValueError("LLM runtime 'load_shrink_factor' must be > 0 and < 1")
    if runtime["mild_load_shrink_factor"] <= 0 or runtime["mild_load_shrink_factor"] >= 1:
        raise ValueError("LLM runtime 'mild_load_shrink_factor' must be > 0 and < 1")

    if runtime["concurrency_growth_cooldown_waves"] < 1:
        raise ValueError("LLM runtime 'concurrency_growth_cooldown_waves' must be >= 1")
    if runtime["concurrency_shrink_cooldown_waves"] < 1:
        raise ValueError("LLM runtime 'concurrency_shrink_cooldown_waves' must be >= 1")

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
    if runtime["min_inter_wave_sleep_seconds"] < 0:
        raise ValueError("LLM runtime 'min_inter_wave_sleep_seconds' must be >= 0")
    if runtime["transport_failure_cooldown_seconds"] < 0:
        raise ValueError("LLM runtime 'transport_failure_cooldown_seconds' must be >= 0")
    if runtime["all_transport_failure_cooldown_seconds"] < 0:
        raise ValueError(
            "LLM runtime 'all_transport_failure_cooldown_seconds' must be >= 0"
        )

    if runtime["target_tokens_per_minute"] < 0:
        raise ValueError("LLM runtime 'target_tokens_per_minute' must be >= 0")
    if runtime["token_estimation_chars_per_token"] <= 0:
        raise ValueError(
            "LLM runtime 'token_estimation_chars_per_token' must be > 0"
        )
    if runtime["estimated_response_tokens_per_unit"] < 0:
        raise ValueError(
            "LLM runtime 'estimated_response_tokens_per_unit' must be >= 0"
        )
    if runtime["estimated_request_overhead_tokens"] < 0:
        raise ValueError(
            "LLM runtime 'estimated_request_overhead_tokens' must be >= 0"
        )
    if runtime["tpm_safety_margin"] <= 0 or runtime["tpm_safety_margin"] > 1:
        raise ValueError("LLM runtime 'tpm_safety_margin' must be > 0 and <= 1")
    if runtime["max_sleep_to_respect_tpm_seconds"] < 0:
        raise ValueError(
            "LLM runtime 'max_sleep_to_respect_tpm_seconds' must be >= 0"
        )

    if runtime["flush_scope"] not in {"unit", "row_complete"}:
        raise ValueError("LLM runtime 'flush_scope' must be 'unit' or 'row_complete'")

    if runtime["input_row_limit"] is not None and runtime["input_row_limit"] < 1:
        raise ValueError("LLM runtime 'input_row_limit' must be >= 1 when provided")
    if runtime["input_row_offset"] < 0:
        raise ValueError("LLM runtime 'input_row_offset' must be >= 0")


def apply_input_row_window(source_df: pl.DataFrame, runtime: dict) -> pl.DataFrame:
    row_offset = runtime.get("input_row_offset", 0) or 0
    row_limit = runtime.get("input_row_limit")

    if row_offset > 0:
        source_df = source_df.slice(row_offset)

    if row_limit is not None:
        source_df = source_df.slice(0, row_limit)

    return source_df


def build_run_metadata(
    *,
    step_name: str,
    runtime: dict,
    current_group_size: int,
    pending_units: int,
    outcome: LLMRunOutcome | None = None,
    workflow_name: str | None = None,
    current_concurrency: int | None = None,
    current_load_budget: int | None = None,
) -> dict:
    payload = {
        "step_name": step_name,
        "workflow_name": workflow_name,
        "runtime": runtime,
        "current_group_size": current_group_size,
        "pending_units": pending_units,
        "updated_at": utc_now_iso(),
        "current_concurrency": current_concurrency,
        "current_load_budget": current_load_budget,
    }
    if outcome is not None:
        payload["outcome"] = {
            "status": outcome.status,
            "processed_units": outcome.processed_units,
            "remaining_units": outcome.remaining_units,
        }
    return payload
