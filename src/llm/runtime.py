import polars as pl

from .models import LLMRunOutcome
from .state import utc_now_iso


TPM_RUNTIME_DEFAULTS = {
    "target_tokens_per_minute": None,
    "token_estimation_chars_per_token": 4.0,
    "estimated_response_tokens_per_unit": 24,
    "estimated_request_overhead_tokens": 250,
    "tpm_safety_margin": 0.80,
    "max_sleep_to_respect_tpm_seconds": 120,
}


def default_runtime(step_config: dict) -> dict:
    runtime = step_config.get("runtime", {})
    initial_concurrency = int(runtime.get("initial_concurrency", 1))
    max_concurrent_requests = int(runtime.get("max_concurrent_requests", 4))

    if initial_concurrency > max_concurrent_requests:
        initial_concurrency = max_concurrent_requests

    merged = {
        "initial_group_size": int(runtime.get("initial_group_size", 4)),
        "min_group_size": int(runtime.get("min_group_size", 1)),
        "max_group_size": int(runtime.get("max_group_size", 16)),
        "grow_after_successes": int(runtime.get("grow_after_successes", 2)),
        "grow_step": int(runtime.get("grow_step", 1)),
        "shrink_factor": float(runtime.get("shrink_factor", 0.75)),
        "mild_shrink_factor": float(runtime.get("mild_shrink_factor", 0.90)),
        "soft_failure_rate": float(runtime.get("soft_failure_rate", 0.02)),
        "hard_failure_rate": float(runtime.get("hard_failure_rate", 0.10)),
        "throughput_tolerance": float(runtime.get("throughput_tolerance", 0.05)),
        "throughput_ema_alpha": float(runtime.get("throughput_ema_alpha", 0.30)),
        "initial_concurrency": initial_concurrency,
        "min_concurrency": int(runtime.get("min_concurrency", 1)),
        "max_concurrent_requests": max_concurrent_requests,
        "initial_load_budget": int(
            runtime.get(
                "initial_load_budget",
                int(runtime.get("initial_group_size", 4)) * initial_concurrency,
            )
        ),
        "load_growth_factor": float(runtime.get("load_growth_factor", 1.35)),
        "load_shrink_factor": float(runtime.get("load_shrink_factor", 0.60)),
        "mild_load_shrink_factor": float(runtime.get("mild_load_shrink_factor", 0.85)),
        "concurrency_growth_cooldown_waves": int(runtime.get("concurrency_growth_cooldown_waves", 2)),
        "concurrency_shrink_cooldown_waves": int(runtime.get("concurrency_shrink_cooldown_waves", 1)),
        "flush_every_n_units": int(runtime.get("flush_every_n_units", 50)),
        "flush_every_n_groups": int(runtime.get("flush_every_n_groups", 1)),
        "flush_every_n_seconds": int(runtime.get("flush_every_n_seconds", 60)),
        "max_flushes_per_run": runtime.get("max_flushes_per_run"),
        "log_every_n_groups": int(runtime.get("log_every_n_groups", 1)),
        "soft_time_limit_minutes": float(runtime.get("soft_time_limit_minutes", 40)),
        "max_request_retries": int(runtime.get("max_request_retries", 3)),
        "retry_backoff_seconds": float(runtime.get("retry_backoff_seconds", 10)),
        "request_timeout_enabled": bool(runtime.get("request_timeout_enabled", True)),
        "request_timeout_seconds": float(runtime.get("request_timeout_seconds", 300)),
        "request_timeout_min_success_samples": int(runtime.get("request_timeout_min_success_samples", 1)),
        "request_timeout_window_size": int(runtime.get("request_timeout_window_size", 8)),
        "request_timeout_window_statistic": runtime.get("request_timeout_window_statistic", "median"),
        "request_timeout_spread_statistic": runtime.get("request_timeout_spread_statistic", "stdev"),
        "request_timeout_spread_multiplier": float(runtime.get("request_timeout_spread_multiplier", 2.0)),
        "request_timeout_min_margin_seconds": float(runtime.get("request_timeout_min_margin_seconds", 15.0)),
        "request_timeout_max_seconds": float(runtime.get("request_timeout_max_seconds", 1800)),
        "inflight_heartbeat_seconds": float(runtime.get("inflight_heartbeat_seconds", 15)),
        "min_inter_wave_sleep_seconds": float(runtime.get("min_inter_wave_sleep_seconds", 0)),
        "transport_failure_cooldown_seconds": float(runtime.get("transport_failure_cooldown_seconds", 0)),
        "all_transport_failure_cooldown_seconds": float(runtime.get("all_transport_failure_cooldown_seconds", 0)),
        "flush_scope": runtime.get("flush_scope", "unit"),
        "write_partial_merged_output": bool(runtime.get("write_partial_merged_output", False)),
        "write_pair_status": bool(runtime.get("write_pair_status", False)),
        "input_row_limit": runtime.get("input_row_limit"),
        "input_row_offset": int(runtime.get("input_row_offset", 0)),
        "lane_strategy": runtime.get("lane_strategy", "hybrid"),
        "initial_active_lanes": int(runtime.get("initial_active_lanes", 1)),
        "max_active_lanes": int(runtime.get("max_active_lanes", 999999)),
        "lane_exploration_success_waves": int(runtime.get("lane_exploration_success_waves", 2)),
        "lane_reduction_cooldown_waves": int(runtime.get("lane_reduction_cooldown_waves", 2)),
        "shared_failure_window_seconds": float(runtime.get("shared_failure_window_seconds", 90)),
        "shared_failure_lane_threshold": int(runtime.get("shared_failure_lane_threshold", 2)),
        "allow_spillover_when_tpm_blocked": bool(runtime.get("allow_spillover_when_tpm_blocked", True)),
        "artifact_retention_mode": runtime.get("artifact_retention_mode", "standard"),
        "keep_last_flushes": int(runtime.get("keep_last_flushes", 3)),
        "keep_last_results_snapshots": int(runtime.get("keep_last_results_snapshots", runtime.get("keep_last_flushes", 3))),
        "keep_last_results_checkpoints": int(runtime.get("keep_last_results_checkpoints", 2)),
        "keep_last_results_deltas": int(runtime.get("keep_last_results_deltas", 1000)),
        "keep_last_permanent_reviews": int(runtime.get("keep_last_permanent_reviews", 2)),
        "keep_last_progress": int(runtime.get("keep_last_progress", 2)),
        "keep_last_metadata": int(runtime.get("keep_last_metadata", 2)),
        "keep_last_manifests": int(runtime.get("keep_last_manifests", 2)),
        "keep_last_partial_outputs": int(runtime.get("keep_last_partial_outputs", 2)),
        "keep_last_pair_status": int(runtime.get("keep_last_pair_status", 2)),
        "keep_last_traces": int(runtime.get("keep_last_traces", 2)),
        "keep_last_reviews": int(runtime.get("keep_last_reviews", 2)),
        "materialize_every_n_flushes": int(runtime.get("materialize_every_n_flushes", 5)),
        "materialize_every_n_seconds": int(runtime.get("materialize_every_n_seconds", 60)),
        "checkpoint_every_n_flushes": int(runtime.get("checkpoint_every_n_flushes", 10)),
        "checkpoint_every_n_seconds": int(runtime.get("checkpoint_every_n_seconds", 300)),
        "write_pair_status": bool(runtime.get("write_pair_status", True)),
        "write_partial_merged_output": bool(runtime.get("write_partial_merged_output", True)),
        "keep_last_final_outputs": int(runtime.get("keep_last_final_outputs", 1)),
        "keep_last_runs": int(runtime.get("keep_last_runs", 10)),
        "keep_all_flushes_within_kept_runs": bool(runtime.get("keep_all_flushes_within_kept_runs", True)),
    }
    merged.update({key: runtime.get(key, default) for key, default in TPM_RUNTIME_DEFAULTS.items()})
    return merged


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
        raise ValueError("LLM runtime 'initial_concurrency' must be <= 'max_concurrent_requests'")
    if runtime["min_concurrency"] > runtime["max_concurrent_requests"]:
        raise ValueError("LLM runtime 'min_concurrency' must be <= 'max_concurrent_requests'")
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
    if runtime["request_timeout_seconds"] <= 0:
        raise ValueError("LLM runtime 'request_timeout_seconds' must be > 0")
    if str(runtime["request_timeout_spread_statistic"]).lower() not in {"stdev", "mad", "none"}:
        raise ValueError("LLM runtime 'request_timeout_spread_statistic' must be one of {'stdev', 'mad', 'none'}")
    if runtime["request_timeout_spread_multiplier"] < 0:
        raise ValueError("LLM runtime 'request_timeout_spread_multiplier' must be >= 0")
    if runtime["request_timeout_min_margin_seconds"] < 0:
        raise ValueError("LLM runtime 'request_timeout_min_margin_seconds' must be >= 0")
    if runtime["request_timeout_min_success_samples"] < 1:
        raise ValueError("LLM runtime 'request_timeout_min_success_samples' must be >= 1")
    if runtime["request_timeout_window_size"] < 1:
        raise ValueError("LLM runtime 'request_timeout_window_size' must be >= 1")
    if str(runtime["request_timeout_window_statistic"]).lower() not in {"median", "mean"}:
        raise ValueError("LLM runtime 'request_timeout_window_statistic' must be one of {'median', 'mean'}")
    if runtime["request_timeout_max_seconds"] < runtime["request_timeout_seconds"]:
        raise ValueError("LLM runtime 'request_timeout_max_seconds' must be >= 'request_timeout_seconds'")
    if runtime["inflight_heartbeat_seconds"] <= 0:
        raise ValueError("LLM runtime 'inflight_heartbeat_seconds' must be > 0")
    if runtime["min_inter_wave_sleep_seconds"] < 0:
        raise ValueError("LLM runtime 'min_inter_wave_sleep_seconds' must be >= 0")
    if runtime["transport_failure_cooldown_seconds"] < 0:
        raise ValueError("LLM runtime 'transport_failure_cooldown_seconds' must be >= 0")
    if runtime["all_transport_failure_cooldown_seconds"] < 0:
        raise ValueError("LLM runtime 'all_transport_failure_cooldown_seconds' must be >= 0")
    if runtime["flush_scope"] not in {"unit", "row_complete"}:
        raise ValueError("LLM runtime 'flush_scope' must be one of {'unit', 'row_complete'}")


    if runtime["lane_strategy"] not in {"hybrid", "safe_single_active"}:
        raise ValueError("LLM runtime 'lane_strategy' must be one of {'hybrid', 'safe_single_active'}")
    if runtime["initial_active_lanes"] < 1:
        raise ValueError("LLM runtime 'initial_active_lanes' must be >= 1")
    if runtime["max_active_lanes"] < 1:
        raise ValueError("LLM runtime 'max_active_lanes' must be >= 1")
    if runtime["initial_active_lanes"] > runtime["max_active_lanes"]:
        raise ValueError("LLM runtime 'initial_active_lanes' must be <= 'max_active_lanes'")
    if runtime["lane_exploration_success_waves"] < 1:
        raise ValueError("LLM runtime 'lane_exploration_success_waves' must be >= 1")
    if runtime["lane_reduction_cooldown_waves"] < 1:
        raise ValueError("LLM runtime 'lane_reduction_cooldown_waves' must be >= 1")
    if runtime["shared_failure_window_seconds"] < 0:
        raise ValueError("LLM runtime 'shared_failure_window_seconds' must be >= 0")
    if runtime["shared_failure_lane_threshold"] < 1:
        raise ValueError("LLM runtime 'shared_failure_lane_threshold' must be >= 1")
    if runtime["artifact_retention_mode"] not in {"standard", "debug"}:
        raise ValueError("LLM runtime 'artifact_retention_mode' must be one of {'standard', 'debug'}")
    if runtime["keep_last_flushes"] < 1:
        raise ValueError("LLM runtime 'keep_last_flushes' must be >= 1")
    if runtime["keep_last_results_snapshots"] < 1:
        raise ValueError("LLM runtime 'keep_last_results_snapshots' must be >= 1")
    if runtime["keep_last_permanent_reviews"] < 1:
        raise ValueError("LLM runtime 'keep_last_permanent_reviews' must be >= 1")
    if runtime["keep_last_progress"] < 1:
        raise ValueError("LLM runtime 'keep_last_progress' must be >= 1")
    if runtime["keep_last_metadata"] < 1:
        raise ValueError("LLM runtime 'keep_last_metadata' must be >= 1")
    if runtime["keep_last_manifests"] < 1:
        raise ValueError("LLM runtime 'keep_last_manifests' must be >= 1")
    if runtime["keep_last_partial_outputs"] < 1:
        raise ValueError("LLM runtime 'keep_last_partial_outputs' must be >= 1")
    if runtime["keep_last_pair_status"] < 1:
        raise ValueError("LLM runtime 'keep_last_pair_status' must be >= 1")
    if runtime["keep_last_traces"] < 1:
        raise ValueError("LLM runtime 'keep_last_traces' must be >= 1")
    if runtime["keep_last_reviews"] < 1:
        raise ValueError("LLM runtime 'keep_last_reviews' must be >= 1")
    if runtime["materialize_every_n_flushes"] < 1:
        raise ValueError("LLM runtime 'materialize_every_n_flushes' must be >= 1")
    if runtime["materialize_every_n_seconds"] < 1:
        raise ValueError("LLM runtime 'materialize_every_n_seconds' must be >= 1")
    if runtime["keep_last_final_outputs"] < 1:
        raise ValueError("LLM runtime 'keep_last_final_outputs' must be >= 1")
    if runtime["keep_last_runs"] < 1:
        raise ValueError("LLM runtime 'keep_last_runs' must be >= 1")

    if runtime["target_tokens_per_minute"] is not None:
        if float(runtime["target_tokens_per_minute"]) <= 0:
            raise ValueError("LLM runtime 'target_tokens_per_minute' must be > 0 when provided")
        if float(runtime["token_estimation_chars_per_token"]) <= 0:
            raise ValueError("LLM runtime 'token_estimation_chars_per_token' must be > 0")
        if float(runtime["estimated_response_tokens_per_unit"]) < 0:
            raise ValueError("LLM runtime 'estimated_response_tokens_per_unit' must be >= 0")
        if float(runtime["estimated_request_overhead_tokens"]) < 0:
            raise ValueError("LLM runtime 'estimated_request_overhead_tokens' must be >= 0")
        if float(runtime["tpm_safety_margin"]) <= 0 or float(runtime["tpm_safety_margin"]) > 1:
            raise ValueError("LLM runtime 'tpm_safety_margin' must be > 0 and <= 1")
        if float(runtime["max_sleep_to_respect_tpm_seconds"]) < 0:
            raise ValueError("LLM runtime 'max_sleep_to_respect_tpm_seconds' must be >= 0")


def apply_input_row_window(source_df: pl.DataFrame, runtime: dict) -> pl.DataFrame:
    offset = int(runtime.get("input_row_offset", 0) or 0)
    limit = runtime.get("input_row_limit")
    if offset < 0:
        raise ValueError("LLM runtime 'input_row_offset' must be >= 0")
    if limit is not None and int(limit) < 0:
        raise ValueError("LLM runtime 'input_row_limit' must be >= 0 when provided")
    if offset:
        source_df = source_df.slice(offset)
    if limit is not None:
        source_df = source_df.slice(0, int(limit))
    return source_df


def success_unit_ids(progress_df: pl.DataFrame) -> set:
    if progress_df is None or progress_df.is_empty():
        return set()
    completed = progress_df.filter(pl.col("status") == "success")
    return set(completed["unit_id"].to_list())


def success_or_terminal_unit_ids(progress_df: pl.DataFrame) -> set:
    if progress_df is None or progress_df.is_empty():
        return set()
    completed = progress_df.filter(pl.col("status").is_in(["success", "permanent_error"]))
    return set(completed["unit_id"].to_list())


def current_retry_count(progress_df: pl.DataFrame, unit_id: str) -> int:
    if progress_df is None or progress_df.is_empty():
        return 0
    rows = progress_df.filter(pl.col("unit_id") == unit_id)
    if rows.is_empty():
        return 0
    value = rows.sort("updated_at").tail(1)["retry_count"].to_list()[0]
    return int(value or 0)


def build_runtime_metadata(
    step_config: dict,
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    outcome: LLMRunOutcome,
    all_results_df: pl.DataFrame | None = None,
) -> dict:
    useful_completed = 0
    if all_results_df is not None and not all_results_df.is_empty():
        useful_completed = int(
            all_results_df
            .filter(pl.col("status") == "success")
            .select(pl.col("unit_id").n_unique())
            .item()
        )
    else:
        useful_completed = len(success_unit_ids(progress_df))

    terminal_count = len(success_or_terminal_unit_ids(progress_df))
    permanent_error_count = 0 if progress_df is None or progress_df.is_empty() else int(
        progress_df
        .filter(pl.col("status") == "permanent_error")
        .select(pl.col("unit_id").n_unique())
        .item()
    )
    return {
        "step_name": step_config["name"],
        "status": outcome.status,
        "processed_units_this_run": int(outcome.processed_units),
        "remaining_units": int(outcome.remaining_units),
        "total_units": int(work_units_df.height),
        "completed_useful_units": useful_completed,
        "completed_or_terminal_units": terminal_count,
        "permanent_error_units": permanent_error_count,
        "updated_at": utc_now_iso(),
    }


def build_run_metadata(step_config: dict, work_units_df: pl.DataFrame, progress_df: pl.DataFrame, outcome: LLMRunOutcome, all_results_df: pl.DataFrame | None = None) -> dict:
    return build_runtime_metadata(step_config, work_units_df, progress_df, outcome, all_results_df=all_results_df)
