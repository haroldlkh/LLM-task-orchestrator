from .runtime import default_runtime


_ARTIFACT_KEYS = {
    "artifact_retention_mode",
    "keep_last_flushes",
    "keep_last_final_outputs",
    "keep_last_runs",
    "keep_all_flushes_within_kept_runs",
}


def merged_artifact_runtime_from_executable(executable):
    if executable.get("kind") != "pipeline":
        return None

    llm_runtimes = []
    for step in executable.get("steps", []):
        if step.get("kind") == "llm":
            llm_runtimes.append(default_runtime(step["config"]))

    if not llm_runtimes:
        return None

    merged = {key: llm_runtimes[0][key] for key in _ARTIFACT_KEYS}
    merged["artifact_retention_mode"] = (
        "debug" if any(rt.get("artifact_retention_mode") == "debug" for rt in llm_runtimes) else "standard"
    )
    merged["keep_last_flushes"] = max(int(rt.get("keep_last_flushes", 3)) for rt in llm_runtimes)
    merged["keep_last_final_outputs"] = max(int(rt.get("keep_last_final_outputs", 1)) for rt in llm_runtimes)
    merged["keep_last_runs"] = max(int(rt.get("keep_last_runs", 10)) for rt in llm_runtimes)
    merged["keep_all_flushes_within_kept_runs"] = any(
        bool(rt.get("keep_all_flushes_within_kept_runs", True)) for rt in llm_runtimes
    )
    return merged
