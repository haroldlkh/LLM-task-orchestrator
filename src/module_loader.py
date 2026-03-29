import importlib
import importlib.util
import os
import sys
from typing import Any, Dict


ENGINE_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(ENGINE_SRC_DIR)

if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)


def add_user_repo_to_path(path_in_repo: str) -> str:
    abs_path = os.path.abspath(path_in_repo)
    repo_root = abs_path.split(os.sep + "user_repo" + os.sep)[0] + os.sep + "user_repo"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def load_python_file_module(file_path: str, module_name: str):
    add_user_repo_to_path(file_path)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_task_module(user_module, task_script: str):
    if not hasattr(user_module, "TASK_CONFIG"):
        raise AttributeError(f"{task_script} must define TASK_CONFIG")
    if not hasattr(user_module, "get_requirements"):
        raise AttributeError(f"{task_script} must define get_requirements()")
    if not hasattr(user_module, "run"):
        raise AttributeError(f"{task_script} must define run(data)")

    task_config = user_module.TASK_CONFIG

    for key in ("name", "mode", "data_type"):
        if key not in task_config:
            raise KeyError(f"{task_script} TASK_CONFIG must include '{key}'")

    if task_config["mode"] not in {"batch", "global"}:
        raise ValueError(
            f"{task_script} TASK_CONFIG['mode'] must be 'batch' or 'global'"
        )

    return task_config


def load_single_task(task_path: str) -> Dict[str, Any]:
    module = load_python_file_module(task_path, "user_task")
    config = validate_task_module(module, task_path)

    return {
        "kind": "task",
        "name": config["name"],
        "mode": config["mode"],
        "data_type": config["data_type"],
        "requirements": module.get_requirements(),
        "runner": module.run,
    }


def validate_pipeline_config(pipeline_module, pipeline_path: str):
    if not hasattr(pipeline_module, "PIPELINE_CONFIG"):
        raise AttributeError(f"{pipeline_path} must define PIPELINE_CONFIG")

    config = pipeline_module.PIPELINE_CONFIG

    for key in ("name", "mode", "data_type", "requirements", "steps"):
        if key not in config:
            raise KeyError(f"{pipeline_path} PIPELINE_CONFIG must include '{key}'")

    if config["mode"] not in {"batch", "global"}:
        raise ValueError(
            f"{pipeline_path} PIPELINE_CONFIG['mode'] must be 'batch' or 'global'"
        )

    requirements = config["requirements"]
    if not isinstance(requirements, dict) or "columns" not in requirements:
        raise ValueError(
            f"{pipeline_path} PIPELINE_CONFIG['requirements'] must be a dict "
            f"containing a 'columns' list"
        )

    if not isinstance(config["steps"], list) or not config["steps"]:
        raise ValueError(f"{pipeline_path} PIPELINE_CONFIG['steps'] must be a non-empty list")

    for i, step in enumerate(config["steps"], start=1):
        if step.get("type") == "llm":
            required = [
                "name",
                "adapter",
                "provider_config_key",
                "model",
                "row_id_column",
                "input_columns",
                "output_columns",
            ]
            missing = [k for k in required if k not in step]
            if missing:
                raise KeyError(
                    f"{pipeline_path} llm step {i} missing required keys: {missing}"
                )
        else:
            if "module" not in step or "function" not in step:
                raise KeyError(
                    f"{pipeline_path} normal step {i} must include 'module' and 'function'"
                )

    return config


def _resolve_normal_step(step: Dict[str, Any]) -> Dict[str, Any]:
    module_name = step["module"]
    function_name = step["function"]
    module = importlib.import_module(module_name)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Pipeline step module '{module_name}' does not have "
            f"function '{function_name}'"
        )

    fn = getattr(module, function_name)

    return {
        "kind": "normal",
        "module": module_name,
        "function": function_name,
        "callable": fn,
        "kwargs": step.get("kwargs", {}),
    }


def _resolve_llm_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "llm",
        "config": step,
    }


def load_pipeline(pipeline_path: str) -> Dict[str, Any]:
    pipeline_module = load_python_file_module(pipeline_path, "user_pipeline")
    config = validate_pipeline_config(pipeline_module, pipeline_path)

    add_user_repo_to_path(pipeline_path)

    loaded_steps = []
    for step in config["steps"]:
        if step.get("type") == "llm":
            loaded_steps.append(_resolve_llm_step(step))
        else:
            loaded_steps.append(_resolve_normal_step(step))

    return {
        "kind": "pipeline",
        "name": config["name"],
        "mode": config["mode"],
        "data_type": config["data_type"],
        "requirements": config["requirements"],
        "steps": loaded_steps,
    }