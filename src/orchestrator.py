import os
import sys
import shutil
import importlib.util
import importlib
from typing import Any, Dict, List

from data_connectors import GDriveConnector
from factory_loader import get_loader


def require_env(name: str, allow_empty: bool = False) -> str:
    value = os.environ.get(name, "")
    if not allow_empty and not value.strip():
        raise ValueError(f"Required environment variable {name} is missing or empty.")
    return value.strip()


def add_user_repo_to_path(path_in_repo: str) -> str:
    """
    Given a user_repo-relative file path, ensure user_repo root is on sys.path.
    Returns absolute repo-root path.
    """
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


def parse_entry_inputs() -> Dict[str, str]:
    task_script = os.environ.get("TASK_SCRIPT", "").strip()
    pipeline_script = os.environ.get("PIPELINE_SCRIPT", "").strip()

    provided = [x for x in [task_script, pipeline_script] if x]
    if len(provided) == 0:
        raise ValueError("You must provide exactly one of TASK_SCRIPT or PIPELINE_SCRIPT.")
    if len(provided) > 1:
        raise ValueError("Provide only one of TASK_SCRIPT or PIPELINE_SCRIPT, not both.")

    if task_script:
        return {"entry_type": "task", "path": task_script}
    return {"entry_type": "pipeline", "path": pipeline_script}


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
        if "module" not in step or "function" not in step:
            raise KeyError(
                f"{pipeline_path} step {i} must include 'module' and 'function'"
            )

    return config


def load_pipeline(pipeline_path: str) -> Dict[str, Any]:
    pipeline_module = load_python_file_module(pipeline_path, "user_pipeline")
    config = validate_pipeline_config(pipeline_module, pipeline_path)

    add_user_repo_to_path(pipeline_path)

    loaded_steps = []
    for step in config["steps"]:
        module = importlib.import_module(step["module"])
        function_name = step["function"]

        if not hasattr(module, function_name):
            raise AttributeError(
                f"Pipeline step module '{step['module']}' does not have "
                f"function '{function_name}'"
            )

        fn = getattr(module, function_name)
        kwargs = step.get("kwargs", {})

        loaded_steps.append(
            {
                "module": step["module"],
                "function": function_name,
                "callable": fn,
                "kwargs": kwargs,
            }
        )

    return {
        "kind": "pipeline",
        "name": config["name"],
        "mode": config["mode"],
        "data_type": config["data_type"],
        "requirements": config["requirements"],
        "steps": loaded_steps,
    }


def run_pipeline_steps(steps: List[Dict[str, Any]], data):
    current_data = data

    for i, step in enumerate(steps, start=1):
        step_name = f"{step['module']}:{step['function']}"
        print(f"Running pipeline step {i}/{len(steps)}: {step_name}")
        current_data = step["callable"](current_data, **step["kwargs"])

        if current_data is None:
            raise ValueError(
                f"Pipeline step '{step_name}' returned None. "
                f"Each pipeline step must return data."
            )

    return current_data


def run_single_task_runner(runner, data):
    result = runner(data)
    if result is None:
        raise ValueError("Task returned None. Expected data output.")
    return result


def save_final_output(loader, data, output_folder, connector, output_name):
    final_filename = loader.save(data, output_name)

    if final_filename and os.path.exists(final_filename):
        remote_name = os.path.basename(final_filename)
        file_id = connector.upload_file(final_filename, output_folder, remote_name)
        print(f"Uploaded: {remote_name} (file_id={file_id})")
        os.remove(final_filename)
    else:
        raise RuntimeError("No output file was created.")


def run_global_execution(
    connector,
    loader,
    executable,
    source_folder,
    output_folder,
    output_name,
    target_cols,
):
    temp_dir = "temp_all"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        all_files = [
            f for f in connector.list_files_in_folder(source_folder)
            if f["name"].endswith(".parquet")
        ]

        print(f"Discovered {len(all_files)} parquet file(s) in source folder.")

        for f_info in all_files:
            local_path = os.path.join(temp_dir, f_info["name"])
            print(f"Downloading {f_info['name']}...")
            connector.download_file(f_info["id"], local_path)

        data = loader.load(target_cols)

        if executable["kind"] == "task":
            result = run_single_task_runner(executable["runner"], data)
        else:
            result = run_pipeline_steps(executable["steps"], data)

        save_final_output(loader, result, output_folder, connector, output_name)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batched_execution(
    connector,
    executable,
    source_folder,
    output_folder,
    output_name,
    batch_size,
    data_type,
    target_cols,
):
    all_files = [
        f for f in connector.list_files_in_folder(source_folder)
        if f["name"].endswith(".parquet")
    ]

    print(f"Discovered {len(all_files)} parquet file(s) in source folder.")

    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        temp_dir = "temp_batch"
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Processing batch {batch_num} with {len(batch)} file(s)...")

        try:
            for f_info in batch:
                local_path = os.path.join(temp_dir, f_info["name"])
                print(f"Downloading {f_info['name']}...")
                connector.download_file(f_info["id"], local_path)

            loader = get_loader(data_type, temp_dir)
            data = loader.load(target_cols)

            if executable["kind"] == "task":
                result = run_single_task_runner(executable["runner"], data)
            else:
                result = run_pipeline_steps(executable["steps"], data)

            batch_output_name = f"{output_name}_batch_{batch_num}"
            save_final_output(loader, result, output_folder, connector, batch_output_name)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_pipeline():
    source_folder = require_env("SOURCE_FOLDER_ID")
    output_folder = require_env("OUTPUT_FOLDER_ID")
    ts = os.environ.get("TIMESTAMP", "000000")
    batch_size = int(os.environ.get("BATCH_SIZE", 5))

    entry = parse_entry_inputs()

    if entry["entry_type"] == "task":
        executable = load_single_task(entry["path"])
    else:
        executable = load_pipeline(entry["path"])

    connector = GDriveConnector()

    mode = executable["mode"]
    data_type = executable["data_type"]
    target_cols = executable["requirements"].get("columns", [])

    pipeline_name = os.environ.get("PIPELINE_NAME", "").strip()
    base_name = pipeline_name if pipeline_name else executable["name"]
    output_name = f"{base_name}_{ts}"

    print("=== ENGINE CONFIG ===")
    print(f"Entry type: {entry['entry_type']}")
    print(f"Entry path: {entry['path']}")
    print(f"Mode: {mode}")
    print(f"Data type: {data_type}")
    print(f"Source folder: {source_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Output name base: {output_name}")

    if mode == "global":
        loader = get_loader(data_type, "temp_all")
        run_global_execution(
            connector=connector,
            loader=loader,
            executable=executable,
            source_folder=source_folder,
            output_folder=output_folder,
            output_name=f"{output_name}",
            target_cols=target_cols,
        )
    elif mode == "batch":
        run_batched_execution(
            connector=connector,
            executable=executable,
            source_folder=source_folder,
            output_folder=output_folder,
            output_name=output_name,
            batch_size=batch_size,
            data_type=data_type,
            target_cols=target_cols,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    run_pipeline()