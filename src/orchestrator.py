import os
import sys
import shutil
import importlib.util
from typing import List

from data_connectors import GDriveConnector
from factory_loader import get_loader


def require_env(name: str, allow_empty: bool = False) -> str:
    value = os.environ.get(name, "")
    if not allow_empty and not value.strip():
        raise ValueError(f"Required environment variable {name} is missing or empty.")
    return value.strip()


def load_user_task(task_path: str):
    """
    Load a task module from the user repo and make the user_repo root importable,
    so imports like `from tasks.lib...` work.
    """
    repo_root = os.path.dirname(os.path.dirname(task_path))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location("user_task", task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task script: {task_path}")

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


def parse_task_paths() -> List[str]:
    """
    Backward compatible:
    - TASK_SCRIPT = one task
    - TASK_SCRIPTS = comma-separated task list
    """
    task_script = os.environ.get("TASK_SCRIPT", "").strip()
    task_scripts = os.environ.get("TASK_SCRIPTS", "").strip()

    if task_script and task_scripts:
        raise ValueError("Use either TASK_SCRIPT or TASK_SCRIPTS, not both.")

    if task_script:
        return [task_script]

    if task_scripts:
        paths = [p.strip() for p in task_scripts.split(",") if p.strip()]
        if not paths:
            raise ValueError("TASK_SCRIPTS was provided but no valid paths were found.")
        return paths

    raise ValueError("You must provide TASK_SCRIPT or TASK_SCRIPTS.")


def load_pipeline(task_paths: List[str]):
    tasks = []

    for task_path in task_paths:
        module = load_user_task(task_path)
        config = validate_task_module(module, task_path)
        tasks.append(
            {
                "path": task_path,
                "module": module,
                "config": config,
            }
        )

    # All tasks in one in-memory pipeline must share mode + data_type
    modes = {t["config"]["mode"] for t in tasks}
    data_types = {t["config"]["data_type"] for t in tasks}

    if len(modes) != 1:
        raise ValueError(
            f"All tasks in a pipeline must share the same mode. Found: {modes}"
        )

    if len(data_types) != 1:
        raise ValueError(
            f"All tasks in a pipeline must share the same data_type. Found: {data_types}"
        )

    return tasks


def run_task_sequence(tasks, data):
    """
    Runs tasks in memory, passing the output of one task directly to the next.
    """
    current_data = data

    for i, task in enumerate(tasks, start=1):
        task_name = task["config"]["name"]
        print(f"Running pipeline task {i}/{len(tasks)}: {task_name}")
        current_data = task["module"].run(current_data)

        if current_data is None:
            raise ValueError(
                f"Task '{task_name}' returned None. In pipeline mode, each task "
                f"must return data for the next task."
            )

    return current_data


def save_final_output(loader, data, output_folder, connector, output_name):
    final_filename = loader.save(data, output_name)

    if final_filename and os.path.exists(final_filename):
        remote_name = os.path.basename(final_filename)
        file_id = connector.upload_file(final_filename, output_folder, remote_name)
        print(f"Uploaded: {remote_name} (file_id={file_id})")
        os.remove(final_filename)
    else:
        raise RuntimeError("No output file was created.")


def run_global_pipeline(
    connector,
    loader,
    tasks,
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
        result = run_task_sequence(tasks, data)
        save_final_output(loader, result, output_folder, connector, output_name)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batched_pipeline(
    connector,
    tasks,
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
            result = run_task_sequence(tasks, data)

            batch_output_name = f"{output_name}_batch_{batch_num}"
            save_final_output(loader, result, output_folder, connector, batch_output_name)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_pipeline():
    source_folder = require_env("SOURCE_FOLDER_ID")
    output_folder = require_env("OUTPUT_FOLDER_ID")
    ts = os.environ.get("TIMESTAMP", "000000")
    batch_size = int(os.environ.get("BATCH_SIZE", 5))

    task_paths = parse_task_paths()
    tasks = load_pipeline(task_paths)

    connector = GDriveConnector()

    pipeline_mode = tasks[0]["config"]["mode"]
    data_type = tasks[0]["config"]["data_type"]

    # For in-memory chaining, the initial raw load should satisfy the FIRST task.
    # Downstream tasks are expected to consume the previous task's output.
    first_requirements = tasks[0]["module"].get_requirements()
    target_cols = first_requirements.get("columns", [])

    pipeline_name = os.environ.get("PIPELINE_NAME", "").strip()
    if pipeline_name:
        base_name = pipeline_name
    elif len(tasks) == 1:
        base_name = tasks[0]["config"]["name"]
    else:
        base_name = "__".join([t["config"]["name"] for t in tasks])

    output_name = f"{base_name}_{ts}"

    print("=== ENGINE CONFIG ===")
    print(f"Task paths: {task_paths}")
    print(f"Pipeline mode: {pipeline_mode}")
    print(f"Data type: {data_type}")
    print(f"Source folder: {source_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Output name base: {output_name}")

    if pipeline_mode == "global":
        loader = get_loader(data_type, "temp_all")
        run_global_pipeline(
            connector=connector,
            loader=loader,
            tasks=tasks,
            source_folder=source_folder,
            output_folder=output_folder,
            output_name=f"{output_name}_full",
            target_cols=target_cols,
        )
    elif pipeline_mode == "batch":
        run_batched_pipeline(
            connector=connector,
            tasks=tasks,
            source_folder=source_folder,
            output_folder=output_folder,
            output_name=output_name,
            batch_size=batch_size,
            data_type=data_type,
            target_cols=target_cols,
        )
    else:
        raise ValueError(f"Unsupported mode: {pipeline_mode}")


if __name__ == "__main__":
    run_pipeline()