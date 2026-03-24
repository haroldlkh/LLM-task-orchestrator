import os
import sys
import shutil
import importlib.util
from data_connectors import GDriveConnector
from factory_loader import get_loader


def load_user_task(task_path):
    repo_root = os.path.dirname(os.path.dirname(task_path))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location("user_task", task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task script: {task_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_task_module(user_module, task_script):
    if not hasattr(user_module, "TASK_CONFIG"):
        raise AttributeError(
            f"{task_script} must define TASK_CONFIG"
        )

    if not hasattr(user_module, "get_requirements"):
        raise AttributeError(
            f"{task_script} must define get_requirements()"
        )

    if not hasattr(user_module, "run"):
        raise AttributeError(
            f"{task_script} must define run(data)"
        )

    task_config = user_module.TASK_CONFIG

    if "name" not in task_config:
        raise KeyError(f"{task_script} TASK_CONFIG must include 'name'")

    if "mode" not in task_config:
        raise KeyError(f"{task_script} TASK_CONFIG must include 'mode'")

    if "data_type" not in task_config:
        raise KeyError(f"{task_script} TASK_CONFIG must include 'data_type'")

    if task_config["mode"] not in {"batch", "global"}:
        raise ValueError(
            f"{task_script} TASK_CONFIG['mode'] must be 'batch' or 'global'"
        )

    return task_config


def run_global_task(
    connector,
    loader,
    user_module,
    source_folder,
    output_folder,
    task_name,
    ts,
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
        processed_data = user_module.run(data)

        if processed_data is not None:
            base_name = f"{task_name}_{ts}_full"
            final_filename = loader.save(processed_data, base_name)

            if final_filename and os.path.exists(final_filename):
                remote_name = os.path.basename(final_filename)
                file_id = connector.upload_file(
                    final_filename,
                    output_folder,
                    remote_name,
                )
                print(f"Uploaded: {remote_name} (file_id={file_id})")
                os.remove(final_filename)
            else:
                print("No output file was created.")
        else:
            print("Task returned None. Nothing to upload.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batched_task(
    connector,
    user_module,
    source_folder,
    output_folder,
    task_name,
    ts,
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
            processed_data = user_module.run(data)

            if processed_data is not None:
                base_name = f"{task_name}_{ts}_batch_{batch_num}"
                final_filename = loader.save(processed_data, base_name)

                if final_filename and os.path.exists(final_filename):
                    remote_name = os.path.basename(final_filename)
                    file_id = connector.upload_file(
                        final_filename,
                        output_folder,
                        remote_name,
                    )
                    print(f"Uploaded: {remote_name} (file_id={file_id})")
                    os.remove(final_filename)
                else:
                    print("No output file was created.")
            else:
                print("Task returned None. Nothing to upload.")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_pipeline():
    source_folder = os.environ["SOURCE_FOLDER_ID"]
    output_folder = os.environ["OUTPUT_FOLDER_ID"]
    task_script = os.environ["TASK_SCRIPT"]
    ts = os.environ.get("TIMESTAMP", "000000")
    batch_size = int(os.environ.get("BATCH_SIZE", 5))

    connector = GDriveConnector()
    user_module = load_user_task(task_script)
    task_config = validate_task_module(user_module, task_script)

    task_name = task_config["name"]
    task_mode = task_config["mode"]
    data_type = task_config["data_type"]

    requirements = user_module.get_requirements()
    target_cols = requirements.get("columns", [])

    print(f"Running task: {task_name}")
    print(f"Task mode: {task_mode}")
    print(f"Data type: {data_type}")

    if task_mode == "global":
        loader = get_loader(data_type, "temp_all")
        run_global_task(
            connector=connector,
            loader=loader,
            user_module=user_module,
            source_folder=source_folder,
            output_folder=output_folder,
            task_name=task_name,
            ts=ts,
            target_cols=target_cols,
        )
    elif task_mode == "batch":
        run_batched_task(
            connector=connector,
            user_module=user_module,
            source_folder=source_folder,
            output_folder=output_folder,
            task_name=task_name,
            ts=ts,
            batch_size=batch_size,
            data_type=data_type,
            target_cols=target_cols,
        )
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")


if __name__ == "__main__":
    run_pipeline()