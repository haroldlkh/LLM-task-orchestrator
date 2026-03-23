import os
import shutil
import importlib.util
from data_connectors import GDriveConnector
from factory_loader import get_loader


def load_user_task(task_path):
    spec = importlib.util.spec_from_file_location("user_task", task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task script: {task_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline():
    # 1. SETUP
    source_folder = os.environ["SOURCE_FOLDER_ID"]
    output_folder = os.environ["OUTPUT_FOLDER_ID"]

    wf_name = os.environ.get("WORKFLOW_NAME", "task").replace(" ", "_")
    ts = os.environ.get("TIMESTAMP", "000000")

    batch_size = int(os.environ.get("BATCH_SIZE", 5))
    task_script = os.environ["TASK_SCRIPT"]
    data_type = os.environ.get("DATA_TYPE", "tabular")

    # 2. INITIALIZE
    connector = GDriveConnector()
    user_module = load_user_task(task_script)

    if not hasattr(user_module, "get_requirements"):
        raise AttributeError(
            f"{task_script} must define get_requirements()"
        )

    if not hasattr(user_module, "run"):
        raise AttributeError(
            f"{task_script} must define run(data)"
        )

    requirements = user_module.get_requirements()
    target_cols = requirements.get("columns", [])

    # 3. DISCOVER
    all_files = [
        f for f in connector.list_files_in_folder(source_folder)
        if f["name"].endswith(".parquet")
    ]

    print(f"Discovered {len(all_files)} parquet file(s) in source folder.")

    # 4. LOOP
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
                base_name = f"{wf_name}_{ts}_batch_{batch_num}"
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


if __name__ == "__main__":
    run_pipeline()