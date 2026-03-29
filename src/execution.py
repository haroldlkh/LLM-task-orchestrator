import os
import shutil
from typing import Any, Dict, List

from factory_loader import get_loader
from llm.executor import execute_llm_step


def run_pipeline_steps(
    steps: List[Dict[str, Any]],
    data,
    runtime_context: Dict[str, Any],
):
    current_data = data

    for i, step in enumerate(steps, start=1):
        if step["kind"] == "llm":
            step_name = step["config"]["name"]
            print(f"Running llm pipeline step {i}/{len(steps)}: {step_name}")
            current_data = execute_llm_step(
                data=current_data,
                step_config=step["config"],
                runtime_context=runtime_context,
            )
        else:
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


def save_final_output(loader, data, dest_location, dest_connector, output_name):
    final_filename = loader.save(data, output_name)

    if final_filename and os.path.exists(final_filename):
        remote_name = os.path.basename(final_filename)
        object_id = dest_connector.upload_object(final_filename, dest_location, remote_name)
        print(f"Uploaded: {remote_name} (object_id={object_id})")
        os.remove(final_filename)
    else:
        raise RuntimeError("No output file was created.")


def run_global_execution(
    source_connector,
    dest_connector,
    executable,
    source_location,
    dest_location,
    output_name,
    data_type,
    target_cols,
    user_runtime_config,
):
    temp_dir = "temp_all"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        all_files = [
            f for f in source_connector.list_objects(source_location)
            if f["name"].endswith(".parquet")
        ]

        print(f"Discovered {len(all_files)} parquet file(s) in source location.")

        for f_info in all_files:
            local_path = os.path.join(temp_dir, f_info["name"])
            print(f"Downloading {f_info['name']}...")
            source_connector.download_object(f_info["id"], local_path)

        loader = get_loader(data_type, temp_dir)
        data = loader.load(target_cols)

        if executable["kind"] == "task":
            result = run_single_task_runner(executable["runner"], data)
        else:
            runtime_context = {
                "dest_connector": dest_connector,
                "dest_location": dest_location,
                "pipeline_name": executable["name"],
                "temp_dir": temp_dir,
                "user_runtime_config": user_runtime_config,
            }
            result = run_pipeline_steps(executable["steps"], data, runtime_context)

        save_final_output(loader, result, dest_location, dest_connector, output_name)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_batched_execution(
    source_connector,
    dest_connector,
    executable,
    source_location,
    dest_location,
    output_name,
    batch_size,
    data_type,
    target_cols,
    user_runtime_config,
):
    all_files = [
        f for f in source_connector.list_objects(source_location)
        if f["name"].endswith(".parquet")
    ]

    print(f"Discovered {len(all_files)} parquet file(s) in source location.")

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
                source_connector.download_object(f_info["id"], local_path)

            loader = get_loader(data_type, temp_dir)
            data = loader.load(target_cols)

            if executable["kind"] == "task":
                result = run_single_task_runner(executable["runner"], data)
            else:
                runtime_context = {
                    "dest_connector": dest_connector,
                    "dest_location": dest_location,
                    "pipeline_name": executable["name"],
                    "temp_dir": temp_dir,
                    "user_runtime_config": user_runtime_config,
                }
                result = run_pipeline_steps(executable["steps"], data, runtime_context)

            batch_output_name = f"{output_name}_batch_{batch_num}"
            save_final_output(loader, result, dest_location, dest_connector, batch_output_name)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)