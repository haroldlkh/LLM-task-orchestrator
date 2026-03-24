from data_connectors import get_connector
from execution import run_batched_execution, run_global_execution
from module_loader import load_pipeline, load_single_task
from runtime_config import parse_entry_inputs, parse_json_env, require_env


def run_pipeline():
    source_connector_name = require_env("SOURCE_CONNECTOR")
    source_location = require_env("SOURCE_LOCATION")
    dest_connector_name = require_env("DEST_CONNECTOR")
    dest_location = require_env("DEST_LOCATION")

    source_connector_config = parse_json_env("SOURCE_CONNECTOR_CONFIG_JSON", allow_empty=True)
    dest_connector_config = parse_json_env("DEST_CONNECTOR_CONFIG_JSON", allow_empty=True)

    ts = require_env("TIMESTAMP")
    batch_size = int(require_env("BATCH_SIZE"))

    entry = parse_entry_inputs()

    if entry["entry_type"] == "task":
        executable = load_single_task(entry["path"])
    else:
        executable = load_pipeline(entry["path"])

    source_connector = get_connector(source_connector_name, source_connector_config)
    dest_connector = get_connector(dest_connector_name, dest_connector_config)

    mode = executable["mode"]
    data_type = executable["data_type"]
    target_cols = executable["requirements"].get("columns", [])

    import os
    pipeline_name = os.environ.get("PIPELINE_NAME", "").strip()
    base_name = pipeline_name if pipeline_name else executable["name"]
    output_name = f"{base_name}_{ts}"

    print("=== ENGINE CONFIG ===")
    print(f"Entry type: {entry['entry_type']}")
    print(f"Entry path: {entry['path']}")
    print(f"Mode: {mode}")
    print(f"Data type: {data_type}")
    print(f"Source connector: {source_connector_name}")
    print(f"Source location: {source_location}")
    print(f"Dest connector: {dest_connector_name}")
    print(f"Dest location: {dest_location}")
    print(f"Output name base: {output_name}")

    if mode == "global":
        run_global_execution(
            source_connector=source_connector,
            dest_connector=dest_connector,
            executable=executable,
            source_location=source_location,
            dest_location=dest_location,
            output_name=f"{output_name}_full",
            data_type=data_type,
            target_cols=target_cols,
        )
    elif mode == "batch":
        run_batched_execution(
            source_connector=source_connector,
            dest_connector=dest_connector,
            executable=executable,
            source_location=source_location,
            dest_location=dest_location,
            output_name=output_name,
            batch_size=batch_size,
            data_type=data_type,
            target_cols=target_cols,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    run_pipeline()