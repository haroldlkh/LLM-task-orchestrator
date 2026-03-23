import os
import json
import shutil
import importlib.util
from data_connectors import GDriveConnector
from loader_factory import get_loader

def load_user_task(task_path):
    """Dynamically imports the user's Python script from their repo."""
    spec = importlib.util.spec_from_file_location("user_task", task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_pipeline():
    # 1. SETUP (Constants from Environment)
    creds = json.loads(os.environ['GDRIVE_KEY'])
    source_folder = os.environ['SOURCE_FOLDER_ID']
    output_folder = os.environ['GDRIVE_FOLDER_ID']
    
    batch_size = int(os.environ.get('BATCH_SIZE', 5))
    task_script = os.environ.get("TASK_SCRIPT", "tasks/process.py")
    data_type = os.environ.get("DATA_TYPE", "tabular")
    
    # 2. INITIALIZE ENGINES
    connector = GDriveConnector(creds)
    user_module = load_user_task(task_script)
    
    # 3. GET THE "CONTRACT"
    # Ask the user task: What do you need from the data?
    requirements = user_module.get_requirements()
    target_cols = requirements.get('columns', [])

    # 4. DISCOVER FILES IN GDRIVE
    all_files = [
        f for f in connector.list_files_in_folder(source_folder) 
        if f['name'].endswith('.parquet') # Note: Discovery is still based on source file extension
    ]
    print(f"Found {len(all_files)} files. Starting processing in batches of {batch_size}...")

    # 5. THE BATCHED LOOP
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        temp_dir = "temp_batch"
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"--- Processing Batch {i//batch_size + 1} ({len(batch)} files) ---")
        
        # Download the current batch
        for f_info in batch:
            dest = os.path.join(temp_dir, f_info['name'])
            connector.download_file(f_info['id'], dest)
        
        # --- THE ABSTRACTION LAYER ---
        # Orchestrator asks for a loader, then tells it to load.
        loader = get_loader(data_type, temp_dir)
        data = loader.load(target_cols)
        
        # 6. HAND OFF TO USER TASK
        processed_data = user_module.run(data)
        
        # 7. SAVE & UPLOAD (Agnostic Version)
        if processed_data is not None:
            # We define the filename; the Loader handles the writing logic.
            # In a tabular world, this saves as a .parquet file.
            file_ext = "parquet" if data_type == "tabular" else "dat"
            out_name = f"results_batch_{i//batch_size + 1}.{file_ext}"
            
            # The Loader handles the disk-writing (Specific to the data type)
            save_success = loader.save(processed_data, out_name)
            
            if save_success:
                # The Connector handles the upload (It just sees 'bytes')
                connector.upload_file(out_name, output_folder, out_name)
                print(f"Successfully uploaded {out_name}")
                os.remove(out_name)

        # 8. CLEANUP
        shutil.rmtree(temp_dir)

    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()