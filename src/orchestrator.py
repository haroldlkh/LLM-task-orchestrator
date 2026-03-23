import os
import json
import shutil
import importlib.util
from data_connectors import GDriveConnector
from factory_loader import get_loader

def load_user_task(task_path):
    spec = importlib.util.spec_from_file_location("user_task", task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_pipeline():
    # 1. SETUP
    creds = json.loads(os.environ['GDRIVE_KEY'])
    source_folder = os.environ['SOURCE_FOLDER_ID']
    output_folder = os.environ['OUTPUT_FOLDER_ID']
    
    # Metadata for naming
    wf_name = os.environ.get("WORKFLOW_NAME", "task").replace(" ", "_")
    ts = os.environ.get("TIMESTAMP", "000000")
    
    batch_size = int(os.environ.get('BATCH_SIZE', 5))
    task_script = os.environ.get("TASK_SCRIPT")
    data_type = os.environ.get("DATA_TYPE", "tabular")
    
    # 2. INITIALIZE
    connector = GDriveConnector(creds)
    user_module = load_user_task(task_script)
    requirements = user_module.get_requirements()
    target_cols = requirements.get('columns', [])

    # 3. DISCOVER
    all_files = [f for f in connector.list_files_in_folder(source_folder) if f['name'].endswith('.parquet')]

    # 4. LOOP
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        temp_dir = "temp_batch"
        os.makedirs(temp_dir, exist_ok=True)
        
        for f_info in batch:
            connector.download_file(f_info['id'], os.path.join(temp_dir, f_info['name']))
        
        loader = get_loader(data_type, temp_dir)
        data = loader.load(target_cols)
        processed_data = user_module.run(data)
        
        # 5. DYNAMIC NAMING & SAVING
        if processed_data is not None:
            # Create a unique base name for this batch
            base_name = f"{wf_name}_{ts}_batch_{i//batch_size + 1}"
            
            # THE CHANGE: Loader decides the filename and handles the save
            final_filename = loader.save(processed_data, base_name)
            
            if final_filename and os.path.exists(final_filename):
                connector.upload_file(final_filename, output_folder, final_filename)
                print(f"Uploaded: {final_filename}")
                os.remove(final_filename)

        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_pipeline()