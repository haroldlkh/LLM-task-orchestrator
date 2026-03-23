import os
import json
import shutil
import polars as pl
from data_connectors import GDriveConnector

def run_pipeline():
    # 1. Environment Setup (Passed from User Repo)
    creds = json.loads(os.environ['GDRIVE_KEY'])
    source_folder = os.environ['SOURCE_FOLDER_ID']
    output_folder = os.environ['GDRIVE_FOLDER_ID']
    
    # Default batch size is 5, but can be overridden by the user
    batch_size = int(os.environ.get('BATCH_SIZE', 5))
    
    connector = GDriveConnector(creds)
    
    # 2. Get the list of chunks
    all_files = connector.list_files_in_folder(source_folder)
    parquet_files = [f for f in all_files if f['name'].endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet chunks.")

    # 3. Sliding Window Loop
    for i in range(0, len(parquet_files), batch_size):
        batch = parquet_files[i : i + batch_size]
        temp_dir = "temp_batch_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download Batch
        print(f"--- Processing Batch {i//batch_size + 1} ---")
        for f_info in batch:
            local_path = os.path.join(temp_dir, f_info['name'])
            connector.download_file(f_info['id'], local_path)
        
        # 4. Lazy Load the whole batch at once
        # This is where the magic happens: Polars treats the folder as one table
        lf = pl.scan_parquet(f"{temp_dir}/*.parquet")
        
        # TEST: Just counting rows to prove connection
        df_result = lf.select(pl.len()).collect()
        total_rows = df_result.item()
        
        # 5. Upload a "Batch Receipt"
        receipt_name = f"receipt_batch_{i//batch_size + 1}.txt"
        with open(receipt_name, "w") as r:
            r.write(f"Processed {len(batch)} files. Total rows in this batch: {total_rows}")
        
        connector.upload_file(receipt_name, output_folder, receipt_name)
        
        # 6. Cleanup local disk for next batch
        shutil.rmtree(temp_dir)
        os.remove(receipt_name)

if __name__ == "__main__":
    run_pipeline()