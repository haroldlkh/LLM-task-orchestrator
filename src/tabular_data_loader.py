import polars as pl

class TabularDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def stream_data(self, target_columns):
        return pl.scan_parquet(self.file_path).select(target_columns).collect().to_dicts()