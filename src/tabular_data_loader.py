import polars as pl
import os

class TabularDataLoader:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    def load(self, target_columns):
        search_path = os.path.join(self.temp_dir, "*.parquet")
        return pl.scan_parquet(search_path).select(target_columns)

    def save(self, data, filename):
        """
        The Orchestrator calls this. 
        Only the Loader knows that 'Tabular' data saves as Parquet.
        """
        if isinstance(data, pl.DataFrame):
            data.write_parquet(filename)
            return True
        return False