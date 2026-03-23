import os
import polars as pl


class TabularDataLoader:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    def load(self, target_columns):
        search_path = os.path.join(self.temp_dir, "*.parquet")
        lf = pl.scan_parquet(search_path)

        if target_columns:
            lf = lf.select(target_columns)

        return lf

    def save(self, data, base_name):
        filename = f"{base_name}.parquet"

        if isinstance(data, pl.LazyFrame):
            data.collect(streaming=True).write_parquet(filename)
            return filename

        if isinstance(data, pl.DataFrame):
            data.write_parquet(filename)
            return filename

        raise TypeError(
            f"Unsupported data type for save(): {type(data)}. "
            f"Expected pl.LazyFrame or pl.DataFrame."
        )