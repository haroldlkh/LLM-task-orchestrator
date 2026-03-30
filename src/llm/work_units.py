import hashlib
from typing import List

import polars as pl

from .models import LLMWorkUnit


def hash_unit_id(parts: List[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_work_units(data: pl.DataFrame, step_config: dict) -> pl.DataFrame:
    row_id_column = step_config["row_id_column"]
    model = step_config["model"]
    step_name = step_config["name"]
    prompt_version = step_config.get("kwargs", {}).get("prompt_version", "v1")

    units = []

    for row in data.iter_rows(named=True):
        row_id = str(row[row_id_column])

        for input_col in step_config["input_columns"]:
            input_text = row.get(input_col)
            if input_text is None:
                continue

            input_text = str(input_text).strip()
            if not input_text:
                continue

            output_column = step_config["output_columns"][input_col]
            unit_id = hash_unit_id([
                step_name,
                row_id,
                input_col,
                output_column,
                model,
                prompt_version,
            ])

            unit = LLMWorkUnit(
                unit_id=unit_id,
                row_id=row_id,
                field_name=input_col,
                input_text=input_text,
                output_column=output_column,
                model=model,
                step_name=step_name,
                prompt_version=prompt_version,
                step_kwargs=step_config.get("kwargs", {}),
            )
            units.append(unit.to_dict())

    if not units:
        return pl.DataFrame(
            schema={
                "unit_id": pl.Utf8,
                "row_id": pl.Utf8,
                "field_name": pl.Utf8,
                "input_text": pl.Utf8,
                "output_column": pl.Utf8,
                "model": pl.Utf8,
                "step_name": pl.Utf8,
                "prompt_version": pl.Utf8,
            }
        )

    return pl.DataFrame(units)