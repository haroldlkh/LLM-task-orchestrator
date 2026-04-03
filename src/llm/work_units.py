import hashlib
from typing import List

import polars as pl

from .models import LLMWorkUnit


WORK_UNIT_SCHEMA = {
    "unit_id": pl.Utf8,
    "row_id": pl.Utf8,
    "field_name": pl.Utf8,
    "input_text": pl.Utf8,
    "output_column": pl.Utf8,
    "model": pl.Utf8,
    "step_name": pl.Utf8,
    "prompt_version": pl.Utf8,
}

MANIFEST_SCHEMA = {
    "unit_id": pl.Utf8,
    "row_id": pl.Utf8,
    "field_name": pl.Utf8,
    "output_column": pl.Utf8,
    "model": pl.Utf8,
    "step_name": pl.Utf8,
    "prompt_version": pl.Utf8,
    "input_hash": pl.Utf8,
}

MANIFEST_COLUMNS = list(MANIFEST_SCHEMA.keys())


def hash_unit_id(parts: List[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def hash_input_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_work_units(data: pl.DataFrame, step_config: dict) -> pl.DataFrame:
    """Build the in-memory runtime work-unit table.

    This table intentionally includes ``input_text`` because task handlers build
    request payloads from it during the current run. It should not be uploaded
    as the manifest artifact because that would duplicate full payload text once
    per unit.
    """
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
        return pl.DataFrame(schema=WORK_UNIT_SCHEMA)

    return pl.DataFrame(units)


def build_manifest_df(work_units_df: pl.DataFrame) -> pl.DataFrame:
    """Return the persisted manifest artifact.

    The manifest is a thin work index. It must not persist full input payloads.
    Payload text is sourced from the input dataframe at runtime when requests are
    built, not from the uploaded manifest artifact.
    """
    if work_units_df is None or work_units_df.is_empty():
        return pl.DataFrame(schema=MANIFEST_SCHEMA)

    required_cols = [
        "unit_id",
        "row_id",
        "field_name",
        "output_column",
        "model",
        "step_name",
        "prompt_version",
    ]
    missing = [col for col in required_cols + ["input_text"] if col not in work_units_df.columns]
    if missing:
        raise KeyError(f"Cannot build manifest; missing work-unit columns: {missing}")

    manifest_df = (
        work_units_df
        .select(required_cols + ["input_text"])
        .with_columns(pl.col("input_text").map_elements(hash_input_text, return_dtype=pl.Utf8).alias("input_hash"))
        .drop("input_text")
        .select(MANIFEST_COLUMNS)
    )
    return manifest_df
