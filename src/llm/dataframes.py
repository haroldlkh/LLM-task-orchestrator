import json
from typing import Any, Dict, List

import polars as pl


PROGRESS_SCHEMA: Dict[str, pl.DataType] = {
    "unit_id": pl.Utf8,
    "status": pl.Utf8,
    "retry_count": pl.Int64,
    "last_error_type": pl.Utf8,
    "last_error_message": pl.Utf8,
    "updated_at": pl.Utf8,
    "key_alias": pl.Utf8,
    "provider": pl.Utf8,
    "model": pl.Utf8,
}

RESULT_SCHEMA: Dict[str, pl.DataType] = {
    "unit_id": pl.Utf8,
    "row_id": pl.Utf8,
    "output_column": pl.Utf8,
    "status": pl.Utf8,
    "parsed_output": pl.Utf8,
    "output_value": pl.Utf8,
    "raw_output": pl.Utf8,
    "error_type": pl.Utf8,
    "error_message": pl.Utf8,
    "review_flag": pl.Boolean,
    "review_reason": pl.Utf8,
    "key_alias": pl.Utf8,
    "provider": pl.Utf8,
    "model": pl.Utf8,
}

DEBUG_SCHEMA: Dict[str, pl.DataType] = {
    "unit_id": pl.Utf8,
    "row_id": pl.Utf8,
    "field_name": pl.Utf8,
    "input_text": pl.Utf8,
    "output_column": pl.Utf8,
    "request_group_id": pl.Utf8,
    "request_group_size": pl.Int64,
    "request_attempt_count": pl.Int64,
    "status": pl.Utf8,
    "review_flag": pl.Boolean,
    "review_reason": pl.Utf8,
    "error_type": pl.Utf8,
    "error_message": pl.Utf8,
    "rendered_prompt": pl.Utf8,
    "raw_output": pl.Utf8,
    "parsed_output": pl.Utf8,
    "output_value": pl.Utf8,
    "updated_at": pl.Utf8,
    "key_alias": pl.Utf8,
    "provider": pl.Utf8,
    "model": pl.Utf8,
}


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_value(dtype: pl.DataType, value: Any):
    if value is None:
        return None

    if dtype == pl.Utf8:
        return _stringify(value)

    if dtype == pl.Int64:
        if isinstance(value, bool):
            return int(value)
        return int(value)

    if dtype == pl.Boolean:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    return value


def _normalize_rows(rows: List[dict], schema: Dict[str, pl.DataType]) -> List[dict]:
    normalized = []
    for row in rows:
        normalized_row = {}
        for col_name, dtype in schema.items():
            normalized_row[col_name] = _coerce_value(dtype, row.get(col_name))
        normalized.append(normalized_row)
    return normalized


def _empty_df(schema: Dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            col_name: pl.Series(name=col_name, values=[], dtype=dtype)
            for col_name, dtype in schema.items()
        }
    )


def rows_to_typed_df(rows: List[dict], schema: Dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return _empty_df(schema)

    normalized_rows = _normalize_rows(rows, schema)
    columns = {
        col_name: pl.Series(
            name=col_name,
            values=[row[col_name] for row in normalized_rows],
            dtype=dtype,
        )
        for col_name, dtype in schema.items()
    }
    return pl.DataFrame(columns)


def ensure_df_schema(df: pl.DataFrame, schema: Dict[str, pl.DataType]) -> pl.DataFrame:
    if df is None or df.is_empty():
        return _empty_df(schema)

    working = df

    for col_name, dtype in schema.items():
        if col_name not in working.columns:
            working = working.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    working = working.select(list(schema.keys()))

    casts = []
    for col_name, dtype in schema.items():
        casts.append(pl.col(col_name).cast(dtype, strict=False).alias(col_name))

    return working.with_columns(casts)


def progress_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    return rows_to_typed_df(rows, PROGRESS_SCHEMA)


def result_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    return rows_to_typed_df(rows, RESULT_SCHEMA)


def debug_rows_to_df(rows: List[dict]) -> pl.DataFrame:
    return rows_to_typed_df(rows, DEBUG_SCHEMA)


def ensure_progress_df(df: pl.DataFrame) -> pl.DataFrame:
    return ensure_df_schema(df, PROGRESS_SCHEMA)


def ensure_result_df(df: pl.DataFrame) -> pl.DataFrame:
    return ensure_df_schema(df, RESULT_SCHEMA)


def ensure_debug_df(df: pl.DataFrame) -> pl.DataFrame:
    return ensure_df_schema(df, DEBUG_SCHEMA)


def empty_progress_df() -> pl.DataFrame:
    return _empty_df(PROGRESS_SCHEMA)


def empty_result_df() -> pl.DataFrame:
    return _empty_df(RESULT_SCHEMA)


def empty_debug_df() -> pl.DataFrame:
    return _empty_df(DEBUG_SCHEMA)
