import polars as pl


def cast_output_columns_from_task(df: pl.DataFrame, task_handler, step_config: dict) -> pl.DataFrame:
    if not hasattr(task_handler, "get_output_dtypes"):
        return df

    output_dtypes = task_handler.get_output_dtypes(step_config) or {}
    if not output_dtypes:
        return df

    casts = []
    for col_name, dtype_name in output_dtypes.items():
        if col_name not in df.columns:
            continue

        if dtype_name == "Int64":
            casts.append(pl.col(col_name).cast(pl.Int64))
        elif dtype_name == "Int32":
            casts.append(pl.col(col_name).cast(pl.Int32))
        elif dtype_name == "Float64":
            casts.append(pl.col(col_name).cast(pl.Float64))
        elif dtype_name == "Utf8":
            casts.append(pl.col(col_name).cast(pl.Utf8))
        else:
            raise ValueError(
                f"Unsupported output dtype '{dtype_name}' for column '{col_name}'"
            )

    if not casts:
        return df

    return df.with_columns(casts)


def merge_results_back(
    source_df: pl.DataFrame,
    all_results_df: pl.DataFrame,
    row_id_column: str,
    task_handler,
    step_config: dict,
) -> pl.DataFrame:
    if all_results_df.is_empty():
        return cast_output_columns_from_task(source_df, task_handler, step_config)

    success_df = all_results_df.filter(pl.col("status") == "success")
    if success_df.is_empty():
        return cast_output_columns_from_task(source_df, task_handler, step_config)

    wide = (
        success_df
        .select(["row_id", "output_column", "output_value"])
        .pivot(
            index="row_id",
            on="output_column",
            values="output_value",
            aggregate_function="first",
        )
        .rename({"row_id": row_id_column})
    )

    merged = source_df.with_columns(pl.col(row_id_column).cast(pl.Utf8)).join(
        wide.with_columns(pl.col(row_id_column).cast(pl.Utf8)),
        on=row_id_column,
        how="left",
    )

    return cast_output_columns_from_task(merged, task_handler, step_config)