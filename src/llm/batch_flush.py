from __future__ import annotations

from functools import reduce

import polars as pl

from .merge import merge_results_back


def released_row_ids_for_flush(
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    flush_scope: str,
) -> list[str]:
    if progress_df.is_empty():
        return []

    joined = work_units_df.select(["unit_id", "row_id"]).join(
        progress_df.select(["unit_id", "status"]),
        on="unit_id",
        how="inner",
    )

    if joined.is_empty():
        return []

    if flush_scope == "unit":
        released = joined.filter(pl.col("status") == "success")
        return sorted(set(released["row_id"].to_list()))

    expected_per_row = (
        work_units_df.group_by("row_id")
        .agg(pl.len().alias("expected_unit_count"))
    )

    success_per_row = (
        joined.filter(pl.col("status") == "success")
        .group_by("row_id")
        .agg(pl.len().alias("success_unit_count"))
    )

    complete = expected_per_row.join(success_per_row, on="row_id", how="left").with_columns(
        pl.col("success_unit_count").fill_null(0)
    )

    released = complete.filter(pl.col("expected_unit_count") == pl.col("success_unit_count"))
    return sorted(set(released["row_id"].to_list()))


def released_results_df_for_row_ids(
    all_results_df: pl.DataFrame,
    released_row_ids: list[str],
) -> pl.DataFrame:
    if all_results_df.is_empty() or not released_row_ids:
        return all_results_df.head(0)
    return all_results_df.filter(pl.col("row_id").is_in(released_row_ids))


def build_partial_output_df(
    source_df: pl.DataFrame,
    all_results_df: pl.DataFrame,
    row_id_column: str,
    task_handler,
    step_config: dict,
    released_row_ids: list[str],
) -> pl.DataFrame:
    if not released_row_ids:
        return source_df.head(0)

    subset_source = source_df.filter(pl.col(row_id_column).cast(pl.Utf8).is_in(released_row_ids))
    if subset_source.is_empty():
        return subset_source

    subset_results = released_results_df_for_row_ids(all_results_df, released_row_ids)
    return merge_results_back(
        source_df=subset_source,
        all_results_df=subset_results,
        row_id_column=row_id_column,
        task_handler=task_handler,
        step_config=step_config,
    )


def build_pair_status_df(
    work_units_df: pl.DataFrame,
    progress_df: pl.DataFrame,
    step_config: dict,
) -> pl.DataFrame:
    if progress_df.is_empty():
        return pl.DataFrame({"row_id": []}, schema={"row_id": pl.Utf8})

    fields = list(step_config["input_columns"])

    touched = work_units_df.select(["unit_id", "row_id", "field_name"]).join(
        progress_df.select(
            [
                "unit_id",
                "status",
                "retry_count",
                "last_error_type",
                "last_error_message",
            ]
        ),
        on="unit_id",
        how="inner",
    )

    if touched.is_empty():
        return pl.DataFrame({"row_id": []}, schema={"row_id": pl.Utf8})

    expected = work_units_df.group_by("row_id").agg(pl.len().alias("expected_unit_count"))
    observed = touched.group_by("row_id").agg(pl.len().alias("observed_unit_count"))
    success = touched.filter(pl.col("status") == "success").group_by("row_id").agg(
        pl.len().alias("success_unit_count")
    )
    retryable = touched.filter(pl.col("status") == "retryable_error").group_by("row_id").agg(
        pl.len().alias("retryable_error_count")
    )
    permanent = touched.filter(pl.col("status") == "permanent_error").group_by("row_id").agg(
        pl.len().alias("permanent_error_count")
    )

    base = expected.join(observed, on="row_id", how="inner")
    for extra in (success, retryable, permanent):
        base = base.join(extra, on="row_id", how="left")

    base = base.with_columns(
        [
            pl.col("success_unit_count").fill_null(0),
            pl.col("retryable_error_count").fill_null(0),
            pl.col("permanent_error_count").fill_null(0),
        ]
    )

    field_frames = []
    for field_name in fields:
        suffix = field_name
        field_df = touched.filter(pl.col("field_name") == field_name).rename(
            {
                "unit_id": f"{suffix}__unit_id",
                "status": f"{suffix}__status",
                "retry_count": f"{suffix}__retry_count",
                "last_error_type": f"{suffix}__last_error_type",
                "last_error_message": f"{suffix}__last_error_message",
            }
        ).select(
            [
                "row_id",
                f"{suffix}__unit_id",
                f"{suffix}__status",
                f"{suffix}__retry_count",
                f"{suffix}__last_error_type",
                f"{suffix}__last_error_message",
            ]
        )
        field_frames.append(field_df)

    if field_frames:
        base = reduce(lambda left, right: left.join(right, on="row_id", how="left"), [base] + field_frames)

    blocked_exprs = []
    for field_name in fields:
        status_col = f"{field_name}__status"
        blocked_exprs.append(
            pl.when(
                pl.col(status_col).is_null() | (pl.col(status_col) != "success")
            )
            .then(pl.lit(field_name))
            .otherwise(pl.lit(None))
            .alias(f"_blocked_{field_name}")
        )

    base = base.with_columns(blocked_exprs)
    blocked_cols = [f"_blocked_{field_name}" for field_name in fields]

    base = base.with_columns(
        [
            pl.concat_list([pl.col(col_name) for col_name in blocked_cols]).alias("_blocked_fields_list"),
            (pl.col("success_unit_count") == pl.col("expected_unit_count")).alias("row_success_complete"),
            (
                (pl.col("success_unit_count") + pl.col("permanent_error_count"))
                == pl.col("expected_unit_count")
            ).alias("row_terminal_complete"),
        ]
    )

    base = base.with_columns(
        pl.col("_blocked_fields_list")
        .list.eval(pl.element().drop_nulls())
        .list.join(",")
        .alias("blocked_fields")
    )

    drop_cols = blocked_cols + ["_blocked_fields_list"]
    return base.drop(drop_cols).sort("row_id")
