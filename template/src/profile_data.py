from __future__ import annotations

from pathlib import Path

import polars as pl


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    require_file(train_path)
    require_file(test_path)
    require_file(sample_submission_path)

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    sample_submission = pl.read_csv(sample_submission_path)

    id_column = infer_id_column(train_df, test_df, sample_submission)
    target_column = infer_target_column(train_df, test_df, sample_submission)
    numeric_features = infer_numeric_features(train_df, target_column, id_column)
    categorical_features = infer_categorical_features(train_df, target_column)

    report_lines = [
        "# Data Profile",
        "",
        "## Files",
        "",
        f"- `train.csv`: {train_df.height} rows x {train_df.width} columns",
        f"- `test.csv`: {test_df.height} rows x {test_df.width} columns",
        f"- `sample_submission.csv`: {sample_submission.height} rows x {sample_submission.width} columns",
        "",
        "## Inferred Roles",
        "",
        f"- id column: `{id_column}`" if id_column else "- id column: could not infer",
        f"- target column: `{target_column}`" if target_column else "- target column: could not infer",
        f"- numeric features: {len(numeric_features)}",
        f"- categorical features: {len(categorical_features)}",
        "",
        "## Column Checks",
        "",
    ]

    train_only = sorted(set(train_df.columns) - set(test_df.columns))
    test_only = sorted(set(test_df.columns) - set(train_df.columns))
    report_lines.append(f"- columns only in train: {format_list(train_only)}")
    report_lines.append(f"- columns only in test: {format_list(test_only)}")
    report_lines.append("")

    report_lines.extend(
        [
            "## Data Quality",
            "",
            f"- duplicate rows in train: {int(train_df.is_duplicated().sum())}",
            f"- duplicate rows in test: {int(test_df.is_duplicated().sum())}",
        ]
    )

    if id_column:
        report_lines.append(f"- `{id_column}` unique in train: {train_df[id_column].n_unique() == train_df.height}")
        report_lines.append(f"- `{id_column}` unique in test: {test_df[id_column].n_unique() == test_df.height}")
        report_lines.append(
            f"- sample submission ids match test ids: {sample_submission[id_column].equals(test_df[id_column])}"
        )

    train_nulls = nonzero_null_counts(train_df)
    test_nulls = nonzero_null_counts(test_df)
    report_lines.append(f"- null counts in train: {format_counts(train_nulls)}")
    report_lines.append(f"- null counts in test: {format_counts(test_nulls)}")
    report_lines.append("")

    if target_column and target_column in train_df.columns:
        report_lines.extend(target_summary(train_df, target_column))

    if categorical_features:
        report_lines.extend(categorical_overlap_summary(train_df, test_df, categorical_features))

    if numeric_features:
        report_lines.extend(numeric_summary(train_df, numeric_features))

    output_path = artifact_dir / "data_profile.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(project_root)}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def infer_id_column(train_df: pl.DataFrame, test_df: pl.DataFrame, sample_submission: pl.DataFrame) -> str | None:
    sample_id = sample_submission.columns[0]
    if sample_id in train_df.columns and sample_id in test_df.columns:
        return sample_id
    if "id" in train_df.columns and "id" in test_df.columns:
        return "id"
    return None


def infer_target_column(train_df: pl.DataFrame, test_df: pl.DataFrame, sample_submission: pl.DataFrame) -> str | None:
    submission_target = sample_submission.columns[1]
    if submission_target in train_df.columns and submission_target not in test_df.columns:
        return submission_target

    train_only = [column for column in train_df.columns if column not in test_df.columns]
    if len(train_only) == 1:
        return train_only[0]
    return None


def infer_numeric_features(train_df: pl.DataFrame, target_column: str | None, id_column: str | None) -> list[str]:
    excluded = {target_column, id_column}
    return [
        name
        for name, dtype in train_df.schema.items()
        if dtype.is_numeric() and name not in excluded
    ]


def infer_categorical_features(train_df: pl.DataFrame, target_column: str | None) -> list[str]:
    return [
        name
        for name, dtype in train_df.schema.items()
        if not dtype.is_numeric() and name != target_column
    ]


def nonzero_null_counts(frame: pl.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in frame.columns:
        null_count = int(frame[column].null_count())
        if null_count:
            counts[column] = null_count
    return counts


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"`{column}`={count}" for column, count in counts.items())


def format_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def target_summary(train_df: pl.DataFrame, target_column: str) -> list[str]:
    lines = ["## Target Summary", ""]
    target = train_df[target_column]

    if target.dtype.is_numeric() and target.n_unique() > 20:
        lines.append("- inferred task type: regression-like")
        lines.append(f"- mean: {float(target.mean()):.6f}")
        lines.append(f"- median: {float(target.median()):.6f}")
        lines.append(f"- std: {float(target.std()):.6f}")
        lines.append(f"- min: {float(target.min()):.6f}")
        lines.append(f"- max: {float(target.max()):.6f}")
        lines.append("")
        return lines

    lines.append("- inferred task type: classification-like")
    counts = (
        train_df.group_by(target_column)
        .len()
        .rename({"len": "count"})
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
        .sort("count", descending=True)
    )
    for row in counts.iter_rows(named=True):
        lines.append(f"- `{row[target_column]}`: {row['count']} rows ({row['share']:.2%})")
    lines.append("")
    return lines


def categorical_overlap_summary(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    categorical_features: list[str],
) -> list[str]:
    lines = ["## Categorical Overlap", ""]
    shared_features = [feature for feature in categorical_features if feature in test_df.columns]
    for feature in shared_features:
        train_levels = set(train_df[feature].drop_nulls().unique().to_list())
        test_levels = set(test_df[feature].drop_nulls().unique().to_list())
        train_only = sorted(train_levels - test_levels)
        test_only = sorted(test_levels - train_levels)
        lines.append(
            f"- `{feature}`: train_only={format_list(train_only)} | test_only={format_list(test_only)}"
        )
    lines.append("")
    return lines


def numeric_summary(train_df: pl.DataFrame, numeric_features: list[str]) -> list[str]:
    lines = ["## Numeric Summary", "", "| feature | mean | std | min | max |", "| --- | ---: | ---: | ---: | ---: |"]
    for feature in numeric_features:
        column = train_df[feature]
        lines.append(
            f"| `{feature}` | {float(column.mean()):.6f} | {float(column.std()):.6f} | {float(column.min()):.6f} | {float(column.max()):.6f} |"
        )
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
