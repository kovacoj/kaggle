from __future__ import annotations

from pathlib import Path

import polars as pl

from benchmark import (
    BENCHMARK_NAMES,
    ID_COLUMN,
    PROJECT_ROOT,
    TARGET,
    benchmark_split_column,
    ensure_benchmark_exists,
    load_spec,
    load_split_frame,
)


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_csv(data_dir / "train.csv")
    test_df = pl.read_csv(data_dir / "test.csv")
    sample_submission = pl.read_csv(data_dir / "sample_submission.csv")

    numeric_features = [name for name, dtype in train_df.schema.items() if dtype.is_numeric() and name not in {ID_COLUMN}]
    categorical_features = [name for name, dtype in train_df.schema.items() if not dtype.is_numeric() and name != TARGET]

    lines = [
        "# Data Profile",
        "",
        "## Files",
        "",
        f"- `train.csv`: {train_df.height} rows x {train_df.width} columns",
        f"- `test.csv`: {test_df.height} rows x {test_df.width} columns",
        f"- `sample_submission.csv`: {sample_submission.height} rows x {sample_submission.width} columns",
        "",
        "## Roles",
        "",
        f"- id column: `{ID_COLUMN}`",
        f"- target column: `{TARGET}`",
        f"- numeric features: {len(numeric_features)}",
        f"- categorical features: {len(categorical_features)}",
        "",
        "## Data Quality",
        "",
        f"- train duplicate rows: {int(train_df.is_duplicated().sum())}",
        f"- test duplicate rows: {int(test_df.is_duplicated().sum())}",
        f"- `{ID_COLUMN}` unique in train: {train_df[ID_COLUMN].n_unique() == train_df.height}",
        f"- `{ID_COLUMN}` unique in test: {test_df[ID_COLUMN].n_unique() == test_df.height}",
        f"- sample submission ids match test ids: {sample_submission[ID_COLUMN].equals(test_df[ID_COLUMN])}",
        f"- null counts in train: {format_counts(nonzero_null_counts(train_df))}",
        f"- null counts in test: {format_counts(nonzero_null_counts(test_df))}",
        "",
        "## Target Distribution",
        "",
    ]

    target_distribution = (
        train_df.group_by(TARGET)
        .len()
        .rename({"len": "count"})
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
        .sort(TARGET)
    )
    for row in target_distribution.iter_rows(named=True):
        lines.append(f"- `{row[TARGET]}`: {row['count']} rows ({row['share']:.2%})")
    lines.append("")

    lines.extend(["## Train/Test Column Checks", ""])
    train_only = sorted(set(train_df.columns) - set(test_df.columns))
    test_only = sorted(set(test_df.columns) - set(train_df.columns))
    lines.append(f"- columns only in train: {format_list(train_only)}")
    lines.append(f"- columns only in test: {format_list(test_only)}")
    lines.append("")

    lines.extend(["## Categorical Overlap", ""])
    for feature in categorical_features:
        train_levels = set(train_df[feature].drop_nulls().unique().to_list())
        test_levels = set(test_df[feature].drop_nulls().unique().to_list())
        lines.append(
            f"- `{feature}`: train_only={format_list(sorted(train_levels - test_levels))} | "
            f"test_only={format_list(sorted(test_levels - train_levels))}"
        )
    lines.append("")

    lines.extend(["## Numeric Summary", "", "| feature | mean | std | min | max |", "| --- | ---: | ---: | ---: | ---: |"])
    for feature in numeric_features:
        if feature == TARGET:
            continue
        column = train_df[feature]
        lines.append(
            f"| `{feature}` | {float(column.mean()):.6f} | {float(column.std()):.6f} | {float(column.min()):.6f} | {float(column.max()):.6f} |"
        )
    lines.append("")

    ensure_benchmark_exists()
    spec = load_spec()
    split_frame = load_split_frame()
    lines.extend(["## Fixed Benchmark", "", f"- version: `{spec['version']}`", f"- metric: `{spec['metric']}`", ""])
    for benchmark_name in BENCHMARK_NAMES:
        split_column = benchmark_split_column(benchmark_name)
        lines.append(f"### {benchmark_name}")
        lines.append("")
        for part in ("train", "valid"):
            part_ids = split_frame.filter(pl.col(split_column) == part).select(ID_COLUMN)
            part_frame = train_df.join(part_ids, on=ID_COLUMN, how="inner")
            target_shares = (
                part_frame.group_by(TARGET)
                .len()
                .rename({"len": "count"})
                .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
                .sort(TARGET)
            )
            share_map = {row[TARGET]: row["share"] for row in target_shares.iter_rows(named=True)}
            lines.append(
                f"- `{part}`: {part_frame.height} rows | "
                f"Low={share_map.get('Low', 0.0):.2%}, Medium={share_map.get('Medium', 0.0):.2%}, High={share_map.get('High', 0.0):.2%}"
            )
        lines.append("")

    lines.extend(
        [
            "## Modeling Constraints",
            "",
            "- Optimize balanced accuracy, not plain accuracy.",
            "- Respect the fixed benchmark split in `benchmark/holdout_v1/` for comparable local evaluations.",
            "- Use as many CPU cores as possible whenever the training library supports it.",
            "- Use `smoke` for quick iteration and `full` to confirm promising changes.",
            "- If iteration is too slow on the full data, test on `smoke` or another smaller stratified subset first.",
            "- `data/sample_submission.csv` remains the source of truth for Kaggle submission columns and order.",
            "",
        ]
    )

    output_path = artifact_dir / "data_profile.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")


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


if __name__ == "__main__":
    main()
