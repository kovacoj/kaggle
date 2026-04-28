from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "holdout_v1"
SPEC_PATH = BENCHMARK_DIR / "spec.json"
SPLIT_PATH = BENCHMARK_DIR / "splits.csv"

TARGET = "Irrigation_Need"
ID_COLUMN = "id"
TARGET_LABELS = ["Low", "Medium", "High"]
BENCHMARK_NAMES = ("full", "smoke")

DEFAULT_SPEC = {
    "version": "holdout_v1",
    "metric": "balanced_accuracy",
    "seed": 42,
    "valid_fraction": 0.2,
    "smoke_train_rows": 80000,
    "smoke_valid_rows": 20000,
    "target": TARGET,
    "id_column": ID_COLUMN,
    "labels": TARGET_LABELS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed benchmark utilities for irrigation modeling experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the fixed benchmark split artifacts.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing split artifacts.")

    subparsers.add_parser("describe", help="Describe the benchmark split artifacts.")

    template_parser = subparsers.add_parser("template", help="Write a prediction template for a benchmark validation split.")
    template_parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default="smoke")
    template_parser.add_argument("--output", required=True, help="Where to write the template CSV.")

    score_parser = subparsers.add_parser("score", help="Score a predictions CSV against a benchmark validation split.")
    score_parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default="smoke")
    score_parser.add_argument("--predictions", required=True, help="CSV with columns id,Irrigation_Need.")

    return parser.parse_args()


def load_spec() -> dict[str, object]:
    if SPEC_PATH.exists():
        return json.loads(SPEC_PATH.read_text())
    return DEFAULT_SPEC.copy()


def load_train_frame() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "train.csv")


def benchmark_split_column(benchmark_name: str) -> str:
    if benchmark_name == "full":
        return "benchmark_split"
    if benchmark_name == "smoke":
        return "smoke_split"
    raise ValueError(f"Unknown benchmark name: {benchmark_name}")


def stratified_subset(indices: list[int], labels: list[str], subset_rows: int, seed: int) -> list[int]:
    if subset_rows >= len(indices):
        return indices
    sampled_indices, _ = train_test_split(indices, train_size=subset_rows, stratify=labels, random_state=seed)
    return list(sampled_indices)


def create_split_frame(train_frame: pl.DataFrame, spec: dict[str, object]) -> pl.DataFrame:
    seed = int(spec["seed"])
    valid_fraction = float(spec["valid_fraction"])
    smoke_train_rows = int(spec["smoke_train_rows"])
    smoke_valid_rows = int(spec["smoke_valid_rows"])

    row_indices = list(range(train_frame.height))
    labels = train_frame[TARGET].to_list()
    train_indices, valid_indices = train_test_split(
        row_indices,
        test_size=valid_fraction,
        stratify=labels,
        random_state=seed,
    )

    train_labels = [labels[index] for index in train_indices]
    valid_labels = [labels[index] for index in valid_indices]
    smoke_train_indices = stratified_subset(train_indices, train_labels, smoke_train_rows, seed + 1)
    smoke_valid_indices = stratified_subset(valid_indices, valid_labels, smoke_valid_rows, seed + 2)

    benchmark_split = ["train"] * train_frame.height
    smoke_split = ["unused"] * train_frame.height
    for index in valid_indices:
        benchmark_split[index] = "valid"
    for index in smoke_train_indices:
        smoke_split[index] = "train"
    for index in smoke_valid_indices:
        smoke_split[index] = "valid"

    return pl.DataFrame(
        {
            ID_COLUMN: train_frame[ID_COLUMN].to_list(),
            "benchmark_split": benchmark_split,
            "smoke_split": smoke_split,
        }
    ).sort(ID_COLUMN)


def ensure_benchmark_exists() -> None:
    if not SPEC_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Benchmark artifacts are missing. Run `uv run python src/benchmark.py init` first."
        )


def load_split_frame() -> pl.DataFrame:
    ensure_benchmark_exists()
    return pl.read_csv(SPLIT_PATH)


def write_split_artifacts(force: bool) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    if not force and (SPEC_PATH.exists() or SPLIT_PATH.exists()):
        raise FileExistsError("Benchmark artifacts already exist. Use --force to overwrite them.")

    spec = DEFAULT_SPEC.copy()
    train_frame = load_train_frame()
    split_frame = create_split_frame(train_frame, spec)
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n")
    split_frame.write_csv(SPLIT_PATH)


def load_benchmark_part(benchmark_name: str, part: str) -> pl.DataFrame:
    if part not in {"train", "valid"}:
        raise ValueError(f"Unknown benchmark part: {part}")

    train_frame = load_train_frame()
    split_frame = load_split_frame()
    split_column = benchmark_split_column(benchmark_name)
    selected_ids = split_frame.filter(pl.col(split_column) == part).select(ID_COLUMN)
    return train_frame.join(selected_ids, on=ID_COLUMN, how="inner")


def prediction_template(benchmark_name: str) -> pl.DataFrame:
    valid_frame = load_benchmark_part(benchmark_name, "valid")
    return valid_frame.select(ID_COLUMN).with_columns(pl.lit(TARGET_LABELS[0]).alias(TARGET))


def summarize_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    report = classification_report(y_true, y_pred, labels=TARGET_LABELS, output_dict=True, zero_division=0)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_low": report["Low"]["recall"],
        "recall_medium": report["Medium"]["recall"],
        "recall_high": report["High"]["recall"],
    }


def score_prediction_frame(predictions: pl.DataFrame, benchmark_name: str) -> dict[str, float | int | str]:
    if predictions.columns != [ID_COLUMN, TARGET]:
        raise ValueError(f"Predictions must have columns exactly [{ID_COLUMN}, {TARGET}].")
    if predictions[ID_COLUMN].n_unique() != predictions.height:
        raise ValueError("Predictions contain duplicate ids.")

    valid_frame = load_benchmark_part(benchmark_name, "valid").select(ID_COLUMN, TARGET)
    extra_ids = predictions.join(valid_frame.select(ID_COLUMN), on=ID_COLUMN, how="anti")
    if extra_ids.height > 0:
        raise ValueError("Predictions contain ids outside the benchmark validation split.")

    scored = valid_frame.join(predictions, on=ID_COLUMN, how="left", suffix="_pred")
    if scored[f"{TARGET}_pred"].null_count() > 0:
        raise ValueError("Predictions are missing ids from the benchmark validation split.")

    invalid_labels = scored.filter(~pl.col(f"{TARGET}_pred").is_in(TARGET_LABELS))
    if invalid_labels.height > 0:
        raise ValueError(f"Predictions contain labels outside {TARGET_LABELS}.")

    y_true = scored[TARGET].to_list()
    y_pred = scored[f"{TARGET}_pred"].to_list()
    metrics = summarize_predictions(y_true, y_pred)
    return {
        "benchmark": benchmark_name,
        "rows": len(y_true),
        **metrics,
    }


def describe_benchmark() -> None:
    spec = load_spec()
    train_frame = load_train_frame()
    split_frame = load_split_frame()
    print(json.dumps(spec, indent=2))

    summary_rows: list[dict[str, object]] = []
    for benchmark_name in BENCHMARK_NAMES:
        split_column = benchmark_split_column(benchmark_name)
        for part in ("train", "valid"):
            part_ids = split_frame.filter(pl.col(split_column) == part).select(ID_COLUMN)
            if part_ids.height == 0:
                continue
            part_frame = train_frame.join(part_ids, on=ID_COLUMN, how="inner")
            target_shares = (
                part_frame.group_by(TARGET)
                .len()
                .rename({"len": "count"})
                .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
                .sort(TARGET)
            )
            shares = {row[TARGET]: row["share"] for row in target_shares.to_dicts()}
            summary_rows.append(
                {
                    "benchmark": benchmark_name,
                    "part": part,
                    "rows": part_frame.height,
                    "share_low": shares.get("Low", 0.0),
                    "share_medium": shares.get("Medium", 0.0),
                    "share_high": shares.get("High", 0.0),
                }
            )

    print(
        pl.DataFrame(summary_rows).with_columns(
            pl.col("share_low").round(6),
            pl.col("share_medium").round(6),
            pl.col("share_high").round(6),
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "init":
        write_split_artifacts(force=args.force)
        describe_benchmark()
        return

    if args.command == "describe":
        describe_benchmark()
        return

    if args.command == "template":
        template = prediction_template(args.benchmark)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        template.write_csv(output_path)
        print(f"Wrote template to {output_path}")
        return

    if args.command == "score":
        predictions = pl.read_csv(args.predictions)
        metrics = score_prediction_frame(predictions, args.benchmark)
        print(pl.DataFrame([metrics]).with_columns(pl.exclude(["benchmark", "rows"]).round(6)))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
