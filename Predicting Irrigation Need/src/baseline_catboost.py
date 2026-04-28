from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from catboost import CatBoostClassifier

from benchmark import ID_COLUMN, PROJECT_ROOT, TARGET, load_benchmark_part, score_prediction_frame

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline_catboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CatBoost baseline against the fixed benchmark split.")
    parser.add_argument("--benchmark", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--feature-set", choices=("raw", "domain_v1"), default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--output", help="Optional output path for validation predictions.")
    return parser.parse_args()


def add_domain_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("Crop_Growth_Stage") == "Sowing")
        .then(0)
        .when(pl.col("Crop_Growth_Stage") == "Vegetative")
        .then(1)
        .when(pl.col("Crop_Growth_Stage") == "Flowering")
        .then(2)
        .when(pl.col("Crop_Growth_Stage") == "Harvest")
        .then(3)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("Crop_Growth_Stage_Ordinal"),
        (pl.col("Temperature_C") * pl.col("Sunlight_Hours") * pl.col("Wind_Speed_kmh") / (pl.col("Humidity") + 1.0)).alias(
            "Dryness_Index"
        ),
        (pl.col("Soil_Moisture") / (pl.col("Temperature_C") + 1.0)).alias("Moisture_Temperature_Ratio"),
        ((pl.col("Rainfall_mm") + pl.col("Previous_Irrigation_mm")) / (pl.col("Sunlight_Hours") + 1.0)).alias(
            "Water_Input_Per_Sunlight"
        ),
    )


def build_feature_frame(frame: pl.DataFrame, feature_set: str) -> pl.DataFrame:
    features = frame.drop(TARGET, ID_COLUMN)
    if feature_set == "raw":
        return features
    if feature_set == "domain_v1":
        return add_domain_features(features)
    raise ValueError(f"Unknown feature set: {feature_set}")


def to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def split_feature_columns(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    categorical_columns = [name for name in frame.columns if name not in numeric_columns]
    return numeric_columns, categorical_columns


def output_path_for(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    filename = f"{args.benchmark}_{args.feature_set}_seed{args.seed}.csv"
    return ARTIFACTS_DIR / filename


def main() -> None:
    args = parse_args()
    train_frame = load_benchmark_part(args.benchmark, "train")
    valid_frame = load_benchmark_part(args.benchmark, "valid")

    x_train_pl = build_feature_frame(train_frame, args.feature_set)
    x_valid_pl = build_feature_frame(valid_frame, args.feature_set)
    _, categorical_columns = split_feature_columns(x_train_pl)

    x_train = to_pandas(x_train_pl)
    x_valid = to_pandas(x_valid_pl)
    y_train = train_frame[TARGET].to_list()

    model = CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        depth=8,
        learning_rate=0.05,
        iterations=args.iterations,
        random_seed=args.seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x_train, y_train, cat_features=categorical_columns, eval_set=(x_valid, valid_frame[TARGET].to_list()), use_best_model=True)

    predictions = model.predict(x_valid).ravel().tolist()
    prediction_frame = pl.DataFrame({ID_COLUMN: valid_frame[ID_COLUMN].to_list(), TARGET: predictions})

    output_path = output_path_for(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.write_csv(output_path)

    metrics = score_prediction_frame(prediction_frame, args.benchmark)
    print(f"Wrote predictions to {output_path}")
    print(pl.DataFrame([metrics]).with_columns(pl.exclude(["benchmark", "rows"]).round(6)))


if __name__ == "__main__":
    main()
