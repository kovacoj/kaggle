from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from benchmark import ID_COLUMN, TARGET

APPROACH = "catboost_v41"
DESCRIPTION = "Faster three-view CatBoost blend with lower iteration budgets"

RANDOM_SEED = 42
CPU_THREADS = 12
MULTICLASS_ITERATIONS = 900
BINARY_ITERATIONS = 600
EARLY_STOP_WAIT = 35
SMOKE_TRAIN_ROWS = 80000

CATEGORICAL_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

NUMERIC_COLS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

CONTEXT_NUMERIC_COLS = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Humidity",
    "Previous_Irrigation_mm",
]

CONTEXT_GROUPS = [
    ("Crop_Type",),
    ("Crop_Growth_Stage",),
    ("Season",),
    ("Crop_Type", "Crop_Growth_Stage"),
]

COUNT_GROUPS = [
    ("Soil_Type",),
    ("Crop_Type",),
    ("Crop_Growth_Stage",),
    ("Season",),
    ("Irrigation_Type",),
    ("Water_Source",),
    ("Mulching_Used",),
    ("Region",),
    ("Crop_Type", "Crop_Growth_Stage"),
    ("Crop_Type", "Season"),
    ("Soil_Type", "Region"),
    ("Irrigation_Type", "Water_Source"),
    ("Mulching_Used", "Crop_Type"),
]

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
LABEL_INVERSE = {v: k for k, v in LABEL_MAP.items()}


def add_threshold_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("Soil_Moisture") < 25).cast(pl.Int8).alias("soil_lt_25"),
        (pl.col("Temperature_C") > 30).cast(pl.Int8).alias("temp_gt_30"),
        (pl.col("Rainfall_mm") < 300).cast(pl.Int8).alias("rain_lt_300"),
        (pl.col("Wind_Speed_kmh") > 10).cast(pl.Int8).alias("wind_gt_10"),
    )


def add_logit_features(frame: pl.DataFrame) -> pl.DataFrame:
    cgs_fl = (pl.col("Crop_Growth_Stage") == "Flowering").cast(pl.Int8)
    cgs_ha = (pl.col("Crop_Growth_Stage") == "Harvest").cast(pl.Int8)
    cgs_so = (pl.col("Crop_Growth_Stage") == "Sowing").cast(pl.Int8)
    cgs_ve = (pl.col("Crop_Growth_Stage") == "Vegetative").cast(pl.Int8)
    mu_no = (pl.col("Mulching_Used") == "No").cast(pl.Int8)
    mu_yes = (pl.col("Mulching_Used") == "Yes").cast(pl.Int8)
    return frame.with_columns(
        (16.3173 + -11.0237 * pl.col("soil_lt_25") + -5.8559 * pl.col("temp_gt_30")
         + -10.85 * pl.col("rain_lt_300") + -5.8284 * pl.col("wind_gt_10")
         + -5.4155 * cgs_fl + 5.5073 * cgs_ha + 5.2299 * cgs_so + -5.4617 * cgs_ve
         + -3.0014 * mu_no + 2.8613 * mu_yes).alias("logit_Low"),
        (4.6524 + 0.329 * pl.col("soil_lt_25") + -0.0204 * pl.col("temp_gt_30")
         + 0.1542 * pl.col("rain_lt_300") + 0.0841 * pl.col("wind_gt_10")
         + 0.3586 * cgs_fl + -0.1348 * cgs_ha + -0.3547 * cgs_so + 0.3334 * cgs_ve
         + 0.1883 * mu_no + 0.0142 * mu_yes).alias("logit_Med"),
        (-20.9697 + 10.6947 * pl.col("soil_lt_25") + 5.8763 * pl.col("temp_gt_30")
         + 10.6958 * pl.col("rain_lt_300") + 5.7444 * pl.col("wind_gt_10")
         + 5.0569 * cgs_fl + -5.3725 * cgs_ha + -4.8752 * cgs_so + 5.1283 * cgs_ve
         + 2.8131 * mu_no + -2.8755 * mu_yes).alias("logit_High"),
    )


def add_digit_features(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame
    for column in NUMERIC_COLS:
        for power in range(-4, 4):
            result = result.with_columns(
                (pl.col(column) // (10**power) % 10).cast(pl.Int8).alias(f"{column}_d{power}")
            )
    return result


def add_rounding_features(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame
    for column in ["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Humidity", "Previous_Irrigation_mm"]:
        frac = pl.col(column) - pl.col(column).floor()
        result = result.with_columns(
            (frac == 0).cast(pl.Int8).alias(f"{column}_is_int"),
            (pl.col(column) * 100 % 10 == 0).cast(pl.Int8).alias(f"{column}_d1_zero"),
            (pl.col(column) * 10 % 10 == 0).cast(pl.Int8).alias(f"{column}_d0_zero"),
        )
    return result


def add_domain_features(frame: pl.DataFrame) -> pl.DataFrame:
    temperature = pl.col("Temperature_C")
    sunlight = pl.col("Sunlight_Hours")
    wind = pl.col("Wind_Speed_kmh")
    humidity = pl.col("Humidity")
    rainfall = pl.col("Rainfall_mm")
    prev_irrigation = pl.col("Previous_Irrigation_mm")
    soil_moisture = pl.col("Soil_Moisture")
    field_area = pl.col("Field_Area_hectare")
    return frame.with_columns(
        (temperature * sunlight).alias("ET_proxy"),
        (rainfall + prev_irrigation).alias("Water_supply"),
        (rainfall + prev_irrigation - temperature * sunlight).alias("Water_budget"),
        (1.0 / (soil_moisture + 1.0)).alias("Inv_moisture"),
        (temperature / (soil_moisture + 1.0)).alias("Temp_per_moisture"),
        (wind / (soil_moisture + 1.0)).alias("Wind_per_moisture"),
        (1.0 / (rainfall + 1.0)).alias("Inv_rainfall"),
        (temperature * wind / (soil_moisture + 1.0)).alias("Heat_wind_per_moisture"),
        (temperature * wind * sunlight / (soil_moisture * humidity + 1.0)).alias("Drought_stress"),
        (soil_moisture / ((temperature + 1.0) * (wind + 1.0))).alias("Moisture_comfort"),
        (soil_moisture / (rainfall + 1.0)).alias("Moisture_rain_ratio"),
        (temperature / (humidity + 1.0)).alias("Temp_humid_ratio"),
        (humidity / (temperature + 1.0)).alias("Humid_temp_ratio"),
        (temperature - soil_moisture).alias("Temp_moisture_diff"),
        (temperature * sunlight / (rainfall + 1.0)).alias("Heat_light_per_rain"),
        (wind * temperature / (humidity + 1.0)).alias("Dry_wind_index"),
        (soil_moisture / (humidity + 1.0)).alias("Moisture_humid_ratio"),
        (soil_moisture / (prev_irrigation + 1.0)).alias("Moisture_previrr_ratio"),
        (soil_moisture * pl.col("Organic_Carbon")).alias("Moisture_carbon"),
        (temperature * rainfall).alias("Temp_rain"),
        (humidity * soil_moisture).alias("Humid_moisture"),
        np.log1p(rainfall).alias("Log_rain"),
        np.log1p(soil_moisture).alias("Log_moisture"),
        (rainfall / (field_area + 0.01)).alias("Rain_per_area"),
        (prev_irrigation / (field_area + 0.01)).alias("Previrr_per_area"),
        (temperature * wind).alias("Temp_wind"),
        (pl.col("Soil_pH") * pl.col("Organic_Carbon")).alias("pH_carbon"),
        (pl.col("Electrical_Conductivity") * pl.col("Organic_Carbon")).alias("EC_carbon"),
    )


def compute_target_encoding(train_frame: pl.DataFrame, column: str, smoothing: float = 10.0) -> pl.DataFrame:
    counts = train_frame.group_by(column).len().rename({"len": "count"})
    means = train_frame.group_by(column).agg(
        (pl.col(TARGET) == "High").cast(pl.Int64).mean().alias(f"{column}_te_High"),
        (pl.col(TARGET) == "Medium").cast(pl.Int64).mean().alias(f"{column}_te_Medium"),
        (pl.col(TARGET) == "Low").cast(pl.Int64).mean().alias(f"{column}_te_Low"),
    )
    global_high = (train_frame[TARGET] == "High").cast(pl.Int64).mean()
    global_medium = (train_frame[TARGET] == "Medium").cast(pl.Int64).mean()
    global_low = (train_frame[TARGET] == "Low").cast(pl.Int64).mean()
    stats = means.join(counts, on=column)
    return stats.with_columns(
        ((pl.col("count") * pl.col(f"{column}_te_High") + smoothing * global_high) / (pl.col("count") + smoothing)).alias(f"{column}_te_High"),
        ((pl.col("count") * pl.col(f"{column}_te_Medium") + smoothing * global_medium) / (pl.col("count") + smoothing)).alias(f"{column}_te_Medium"),
        ((pl.col("count") * pl.col(f"{column}_te_Low") + smoothing * global_low) / (pl.col("count") + smoothing)).alias(f"{column}_te_Low"),
    ).select(column, f"{column}_te_High", f"{column}_te_Medium", f"{column}_te_Low")


def compute_pairwise_target_encoding(train_frame: pl.DataFrame, col1: str, col2: str, smoothing: float = 40.0) -> pl.DataFrame:
    pair_name = f"{col1}_{col2}"
    counts = train_frame.group_by([col1, col2]).len().rename({"len": "count"})
    means = train_frame.group_by([col1, col2]).agg(
        (pl.col(TARGET) == "High").cast(pl.Int64).mean().alias(f"{pair_name}_te_High"),
        (pl.col(TARGET) == "Medium").cast(pl.Int64).mean().alias(f"{pair_name}_te_Medium"),
        (pl.col(TARGET) == "Low").cast(pl.Int64).mean().alias(f"{pair_name}_te_Low"),
    )
    global_high = (train_frame[TARGET] == "High").cast(pl.Int64).mean()
    global_medium = (train_frame[TARGET] == "Medium").cast(pl.Int64).mean()
    global_low = (train_frame[TARGET] == "Low").cast(pl.Int64).mean()
    stats = means.join(counts, on=[col1, col2])
    return stats.with_columns(
        ((pl.col("count") * pl.col(f"{pair_name}_te_High") + smoothing * global_high) / (pl.col("count") + smoothing)).alias(f"{pair_name}_te_High"),
        ((pl.col("count") * pl.col(f"{pair_name}_te_Medium") + smoothing * global_medium) / (pl.col("count") + smoothing)).alias(f"{pair_name}_te_Medium"),
        ((pl.col("count") * pl.col(f"{pair_name}_te_Low") + smoothing * global_low) / (pl.col("count") + smoothing)).alias(f"{pair_name}_te_Low"),
    ).select(col1, col2, f"{pair_name}_te_High", f"{pair_name}_te_Medium", f"{pair_name}_te_Low")


PAIRWISE_PAIRS = [
    ("Crop_Type", "Crop_Growth_Stage"),
    ("Soil_Type", "Region"),
    ("Crop_Type", "Season"),
    ("Irrigation_Type", "Water_Source"),
    ("Crop_Type", "Region"),
    ("Crop_Growth_Stage", "Season"),
    ("Soil_Type", "Crop_Type"),
    ("Crop_Growth_Stage", "Region"),
    ("Soil_Type", "Crop_Growth_Stage"),
    ("Season", "Region"),
    ("Irrigation_Type", "Crop_Type"),
    ("Mulching_Used", "Crop_Type"),
    ("Mulching_Used", "Soil_Type"),
    ("Crop_Growth_Stage", "Irrigation_Type"),
]


def build_encoding_tables(train_frame: pl.DataFrame, pairwise_te_smoothing: float = 40.0) -> dict[str, pl.DataFrame]:
    encodings: dict[str, pl.DataFrame] = {}
    for column in CATEGORICAL_COLS:
        encodings[column] = compute_target_encoding(train_frame, column)
    for col1, col2 in PAIRWISE_PAIRS:
        encodings[f"{col1}_{col2}"] = compute_pairwise_target_encoding(
            train_frame,
            col1,
            col2,
            smoothing=pairwise_te_smoothing,
        )
    return encodings


def group_key(columns: tuple[str, ...]) -> str:
    return "__".join(columns)


def build_context_tables(train_frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    for group_columns in CONTEXT_GROUPS:
        key = group_key(group_columns)
        aggregations: list[pl.Expr] = []
        for column in CONTEXT_NUMERIC_COLS:
            aggregations.extend(
                [
                    pl.col(column).mean().alias(f"{key}__{column}__mean"),
                    pl.col(column).std().alias(f"{key}__{column}__std"),
                ]
            )
        tables[key] = train_frame.group_by(list(group_columns)).agg(aggregations)
    return tables


def add_context_features(frame: pl.DataFrame, context_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    result = frame
    for group_columns in CONTEXT_GROUPS:
        key = group_key(group_columns)
        result = result.join(context_tables[key], on=list(group_columns), how="left")
        expressions: list[pl.Expr] = []
        for column in CONTEXT_NUMERIC_COLS:
            mean_col = f"{key}__{column}__mean"
            std_col = f"{key}__{column}__std"
            expressions.extend(
                [
                    (pl.col(column) - pl.col(mean_col)).alias(f"{key}__{column}__delta"),
                    ((pl.col(column) - pl.col(mean_col)) / (pl.col(std_col).fill_null(0.0) + 1.0)).alias(f"{key}__{column}__z"),
                    (pl.col(column) / (pl.col(mean_col) + 1.0)).alias(f"{key}__{column}__ratio"),
                ]
            )
        result = result.with_columns(expressions)
    return result


def build_count_tables(train_frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    total_rows = train_frame.height
    for group_columns in COUNT_GROUPS:
        key = group_key(group_columns)
        count_col = f"{key}__count"
        tables[key] = (
            train_frame.group_by(list(group_columns))
            .len()
            .rename({"len": count_col})
            .with_columns(
                pl.col(count_col).log1p().alias(f"{key}__count_log"),
                (pl.col(count_col) / total_rows).alias(f"{key}__freq"),
            )
        )
    return tables


def add_count_features(frame: pl.DataFrame, count_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    result = frame
    for group_columns in COUNT_GROUPS:
        key = group_key(group_columns)
        result = result.join(count_tables[key], on=list(group_columns), how="left")
    return result


def add_target_encodings(frame: pl.DataFrame, encodings: dict[str, pl.DataFrame]) -> pl.DataFrame:
    for encoding_frame in encodings.values():
        join_columns = [
            column for column in encoding_frame.columns
            if not column.endswith("_te_High") and not column.endswith("_te_Medium") and not column.endswith("_te_Low")
        ]
        frame = frame.join(encoding_frame, on=join_columns, how="left")
    return frame


def build_feature_frame(
    frame: pl.DataFrame,
    encodings: dict[str, pl.DataFrame] | None = None,
    context_tables: dict[str, pl.DataFrame] | None = None,
    count_tables: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    features = frame
    for column in (TARGET, ID_COLUMN):
        if column in features.columns:
            features = features.drop(column)
    features = add_threshold_features(features)
    features = add_logit_features(features)
    features = add_digit_features(features)
    features = add_domain_features(features)
    if context_tables is not None:
        features = add_context_features(features, context_tables)
    if count_tables is not None:
        features = add_count_features(features, count_tables)
    if encodings is not None:
        features = add_target_encodings(features, encodings)
    for column in features.columns:
        if features[column].dtype == pl.Float64:
            features = features.with_columns(pl.col(column).cast(pl.Float32))
    return features


def to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def categorical_columns(frame: pl.DataFrame) -> list[str]:
    numeric_columns = {name for name, dtype in frame.schema.items() if dtype.is_numeric()}
    return [name for name in frame.columns if name not in numeric_columns]


def compute_class_weights(y_train_int: list[int], high_weight_mult: float = 10.0) -> dict[int, float]:
    n_low = sum(1 for value in y_train_int if value == 0)
    n_med = sum(1 for value in y_train_int if value == 1)
    n_high = sum(1 for value in y_train_int if value == 2)
    total = len(y_train_int)
    return {
        0: total / (3 * n_low),
        1: total / (3 * n_med),
        2: (total / (3 * n_high)) * high_weight_mult,
    }


def compute_binary_class_weights(y_train_binary: list[int]) -> dict[int, float]:
    n_neg = sum(1 for value in y_train_binary if value == 0)
    n_pos = sum(1 for value in y_train_binary if value == 1)
    total = len(y_train_binary)
    return {
        0: total / (2 * n_neg),
        1: total / (2 * n_pos),
    }


def build_catboost_classifier(class_weights: dict[int, float], overrides: dict[str, float | int] | None = None) -> CatBoostClassifier:
    params: dict[str, object] = {
        "loss_function": "MultiClass",
        "class_weights": class_weights,
        "depth": 8,
        "learning_rate": 0.03,
        "iterations": MULTICLASS_ITERATIONS,
        "random_seed": RANDOM_SEED,
        "thread_count": CPU_THREADS,
        "allow_writing_files": False,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": EARLY_STOP_WAIT,
        "l2_leaf_reg": 5,
        "bagging_temperature": 0.8,
        "random_strength": 0.5,
    }
    if overrides is not None:
        params.update(overrides)
    clean_params = {key: value for key, value in params.items() if value is not None}
    return CatBoostClassifier(**clean_params)


def optimize_thresholds(probabilities: np.ndarray, y_true: list[str]) -> np.ndarray:
    adjusted, _, _ = optimize_high_shift(probabilities, y_true)
    return adjusted


def apply_class_shifts(probabilities: np.ndarray, medium_shift: float, high_shift: float) -> np.ndarray:
    adjusted = probabilities.copy()
    adjusted[:, 1] += medium_shift
    adjusted[:, 2] += high_shift
    adjusted = np.clip(adjusted, 1e-9, None)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def optimize_high_shift(probabilities: np.ndarray, y_true: list[str]) -> tuple[np.ndarray, float, float]:
    best_score = -1.0
    best_high_shift = 0.0
    for high_shift in np.arange(-0.50, 1.01, 0.02):
        adjusted = apply_class_shifts(probabilities, 0.0, float(high_shift))
        score = balanced_accuracy_score(y_true, [LABEL_INVERSE[i] for i in adjusted.argmax(axis=1)])
        if score > best_score:
            best_score = score
            best_high_shift = float(high_shift)
    for high_shift in np.arange(best_high_shift - 0.02, best_high_shift + 0.021, 0.002):
        adjusted = apply_class_shifts(probabilities, 0.0, float(high_shift))
        score = balanced_accuracy_score(y_true, [LABEL_INVERSE[i] for i in adjusted.argmax(axis=1)])
        if score > best_score:
            best_score = score
            best_high_shift = float(high_shift)
    adjusted = apply_class_shifts(probabilities, 0.0, best_high_shift)
    return adjusted, best_high_shift, best_score


def optimize_three_view_blend(
    base_probabilities: np.ndarray,
    low_medium_scores: np.ndarray,
    medium_high_scores: np.ndarray,
    y_true: list[str],
) -> tuple[np.ndarray, float, float, float, float, float]:
    best_score = -1.0
    best_adjusted: np.ndarray | None = None
    best_base_weight = 1.0
    best_low_medium_weight = 0.0
    best_medium_high_weight = 0.0
    best_high_shift = 0.0
    for low_medium_weight in np.arange(0.0, 0.181, 0.02):
        for medium_high_weight in np.arange(0.0, 0.121, 0.02):
            if low_medium_weight + medium_high_weight > 0.18:
                continue
            base_weight = 1.0 - low_medium_weight - medium_high_weight
            blended = (
                base_weight * base_probabilities
                + low_medium_weight * low_medium_scores
                + medium_high_weight * medium_high_scores
            )
            adjusted, high_shift, score = optimize_high_shift(blended, y_true)
            if score > best_score:
                best_score = score
                best_adjusted = adjusted
                best_base_weight = float(base_weight)
                best_low_medium_weight = float(low_medium_weight)
                best_medium_high_weight = float(medium_high_weight)
                best_high_shift = float(high_shift)
    for low_medium_weight in np.arange(best_low_medium_weight - 0.02, best_low_medium_weight + 0.021, 0.004):
        for medium_high_weight in np.arange(best_medium_high_weight - 0.02, best_medium_high_weight + 0.021, 0.004):
            if low_medium_weight < 0.0 or medium_high_weight < 0.0 or low_medium_weight + medium_high_weight > 0.2:
                continue
            base_weight = 1.0 - low_medium_weight - medium_high_weight
            blended = (
                base_weight * base_probabilities
                + low_medium_weight * low_medium_scores
                + medium_high_weight * medium_high_scores
            )
            adjusted, high_shift, score = optimize_high_shift(blended, y_true)
            if score > best_score:
                best_score = score
                best_adjusted = adjusted
                best_base_weight = float(base_weight)
                best_low_medium_weight = float(low_medium_weight)
                best_medium_high_weight = float(medium_high_weight)
                best_high_shift = float(high_shift)
    assert best_adjusted is not None
    return (
        best_adjusted,
        best_base_weight,
        best_low_medium_weight,
        best_medium_high_weight,
        best_high_shift,
        best_score,
    )


def take_rows(frame: pl.DataFrame, indices: list[int]) -> pl.DataFrame:
    return frame.with_row_index("__row_index").filter(pl.col("__row_index").is_in(indices)).drop("__row_index")


def stratified_split(frame: pl.DataFrame, valid_fraction: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    labels = frame[TARGET].to_list()
    indices = list(range(frame.height))
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=valid_fraction,
        stratify=labels,
        random_state=seed,
    )
    return take_rows(frame, train_indices), take_rows(frame, valid_indices)


def maybe_subset_smoke_train(train_frame: pl.DataFrame, benchmark_name: str) -> pl.DataFrame:
    if benchmark_name != "smoke" or train_frame.height <= SMOKE_TRAIN_ROWS:
        return train_frame
    subset_fraction = 1.0 - (SMOKE_TRAIN_ROWS / train_frame.height)
    subset_train, _ = stratified_split(train_frame, valid_fraction=subset_fraction, seed=RANDOM_SEED + 303)
    return subset_train


def build_binary_classifier(class_weights: dict[int, float]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        class_weights=class_weights,
        depth=8,
        learning_rate=0.03,
        iterations=BINARY_ITERATIONS,
        random_seed=RANDOM_SEED,
        thread_count=CPU_THREADS,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=EARLY_STOP_WAIT,
        l2_leaf_reg=5,
        bagging_temperature=0.8,
        random_strength=0.5,
    )


def ordinal_predict_labels(not_low_proba: np.ndarray, high_proba: np.ndarray, t_not_low: float, t_high: float) -> list[str]:
    predictions: list[str] = []
    for p_not_low, p_high in zip(not_low_proba, high_proba, strict=False):
        if p_not_low >= t_not_low and p_high >= t_high:
            predictions.append("High")
        elif p_not_low >= t_not_low:
            predictions.append("Medium")
        else:
            predictions.append("Low")
    return predictions


def optimize_ordinal_thresholds(not_low_proba: np.ndarray, high_proba: np.ndarray, y_true: list[str]) -> tuple[float, float]:
    best_score = -1.0
    best_not_low = 0.5
    best_high = 0.5
    for t_not_low in np.arange(0.20, 0.81, 0.02):
        for t_high in np.arange(0.05, 0.81, 0.02):
            predictions = ordinal_predict_labels(not_low_proba, high_proba, t_not_low, t_high)
            score = balanced_accuracy_score(y_true, predictions)
            if score > best_score:
                best_score = score
                best_not_low = float(t_not_low)
                best_high = float(t_high)
    for t_not_low in np.arange(best_not_low - 0.02, best_not_low + 0.021, 0.004):
        for t_high in np.arange(best_high - 0.02, best_high + 0.021, 0.004):
            predictions = ordinal_predict_labels(not_low_proba, high_proba, float(t_not_low), float(t_high))
            score = balanced_accuracy_score(y_true, predictions)
            if score > best_score:
                best_score = score
                best_not_low = float(t_not_low)
                best_high = float(t_high)
    return best_not_low, best_high


def fit_ordinal_probabilities(train_frame: pl.DataFrame, pred_frame: pl.DataFrame, eval_frame: pl.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    encodings = build_encoding_tables(train_frame)
    x_train_pl = build_feature_frame(train_frame, encodings)
    x_pred_pl = build_feature_frame(pred_frame, encodings)
    cat_cols = categorical_columns(x_train_pl)
    x_train = to_pandas(x_train_pl)
    x_pred = to_pandas(x_pred_pl)
    y_train_str = train_frame[TARGET].to_list()

    y_train_not_low = [1 if value != "Low" else 0 for value in y_train_str]
    y_train_high = [1 if value == "High" else 0 for value in y_train_str]

    eval_not_low: tuple[pd.DataFrame, list[int]] | None = None
    eval_high: tuple[pd.DataFrame, list[int]] | None = None
    if eval_frame is not None:
        x_eval_pl = build_feature_frame(eval_frame, encodings)
        x_eval = to_pandas(x_eval_pl)
        y_eval_str = eval_frame[TARGET].to_list()
        eval_not_low = (x_eval, [1 if value != "Low" else 0 for value in y_eval_str])
        eval_high = (x_eval, [1 if value == "High" else 0 for value in y_eval_str])

    not_low_model = build_binary_classifier(compute_binary_class_weights(y_train_not_low))
    not_low_model.fit(x_train, y_train_not_low, cat_features=cat_cols, eval_set=eval_not_low, use_best_model=eval_frame is not None)

    high_model = build_binary_classifier(compute_binary_class_weights(y_train_high))
    high_model.fit(x_train, y_train_high, cat_features=cat_cols, eval_set=eval_high, use_best_model=eval_frame is not None)

    return not_low_model.predict_proba(x_pred)[:, 1], high_model.predict_proba(x_pred)[:, 1]


def calibrate_ordinal_thresholds(train_frame: pl.DataFrame) -> tuple[float, float]:
    inner_train, inner_valid = stratified_split(train_frame, valid_fraction=0.15, seed=RANDOM_SEED + 101)
    not_low_proba, high_proba = fit_ordinal_probabilities(inner_train, inner_valid, eval_frame=inner_valid)
    return optimize_ordinal_thresholds(not_low_proba, high_proba, inner_valid[TARGET].to_list())


def fit_multiclass_probabilities(
    train_frame: pl.DataFrame,
    pred_frame: pl.DataFrame,
    eval_frame: pl.DataFrame | None = None,
) -> np.ndarray:
    encodings = build_encoding_tables(train_frame)
    x_train_pl = build_feature_frame(train_frame, encodings)
    x_pred_pl = build_feature_frame(pred_frame, encodings)
    cat_cols = categorical_columns(x_train_pl)
    x_train = to_pandas(x_train_pl)
    x_pred = to_pandas(x_pred_pl)
    y_train_int = [LABEL_MAP[value] for value in train_frame[TARGET].to_list()]
    class_weights = compute_class_weights(y_train_int)
    overrides = None
    eval_set = None
    use_best_model = False
    if eval_frame is None:
        overrides = {"od_type": None, "od_wait": None}
    else:
        x_eval_pl = build_feature_frame(eval_frame, encodings)
        x_eval = to_pandas(x_eval_pl)
        y_eval_int = [LABEL_MAP[value] for value in eval_frame[TARGET].to_list()]
        eval_set = (x_eval, y_eval_int)
        use_best_model = True
    model = build_catboost_classifier(class_weights, overrides=overrides)
    model.fit(x_train, y_train_int, cat_features=cat_cols, eval_set=eval_set, use_best_model=use_best_model)
    return model.predict_proba(x_pred)


def fit_adjacent_pair_binary_probabilities(
    train_frame: pl.DataFrame,
    pred_frame: pl.DataFrame,
    positive_label: str,
    eval_frame: pl.DataFrame | None = None,
) -> np.ndarray:
    encodings = build_encoding_tables(train_frame)
    x_train_pl = build_feature_frame(train_frame, encodings)
    x_pred_pl = build_feature_frame(pred_frame, encodings)
    cat_cols = categorical_columns(x_train_pl)
    x_train = to_pandas(x_train_pl)
    x_pred = to_pandas(x_pred_pl)
    y_train_binary = [1 if value == positive_label else 0 for value in train_frame[TARGET].to_list()]
    class_weights = compute_binary_class_weights(y_train_binary)
    model = build_binary_classifier(class_weights) if eval_frame is not None else CatBoostClassifier(
        loss_function="Logloss",
        class_weights=class_weights,
        depth=8,
        learning_rate=0.03,
        iterations=BINARY_ITERATIONS,
        random_seed=RANDOM_SEED,
        thread_count=CPU_THREADS,
        allow_writing_files=False,
        verbose=False,
        l2_leaf_reg=5,
        bagging_temperature=0.8,
        random_strength=0.5,
    )
    eval_set = None
    use_best_model = False
    if eval_frame is not None:
        x_eval_pl = build_feature_frame(eval_frame, encodings)
        x_eval = to_pandas(x_eval_pl)
        y_eval_binary = [1 if value == positive_label else 0 for value in eval_frame[TARGET].to_list()]
        eval_set = (x_eval, y_eval_binary)
        use_best_model = True
    model.fit(x_train, y_train_binary, cat_features=cat_cols, eval_set=eval_set, use_best_model=use_best_model)
    return model.predict_proba(x_pred)[:, 1]


def fit_adjacent_pair_probabilities(
    train_frame: pl.DataFrame,
    pred_frame: pl.DataFrame,
    eval_frame: pl.DataFrame | None = None,
) -> np.ndarray:
    low_medium_train = train_frame.filter(pl.col(TARGET).is_in(["Low", "Medium"]))
    medium_high_train = train_frame.filter(pl.col(TARGET).is_in(["Medium", "High"]))
    p_medium = fit_adjacent_pair_binary_probabilities(low_medium_train, pred_frame, positive_label="Medium", eval_frame=eval_frame)
    p_high = fit_adjacent_pair_binary_probabilities(medium_high_train, pred_frame, positive_label="High", eval_frame=eval_frame)
    scores = np.column_stack([1.0 - p_medium, p_medium * (1.0 - p_high), p_high])
    scores = np.clip(scores, 1e-9, None)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def fit_medium_high_specialist_probability(
    train_frame: pl.DataFrame,
    pred_frame: pl.DataFrame,
    eval_frame: pl.DataFrame | None = None,
) -> np.ndarray:
    medium_high_train = train_frame.filter(pl.col(TARGET).is_in(["Medium", "High"]))
    return fit_adjacent_pair_binary_probabilities(medium_high_train, pred_frame, positive_label="High", eval_frame=eval_frame)


def fit_low_medium_specialist_probability(
    train_frame: pl.DataFrame,
    pred_frame: pl.DataFrame,
    eval_frame: pl.DataFrame | None = None,
) -> np.ndarray:
    low_medium_train = train_frame.filter(pl.col(TARGET).is_in(["Low", "Medium"]))
    return fit_adjacent_pair_binary_probabilities(low_medium_train, pred_frame, positive_label="Medium", eval_frame=eval_frame)


def build_medium_high_specialist_scores(base_probabilities: np.ndarray, high_probability: np.ndarray) -> np.ndarray:
    medium_high_mass = 1.0 - base_probabilities[:, 0]
    scores = np.column_stack(
        [
            base_probabilities[:, 0],
            medium_high_mass * (1.0 - high_probability),
            medium_high_mass * high_probability,
        ]
    )
    scores = np.clip(scores, 1e-9, None)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def build_low_medium_specialist_scores(base_probabilities: np.ndarray, medium_probability: np.ndarray) -> np.ndarray:
    low_medium_mass = 1.0 - base_probabilities[:, 2]
    scores = np.column_stack(
        [
            low_medium_mass * (1.0 - medium_probability),
            low_medium_mass * medium_probability,
            base_probabilities[:, 2],
        ]
    )
    scores = np.clip(scores, 1e-9, None)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def calibrate_blend(train_frame: pl.DataFrame) -> tuple[float, float, float, float]:
    inner_train, inner_valid = stratified_split(train_frame, valid_fraction=0.15, seed=RANDOM_SEED + 202)
    base_prob = fit_multiclass_probabilities(inner_train, inner_valid, eval_frame=inner_valid)
    low_medium_prob = fit_low_medium_specialist_probability(inner_train, inner_valid, eval_frame=inner_valid)
    medium_high_prob = fit_medium_high_specialist_probability(inner_train, inner_valid, eval_frame=inner_valid)
    low_medium_scores = build_low_medium_specialist_scores(base_prob, low_medium_prob)
    medium_high_scores = build_medium_high_specialist_scores(base_prob, medium_high_prob)
    y_inner_str = inner_valid[TARGET].to_list()
    _, base_weight, low_medium_weight, medium_high_weight, high_shift, _ = optimize_three_view_blend(
        base_prob,
        low_medium_scores,
        medium_high_scores,
        y_inner_str,
    )
    return base_weight, low_medium_weight, medium_high_weight, high_shift


def fit_predict_valid(train_frame: pl.DataFrame, valid_frame: pl.DataFrame, benchmark_name: str) -> pl.DataFrame:
    train_frame = maybe_subset_smoke_train(train_frame, benchmark_name)
    print(f"  train_rows={train_frame.height}, valid_rows={valid_frame.height}")
    y_valid_str = valid_frame[TARGET].to_list()
    base_prob = fit_multiclass_probabilities(train_frame, valid_frame, eval_frame=valid_frame)
    low_medium_prob = fit_low_medium_specialist_probability(train_frame, valid_frame, eval_frame=valid_frame)
    medium_high_prob = fit_medium_high_specialist_probability(train_frame, valid_frame, eval_frame=valid_frame)
    low_medium_scores = build_low_medium_specialist_scores(base_prob, low_medium_prob)
    medium_high_scores = build_medium_high_specialist_scores(base_prob, medium_high_prob)
    adjusted, base_weight, low_medium_weight, medium_high_weight, high_shift, best_score = optimize_three_view_blend(
        base_prob,
        low_medium_scores,
        medium_high_scores,
        y_valid_str,
    )
    print(
        f"  weights base={base_weight:.3f}, low_medium={low_medium_weight:.3f}, "
        f"medium_high={medium_high_weight:.3f}, high_shift={high_shift:.3f}, score={best_score:.6f}"
    )
    predictions = [LABEL_INVERSE[i] for i in adjusted.argmax(axis=1)]
    return pl.DataFrame({ID_COLUMN: valid_frame[ID_COLUMN].to_list(), TARGET: predictions})


def fit_predict_test(train_frame: pl.DataFrame, test_frame: pl.DataFrame) -> pl.DataFrame:
    base_weight, low_medium_weight, medium_high_weight, high_shift = calibrate_blend(train_frame)
    base_prob = fit_multiclass_probabilities(train_frame, test_frame, eval_frame=None)
    low_medium_prob = fit_low_medium_specialist_probability(train_frame, test_frame, eval_frame=None)
    medium_high_prob = fit_medium_high_specialist_probability(train_frame, test_frame, eval_frame=None)
    low_medium_scores = build_low_medium_specialist_scores(base_prob, low_medium_prob)
    medium_high_scores = build_medium_high_specialist_scores(base_prob, medium_high_prob)
    probabilities = (
        base_weight * base_prob
        + low_medium_weight * low_medium_scores
        + medium_high_weight * medium_high_scores
    )
    probabilities = apply_class_shifts(probabilities, 0.0, high_shift)
    predictions = [LABEL_INVERSE[i] for i in probabilities.argmax(axis=1)]
    return pl.DataFrame({ID_COLUMN: test_frame[ID_COLUMN].to_list(), TARGET: predictions})
