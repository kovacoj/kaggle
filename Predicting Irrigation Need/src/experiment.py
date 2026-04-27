from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import optuna

from benchmark import ID_COLUMN, TARGET

optuna.logging.set_verbosity(optuna.logging.WARNING)

APPROACH = "catboost_v12"
DESCRIPTION = "3-model ensemble (CB+LGB+XGB), logit+domain+threshold features, 3-fold OTE on top cats, Optuna weight tuning"

RANDOM_SEED = 42
CPU_THREADS = -1
N_FOLDS = 3

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

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
LABEL_INVERSE = {v: k for k, v in LABEL_MAP.items()}

PAIRWISE_TE_PAIRS = [
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
]


def add_threshold_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("Soil_Moisture") < 25).cast(pl.Int8).alias("soil_lt_25"),
        (pl.col("Temperature_C") > 30).cast(pl.Int8).alias("temp_gt_30"),
        (pl.col("Rainfall_mm") < 300).cast(pl.Int8).alias("rain_lt_300"),
        (pl.col("Wind_Speed_kmh") > 10).cast(pl.Int8).alias("wind_gt_10"),
    )


def add_logit_features(frame: pl.DataFrame) -> pl.DataFrame:
    cgs_flowering = (pl.col("Crop_Growth_Stage") == "Flowering").cast(pl.Int8)
    cgs_harvest = (pl.col("Crop_Growth_Stage") == "Harvest").cast(pl.Int8)
    cgs_sowing = (pl.col("Crop_Growth_Stage") == "Sowing").cast(pl.Int8)
    cgs_vegetative = (pl.col("Crop_Growth_Stage") == "Vegetative").cast(pl.Int8)
    mulch_no = (pl.col("Mulching_Used") == "No").cast(pl.Int8)
    mulch_yes = (pl.col("Mulching_Used") == "Yes").cast(pl.Int8)

    return frame.with_columns(
        (16.3173
         + (-11.0237 * pl.col("soil_lt_25"))
         + (-5.8559 * pl.col("temp_gt_30"))
         + (-10.85 * pl.col("rain_lt_300"))
         + (-5.8284 * pl.col("wind_gt_10"))
         + (-5.4155 * cgs_flowering)
         + (5.5073 * cgs_harvest)
         + (5.2299 * cgs_sowing)
         + (-5.4617 * cgs_vegetative)
         + (-3.0014 * mulch_no)
         + (2.8613 * mulch_yes)
        ).alias("logit_Low"),
        (4.6524
         + (0.329 * pl.col("soil_lt_25"))
         + (-0.0204 * pl.col("temp_gt_30"))
         + (0.1542 * pl.col("rain_lt_300"))
         + (0.0841 * pl.col("wind_gt_10"))
         + (0.3586 * cgs_flowering)
         + (-0.1348 * cgs_harvest)
         + (-0.3547 * cgs_sowing)
         + (0.3334 * cgs_vegetative)
         + (0.1883 * mulch_no)
         + (0.0142 * mulch_yes)
        ).alias("logit_Med"),
        (-20.9697
         + (10.6947 * pl.col("soil_lt_25"))
         + (5.8763 * pl.col("temp_gt_30"))
         + (10.6958 * pl.col("rain_lt_300"))
         + (5.7444 * pl.col("wind_gt_10"))
         + (5.0569 * cgs_flowering)
         + (-5.3725 * cgs_harvest)
         + (-4.8752 * cgs_sowing)
         + (5.1283 * cgs_vegetative)
         + (2.8131 * mulch_no)
         + (-2.8755 * mulch_yes)
        ).alias("logit_High"),
    )


def add_domain_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("Temperature_C") * pl.col("Sunlight_Hours")).alias("ET_proxy"),
        (pl.col("Rainfall_mm") + pl.col("Previous_Irrigation_mm")).alias("Water_supply"),
        (pl.col("Rainfall_mm") + pl.col("Previous_Irrigation_mm") - pl.col("Temperature_C") * pl.col("Sunlight_Hours")).alias("Water_budget"),
        (1.0 / (pl.col("Soil_Moisture") + 1.0)).alias("Inv_moisture"),
        (pl.col("Temperature_C") / (pl.col("Soil_Moisture") + 1.0)).alias("Temp_per_moisture"),
        (pl.col("Wind_Speed_kmh") / (pl.col("Soil_Moisture") + 1.0)).alias("Wind_per_moisture"),
        (1.0 / (pl.col("Rainfall_mm") + 1.0)).alias("Inv_rainfall"),
        (pl.col("Temperature_C") * pl.col("Wind_Speed_kmh") / (pl.col("Soil_Moisture") + 1.0)).alias("Heat_wind_per_moisture"),
        (pl.col("Temperature_C") * pl.col("Wind_Speed_kmh") * pl.col("Sunlight_Hours") / (pl.col("Soil_Moisture") * pl.col("Humidity") + 1.0)).alias("Drought_stress"),
        (pl.col("Soil_Moisture") / ((pl.col("Temperature_C") + 1.0) * (pl.col("Wind_Speed_kmh") + 1.0))).alias("Moisture_comfort"),
        (pl.col("Soil_Moisture") / (pl.col("Rainfall_mm") + 1.0)).alias("Moisture_rain_ratio"),
        (pl.col("Temperature_C") / (pl.col("Humidity") + 1.0)).alias("Temp_humid_ratio"),
        (pl.col("Humidity") / (pl.col("Temperature_C") + 1.0)).alias("Humid_temp_ratio"),
        (pl.col("Temperature_C") - pl.col("Soil_Moisture")).alias("Temp_moisture_diff"),
        (pl.col("Temperature_C") * pl.col("Sunlight_Hours") / (pl.col("Rainfall_mm") + 1.0)).alias("Heat_light_per_rain"),
        (pl.col("Wind_Speed_kmh") * pl.col("Temperature_C") / (pl.col("Humidity") + 1.0)).alias("Dry_wind_index"),
        (pl.col("Soil_Moisture") / (pl.col("Humidity") + 1.0)).alias("Moisture_humid_ratio"),
        (pl.col("Soil_Moisture") / (pl.col("Previous_Irrigation_mm") + 1.0)).alias("Moisture_previrr_ratio"),
        (pl.col("Soil_Moisture") * pl.col("Organic_Carbon")).alias("Moisture_carbon"),
        (pl.col("Temperature_C") * pl.col("Rainfall_mm")).alias("Temp_rain"),
        (pl.col("Humidity") * pl.col("Soil_Moisture")).alias("Humid_moisture"),
        np.log1p(pl.col("Rainfall_mm")).alias("Log_rain"),
        np.log1p(pl.col("Soil_Moisture")).alias("Log_moisture"),
        (pl.col("Rainfall_mm") / (pl.col("Field_Area_hectare") + 0.01)).alias("Rain_per_area"),
        (pl.col("Previous_Irrigation_mm") / (pl.col("Field_Area_hectare") + 0.01)).alias("Previrr_per_area"),
        (pl.col("Soil_Moisture") / (pl.col("Rainfall_mm") + pl.col("Previous_Irrigation_mm") + 1.0)).alias("Moisture_per_water_input"),
        (pl.col("Soil_pH") * pl.col("Electrical_Conductivity") / (pl.col("Organic_Carbon") + 0.01)).alias("Soil_quality"),
        (pl.col("Soil_Moisture") / (pl.col("Temperature_C") * pl.col("Sunlight_Hours") + 1.0)).alias("Moisture_per_heat_light"),
        (pl.col("Humidity") * pl.col("Wind_Speed_kmh")).alias("Humid_wind"),
        (pl.col("Temperature_C") * pl.col("Sunlight_Hours")).alias("Heat_light"),
        (pl.col("Soil_Moisture") * pl.col("Rainfall_mm")).alias("Moisture_rain"),
        (pl.col("Humidity") * pl.col("Previous_Irrigation_mm")).alias("Humid_previrr"),
    )


def compute_target_encoding(train_frame: pl.DataFrame, col: str, smoothing: float = 10.0) -> pl.DataFrame:
    counts = train_frame.group_by(col).len().rename({"len": "count"})
    means = train_frame.group_by(col).agg(
        (pl.col(TARGET) == "High").cast(pl.Int64).mean().alias(f"{col}_te_High"),
        (pl.col(TARGET) == "Medium").cast(pl.Int64).mean().alias(f"{col}_te_Medium"),
        (pl.col(TARGET) == "Low").cast(pl.Int64).mean().alias(f"{col}_te_Low"),
    )
    global_high = (train_frame[TARGET] == "High").cast(pl.Int64).mean()
    global_medium = (train_frame[TARGET] == "Medium").cast(pl.Int64).mean()
    global_low = (train_frame[TARGET] == "Low").cast(pl.Int64).mean()
    stats = means.join(counts, on=col)
    return stats.with_columns(
        ((pl.col("count") * pl.col(f"{col}_te_High") + smoothing * global_high) / (pl.col("count") + smoothing)).alias(f"{col}_te_High"),
        ((pl.col("count") * pl.col(f"{col}_te_Medium") + smoothing * global_medium) / (pl.col("count") + smoothing)).alias(f"{col}_te_Medium"),
        ((pl.col("count") * pl.col(f"{col}_te_Low") + smoothing * global_low) / (pl.col("count") + smoothing)).alias(f"{col}_te_Low"),
    ).select(col, f"{col}_te_High", f"{col}_te_Medium", f"{col}_te_Low")


def compute_pairwise_target_encoding(train_frame: pl.DataFrame, col1: str, col2: str, smoothing: float = 20.0) -> pl.DataFrame:
    pair_col = f"{col1}_{col2}"
    counts = train_frame.group_by([col1, col2]).len().rename({"len": "count"})
    means = train_frame.group_by([col1, col2]).agg(
        (pl.col(TARGET) == "High").cast(pl.Int64).mean().alias(f"{pair_col}_te_High"),
        (pl.col(TARGET) == "Medium").cast(pl.Int64).mean().alias(f"{pair_col}_te_Medium"),
        (pl.col(TARGET) == "Low").cast(pl.Int64).mean().alias(f"{pair_col}_te_Low"),
    )
    global_high = (train_frame[TARGET] == "High").cast(pl.Int64).mean()
    global_medium = (train_frame[TARGET] == "Medium").cast(pl.Int64).mean()
    global_low = (train_frame[TARGET] == "Low").cast(pl.Int64).mean()
    stats = means.join(counts, on=[col1, col2])
    return stats.with_columns(
        ((pl.col("count") * pl.col(f"{pair_col}_te_High") + smoothing * global_high) / (pl.col("count") + smoothing)).alias(f"{pair_col}_te_High"),
        ((pl.col("count") * pl.col(f"{pair_col}_te_Medium") + smoothing * global_medium) / (pl.col("count") + smoothing)).alias(f"{pair_col}_te_Medium"),
        ((pl.col("count") * pl.col(f"{pair_col}_te_Low") + smoothing * global_low) / (pl.col("count") + smoothing)).alias(f"{pair_col}_te_Low"),
    ).select(col1, col2, f"{pair_col}_te_High", f"{pair_col}_te_Medium", f"{pair_col}_te_Low")


def build_encoding_tables(train_frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tables = {}
    for col in CATEGORICAL_COLS:
        tables[col] = compute_target_encoding(train_frame, col)
    for col1, col2 in PAIRWISE_TE_PAIRS:
        key = f"{col1}_{col2}"
        tables[key] = compute_pairwise_target_encoding(train_frame, col1, col2)
    return tables


def add_target_encodings(frame: pl.DataFrame, enc_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    for key, enc_df in enc_tables.items():
        join_cols = [c for c in enc_df.columns if not c.endswith("_te_High") and not c.endswith("_te_Medium") and not c.endswith("_te_Low")]
        frame = frame.join(enc_df, on=join_cols, how="left")
    return frame


THRESHOLD_FEATS = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
LOGIT_FEATS = ["logit_Low", "logit_Med", "logit_High"]

DOMAIN_COLS = [
    "ET_proxy", "Water_supply", "Water_budget", "Inv_moisture", "Temp_per_moisture",
    "Wind_per_moisture", "Inv_rainfall", "Heat_wind_per_moisture", "Drought_stress",
    "Moisture_comfort", "Moisture_rain_ratio", "Temp_humid_ratio", "Humid_temp_ratio",
    "Temp_moisture_diff", "Heat_light_per_rain", "Dry_wind_index", "Moisture_humid_ratio",
    "Moisture_previrr_ratio", "Moisture_carbon", "Temp_rain", "Humid_moisture",
    "Log_rain", "Log_moisture", "Rain_per_area", "Previrr_per_area",
    "Moisture_per_water_input", "Soil_quality", "Moisture_per_heat_light",
    "Humid_wind", "Heat_light", "Moisture_rain", "Humid_previrr",
]


def build_feature_frame(frame: pl.DataFrame, enc_tables: dict[str, pl.DataFrame] | None = None) -> tuple[pl.DataFrame, list[str]]:
    features = frame
    for column in (TARGET, ID_COLUMN):
        if column in features.columns:
            features = features.drop(column)

    features = add_threshold_features(features)
    features = add_logit_features(features)
    features = add_domain_features(features)

    te_cols = []
    if enc_tables is not None:
        features = add_target_encodings(features, enc_tables)
        for key in enc_tables:
            te_cols.extend(c for c in features.columns if c.startswith(key) and c.endswith(("_te_High", "_te_Medium", "_te_Low")))

    for col in features.columns:
        if features[col].dtype == pl.Float64:
            features = features.with_columns(pl.col(col).cast(pl.Float32))

    cat_cols = [c for c in features.columns if not features.schema[c].is_numeric()]
    return features, cat_cols


def to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def fit_predict_valid(train_frame: pl.DataFrame, valid_frame: pl.DataFrame, benchmark_name: str) -> pl.DataFrame:
    del benchmark_name

    enc_tables = build_encoding_tables(train_frame)

    x_train_pl, cat_cols_train = build_feature_frame(train_frame, enc_tables)
    x_valid_pl, _ = build_feature_frame(valid_frame, enc_tables)

    use_cols = [c for c in x_train_pl.columns if c in x_valid_pl.columns]
    cat_cols = [c for c in cat_cols_train if c in use_cols]

    x_train = to_pandas(x_train_pl[use_cols])
    x_valid = to_pandas(x_valid_pl[use_cols])
    y_train_int = [LABEL_MAP[v] for v in train_frame[TARGET].to_list()]
    y_valid_str = valid_frame[TARGET].to_list()
    y_valid_int = [LABEL_MAP[v] for v in y_valid_str]

    classes = np.unique(y_train_int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)
    cw_dict = dict(zip(classes, weights))
    sw = np.array([cw_dict[l] for l in y_train_int])

    oof_cb = np.zeros((len(y_train_int), 3))
    oof_lgb = np.zeros((len(y_train_int), 3))
    oof_xgb = np.zeros((len(y_train_int), 3))
    pred_cb = np.zeros((len(y_valid_str), 3))
    pred_lgb = np.zeros((len(y_valid_str), 3))
    pred_xgb = np.zeros((len(y_valid_str), 3))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (ti, vi) in enumerate(kf.split(x_train, y_train_int)):
        print(f"  Fold {fold+1}/{N_FOLDS}")
        Xtr = x_train.iloc[ti].copy()
        Xv = x_train.iloc[vi].copy()
        Xt = x_valid.copy()
        ytr = np.array(y_train_int)[ti]
        yv = np.array(y_train_int)[vi]

        cb = CatBoostClassifier(
            loss_function="MultiClass",
            class_weights=cw_dict,
            depth=6,
            learning_rate=0.05,
            iterations=2000,
            random_seed=RANDOM_SEED,
            thread_count=CPU_THREADS,
            allow_writing_files=False,
            verbose=False,
            od_type="Iter",
            od_wait=100,
            l2_leaf_reg=5,
        )
        cb.fit(Xtr, ytr, cat_features=cat_cols, eval_set=(Xv, yv), use_best_model=True)
        oof_cb[vi] = cb.predict_proba(Xv)
        pred_cb += cb.predict_proba(Xt) / N_FOLDS

        for c in cat_cols:
            Xtr[c] = Xtr[c].astype("category")
            Xv[c] = Xv[c].astype("category")
            Xt[c] = Xt[c].astype("category")

        import lightgbm
        lgb = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight=cw_dict,
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            reg_lambda=5,
            colsample_bytree=0.7,
            subsample=0.7,
        )
        lgb.fit(Xtr, ytr, eval_set=[(Xv, yv)],
                callbacks=[lightgbm.early_stopping(100, verbose=False), lightgbm.log_evaluation(0)])
        oof_lgb[vi] = lgb.predict_proba(Xv)
        pred_lgb += lgb.predict_proba(Xt) / N_FOLDS

        xgb_m = XGBClassifier(
            n_estimators=2000, max_depth=4, max_leaves=30, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=5, reg_lambda=5,
            max_bin=256, tree_method="hist", objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", random_state=RANDOM_SEED,
            n_jobs=-1, early_stopping_rounds=100, verbosity=0,
        )
        xgb_m.fit(Xtr, ytr, eval_set=[(Xv, yv)], sample_weight=sw[ti], verbose=False)
        oof_xgb[vi] = xgb_m.predict_proba(Xv)
        pred_xgb += xgb_m.predict_proba(Xt) / N_FOLDS

    def objective(trial):
        w_cb = trial.suggest_float("w_cb", 0.1, 3.0)
        w_lgb = trial.suggest_float("w_lgb", 0.1, 3.0)
        w_xgb = trial.suggest_float("w_xgb", 0.1, 3.0)
        oof_blend = (w_cb * oof_cb + w_lgb * oof_lgb + w_xgb * oof_xgb)
        p0 = trial.suggest_float("p0", 0.5, 2.0)
        p1 = trial.suggest_float("p1", 0.5, 2.0)
        p2 = trial.suggest_float("p2", 0.5, 2.0)
        oof_blend = oof_blend * np.array([p0, p1, p2])
        oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True)
        return balanced_accuracy_score(y_train_int, oof_blend.argmax(axis=1))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    print(f"  Optuna best: {study.best_value:.6f}")

    bp = study.best_params
    pred_blend = (bp["w_cb"] * pred_cb + bp["w_lgb"] * pred_lgb + bp["w_xgb"] * pred_xgb)
    pred_blend = pred_blend * np.array([bp["p0"], bp["p1"], bp["p2"]])
    pred_blend = pred_blend / pred_blend.sum(axis=1, keepdims=True)
    predictions = [LABEL_INVERSE[i] for i in pred_blend.argmax(axis=1)]

    return pl.DataFrame({ID_COLUMN: valid_frame[ID_COLUMN].to_list(), TARGET: predictions})


def fit_predict_test(train_frame: pl.DataFrame, test_frame: pl.DataFrame) -> pl.DataFrame:
    enc_tables = build_encoding_tables(train_frame)

    x_train_pl, cat_cols_train = build_feature_frame(train_frame, enc_tables)
    x_test_pl, _ = build_feature_frame(test_frame, enc_tables)

    use_cols = [c for c in x_train_pl.columns if c in x_test_pl.columns]
    cat_cols = [c for c in cat_cols_train if c in use_cols]

    x_train = to_pandas(x_train_pl[use_cols])
    x_test = to_pandas(x_test_pl[use_cols])
    y_train_int = [LABEL_MAP[v] for v in train_frame[TARGET].to_list()]

    classes = np.unique(y_train_int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)
    cw_dict = dict(zip(classes, weights))
    sw = np.array([cw_dict[l] for l in y_train_int])

    pred_cb = np.zeros((len(x_test), 3))
    pred_lgb = np.zeros((len(x_test), 3))
    pred_xgb = np.zeros((len(x_test), 3))

    cb_final = CatBoostClassifier(
        loss_function="MultiClass",
        class_weights=cw_dict,
        depth=6,
        learning_rate=0.05,
        iterations=2000,
        random_seed=RANDOM_SEED,
        thread_count=CPU_THREADS,
        allow_writing_files=False,
        verbose=False,
        l2_leaf_reg=5,
    )
    cb_final.fit(x_train, y_train_int, cat_features=cat_cols)
    pred_cb = cb_final.predict_proba(x_test)

    for c in cat_cols:
        x_train[c] = x_train[c].astype("category")
        x_test[c] = x_test[c].astype("category")

    lgb_final = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight=cw_dict,
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        random_seed=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        reg_lambda=5,
        colsample_bytree=0.7,
        subsample=0.7,
    )
    lgb_final.fit(x_train, y_train_int)
    pred_lgb = lgb_final.predict_proba(x_test)

    xgb_final = XGBClassifier(
        n_estimators=2000, max_depth=4, max_leaves=30, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=5, reg_lambda=5,
        max_bin=256, tree_method="hist", objective="multi:softprob",
        num_class=3, eval_metric="mlogloss", random_state=RANDOM_SEED,
        n_jobs=-1, verbosity=0,
    )
    xgb_final.fit(x_train, y_train_int, sample_weight=sw, verbose=False)
    pred_xgb = xgb_final.predict_proba(x_test)

    avg = (pred_cb + pred_lgb + pred_xgb) / 3.0
    predictions = [LABEL_INVERSE[i] for i in avg.argmax(axis=1)]

    return pl.DataFrame({ID_COLUMN: test_frame[ID_COLUMN].to_list(), TARGET: predictions})
