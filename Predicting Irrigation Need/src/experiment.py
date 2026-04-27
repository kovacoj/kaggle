from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import optuna

from benchmark import ID_COLUMN, TARGET

optuna.logging.set_verbosity(optuna.logging.WARNING)

APPROACH = "xgb_cb_v12"
DESCRIPTION = "XGB+CB blend, digit features, logit features, bool thresholds, 5-fold OTE, Optuna weight tuning"

RANDOM_SEED = 42
CPU_THREADS = -1

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

NUMERICAL_COLS = [
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


def add_boolean_thresholds(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("Soil_Moisture") < 25).cast(pl.Int8).alias("soil_lt_25"),
        (pl.col("Temperature_C") > 30).cast(pl.Int8).alias("temp_gt_30"),
        (pl.col("Rainfall_mm") < 300).cast(pl.Int8).alias("rain_lt_300"),
        (pl.col("Wind_Speed_kmh") > 10).cast(pl.Int8).alias("wind_gt_10"),
    )


def add_logit_features(frame: pl.DataFrame) -> pl.DataFrame:
    soil_lt_25 = frame["soil_lt_25"].to_numpy().astype(np.float64)
    temp_gt_30 = frame["temp_gt_30"].to_numpy().astype(np.float64)
    rain_lt_300 = frame["rain_lt_300"].to_numpy().astype(np.float64)
    wind_gt_10 = frame["wind_gt_10"].to_numpy().astype(np.float64)
    cgs = frame["Crop_Growth_Stage"].to_list()
    mu = frame["Mulching_Used"].to_list()
    cgs_flowering = np.array([1.0 if v == "Flowering" else 0.0 for v in cgs])
    cgs_harvest = np.array([1.0 if v == "Harvest" else 0.0 for v in cgs])
    cgs_sowing = np.array([1.0 if v == "Sowing" else 0.0 for v in cgs])
    cgs_vegetative = np.array([1.0 if v == "Vegetative" else 0.0 for v in cgs])
    mu_no = np.array([1.0 if v == "No" else 0.0 for v in mu])
    mu_yes = np.array([1.0 if v == "Yes" else 0.0 for v in mu])

    logit_low = (16.3173
        + (-11.0237 * soil_lt_25) + (-5.8559 * temp_gt_30)
        + (-10.85 * rain_lt_300) + (-5.8284 * wind_gt_10)
        + (-5.4155 * cgs_flowering) + (5.5073 * cgs_harvest)
        + (5.2299 * cgs_sowing) + (-5.4617 * cgs_vegetative)
        + (-3.0014 * mu_no) + (2.8613 * mu_yes))

    logit_med = (4.6524
        + (0.329 * soil_lt_25) + (-0.0204 * temp_gt_30)
        + (0.1542 * rain_lt_300) + (0.0841 * wind_gt_10)
        + (0.3586 * cgs_flowering) + (-0.1348 * cgs_harvest)
        + (-0.3547 * cgs_sowing) + (0.3334 * cgs_vegetative)
        + (0.1883 * mu_no) + (0.0142 * mu_yes))

    logit_high = (-20.9697
        + (10.6947 * soil_lt_25) + (5.8763 * temp_gt_30)
        + (10.6958 * rain_lt_300) + (5.7444 * wind_gt_10)
        + (5.0569 * cgs_flowering) + (-5.3725 * cgs_harvest)
        + (-4.8752 * cgs_sowing) + (5.1283 * cgs_vegetative)
        + (2.8131 * mu_no) + (-2.8755 * mu_yes))

    return frame.with_columns(
        pl.Series("logit_Low", logit_low, dtype=pl.Float32),
        pl.Series("logit_Med", logit_med, dtype=pl.Float32),
        pl.Series("logit_High", logit_high, dtype=pl.Float32),
    )


def add_digit_features(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for c in NUMERICAL_COLS:
        for k in range(-4, 4):
            exprs.append((pl.col(c) // (10**k) % 10).cast(pl.Int8).alias(f"{c}_d{k}"))
    return frame.with_columns(exprs)


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
        (pl.col("Soil_pH") * pl.col("Electrical_Conductivity") / (pl.col("Organic_Carbon") + 0.01)).alias("Soil_quality_index"),
        (pl.col("Soil_Moisture") / (pl.col("Rainfall_mm") + pl.col("Previous_Irrigation_mm") + 1.0)).alias("Moisture_per_water_input"),
    )


class OrderedTE:
    def __init__(self, a: float = 1.0):
        self.a = a

    def fit(self, train_df: pd.DataFrame, cat_cols: list[str], target_col: str = TARGET):
        self.cat_cols = cat_cols
        self.classes_ = sorted(train_df[target_col].unique())
        self.global_prior_ = train_df[target_col].value_counts(normalize=True).sort_index().values
        self.stats_ = {}
        for c in cat_cols:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y = (train_df[target_col] == cls).astype(int)
                grp = train_df[[c]].copy()
                grp["y"] = y.values
                cum_cnt = grp.groupby(c, observed=False)["y"].cumcount()
                cum_sum = grp.groupby(c, observed=False)["y"].cumsum() - grp["y"]
                prior = self.global_prior_[k]
                te = (cum_sum + self.a * prior) / (cum_cnt + self.a)
                train_df[f"{c}_TE_cls{cls}"] = te.values
                agg = grp.groupby(c, observed=False)["y"].agg(count="count", total="sum").reset_index()
                agg.columns = [c, f"{c}_n_{cls}", f"{c}_s_{cls}"]
                stats_list.append(agg)
            from functools import reduce
            self.stats_[c] = reduce(lambda l, r: l.merge(r, on=c, how="outer"), stats_list)
        return train_df

    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        for c in self.cat_cols:
            test_df = test_df.merge(self.stats_[c], on=c, how="left")
            for k, cls in enumerate(self.classes_):
                n_col, s_col = f"{c}_n_{cls}", f"{c}_s_{cls}"
                prior = self.global_prior_[k]
                test_df[f"{c}_TE_cls{cls}"] = (
                    (test_df[s_col].fillna(0) + self.a * prior) /
                    (test_df[n_col].fillna(0) + self.a)
                )
                test_df.drop([n_col, s_col], axis=1, inplace=True)
        return test_df


BOOL_FEATS = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
LOGIT_FEATS = ["logit_Low", "logit_Med", "logit_High"]
DOMAIN_FEATS = ["ET_proxy", "Water_supply", "Water_budget", "Inv_moisture", "Temp_per_moisture",
                "Wind_per_moisture", "Inv_rainfall", "Heat_wind_per_moisture", "Drought_stress",
                "Moisture_comfort", "Moisture_rain_ratio", "Temp_humid_ratio", "Humid_temp_ratio",
                "Temp_moisture_diff", "Heat_light_per_rain", "Dry_wind_index", "Moisture_humid_ratio",
                "Moisture_previrr_ratio", "Moisture_carbon", "Temp_rain", "Humid_moisture",
                "Log_rain", "Log_moisture", "Rain_per_area", "Previrr_per_area",
                "Soil_quality_index", "Moisture_per_water_input"]


def _build_features_polars(frame: pl.DataFrame) -> pl.DataFrame:
    features = frame
    for column in (TARGET, ID_COLUMN):
        if column in features.columns:
            features = features.drop(column)
    features = add_boolean_thresholds(features)
    features = add_logit_features(features)
    features = add_digit_features(features)
    features = add_domain_features(features)
    for col in features.columns:
        if features[col].dtype == pl.Float64:
            features = features.with_columns(pl.col(col).cast(pl.Float32))
    return features


def _remove_constant_cols(train: pl.DataFrame, test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    drop = [c for c in train.columns if train[c].n_unique() <= 1]
    train = train.drop(drop)
    test = test.drop(drop)
    return train, test, drop


def _align_columns(train: pl.DataFrame, test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    for c in train_cols - test_cols:
        test = test.with_columns(pl.lit(0).cast(train[c].dtype).alias(c))
    for c in test_cols - train_cols:
        train = train.with_columns(pl.lit(0).cast(test[c].dtype).alias(c))
    test = test.select(train.columns)
    return train, test


def _get_feature_lists(x_train_pl: pl.DataFrame) -> tuple[list[str], list[str]]:
    digit_cols = [c for c in x_train_pl.columns if any(c.startswith(n + "_d") for n in NUMERICAL_COLS)]
    te_cat_cols = CATEGORICAL_COLS + BOOL_FEATS + digit_cols
    te_feat_names = [f"{c}_TE_cls{cls}" for c in te_cat_cols for cls in ["Low", "Medium", "High"]]
    use_features = NUMERICAL_COLS + BOOL_FEATS + LOGIT_FEATS + DOMAIN_FEATS + digit_cols + te_feat_names
    return te_cat_cols, use_features


def fit_predict_valid(train_frame: pl.DataFrame, valid_frame: pl.DataFrame, benchmark_name: str) -> pl.DataFrame:
    del benchmark_name

    x_train_pl = _build_features_polars(train_frame)
    x_valid_pl = _build_features_polars(valid_frame)
    x_train_pl, x_valid_pl, dropped = _remove_constant_cols(x_train_pl, x_valid_pl)
    x_train_pl, x_valid_pl = _align_columns(x_train_pl, x_valid_pl)

    te_cat_cols, use_features = _get_feature_lists(x_train_pl)

    x_train_pd = x_train_pl.to_pandas()
    x_valid_pd = x_valid_pl.to_pandas()
    y_train_int = [LABEL_MAP[v] for v in train_frame[TARGET].to_list()]
    y_valid_str = valid_frame[TARGET].to_list()
    y_valid_int = [LABEL_MAP[v] for v in y_valid_str]

    classes = np.unique(y_train_int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)
    cw_dict = dict(zip(classes, weights))
    sw = np.array([cw_dict[l] for l in y_train_int])

    oof_cb = np.zeros((len(train_frame), 3))
    oof_xgb = np.zeros((len(train_frame), 3))
    pred_cb = np.zeros((len(valid_frame), 3))
    pred_xgb = np.zeros((len(valid_frame), 3))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for fold, (ti, vi) in enumerate(kf.split(x_train_pd, y_train_int)):
        print(f"  Fold {fold+1}/5")
        Xtr = x_train_pd.iloc[ti].reset_index(drop=True).copy()
        Xv = x_train_pd.iloc[vi].reset_index(drop=True).copy()
        Xt = x_valid_pd.copy()
        ytr = np.array(y_train_int)[ti]
        yv = np.array(y_train_int)[vi]

        Xtr[TARGET] = ytr
        te = OrderedTE(a=1)
        Xtr = te.fit(Xtr, cat_cols=te_cat_cols, target_col=TARGET)
        Xv = te.transform(Xv)
        Xt = te.transform(Xt)
        Xtr.drop(TARGET, axis=1, inplace=True)

        avail = [f for f in use_features if f in Xtr.columns]
        Xtr_f = Xtr[avail].values
        Xv_f = Xv[avail].values
        Xt_f = Xt[avail].values

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
        cb.fit(Xtr_f, ytr, eval_set=(Xv_f, yv), use_best_model=True)
        oof_cb[vi] = cb.predict_proba(Xv_f)
        pred_cb += cb.predict_proba(Xt_f) / 5

        xgb_m = XGBClassifier(
            n_estimators=2000, max_depth=4, max_leaves=30, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=5, reg_lambda=5,
            max_bin=10000, tree_method="hist", objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", random_state=RANDOM_SEED,
            n_jobs=-1, early_stopping_rounds=100, verbosity=0,
        )
        xgb_m.fit(Xtr_f, ytr, eval_set=[(Xv_f, yv)], sample_weight=sw[ti], verbose=False)
        oof_xgb[vi] = xgb_m.predict_proba(Xv_f)
        pred_xgb += xgb_m.predict_proba(Xt_f) / 5

    oof_avg = (oof_cb + oof_xgb) / 2.0
    cv_before = balanced_accuracy_score(y_train_int, oof_avg.argmax(axis=1))
    print(f"  OOF BA before tuning: {cv_before:.6f}")

    def objective(trial):
        w0 = trial.suggest_float("w0", 0.5, 2.0)
        w1 = trial.suggest_float("w1", 0.5, 2.0)
        w2 = trial.suggest_float("w2", 0.5, 8.0)
        adj = oof_avg * np.array([w0, w1, w2])
        adj = adj / adj.sum(axis=1, keepdims=True)
        return balanced_accuracy_score(y_train_int, adj.argmax(axis=1))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)
    best_w = np.array([study.best_params["w0"], study.best_params["w1"], study.best_params["w2"]])
    print(f"  Optuna best weights: {best_w}, OOF BA: {study.best_value:.6f}")

    pred_avg = (pred_cb + pred_xgb) / 2.0
    adjusted = pred_avg * best_w
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    predictions = [LABEL_INVERSE[i] for i in adjusted.argmax(axis=1)]

    return pl.DataFrame({ID_COLUMN: valid_frame[ID_COLUMN].to_list(), TARGET: predictions})


def fit_predict_test(train_frame: pl.DataFrame, test_frame: pl.DataFrame) -> pl.DataFrame:
    x_train_pl = _build_features_polars(train_frame)
    x_test_pl = _build_features_polars(test_frame)
    x_train_pl, x_test_pl, dropped = _remove_constant_cols(x_train_pl, x_test_pl)
    x_train_pl, x_test_pl = _align_columns(x_train_pl, x_test_pl)

    te_cat_cols, use_features = _get_feature_lists(x_train_pl)

    x_train_pd = x_train_pl.to_pandas()
    x_test_pd = x_test_pl.to_pandas()
    y_train_int = [LABEL_MAP[v] for v in train_frame[TARGET].to_list()]

    classes = np.unique(y_train_int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)
    cw_dict = dict(zip(classes, weights))
    sw = np.array([cw_dict[l] for l in y_train_int])

    oof_cb = np.zeros((len(train_frame), 3))
    oof_xgb = np.zeros((len(train_frame), 3))
    pred_cb = np.zeros((len(test_frame), 3))
    pred_xgb = np.zeros((len(test_frame), 3))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for fold, (ti, vi) in enumerate(kf.split(x_train_pd, y_train_int)):
        print(f"  Fold {fold+1}/5")
        Xtr = x_train_pd.iloc[ti].reset_index(drop=True).copy()
        Xv = x_train_pd.iloc[vi].reset_index(drop=True).copy()
        Xt = x_test_pd.copy()
        ytr = np.array(y_train_int)[ti]
        yv = np.array(y_train_int)[vi]

        Xtr[TARGET] = ytr
        te = OrderedTE(a=1)
        Xtr = te.fit(Xtr, cat_cols=te_cat_cols, target_col=TARGET)
        Xv = te.transform(Xv)
        Xt = te.transform(Xt)
        Xtr.drop(TARGET, axis=1, inplace=True)

        avail = [f for f in use_features if f in Xtr.columns]
        Xtr_f = Xtr[avail].values
        Xv_f = Xv[avail].values
        Xt_f = Xt[avail].values

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
            l2_leaf_reg=5,
        )
        cb.fit(Xtr_f, ytr, eval_set=(Xv_f, yv), use_best_model=True)
        oof_cb[vi] = cb.predict_proba(Xv_f)
        pred_cb += cb.predict_proba(Xt_f) / 5

        xgb_m = XGBClassifier(
            n_estimators=2000, max_depth=4, max_leaves=30, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=5, reg_lambda=5,
            max_bin=10000, tree_method="hist", objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", random_state=RANDOM_SEED,
            n_jobs=-1, early_stopping_rounds=100, verbosity=0,
        )
        xgb_m.fit(Xtr_f, ytr, eval_set=[(Xv_f, yv)], sample_weight=sw[ti], verbose=False)
        oof_xgb[vi] = xgb_m.predict_proba(Xv_f)
        pred_xgb += xgb_m.predict_proba(Xt_f) / 5

    oof_avg = (oof_cb + oof_xgb) / 2.0

    def objective(trial):
        w0 = trial.suggest_float("w0", 0.5, 2.0)
        w1 = trial.suggest_float("w1", 0.5, 2.0)
        w2 = trial.suggest_float("w2", 0.5, 8.0)
        adj = oof_avg * np.array([w0, w1, w2])
        adj = adj / adj.sum(axis=1, keepdims=True)
        return balanced_accuracy_score(y_train_int, adj.argmax(axis=1))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)
    best_w = np.array([study.best_params["w0"], study.best_params["w1"], study.best_params["w2"]])

    pred_avg = (pred_cb + pred_xgb) / 2.0
    adjusted = pred_avg * best_w
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    predictions = [LABEL_INVERSE[i] for i in adjusted.argmax(axis=1)]

    return pl.DataFrame({ID_COLUMN: test_frame[ID_COLUMN].to_list(), TARGET: predictions})
