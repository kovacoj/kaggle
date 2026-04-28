from __future__ import annotations

from collections import Counter

import polars as pl


DESCRIPTION = "majority-class / train-mean baseline"


def fit_model(train_df: pl.DataFrame, config) -> dict[str, object]:
    target_column = config.target_column

    if config.task_type == "classification":
        majority_label = Counter(train_df[target_column].to_list()).most_common(1)[0][0]
        return {"prediction": majority_label}

    mean_value = float(train_df[target_column].mean())
    return {"prediction": mean_value}


def predict(model: dict[str, object], frame: pl.DataFrame, config) -> pl.Series:
    return pl.Series(config.prediction_column, [model["prediction"]] * frame.height)
