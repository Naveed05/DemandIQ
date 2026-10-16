"""Baseline forecasting and chronological model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dem andiq.features import make_features

FEATURE_COLUMNS = [
    "day_of_week", "day_of_month", "month", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_std_7",
]


@dataclass
class EvaluationResult:
    """Model evaluation output with predictions and summary metrics."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    model: RandomForestRegressor


def _score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual.to_numpy() - predicted).sum() / max(actual.sum(), 1.0)),
    }


def evaluate_forecast(frame: pd.DataFrame, validation_days: int = 28) -> EvaluationResult:
    """Fit on earlier dates and evaluate on the final N unique dates.

    This is a one-step-ahead holdout evaluation. Lag features in the validation
    rows use observed sales from prior dates, which is appropriate for rolling
    daily forecasting where the previous day's sales are available.
    """
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")
    data = make_features(frame).dropna(subset=FEATURE_COLUMNS).copy()
    dates = sorted(data["date"].unique())
    if len(dates) <= validation_days:
        raise ValueError("Not enough dates for the requested chronological holdout")

    cutoff = dates[-validation_days]
    train = data[data["date"] < cutoff]
    valid = data[data["date"] >= cutoff]
    if train.empty or valid.empty:
        raise ValueError("Chronological split produced an empty train or validation set")

    model = RandomForestRegressor(
        n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURE_COLUMNS], train["sales"])
    predicted = np.maximum(model.predict(valid[FEATURE_COLUMNS]), 0.0)
    result = valid[["date", "store_id", "product_id", "sales"]].copy()
    result["prediction"] = predicted
    result["baseline_prediction"] = valid["lag_7"].fillna(valid["lag_1"]).to_numpy()
    metrics = _score(valid["sales"], predicted)
    metrics.update({f"baseline_{key}": value for key, value in _score(
        valid["sales"], result["baseline_prediction"].to_numpy()
    ).items()})
    metrics["validation_days"] = float(validation_days)
    return EvaluationResult(result.reset_index(drop=True), metrics, model)
