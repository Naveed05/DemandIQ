"""Baseline forecasting, chronological evaluation, and recursive future forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from demandiq.features import make_features

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


def _new_model() -> RandomForestRegressor:
    """Build a reproducible model configuration shared by evaluation and forecasting."""
    return RandomForestRegressor(
        n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1
    )


def _score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = actual_values - predicted_values
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "wape": float(np.abs(errors).sum() / max(float(np.abs(actual_values).sum()), 1.0)),
    }


def evaluate_forecast(frame: pd.DataFrame, validation_days: int = 28) -> EvaluationResult:
    """Fit on earlier dates and evaluate on the final N unique dates.

    This is a rolling one-step-ahead holdout: prior observed validation sales
    are available to construct lag features for later validation dates.
    """
    if isinstance(validation_days, bool) or not isinstance(validation_days, int) or validation_days < 1:
        raise ValueError("validation_days must be a positive integer")
    data = make_features(frame).dropna(subset=FEATURE_COLUMNS).copy()
    dates = sorted(data["date"].unique())
    if len(dates) <= validation_days:
        raise ValueError("Not enough dates for the requested chronological holdout")

    cutoff = dates[-validation_days]
    train = data[data["date"] < cutoff]
    valid = data[data["date"] >= cutoff]
    if train.empty or valid.empty:
        raise ValueError("Chronological split produced an empty train or validation set")

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS], train["sales"])
    predicted = np.maximum(model.predict(valid[FEATURE_COLUMNS]), 0.0)
    result = valid[["date", "store_id", "product_id", "sales"]].copy()
    result["prediction"] = predicted
    result["baseline_prediction"] = valid["lag_7"].fillna(valid["lag_1"]).to_numpy()
    metrics = _score(valid["sales"], predicted)
    metrics.update({
        f"baseline_{key}": value
        for key, value in _score(
            valid["sales"], result["baseline_prediction"].to_numpy()
        ).items()
    })
    metrics["validation_days"] = float(validation_days)
    return EvaluationResult(result.reset_index(drop=True), metrics, model)


def forecast_demand(frame: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """Train on all available history and recursively forecast each store/SKU.

    Each predicted day is appended to that series' history before the next
    day's lag and rolling features are calculated. Future actual sales are
    never used. The input must contain complete daily series, and enough history
    to create at least one training row with a 28-day lag.
    """
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer")

    featured = make_features(frame)
    train = featured.dropna(subset=FEATURE_COLUMNS).copy()
    if train.empty:
        raise ValueError("Not enough history to train a forecast model; at least 29 days are needed")

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS], train["sales"])
    forecast_rows: list[dict[str, object]] = []

    for (store_id, product_id), group in featured.groupby(
        ["store_id", "product_id"], sort=False
    ):
        history = group.sort_values("date")["sales"].astype(float).tolist()
        last_date = group["date"].max()
        for step in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=step)
            recent_seven = np.asarray(history[-7:], dtype=float)
            row = {
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "month": forecast_date.month,
                "is_weekend": int(forecast_date.dayofweek in (5, 6)),
                "lag_1": history[-1],
                "lag_7": history[-7],
                "lag_14": history[-14],
                "lag_28": history[-28],
                "rolling_mean_7": float(recent_seven.mean()),
                "rolling_std_7": float(recent_seven.std(ddof=1)),
            }
            prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURE_COLUMNS))[0]))
            forecast_rows.append({
                "date": forecast_date,
                "store_id": store_id,
                "product_id": product_id,
                "prediction": prediction,
                "forecast_step": step,
            })
            history.append(prediction)

    return pd.DataFrame(forecast_rows).sort_values(
        ["date", "store_id", "product_id"]
    ).reset_index(drop=True)
