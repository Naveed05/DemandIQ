"""Reproducible benchmark helpers for DemandIQ daily sales forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demandiq.data import validate_sales_frame
from demandiq.forecasting import evaluate_forecast

PREDICTION_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "sales",
    "prediction",
    "naive_prediction",
    "seasonal_naive_prediction",
]


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate comparable metrics for one prediction series."""
    errors = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "wape": float(np.abs(errors).sum() / max(float(np.abs(actual).sum()), 1.0)),
    }


def build_benchmark_report(
    frame: pd.DataFrame, validation_days: int = 28
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate the model and simple baselines on the same chronological holdout."""
    if (
        isinstance(validation_days, bool)
        or not isinstance(validation_days, int)
        or validation_days < 1
    ):
        raise ValueError("validation_days must be a positive integer")

    clean = validate_sales_frame(frame)
    result = evaluate_forecast(clean, validation_days=validation_days)
    predictions = result.predictions.copy()

    metrics = {
        "random_forest": _metrics(predictions["sales"], predictions["prediction"]),
        "naive_last_day": _metrics(
            predictions["sales"], predictions["naive_prediction"]
        ),
        "seasonal_naive_7_day": _metrics(
            predictions["sales"], predictions["seasonal_naive_prediction"]
        ),
    }

    per_series_rows: list[dict[str, object]] = []
    for (store_id, product_id), group in predictions.groupby(
        ["store_id", "product_id"], sort=True
    ):
        row: dict[str, object] = {
            "store_id": str(store_id),
            "product_id": str(product_id),
            "validation_rows": len(group),
        }
        for name, column in (
            ("random_forest", "prediction"),
            ("naive_last_day", "naive_prediction"),
            ("seasonal_naive_7_day", "seasonal_naive_prediction"),
        ):
            for metric, value in _metrics(group["sales"], group[column]).items():
                row[f"{name}_{metric}"] = value
        per_series_rows.append(row)

    report: dict[str, object] = {
        "dataset_rows": len(clean),
        "store_count": int(clean["store_id"].nunique()),
        "product_count": int(clean["product_id"].nunique()),
        "series_count": int(
            clean[["store_id", "product_id"]].drop_duplicates().shape[0]
        ),
        "date_start": clean["date"].min().date().isoformat(),
        "date_end": clean["date"].max().date().isoformat(),
        "validation_days_requested": validation_days,
        "validation_rows": len(predictions),
        "evaluation_method": (
            "Chronological rolling one-step-ahead holdout; observed sales from "
            "earlier validation dates are used as lag inputs for later dates."
        ),
        "metrics": metrics,
        "interpretation_note": (
            "Lower MAE, RMSE, and WAPE are better. Results describe this dataset "
            "and evaluation setup only; they do not guarantee future business performance."
        ),
    }
    return report, predictions[PREDICTION_COLUMNS], pd.DataFrame(per_series_rows)
