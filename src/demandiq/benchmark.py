"""Reproducible benchmark helpers and human-readable insights for DemandIQ."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demandiq.data import validate_sales_frame
from demandiq.forecasting import evaluate_forecast
from demandiq.intermittent import croston_sba_forecast

PREDICTION_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "sales",
    "prediction",
    "naive_prediction",
    "seasonal_naive_prediction",
    "croston_sba_prediction",
]


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate comparable metrics for one prediction series."""
    errors = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "wape": float(np.abs(errors).sum() / max(float(np.abs(actual).sum()), 1.0)),
    }


def _add_croston_predictions(clean: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Generate leakage-safe one-step SBA predictions using prior observations only."""
    result = predictions.copy()
    result["croston_sba_prediction"] = 0.0
    histories = {
        (str(store_id), str(product_id)): group.sort_values("date")
        for (store_id, product_id), group in clean.groupby(
            ["store_id", "product_id"], sort=True
        )
    }
    for key, indexes in result.groupby(["store_id", "product_id"], sort=True).groups.items():
        history = histories[(str(key[0]), str(key[1]))]
        for index in indexes:
            forecast_date = result.at[index, "date"]
            observed = history.loc[history["date"] < forecast_date, "sales"]
            result.at[index, "croston_sba_prediction"] = croston_sba_forecast(
                observed.to_numpy(dtype=float)
            )
    return result


def build_benchmark_report(
    frame: pd.DataFrame, validation_days: int = 28
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate the model and three baselines on the same chronological holdout."""
    if (
        isinstance(validation_days, bool)
        or not isinstance(validation_days, int)
        or validation_days < 1
    ):
        raise ValueError("validation_days must be a positive integer")

    clean = validate_sales_frame(frame)
    result = evaluate_forecast(clean, validation_days=validation_days)
    predictions = _add_croston_predictions(clean, result.predictions.copy())

    columns = {
        "random_forest": "prediction",
        "naive_last_day": "naive_prediction",
        "seasonal_naive_7_day": "seasonal_naive_prediction",
        "croston_sba": "croston_sba_prediction",
    }
    metrics = {
        name: _metrics(predictions["sales"], predictions[column])
        for name, column in columns.items()
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
        for name, column in columns.items():
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
            "earlier validation dates are used as lag inputs for later dates. "
            "Croston SBA is recalculated using observations strictly before each "
            "forecast date."
        ),
        "metrics": metrics,
        "interpretation_note": (
            "Lower MAE, RMSE, and WAPE are better. Results describe this dataset "
            "and evaluation setup only; they do not guarantee future business performance."
        ),
    }
    return report, predictions[PREDICTION_COLUMNS], pd.DataFrame(per_series_rows)


def build_benchmark_summary(report: dict[str, object]) -> str:
    """Create a concise Markdown interpretation of a benchmark report."""
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("report must contain a non-empty metrics mapping")

    metric_names = ("mae", "rmse", "wape")
    methods = (
        "random_forest",
        "naive_last_day",
        "seasonal_naive_7_day",
        "croston_sba",
    )
    for method in methods:
        if method not in metrics or not isinstance(metrics[method], dict):
            raise ValueError(f"report is missing metrics for {method}")
        for metric in metric_names:
            value = metrics[method].get(metric)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"report contains an invalid {metric} value for {method}")

    labels = {
        "random_forest": "Random Forest",
        "naive_last_day": "Naive (last day)",
        "seasonal_naive_7_day": "Seasonal naive (7-day)",
        "croston_sba": "Croston SBA",
    }
    lines = [
        "# DemandIQ benchmark insights",
        "",
        "## Dataset and evaluation",
        "",
        f"- Rows: {report.get('dataset_rows', 'n/a'):,}",
        f"- Stores / location proxies: {report.get('store_count', 'n/a')}",
        f"- Products: {report.get('product_count', 'n/a')}",
        f"- Store-product series: {report.get('series_count', 'n/a')}",
        f"- Date range: {report.get('date_start', 'n/a')} to {report.get('date_end', 'n/a')}",
        f"- Validation rows: {report.get('validation_rows', 'n/a'):,}",
        f"- Requested validation days: {report.get('validation_days_requested', 'n/a')}",
        "",
        str(report.get("evaluation_method", "")),
        "",
        "## Overall metrics",
        "",
        "| Method | MAE (units) | RMSE (units) | WAPE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in methods:
        values = metrics[method]
        lines.append(
            f"| {labels[method]} | {values['mae']:.3f} | {values['rmse']:.3f} "
            f"| {values['wape']:.2%} |"
        )

    lines.extend(["", "## What performed best?", ""])
    for metric in metric_names:
        winner = min(methods, key=lambda method: metrics[method][metric])
        formatted = (
            f"{metrics[winner][metric]:.2%}"
            if metric == "wape"
            else f"{metrics[winner][metric]:.3f}"
        )
        lines.append(f"- **{metric.upper()}**: {labels[winner]} ({formatted}).")

    best_baseline = min(
        ("naive_last_day", "seasonal_naive_7_day", "croston_sba"),
        key=lambda method: metrics[method]["wape"],
    )
    baseline_wape = metrics[best_baseline]["wape"]
    model_wape = metrics["random_forest"]["wape"]
    if baseline_wape > 0:
        change = (baseline_wape - model_wape) / baseline_wape * 100
        if change > 0:
            comparison = f"Random Forest WAPE is {change:.1f}% lower than the best baseline ({labels[best_baseline]})."
        elif change < 0:
            comparison = f"Random Forest WAPE is {abs(change):.1f}% higher than the best baseline ({labels[best_baseline]})."
        else:
            comparison = f"Random Forest and the best baseline ({labels[best_baseline]}) have equal WAPE."
    else:
        comparison = (
            "The best baseline has zero WAPE, so a relative WAPE comparison is undefined."
        )
    lines.extend(["", comparison, ""])
    lines.extend(
        [
            "## Interpretation and limitations",
            "",
            str(report.get("interpretation_note", "")),
            "",
            (
                "This report is specific to the supplied dataset, preprocessing choices, "
                "and chronological holdout. It is not a guarantee of future accuracy. "
                "Review data quality and per-series metrics before drawing business conclusions."
            ),
            "",
        ]
    )
    return "\n".join(lines)
