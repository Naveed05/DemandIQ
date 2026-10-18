"""Benchmark DemandIQ against simple daily-demand baselines on a prepared CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from demandiq.data import validate_sales_frame
from demandiq.forecasting import evaluate_forecast


PREDICTION_COLUMNS = [
    "date", "store_id", "product_id", "sales", "prediction",
    "naive_prediction", "seasonal_naive_prediction",
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
    clean = validate_sales_frame(frame)
    result = evaluate_forecast(clean, validation_days=validation_days)
    predictions = result.predictions.copy()

    model_metrics = _metrics(predictions["sales"], predictions["prediction"])
    naive_metrics = _metrics(predictions["sales"], predictions["naive_prediction"])
    seasonal_metrics = _metrics(
        predictions["sales"], predictions["seasonal_naive_prediction"]
    )
    metrics = {
        "random_forest": model_metrics,
        "naive_last_day": naive_metrics,
        "seasonal_naive_7_day": seasonal_metrics,
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
        "store_count": clean["store_id"].nunique(),
        "product_count": clean["product_id"].nunique(),
        "series_count": int(clean[["store_id", "product_id"]].drop_duplicates().shape[0]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared daily sales CSV")
    parser.add_argument(
        "--validation-days", type=int, default=28,
        help="Number of final calendar dates held out for evaluation (default: 28)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "benchmark",
        help="Directory for JSON metrics and CSV prediction reports",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    report, predictions, per_series = build_benchmark_report(
        frame, validation_days=args.validation_days
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    per_series.to_csv(args.output_dir / "per_series_metrics.csv", index=False)

    print(f"Benchmark reports written to: {args.output_dir}")
    print(json.dumps(report["metrics"], indent=2))
    print("No performance claim is made beyond this dataset and evaluation setup.")


if __name__ == "__main__":
    main()
