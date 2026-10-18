import pandas as pd
import pytest

from scripts.benchmark import build_benchmark_report


def daily_sales(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S1"] * days,
        "product_id": ["P1"] * days,
        "sales": [10 + index % 7 for index in range(days)],
    })


def test_benchmark_compares_model_and_two_baselines():
    report, predictions, per_series = build_benchmark_report(
        daily_sales(), validation_days=7
    )

    assert report["dataset_rows"] == 100
    assert report["series_count"] == 1
    assert report["validation_rows"] == 7
    assert set(report["metrics"]) == {
        "random_forest", "naive_last_day", "seasonal_naive_7_day"
    }
    for metric_values in report["metrics"].values():
        assert set(metric_values) == {"mae", "rmse", "wape"}
        assert all(value >= 0 for value in metric_values.values())
    assert len(predictions) == 7
    assert len(per_series) == 1
    assert "random_forest_wape" in per_series.columns
    assert "seasonal_naive_7_day_wape" in per_series.columns


def test_benchmark_rejects_duplicate_daily_observations():
    frame = pd.concat([daily_sales(), daily_sales().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_benchmark_report(frame, validation_days=7)


def test_benchmark_rejects_nonpositive_validation_window():
    with pytest.raises(ValueError, match="positive integer"):
        build_benchmark_report(daily_sales(), validation_days=0)
