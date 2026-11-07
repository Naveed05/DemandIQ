import pandas as pd
import pytest

from demandiq.benchmark import build_benchmark_report, build_benchmark_summary


def daily_sales(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    values = [0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 15, 0, 0, 0]
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S1"] * days,
        "product_id": ["P1"] * days,
        "sales": (values * ((days + len(values) - 1) // len(values)))[:days],
    })


def test_benchmark_compares_model_and_three_baselines():
    report, predictions, per_series = build_benchmark_report(
        daily_sales(), validation_days=7
    )

    assert report["dataset_rows"] == 100
    assert report["series_count"] == 1
    assert report["validation_rows"] == 7
    assert set(report["metrics"]) == {
        "random_forest",
        "naive_last_day",
        "seasonal_naive_7_day",
        "croston_sba",
        "adaptive_selection",
    }
    for metric_values in report["metrics"].values():
        assert set(metric_values) == {"mae", "rmse", "wape"}
        assert all(value >= 0 for value in metric_values.values())
    assert len(predictions) == 7
    assert "croston_sba_prediction" in predictions.columns
    assert "adaptive_prediction" in predictions.columns
    assert len(per_series) == 1
    assert "random_forest_wape" in per_series.columns
    assert "seasonal_naive_7_day_wape" in per_series.columns
    assert "croston_sba_wape" in per_series.columns
    assert "selected_method" in per_series.columns
    assert {"demand_pattern", "adi", "cv2"}.issubset(per_series.columns)
    assert sum(report["demand_pattern_counts"].values()) == 1


def test_croston_predictions_use_only_observations_before_each_date():
    frame = daily_sales()
    report, predictions, _ = build_benchmark_report(frame, validation_days=7)
    first_validation_date = predictions["date"].min()
    expected_history = frame.loc[frame["date"] < first_validation_date, "sales"]

    from demandiq.intermittent import croston_sba_forecast

    first_prediction = predictions.loc[
        predictions["date"] == first_validation_date, "croston_sba_prediction"
    ].iloc[0]
    assert first_prediction == pytest.approx(croston_sba_forecast(expected_history))
    assert report["validation_rows"] == 7


def test_benchmark_rejects_duplicate_daily_observations():
    frame = pd.concat([daily_sales(), daily_sales().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_benchmark_report(frame, validation_days=7)


def test_benchmark_rejects_nonpositive_validation_window():
    with pytest.raises(ValueError, match="positive integer"):
        build_benchmark_report(daily_sales(), validation_days=0)


def test_benchmark_summary_identifies_best_method_and_comparison():
    report, _, _ = build_benchmark_report(daily_sales(), validation_days=7)

    summary = build_benchmark_summary(report)

    assert "# DemandIQ benchmark insights" in summary
    assert "| Random Forest |" in summary
    assert "| Croston SBA |" in summary
    assert "What performed best?" in summary
    assert "WAPE" in summary
    assert "best baseline" in summary.lower()
    assert "chronological holdout" in summary.lower()


def test_benchmark_summary_rejects_missing_or_invalid_metrics():
    with pytest.raises(ValueError, match="non-empty metrics"):
        build_benchmark_summary({})

    invalid = {
        "metrics": {
            "random_forest": {"mae": -1, "rmse": 1, "wape": 0.1},
            "naive_last_day": {"mae": 1, "rmse": 1, "wape": 0.1},
            "seasonal_naive_7_day": {"mae": 1, "rmse": 1, "wape": 0.1},
            "croston_sba": {"mae": 1, "rmse": 1, "wape": 0.1},
        }
    }
    with pytest.raises(ValueError, match="invalid mae"):
        build_benchmark_summary(invalid)


def test_adaptive_selector_uses_documented_pattern_policy():
    from demandiq.model_selection import select_method_for_pattern

    assert select_method_for_pattern("smooth") == "random_forest"
    assert select_method_for_pattern("erratic") == "random_forest"
    assert select_method_for_pattern("intermittent") == "croston_sba"
    assert select_method_for_pattern("lumpy") == "croston_sba"
    assert select_method_for_pattern("no_demand") == "zero"
    with pytest.raises(ValueError, match="Unsupported demand pattern"):
        select_method_for_pattern("unknown")


def test_adaptive_predictions_follow_training_pattern():
    report, predictions, per_series = build_benchmark_report(daily_sales(), validation_days=7)
    selected = per_series.iloc[0]["selected_method"]
    assert report["selected_method_counts"][selected] == 1
    if selected == "croston_sba":
        assert predictions["adaptive_prediction"].equals(predictions["croston_sba_prediction"])
    elif selected == "random_forest":
        assert predictions["adaptive_prediction"].equals(predictions["prediction"])
    else:
        assert predictions["adaptive_prediction"].eq(0).all()
