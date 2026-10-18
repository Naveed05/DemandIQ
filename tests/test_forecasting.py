import pandas as pd
import pytest

from demandiq.forecasting import evaluate_forecast, forecast_demand


def sales_history(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    quantities = [12 + (index % 7) + (index % 11 == 0) * 5 for index in range(days)]
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S1"] * days,
        "product_id": ["P1"] * days,
        "sales": quantities,
    })


def test_evaluation_uses_chronological_holdout_and_returns_metrics():
    result = evaluate_forecast(sales_history(), validation_days=7)

    assert len(result.predictions) == 7
    assert result.predictions["date"].min() >= pd.Timestamp("2025-04-04")
    assert result.predictions["prediction"].ge(0).all()
    assert {
        "mae", "rmse", "wape", "baseline_mae", "baseline_rmse", "baseline_wape"
    } <= set(result.metrics)
    assert result.metrics["mae"] >= 0
    assert result.metrics["rmse"] >= 0


def test_evaluation_rejects_too_short_holdout_history():
    with pytest.raises(ValueError, match="Not enough dates"):
        evaluate_forecast(sales_history(days=35), validation_days=10)


def test_evaluation_rejects_zero_validation_days():
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_forecast(sales_history(), validation_days=0)


def test_recursive_forecast_returns_requested_future_horizon_per_series():
    first = sales_history()
    second = first.assign(
        store_id="S2",
        product_id="P2",
        sales=first["sales"] * 2,
    )
    history = pd.concat([first, second], ignore_index=True)

    result = forecast_demand(history, horizon_days=7)

    assert len(result) == 14
    assert set(result["forecast_step"]) == set(range(1, 8))
    assert result["prediction"].ge(0).all()
    assert result["prediction"].map(pd.notna).all()
    assert result.groupby(["store_id", "product_id"]).size().eq(7).all()
    assert result["date"].min() == pd.Timestamp("2025-04-11")
    assert result["date"].max() == pd.Timestamp("2025-04-17")


def test_recursive_forecast_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="positive integer"):
        forecast_demand(sales_history(), horizon_days=0)


def test_recursive_forecast_rejects_series_with_insufficient_history():
    short_series = sales_history(days=28)
    with pytest.raises(ValueError, match="at least 29 days"):
        forecast_demand(short_series, horizon_days=3)
