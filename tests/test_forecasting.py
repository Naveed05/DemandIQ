import pandas as pd
import pytest

from demandiq.forecasting import evaluate_forecast, forecast_demand
from demandiq.intermittent import croston_sba_forecast


def sales_history(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    quantities = [12 + (index % 7) + (index % 11 == 0) * 5 for index in range(days)]
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S1"] * days,
        "product_id": ["P1"] * days,
        "sales": quantities,
    })



def intermittent_sales_history(days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    quantities = [0.0] * days
    for index in range(8, days, 10):
        quantities[index] = 8.0 if (index // 10) % 2 else 12.0
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
    assert set(result["forecast_method"]) == {"random_forest"}
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

def test_adaptive_future_forecast_uses_croston_for_intermittent_demand():
    frame = intermittent_sales_history()
    result = forecast_demand(frame, horizon_days=5)
    expected = croston_sba_forecast(frame["sales"].to_numpy(dtype=float))

    assert set(result["demand_pattern"]) == {"intermittent"}
    assert set(result["forecast_method"]) == {"croston_sba"}
    assert result["prediction"].tolist() == pytest.approx([expected] * 5)


def test_adaptive_future_forecast_keeps_no_demand_skus_at_zero():
    frame = sales_history().assign(sales=0.0)

    result = forecast_demand(frame, horizon_days=4)

    assert set(result["demand_pattern"]) == {"no_demand"}
    assert set(result["forecast_method"]) == {"zero"}
    assert result["prediction"].eq(0).all()


def test_adaptive_future_forecast_selects_a_method_for_each_sku():
    smooth = sales_history(days=100)
    intermittent = intermittent_sales_history(days=100).assign(
        store_id="S2", product_id="P2"
    )
    no_demand = sales_history(days=100).assign(
        store_id="S3", product_id="P3", sales=0.0
    )
    frame = pd.concat([smooth, intermittent, no_demand], ignore_index=True)

    result = forecast_demand(frame, horizon_days=3)
    selected = result.drop_duplicates(["store_id", "product_id"]).set_index("product_id")

    assert selected.loc["P1", "demand_pattern"] == "smooth"
    assert selected.loc["P1", "forecast_method"] == "random_forest"
    assert selected.loc["P2", "demand_pattern"] == "intermittent"
    assert selected.loc["P2", "forecast_method"] == "croston_sba"
    assert selected.loc["P3", "demand_pattern"] == "no_demand"
    assert selected.loc["P3", "forecast_method"] == "zero"
    zero_predictions = result.loc[result["product_id"] == "P3", "prediction"]
    assert zero_predictions.eq(0).all()


def test_random_forest_strategy_preserves_all_series_model_option():
    frame = intermittent_sales_history(days=100)

    result = forecast_demand(frame, horizon_days=2, strategy="random_forest")

    assert set(result["demand_pattern"]) == {"intermittent"}
    assert set(result["forecast_method"]) == {"random_forest"}


def test_recursive_forecast_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy must be"):
        forecast_demand(sales_history(), horizon_days=2, strategy="unknown")
