import pandas as pd
import pytest

from demandiq.forecasting import evaluate_forecast


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
    assert {"mae", "rmse", "wape", "baseline_mae", "baseline_rmse", "baseline_wape"} <= set(result.metrics)
    assert result.metrics["mae"] >= 0
    assert result.metrics["rmse"] >= 0


def test_evaluation_rejects_too_short_holdout_history():
    with pytest.raises(ValueError, match="Not enough dates"):
        evaluate_forecast(sales_history(days=35), validation_days=10)


def test_evaluation_rejects_zero_validation_days():
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_forecast(sales_history(), validation_days=0)
