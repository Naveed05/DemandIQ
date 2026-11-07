import numpy as np
import pytest

from demandiq.intermittent import croston_sba_forecast


def test_all_zero_history_forecasts_zero():
    assert croston_sba_forecast([0, 0, 0, 0]) == 0.0


def test_constant_daily_demand_applies_sba_bias_correction():
    forecast = croston_sba_forecast([10, 10, 10, 10], alpha=0.1)
    assert forecast == pytest.approx(9.5)


def test_intermittent_history_returns_finite_nonnegative_forecast():
    forecast = croston_sba_forecast([0, 0, 12, 0, 0, 0, 8, 0, 0])
    assert np.isfinite(forecast)
    assert forecast > 0


def test_rejects_negative_or_non_finite_history():
    with pytest.raises(ValueError, match="non-negative"):
        croston_sba_forecast([1, -1, 0])
    with pytest.raises(ValueError, match="finite"):
        croston_sba_forecast([1, np.nan, 0])


def test_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="interval"):
        croston_sba_forecast([1, 0, 2], alpha=0)
