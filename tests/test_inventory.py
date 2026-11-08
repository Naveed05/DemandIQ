import math

import pytest

from demandiq.inventory import reorder_recommendation


def test_recommends_order_when_stock_is_below_reorder_point():
    result = reorder_recommendation(
        average_daily_demand=10, lead_time_days=5, current_stock=20, safety_stock_days=3
    )
    assert result["lead_time_demand"] == 50
    assert result["safety_stock"] == 30
    assert result["reorder_point"] == 80
    assert result["order_quantity"] == 60
    assert result["reorder_required"] is True
    assert result["stockout_risk"] == "High"


def test_no_order_when_stock_exceeds_reorder_point():
    result = reorder_recommendation(4, 2, 30, 3)
    assert result["order_quantity"] == 0
    assert result["reorder_required"] is False
    assert result["stockout_risk"] == "Low"


def test_variability_increases_safety_stock():
    stable = reorder_recommendation(10, 4, 100, 2, demand_std_dev=0)
    variable = reorder_recommendation(10, 4, 100, 2, demand_std_dev=3)
    assert variable["safety_stock"] > stable["safety_stock"]
    assert variable["variability_buffer"] == pytest.approx(9.9)


def test_days_of_cover_and_medium_risk_are_reported():
    result = reorder_recommendation(10, 5, 70, 3)
    assert result["days_of_cover"] == 7
    assert result["stockout_risk"] == "Medium"


@pytest.mark.parametrize("kwargs", [
    {"average_daily_demand": -1, "lead_time_days": 2, "current_stock": 4},
    {"average_daily_demand": 1, "lead_time_days": 2, "current_stock": math.inf},
    {"average_daily_demand": 1, "lead_time_days": -1, "current_stock": 4},
    {"average_daily_demand": 1, "lead_time_days": 2, "current_stock": 4, "demand_std_dev": -1},
])
def test_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        reorder_recommendation(**kwargs)


def test_zero_demand_does_not_trigger_replenishment():
    result = reorder_recommendation(
        average_daily_demand=0,
        lead_time_days=7,
        current_stock=0,
        safety_stock_days=3,
    )
    assert result["lead_time_demand"] == 0
    assert result["reorder_point"] == 0
    assert result["order_quantity"] == 0
    assert result["days_of_cover"] == -1
    assert result["stockout_risk"] == "Low"
    assert result["reorder_required"] is False


def test_zero_lead_time_has_no_variability_buffer():
    result = reorder_recommendation(
        average_daily_demand=8,
        lead_time_days=0,
        current_stock=0,
        demand_std_dev=4,
        service_level_z=2,
    )
    assert result["lead_time_demand"] == 0
    assert result["variability_buffer"] == 0
    assert result["reorder_point"] == 0
    assert result["order_quantity"] == 0
    assert result["reorder_required"] is True
