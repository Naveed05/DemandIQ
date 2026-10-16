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


def test_no_order_when_stock_exceeds_reorder_point():
    result = reorder_recommendation(4, 2, 30, 3)
    assert result["order_quantity"] == 0
    assert result["reorder_required"] is False


def test_rejects_negative_inputs():
    with pytest.raises(ValueError):
        reorder_recommendation(-1, 2, 4)


def test_rejects_invalid_lead_time():
    with pytest.raises(ValueError):
        reorder_recommendation(3, -1, 4)
