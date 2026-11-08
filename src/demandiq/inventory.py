"""Interpretable, variability-aware inventory replenishment calculations."""

from __future__ import annotations

import math


def reorder_recommendation(
    average_daily_demand: float,
    lead_time_days: int,
    current_stock: float,
    safety_stock_days: float = 3.0,
    demand_std_dev: float = 0.0,
    service_level_z: float = 1.65,
) -> dict[str, float | int | bool | str]:
    """Estimate reorder point and order-up-to quantity.

    Safety stock combines a transparent days-of-demand buffer with a
    variability buffer (z * daily standard deviation * sqrt(lead time)).
    The z value is an approximate service-level multiplier, not a calibrated
    guarantee. Order quantity raises inventory to lead-time demand plus safety
    stock; no pack-size, on-order, expiry, or supplier-MOQ data is assumed.
    """
    values = [average_daily_demand, current_stock, safety_stock_days,
              demand_std_dev, service_level_z]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Demand, stock, safety-stock, variability, and z inputs must be finite.")
    if average_daily_demand < 0 or current_stock < 0 or safety_stock_days < 0:
        raise ValueError("Demand, stock, and safety-stock inputs must be non-negative.")
    if demand_std_dev < 0 or service_level_z < 0:
        raise ValueError("Demand variability and service-level z must be non-negative.")
    if isinstance(lead_time_days, bool) or not isinstance(lead_time_days, int) or lead_time_days < 0:
        raise ValueError("lead_time_days must be a non-negative integer.")

    lead_time_demand = average_daily_demand * lead_time_days
    time_buffer = average_daily_demand * safety_stock_days
    variability_buffer = service_level_z * demand_std_dev * math.sqrt(lead_time_days)
    safety_stock = time_buffer + variability_buffer
    reorder_point = lead_time_demand + safety_stock
    order_quantity = max(0, math.ceil(reorder_point - current_stock))
    coverage_days = current_stock / average_daily_demand if average_daily_demand > 0 else math.inf
    if average_daily_demand == 0:
        # No observed demand means there is no consumption signal to replenish.
        # Keep the conventional risk labels for dashboard sorting/compatibility.
        risk = "Low"
        order_quantity = 0
        reorder_required = False
    elif current_stock <= lead_time_demand:
        risk = "High"
        reorder_required = current_stock <= reorder_point
    elif current_stock <= reorder_point:
        risk = "Medium"
        reorder_required = current_stock <= reorder_point
    else:
        risk = "Low"
        reorder_required = False
    return {
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "variability_buffer": round(variability_buffer, 2),
        "reorder_point": round(reorder_point, 2),
        "order_quantity": order_quantity,
        "days_of_cover": round(coverage_days, 2) if math.isfinite(coverage_days) else -1.0,
        "stockout_risk": risk,
        "reorder_required": reorder_required,
    }
