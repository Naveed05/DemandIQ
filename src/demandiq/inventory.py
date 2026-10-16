"""Simple, interpretable inventory replenishment calculations."""

from __future__ import annotations

import math


def reorder_recommendation(
    average_daily_demand: float,
    lead_time_days: int,
    current_stock: float,
    safety_stock_days: float = 3.0,
) -> dict[str, float | int | bool]:
    """Estimate reorder point and suggested order quantity.

    The calculation uses average daily demand as a simple proxy, not a
    probabilistic service-level model. Order quantity is the gap between target
    stock (lead-time demand plus safety stock) and current stock, floored at 0.
    """
    values = [average_daily_demand, current_stock, safety_stock_days]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Demand, stock, and safety-stock inputs must be finite.")
    if average_daily_demand < 0 or current_stock < 0 or safety_stock_days < 0:
        raise ValueError("Demand, stock, and safety-stock inputs must be non-negative.")
    if isinstance(lead_time_days, bool) or not isinstance(lead_time_days, int) or lead_time_days < 0:
        raise ValueError("lead_time_days must be a non-negative integer.")

    lead_time_demand = average_daily_demand * lead_time_days
    safety_stock = average_daily_demand * safety_stock_days
    reorder_point = lead_time_demand + safety_stock
    order_quantity = max(0, math.ceil(reorder_point - current_stock))
    return {
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "order_quantity": order_quantity,
        "reorder_required": current_stock <= reorder_point,
    }
