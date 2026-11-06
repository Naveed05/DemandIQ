"""Intermittent-demand forecasting baselines.

Croston's method smooths non-zero demand sizes and intervals separately.
The Syntetos-Boylan Approximation (SBA) applies a bias correction.
"""

from __future__ import annotations

import numpy as np


def croston_sba_forecast(history: object, alpha: float = 0.1) -> float:
    """Return a one-step SBA forecast from non-negative daily demand history.

    The input must contain only observations available before the forecast
    timestamp. All-zero history produces a zero forecast. The estimate is
    updated only when positive demand occurs.
    """
    values = np.asarray(history, dtype=float)
    if values.ndim != 1:
        raise ValueError("history must be a one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values")
    if (values < 0).any():
        raise ValueError("history must contain non-negative demand")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")

    nonzero_positions = np.flatnonzero(values > 0)
    if len(nonzero_positions) == 0:
        return 0.0

    first = int(nonzero_positions[0])
    demand_size = float(values[first])
    interval = float(first + 1)
    previous_position = first

    for position in nonzero_positions[1:]:
        position = int(position)
        gap = float(position - previous_position)
        demand_size += alpha * (float(values[position]) - demand_size)
        interval += alpha * (gap - interval)
        previous_position = position

    forecast = (1.0 - alpha / 2.0) * demand_size / interval
    return max(0.0, float(forecast))
