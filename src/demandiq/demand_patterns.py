"""Classify SKU demand patterns using ADI and squared coefficient of variation."""

from __future__ import annotations

import numpy as np

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def demand_pattern_metrics(history: object) -> dict[str, float | str]:
    """Return ADI, CV², and demand-pattern class for non-negative daily demand.

    ADI is the number of observed periods divided by the number of non-zero
    demand periods. CV² is the population variance of positive demand sizes
    divided by their squared mean. The conventional 1.32 / 0.49 thresholds
    separate smooth, erratic, intermittent, and lumpy demand.
    """
    values = np.asarray(history, dtype=float)
    if values.ndim != 1:
        raise ValueError("history must be a one-dimensional sequence")
    if values.size == 0:
        raise ValueError("history must contain at least one observation")
    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values")
    if (values < 0).any():
        raise ValueError("history must contain non-negative demand")

    positive = values[values > 0]
    if positive.size == 0:
        return {"adi": float("inf"), "cv2": float("inf"), "pattern": "no_demand"}

    adi = float(values.size / positive.size)
    mean_size = float(positive.mean())
    cv2 = float(positive.var(ddof=0) / (mean_size * mean_size))

    if adi < ADI_THRESHOLD:
        pattern = "smooth" if cv2 < CV2_THRESHOLD else "erratic"
    else:
        pattern = "intermittent" if cv2 < CV2_THRESHOLD else "lumpy"

    return {"adi": adi, "cv2": cv2, "pattern": pattern}
