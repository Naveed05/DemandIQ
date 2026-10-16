"""Validation utilities for daily retail sales data."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "store_id", "product_id", "sales"}


def validate_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized, sorted copy of a valid daily sales dataframe.

    Duplicate store/product/date records are rejected rather than silently
    aggregated, so upstream data-quality issues remain visible.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["store_id"] = result["store_id"].astype("string")
    result["product_id"] = result["product_id"].astype("string")
    result["sales"] = pd.to_numeric(result["sales"], errors="raise")

    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required sales fields cannot contain missing values.")
    if not np.isfinite(result["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Sales must contain finite numeric values.")
    if (result["sales"] < 0).any():
        raise ValueError("Sales must be non-negative.")
    if result.duplicated(["date", "store_id", "product_id"]).any():
        raise ValueError("Duplicate date/store_id/product_id rows found.")

    return result.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
