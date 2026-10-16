"""Validation utilities for daily retail sales data."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "store_id", "product_id", "sales"}


def validate_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized, sorted copy of a valid daily sales dataframe.

    Dates are normalized to calendar days because each row represents daily
    demand. Duplicate store/product/day records are rejected rather than
    silently aggregated, so upstream data-quality issues remain visible.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Sales data must contain at least one row.")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["store_id"] = result["store_id"].astype("string").str.strip()
    result["product_id"] = result["product_id"].astype("string").str.strip()
    result["sales"] = pd.to_numeric(result["sales"], errors="raise")

    if result[["date", "store_id", "product_id", "sales"]].isna().any().any():
        raise ValueError("Required sales fields cannot contain missing values.")
    if result["store_id"].eq("").any() or result["product_id"].eq("").any():
        raise ValueError("store_id and product_id cannot be empty.")
    if not np.isfinite(result["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Sales must contain finite numeric values.")
    if (result["sales"] < 0).any():
        raise ValueError("Sales must be non-negative.")
    if result.duplicated(["date", "store_id", "product_id"]).any():
        raise ValueError("Duplicate date/store_id/product_id rows found.")

    return result.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
