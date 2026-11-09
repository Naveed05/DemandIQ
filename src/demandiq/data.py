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



INVENTORY_REQUIRED_COLUMNS = {"store_id", "product_id", "current_stock"}


def validate_inventory_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized inventory snapshot with unique, finite, non-negative stock.

    Extra columns are ignored so supplier exports can contain unrelated metadata.
    Duplicate store/SKU rows are rejected because silently summing snapshots can
    double-count stock when the source contains repeated records.
    """
    missing = INVENTORY_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Inventory data is missing columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("Inventory data must contain at least one row.")

    result = frame[["store_id", "product_id", "current_stock"]].copy()
    result["store_id"] = result["store_id"].astype("string").str.strip()
    result["product_id"] = result["product_id"].astype("string").str.strip()
    result["current_stock"] = pd.to_numeric(result["current_stock"], errors="raise")

    if result.isna().any().any():
        raise ValueError("Inventory fields cannot contain missing values.")
    if result["store_id"].eq("").any() or result["product_id"].eq("").any():
        raise ValueError("Inventory store_id and product_id cannot be empty.")
    if not np.isfinite(result["current_stock"].to_numpy(dtype=float)).all():
        raise ValueError("Inventory stock must contain finite numeric values.")
    if (result["current_stock"] < 0).any():
        raise ValueError("Inventory stock must be non-negative.")
    if result.duplicated(["store_id", "product_id"]).any():
        raise ValueError("Duplicate inventory store_id/product_id rows found.")

    return result.sort_values(["store_id", "product_id"]).reset_index(drop=True)
