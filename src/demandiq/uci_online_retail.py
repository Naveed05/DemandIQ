"""Prepare UCI Online Retail transactions as continuous daily-demand rows."""

from __future__ import annotations

import pandas as pd


def prepare_online_retail(
    transactions: pd.DataFrame,
    country: str = "United Kingdom",
    min_history_days: int = 60,
) -> pd.DataFrame:
    """Aggregate positive, non-cancelled transactions to daily country/SKU demand.

    Country is used as a location proxy because this dataset has no store ID.
    Returns/refunds (non-positive quantities) and cancellation invoices (InvoiceNo
    prefixed with C) are excluded. Missing dates between each SKU's first and last
    transaction are filled with zero; this is an explicit continuous-demand
    assumption, not evidence that inventory was available on those dates.
    """
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, int):
        raise ValueError("min_history_days must be an integer")
    if min_history_days < 29:
        raise ValueError("min_history_days must be at least 29 for DemandIQ lag features")
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "Country"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(sorted(missing))}")
    if transactions.empty:
        raise ValueError("Transaction data must contain at least one row")

    frame = transactions[list(required)].copy()
    frame["Country"] = frame["Country"].astype("string").str.strip()
    frame["StockCode"] = frame["StockCode"].astype("string").str.strip()
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["Country"].eq(country)].copy()
    frame = frame.loc[
        frame["date"].notna()
        & frame["StockCode"].notna()
        & frame["StockCode"].ne("")
        & frame["Quantity"].notna()
        & frame["Quantity"].gt(0)
        & ~frame["InvoiceNo"].str.upper().str.startswith("C", na=False)
    ].copy()
    if frame.empty:
        raise ValueError(f"No eligible positive sales transactions found for country {country!r}")

    daily = (
        frame.groupby(["date", "Country", "StockCode"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Country": "store_id", "StockCode": "product_id", "Quantity": "sales"})
    )
    daily["sales"] = daily["sales"].astype(float)

    rows: list[pd.DataFrame] = []
    for (store_id, product_id), group in daily.groupby(["store_id", "product_id"], sort=True):
        group = group.sort_values("date").set_index("date")
        start, end = group.index.min(), group.index.max()
        if (end - start).days + 1 < min_history_days:
            continue
        series = group["sales"].reindex(pd.date_range(start, end, freq="D"), fill_value=0.0)
        rows.append(
            pd.DataFrame(
                {
                    "date": series.index,
                    "store_id": str(store_id),
                    "product_id": str(product_id),
                    "sales": series.to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        raise ValueError(
            "No product series met min_history_days; lower the threshold only if "
            "the resulting history is suitable for evaluation."
        )
    result = pd.concat(rows, ignore_index=True)
    return result[["date", "store_id", "product_id", "sales"]].sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)


