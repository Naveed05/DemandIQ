"""Prepare UCI Online Retail transactions as continuous daily-demand rows."""

from __future__ import annotations

import pandas as pd


def prepare_online_retail(
    transactions: pd.DataFrame,
    country: str = "United Kingdom",
    min_history_days: int = 60,
    max_products: int | None = 100,
) -> pd.DataFrame:
    """Aggregate eligible transactions to daily country/SKU demand.

    Country is a location proxy because the source has no store ID. Returns,
    refunds, and cancellation invoices are excluded. Missing dates between a
    product's first and last eligible transaction are filled with zero; this
    is an explicit continuous-demand assumption, not evidence of availability.

    By default, keep the 100 products with the highest eligible unit volume.
    This makes the workbook practical to benchmark on a laptop while keeping
    the selection deterministic. Set max_products=None to retain all
    eligible product series.
    """
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, int):
        raise TypeError("min_history_days must be an integer")
    if min_history_days < 29:
        raise ValueError("min_history_days must be at least 29 for DemandIQ lag features")
    if max_products is not None and (
        isinstance(max_products, bool) or not isinstance(max_products, int) or max_products < 1
    ):
        raise ValueError("max_products must be a positive integer or None")

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
        .rename(
            columns={
                "Country": "store_id",
                "StockCode": "product_id",
                "Quantity": "sales",
            }
        )
    )
    daily["sales"] = daily["sales"].astype(float)

    eligible_products = []
    for product_id, group in daily.groupby("product_id", sort=True):
        span_days = (group["date"].max() - group["date"].min()).days + 1
        if span_days >= min_history_days:
            eligible_products.append((str(product_id), float(group["sales"].sum())))
    if not eligible_products:
        raise ValueError(
            "No product series met min_history_days; lower the threshold only if "
            "the resulting history is suitable for evaluation."
        )

    if max_products is not None:
        selected = sorted(eligible_products, key=lambda item: (-item[1], item[0]))[
            :max_products
        ]
        selected_ids = {product_id for product_id, _ in selected}
        daily = daily.loc[daily["product_id"].isin(selected_ids)].copy()

    rows: list[pd.DataFrame] = []
    for (store_id, product_id), group in daily.groupby(
        ["store_id", "product_id"], sort=True
    ):
        group = group.sort_values("date").set_index("date")
        start, end = group.index.min(), group.index.max()
        series = group["sales"].reindex(
            pd.date_range(start, end, freq="D"), fill_value=0.0
        )
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

    result = pd.concat(rows, ignore_index=True)
    return result[["date", "store_id", "product_id", "sales"]].sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)
