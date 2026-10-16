"""Time-aware feature engineering for daily demand forecasting."""

from __future__ import annotations

import pandas as pd

from demandiq.data import validate_sales_frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling features per store/product time series.

    Every store/product series must contain one row for every calendar day.
    Missing dates are ambiguous (zero sales versus missing records), so callers
    must resolve them explicitly before feature generation. This prevents
    observation-based lags from being mistaken for calendar-day lags.

    Rolling features are shifted by one row so a row's features never include
    that row's target sales. The first rows may have missing lag values by design.
    """
    data = validate_sales_frame(frame)
    group_keys = ["store_id", "product_id"]
    for (store_id, product_id), group in data.groupby(group_keys, sort=False):
        expected = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        if len(expected) != len(group) or not group["date"].reset_index(drop=True).equals(
            pd.Series(expected, name="date")
        ):
            raise ValueError(
                "Daily sales history has missing calendar dates for "
                f"store_id={store_id!r}, product_id={product_id!r}. "
                "Add explicit zero-sales rows where appropriate or correct the source data."
            )

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype("int8")

    grouped = data.groupby(group_keys, sort=False)["sales"]
    for lag in (1, 7, 14, 28):
        data[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    shifted_groups = shifted.groupby([data["store_id"], data["product_id"]], sort=False)
    data["rolling_mean_7"] = shifted_groups.transform(
        lambda values: values.rolling(7, min_periods=3).mean()
    )
    data["rolling_std_7"] = shifted_groups.transform(
        lambda values: values.rolling(7, min_periods=3).std()
    )
    return data
