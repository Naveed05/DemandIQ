from __future__ import annotations

import pandas as pd
import pytest

from demandiq.uci_online_retail import prepare_online_retail


def sample_transactions() -> pd.DataFrame:
    dates = pd.date_range("2010-12-01", periods=35, freq="D")
    rows = [
        {
            "InvoiceNo": f"500{i}",
            "StockCode": "SKU-A",
            "Quantity": 2,
            "InvoiceDate": date + pd.Timedelta(hours=10),
            "Country": "United Kingdom",
        }
        for i, date in enumerate(dates)
    ]
    rows.extend(
        [
            {
                "InvoiceNo": "C5001",
                "StockCode": "SKU-A",
                "Quantity": 10,
                "InvoiceDate": dates[5],
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "500X",
                "StockCode": "SKU-A",
                "Quantity": -3,
                "InvoiceDate": dates[6],
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "6001",
                "StockCode": "SKU-B",
                "Quantity": 5,
                "InvoiceDate": dates[0],
                "Country": "France",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_preparation_filters_returns_cancellations_and_other_countries() -> None:
    result = prepare_online_retail(sample_transactions(), min_history_days=29)

    assert set(result["store_id"]) == {"United Kingdom"}
    assert set(result["product_id"]) == {"SKU-A"}
    assert len(result) == 35
    assert result["sales"].sum() == 70
    assert result["sales"].ge(0).all()


def test_preparation_fills_missing_calendar_days_with_zero() -> None:
    frame = sample_transactions()
    frame = frame.loc[frame["InvoiceDate"].dt.day != 10]
    result = prepare_online_retail(frame, min_history_days=29)

    missing_date = pd.Timestamp("2010-12-10")
    row = result.loc[result["date"].eq(missing_date)]
    assert len(row) == 1
    assert row.iloc[0]["sales"] == 0


def test_preparation_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Missing required transaction columns"):
        prepare_online_retail(pd.DataFrame({"InvoiceNo": ["1"]}))


def test_preparation_rejects_history_threshold_under_29_days() -> None:
    with pytest.raises(ValueError, match="at least 29"):
        prepare_online_retail(sample_transactions(), min_history_days=28)
