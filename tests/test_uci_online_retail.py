import pandas as pd
import pytest

from demandiq.uci_online_retail import prepare_online_retail


def transactions() -> pd.DataFrame:
    dates = pd.date_range("2011-01-01", periods=35, freq="D")
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "InvoiceNo": str(1000 + index),
                "StockCode": "A",
                "Quantity": 2,
                "InvoiceDate": date,
                "Country": "United Kingdom",
            }
        )
        rows.append(
            {
                "InvoiceNo": str(2000 + index),
                "StockCode": "B",
                "Quantity": 5,
                "InvoiceDate": date,
                "Country": "United Kingdom",
            }
        )
    rows.extend(
        [
            {
                "InvoiceNo": "C9999",
                "StockCode": "B",
                "Quantity": 100,
                "InvoiceDate": dates[-1],
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "3000",
                "StockCode": "B",
                "Quantity": -10,
                "InvoiceDate": dates[-1],
                "Country": "United Kingdom",
            },
            {
                "InvoiceNo": "4000",
                "StockCode": "A",
                "Quantity": 100,
                "InvoiceDate": dates[-1],
                "Country": "France",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_prepares_daily_sales_and_uses_country_as_location_proxy():
    result = prepare_online_retail(transactions(), min_history_days=29)

    assert set(result.columns) == {"date", "store_id", "product_id", "sales"}
    assert result["store_id"].unique().tolist() == ["United Kingdom"]
    assert result["product_id"].nunique() == 2
    assert len(result) == 70
    assert result["sales"].sum() == 245
    assert result["date"].min() == pd.Timestamp("2011-01-01")


def test_top_product_limit_is_deterministic_and_uses_positive_units():
    result = prepare_online_retail(
        transactions(), min_history_days=29, max_products=1
    )

    assert result["product_id"].unique().tolist() == ["B"]
    assert result["sales"].sum() == 175


def test_none_keeps_all_eligible_products():
    result = prepare_online_retail(
        transactions(), min_history_days=29, max_products=None
    )

    assert result["product_id"].nunique() == 2


def test_rejects_short_history_threshold():
    with pytest.raises(ValueError, match="at least 29"):
        prepare_online_retail(transactions(), min_history_days=28)


def test_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Missing required transaction columns"):
        prepare_online_retail(pd.DataFrame({"InvoiceNo": ["1"]}))
