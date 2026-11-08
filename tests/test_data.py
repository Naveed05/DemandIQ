import pandas as pd
import pytest

from demandiq.data import validate_sales_frame


def sample_frame():
    return pd.DataFrame({
        "date": ["2025-01-02", "2025-01-01"],
        "store_id": ["S1", "S1"],
        "product_id": ["P1", "P1"],
        "sales": [3.0, 2.0],
    })


def test_validation_sorts_rows_and_normalizes_dates():
    result = validate_sales_frame(sample_frame())
    assert result["date"].is_monotonic_increasing
    assert str(result["date"].dtype).startswith("datetime64")


def test_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales_frame(pd.DataFrame({"date": ["2025-01-01"]}))


def test_validation_rejects_empty_frame():
    empty = pd.DataFrame(columns=["date", "store_id", "product_id", "sales"])
    with pytest.raises(ValueError, match="at least one row"):
        validate_sales_frame(empty)


@pytest.mark.parametrize("bad_sales", [-1, float("inf"), float("nan")])
def test_validation_rejects_invalid_sales(bad_sales):
    frame = sample_frame()
    frame.loc[0, "sales"] = bad_sales
    with pytest.raises(ValueError):
        validate_sales_frame(frame)


@pytest.mark.parametrize("column", ["store_id", "product_id"])
def test_validation_rejects_blank_identifiers(column):
    frame = sample_frame()
    frame.loc[0, column] = "   "
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_sales_frame(frame)


def test_validation_rejects_duplicate_series_date():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales_frame(frame)


def test_validation_normalizes_timestamps_before_duplicate_check():
    frame = pd.DataFrame({
        "date": ["2025-01-01 08:00:00", "2025-01-01 17:30:00"],
        "store_id": ["S1", "S1"],
        "product_id": ["P1", "P1"],
        "sales": [2, 3],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales_frame(frame)



def test_inventory_validation_normalizes_and_sorts_rows():
    from demandiq.data import validate_inventory_frame

    frame = pd.DataFrame({
        "store_id": [" S2 ", "S1"],
        "product_id": [" P2 ", "P1"],
        "current_stock": ["12", 4],
        "supplier_note": ["keep in source", "extra column"],
    })

    result = validate_inventory_frame(frame)

    assert result[["store_id", "product_id"]].values.tolist() == [
        ["S1", "P1"],
        ["S2", "P2"],
    ]
    assert result["current_stock"].tolist() == [4, 12]
    assert list(result.columns) == ["store_id", "product_id", "current_stock"]


@pytest.mark.parametrize(
    "frame, message",
    [
        (pd.DataFrame({"store_id": ["S1"]}), "missing columns"),
        (
            pd.DataFrame(columns=["store_id", "product_id", "current_stock"]),
            "at least one row",
        ),
        (
            pd.DataFrame({
                "store_id": ["S1", "S1"],
                "product_id": ["P1", "P1"],
                "current_stock": [4, 5],
            }),
            "Duplicate",
        ),
        (
            pd.DataFrame({
                "store_id": ["S1"],
                "product_id": ["P1"],
                "current_stock": [-1],
            }),
            "non-negative",
        ),
        (
            pd.DataFrame({
                "store_id": ["S1"],
                "product_id": ["P1"],
                "current_stock": [float("inf")],
            }),
            "finite",
        ),
        (
            pd.DataFrame({
                "store_id": [" "],
                "product_id": ["P1"],
                "current_stock": [4],
            }),
            "cannot be empty",
        ),
    ],
)
def test_inventory_validation_rejects_invalid_snapshots(frame, message):
    from demandiq.data import validate_inventory_frame

    with pytest.raises(ValueError, match=message):
        validate_inventory_frame(frame)


def test_inventory_validation_rejects_non_numeric_stock():
    from demandiq.data import validate_inventory_frame

    frame = pd.DataFrame({
        "store_id": ["S1"],
        "product_id": ["P1"],
        "current_stock": ["not-a-number"],
    })

    with pytest.raises((ValueError, TypeError)):
        validate_inventory_frame(frame)
