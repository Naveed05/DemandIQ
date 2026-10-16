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


@pytest.mark.parametrize("bad_sales", [-1, float("inf")])
def test_validation_rejects_invalid_sales(bad_sales):
    frame = sample_frame()
    frame.loc[0, "sales"] = bad_sales
    with pytest.raises(ValueError):
        validate_sales_frame(frame)


def test_validation_rejects_duplicate_series_date():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales_frame(frame)
