import pandas as pd
import pytest

from demandiq.features import make_features


def test_lag_one_uses_previous_observation():
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=35),
        "store_id": ["S1"] * 35,
        "product_id": ["P1"] * 35,
        "sales": list(range(1, 36)),
    })
    result = make_features(frame)
    row = result.loc[result["date"] == pd.Timestamp("2025-01-30")].iloc[0]
    assert row["sales"] == 30
    assert row["lag_1"] == 29
    assert row["lag_7"] == 23
    assert row["rolling_mean_7"] == 26


def test_features_are_isolated_between_store_product_series():
    dates = pd.date_range("2025-01-01", periods=35)
    frame = pd.concat([
        pd.DataFrame({
            "date": dates,
            "store_id": store,
            "product_id": product,
            "sales": values,
        })
        for store, product, values in [
            ("S1", "P1", list(range(1, 36))),
            ("S1", "P2", [100] * 35),
            ("S2", "P1", [200] * 35),
        ]
    ], ignore_index=True)
    result = make_features(frame)
    row = result.loc[
        (result["store_id"] == "S1")
        & (result["product_id"] == "P2")
        & (result["date"] == pd.Timestamp("2025-01-30"))
    ].iloc[0]
    assert row["lag_1"] == 100
    assert row["lag_7"] == 100


def test_missing_calendar_date_is_rejected_instead_of_mislabelled_as_lag():
    dates = pd.date_range("2025-01-01", periods=35).delete(10)
    frame = pd.DataFrame({
        "date": dates,
        "store_id": ["S1"] * len(dates),
        "product_id": ["P1"] * len(dates),
        "sales": list(range(1, len(dates) + 1)),
    })
    with pytest.raises(ValueError, match="missing calendar dates"):
        make_features(frame)
