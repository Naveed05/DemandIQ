import pandas as pd

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
