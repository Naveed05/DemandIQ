import numpy as np
import pytest

from demandiq.demand_patterns import demand_pattern_metrics


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([10, 10, 10, 10, 10], "smooth"),
        ([2, 18, 3, 20, 1], "erratic"),
        ([0, 0, 10, 0, 0, 10, 0, 0], "intermittent"),
        ([0, 0, 1, 0, 0, 25, 0, 0], "lumpy"),
        ([0, 0, 0, 0], "no_demand"),
    ],
)
def test_classifies_demand_patterns(history, expected):
    result = demand_pattern_metrics(history)
    assert result["pattern"] == expected


def test_reports_adi_and_cv_squared():
    result = demand_pattern_metrics([0, 10, 0, 10, 0, 10])
    assert result["adi"] == pytest.approx(2.0)
    assert result["cv2"] == pytest.approx(0.0)


@pytest.mark.parametrize("history", [[], [1, -1], [1, np.nan], [[1, 2], [3, 4]]])
def test_rejects_invalid_history(history):
    with pytest.raises(ValueError):
        demand_pattern_metrics(history)
