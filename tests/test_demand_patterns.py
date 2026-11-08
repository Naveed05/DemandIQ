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


def test_cv_squared_threshold_is_classified_as_erratic():
    result = demand_pattern_metrics([3, 17])
    assert result["cv2"] == pytest.approx(0.49)
    assert result["pattern"] == "erratic"


def test_positive_scale_does_not_change_demand_pattern():
    history = [0, 0, 4, 0, 0, 12, 0, 0, 0, 8]
    original = demand_pattern_metrics(history)
    scaled = demand_pattern_metrics([value * 100 for value in history])
    assert scaled["pattern"] == original["pattern"]
    assert scaled["adi"] == original["adi"]
    assert scaled["cv2"] == pytest.approx(original["cv2"])
