"""Transparent demand-pattern-based forecasting strategy selection."""

from __future__ import annotations


def select_method_for_pattern(pattern: str) -> str:
    """Choose a method using a training-history demand-pattern label."""
    methods = {
        "smooth": "random_forest",
        "erratic": "random_forest",
        "intermittent": "croston_sba",
        "lumpy": "croston_sba",
        "no_demand": "zero",
    }
    try:
        return methods[pattern]
    except KeyError as exc:
        raise ValueError(f"Unsupported demand pattern: {pattern}") from exc
