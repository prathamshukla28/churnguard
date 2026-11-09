"""Tests for the cost-based threshold and threshold-consistent risk bands."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard.decision import choose_threshold, expected_cost, high_risk_cutoff, risk_band


def test_choose_threshold_prefers_cheaper_cut():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    best, grid, costs = choose_threshold(y, p, cost_fn=5, cost_fp=1)
    assert 0.0 < best < 1.0
    assert costs.min() == 0.0  # perfect separation is achievable here
    assert len(grid) == len(costs)


def test_expected_cost_counts_fn_and_fp():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    # @0.5: one FN (cost 5) + one FP (cost 1)
    assert expected_cost(y, p, 0.5, cost_fn=5, cost_fp=1) == 6.0


@pytest.mark.parametrize("threshold", [0.05, 0.19, 0.2, 0.35, 0.5, 0.7, 0.95])
def test_band_never_contradicts_the_decision(threshold):
    for p in np.linspace(0.0, 1.0, 201):
        band = risk_band(float(p), threshold)
        assert (band != "low") == (p >= threshold), (p, threshold, band)


def test_bands_at_production_threshold():
    # The old fixed 0.33/0.66 bands called p=0.25 "low" while will_churn was true.
    assert risk_band(0.25, 0.2) == "medium"
    assert risk_band(0.15, 0.2) == "low"
    assert risk_band(0.55, 0.2) == "high"


def test_high_cutoff_never_below_threshold():
    assert high_risk_cutoff(0.7, 0.5) == 0.7
    assert risk_band(0.75, 0.7) == "high"
    assert risk_band(0.65, 0.7) == "low"
