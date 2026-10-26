"""Tests for drift monitoring (PSI, reference profile, drift grading)."""

from __future__ import annotations

import numpy as np

from churnguard.monitor import _psi, build_reference, compute_drift
from churnguard.train import train


def _trained(cfg):
    res = train(cfg)
    proba = res.bundle.pipeline.predict_proba(res.X_test)[:, 1]
    return res, proba


def test_psi_is_zero_for_identical_distributions():
    dist = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(_psi(dist, dist)) < 1e-9


def test_psi_grows_with_divergence():
    ref = np.array([0.5, 0.5])
    near = _psi(ref, np.array([0.55, 0.45]))
    far = _psi(ref, np.array([0.9, 0.1]))
    assert 0 < near < far


def test_no_drift_on_the_same_data(small_cfg):
    res, proba = _trained(small_cfg)
    y = res.y_test.to_numpy()
    ref = build_reference(res.X_test, proba, small_cfg, y=y, baseline_roc_auc=0.8)
    report = compute_drift(ref, res.X_test, proba, small_cfg, y=y)
    assert report["overall_status"] == "ok"
    assert report["prediction_drift"]["psi"] < 0.1
    assert report["summary"]["alert"] == 0


def test_drift_detected_on_shifted_population(small_cfg):
    res, proba = _trained(small_cfg)
    ref = build_reference(res.X_test, proba, small_cfg, baseline_roc_auc=0.8)

    shifted = res.X_test.copy()
    shifted["tenure"] = 0  # everyone brand-new
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] + 60
    if "Contract" in shifted.columns:
        shifted["Contract"] = "Month-to-month"
    aligned = shifted.reindex(columns=res.bundle.feature_names)
    shifted_proba = res.bundle.pipeline.predict_proba(aligned)[:, 1]

    report = compute_drift(ref, shifted, shifted_proba, small_cfg)
    assert report["overall_status"] in {"warn", "alert"}
    assert report["summary"]["warn"] + report["summary"]["alert"] >= 1
    # tenure should be the most-drifted feature.
    assert report["features"][0]["feature"] in {"tenure", "MonthlyCharges"}


def test_reference_profile_round_trips(small_cfg):
    res, proba = _trained(small_cfg)
    ref = build_reference(res.X_test, proba, small_cfg, baseline_roc_auc=0.81)
    assert ref["n"] == len(res.X_test)
    assert "tenure" in ref["numeric"]
    assert ref["baseline_roc_auc"] == 0.81
