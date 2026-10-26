"""Tests for training, selection, tuning, calibration and evaluation."""

from __future__ import annotations

import copy

from churnguard.config import Config
from churnguard.evaluate import choose_threshold, evaluate
from churnguard.train import train


def test_training_selects_a_model_and_learns(small_cfg):
    result = train(small_cfg)
    assert result.best_name in small_cfg.get("model", "candidates")
    assert result.cv_scores[result.best_name] > 0.70


def test_bundle_has_explainer_payload(small_cfg):
    bundle = train(small_cfg).bundle
    assert bundle.background is not None
    assert len(bundle.transformed_feature_names) == bundle.background.shape[1]
    assert bundle.feature_names  # raw input columns recorded


def test_evaluate_writes_metrics_figures_and_threshold(small_cfg):
    metrics = evaluate(small_cfg)
    assert metrics["roc_auc"] > 0.70
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 < metrics["threshold"] < 1.0
    # Business threshold should not cost more than the naive 0.5 cut.
    assert metrics["cost_at_threshold"] <= metrics["cost_at_0.5"]
    figures = small_cfg.path("artifacts", "figures_dir")
    for name in ("roc_curve", "calibration_curve", "cost_curve", "feature_importance"):
        assert (figures / f"{name}.png").exists()
    # Threshold is persisted back into the saved bundle.
    from churnguard.bundle import ModelBundle
    saved = ModelBundle.load(small_cfg.path("artifacts", "model_path"))
    assert abs(saved.threshold - metrics["threshold"]) < 1e-6


def test_choose_threshold_prefers_cheaper_cut():
    import numpy as np
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    best, grid, costs = choose_threshold(y, p, cost_fn=5, cost_fp=1)
    assert 0.0 < best < 1.0
    assert costs.min() == 0.0  # perfect separation is achievable here


def test_calibration_and_tuning_run(small_cfg):
    raw = copy.deepcopy(small_cfg.raw)
    raw["model"]["tuning"]["enabled"] = True
    raw["model"]["tuning"]["n_iter"] = 3
    raw["model"]["calibration"]["enabled"] = True
    result = train(Config(raw=raw))
    # Serving model still produces valid probabilities after calibration.
    proba = result.bundle.pipeline.predict_proba(result.X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
