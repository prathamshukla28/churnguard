"""Tests for training and evaluation on a fast config."""

from __future__ import annotations

from churnguard.evaluate import evaluate
from churnguard.train import train


def test_training_selects_a_model_and_learns(small_cfg):
    result = train(small_cfg)
    assert result.best_name in small_cfg.get("model", "candidates")
    # The signal is real, so CV AUC must beat random (0.5) comfortably.
    assert result.cv_scores[result.best_name] > 0.70


def test_evaluate_writes_metrics_and_figures(small_cfg):
    metrics = evaluate(small_cfg)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["roc_auc"] > 0.70
    assert small_cfg.path("artifacts", "metrics_path").exists()
    assert small_cfg.path("artifacts", "model_card_path").exists()
    figures = small_cfg.path("artifacts", "figures_dir")
    assert (figures / "roc_curve.png").exists()
    assert (figures / "feature_importance.png").exists()
