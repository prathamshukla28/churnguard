"""Tests for training, selection, tuning, calibration, threshold choice and evaluation.

Runs on the synthetic test dataset: scores here are smoke checks, not results.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from churnguard import train as train_mod
from churnguard.bundle import ModelBundle
from churnguard.config import Config
from churnguard.evaluate import StaleModelError, evaluate
from churnguard.predict import ModelNotTrainedError
from churnguard.train import train, train_and_save


def test_training_selects_a_model_and_learns(small_cfg):
    result = train(small_cfg)
    assert result.best_name in small_cfg.get("model", "candidates")
    assert result.cv_scores[result.best_name] > 0.70


def test_bundle_has_explainer_payload_and_provenance(small_cfg):
    result = train(small_cfg)
    bundle = result.bundle
    assert bundle.background is not None
    assert len(bundle.transformed_feature_names) == bundle.background.shape[1]
    assert bundle.feature_names  # raw input columns recorded
    assert bundle.data_source == "synthetic"
    assert bundle.data_sha256 and len(bundle.data_sha256) == 64
    assert bundle.test_index == [int(i) for i in result.X_test.index]
    assert bundle.cv_scores == result.cv_scores


def test_threshold_is_selected_out_of_fold_on_train_only(small_cfg, monkeypatch):
    """The rows used to pick the threshold never include a test row."""
    seen: dict[str, object] = {}
    real_cvp = train_mod.cross_val_predict
    real_choose = train_mod.choose_threshold

    def spy_cvp(estimator, X, y, **kwargs):
        seen["oof_index"] = set(X.index)
        return real_cvp(estimator, X, y, **kwargs)

    def spy_choose(y_true, y_proba, cost_fn, cost_fp):
        seen["n_scored"] = len(y_true)
        return real_choose(y_true, y_proba, cost_fn, cost_fp)

    monkeypatch.setattr(train_mod, "cross_val_predict", spy_cvp)
    monkeypatch.setattr(train_mod, "choose_threshold", spy_choose)

    result = train(small_cfg)
    test_rows = set(result.bundle.test_index or [])
    assert test_rows, "the held-out split must be recorded"
    assert seen["oof_index"].isdisjoint(test_rows)
    n_total = small_cfg.get("data", "n_rows")
    assert seen["n_scored"] == n_total - len(test_rows)
    assert result.bundle.threshold_source.startswith("out-of-fold")
    assert 0.0 < result.bundle.threshold < 1.0
    assert result.bundle.threshold_curve is not None


def test_evaluate_never_chooses_a_threshold_or_retrains(fresh_cfg, monkeypatch):
    bundle = train_and_save(fresh_cfg).bundle

    def boom(*_a, **_k):
        raise AssertionError("evaluate must not train or choose a threshold")

    monkeypatch.setattr(train_mod, "train", boom)
    monkeypatch.setattr(train_mod, "select_threshold_oof", boom)
    monkeypatch.setattr(train_mod, "choose_threshold", boom)
    import churnguard.decision as decision

    monkeypatch.setattr(decision, "choose_threshold", boom)

    metrics = evaluate(fresh_cfg)
    assert metrics["threshold"] == round(bundle.threshold, 3)
    assert metrics["n_test"] == len(bundle.test_index or [])
    # The saved artifact is untouched by evaluation.
    saved = ModelBundle.load(fresh_cfg.path("artifacts", "model_path"))
    assert saved.threshold == bundle.threshold
    assert saved.trained_at == bundle.trained_at


def test_evaluate_writes_reports_but_never_the_baseline(fresh_cfg):
    train_and_save(fresh_cfg)
    baseline = fresh_cfg.path("artifacts", "baseline_path")
    baseline.parent.mkdir(parents=True, exist_ok=True)
    frozen = json.dumps({"roc_auc": 0.5, "pr_auc": 0.1, "brier": 0.3})
    baseline.write_text(frozen, encoding="utf-8")

    metrics = evaluate(fresh_cfg)

    assert baseline.read_text(encoding="utf-8") == frozen
    assert metrics["roc_auc"] > 0.70
    assert 0.0 <= metrics["brier"] <= 1.0
    lo, hi = metrics["roc_auc_ci95"]
    assert lo <= metrics["roc_auc"] <= hi
    assert {"Contract", "tenure_band"} <= set(metrics["segments"])
    assert metrics["cost_at_threshold"] >= 0
    figures = fresh_cfg.path("artifacts", "figures_dir")
    for name in ("roc_curve", "calibration_curve", "cost_curve", "feature_importance"):
        assert (figures / f"{name}.png").exists()
    card = fresh_cfg.path("artifacts", "model_card_path").read_text(encoding="utf-8")
    assert "SYNTHETIC" in card  # synthetic numbers are labelled as such
    assert "no concept-drift monitoring" not in card
    assert fresh_cfg.path("artifacts", "reference_path").exists()


def test_evaluate_requires_a_trained_model(fresh_cfg):
    with pytest.raises(ModelNotTrainedError):
        evaluate(fresh_cfg)


def test_evaluate_refuses_a_model_trained_on_other_data(fresh_cfg, tmp_path):
    train_and_save(fresh_cfg)
    raw = copy.deepcopy(fresh_cfg.raw)
    raw["data"]["synthetic_path"] = str(tmp_path / "other.csv")
    raw["seed"] = 7  # different generated data
    with pytest.raises(StaleModelError):
        evaluate(Config(raw=raw))


def test_calibration_and_tuning_run(small_cfg):
    raw = copy.deepcopy(small_cfg.raw)
    raw["model"]["tuning"]["enabled"] = True
    raw["model"]["tuning"]["n_iter"] = 3
    raw["model"]["calibration"]["enabled"] = True
    result = train(Config(raw=raw))
    # Serving model still produces valid probabilities after calibration.
    proba = result.bundle.pipeline.predict_proba(result.X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.isfinite(result.bundle.threshold)


def test_one_standard_error_rule_prefers_simpler_model():
    from churnguard.train import select_candidate

    folds = {
        "logistic_regression": np.array([0.84, 0.85, 0.845, 0.842, 0.848]),
        "random_forest": np.array([0.86, 0.835, 0.855, 0.838, 0.854]),
    }
    order = ["logistic_regression", "random_forest"]
    assert select_candidate(folds, order, "best") == "random_forest"
    assert select_candidate(folds, order, "one_standard_error") == "logistic_regression"
    clearly_better = {**folds, "random_forest": folds["random_forest"] + 0.05}
    assert select_candidate(clearly_better, order, "one_standard_error") == "random_forest"
    with pytest.raises(ValueError):
        select_candidate(folds, order, "coin_flip")


def test_saved_tree_models_predict_single_threaded(small_cfg):
    raw = copy.deepcopy(small_cfg.raw)
    raw["model"]["candidates"] = ["random_forest"]
    raw["model"]["calibration"]["enabled"] = True
    bundle = train(Config(raw=raw)).bundle
    assert bundle.base_pipeline.named_steps["clf"].n_jobs == 1
    for cc in bundle.pipeline.calibrated_classifiers_:
        assert cc.estimator.named_steps["clf"].n_jobs == 1
