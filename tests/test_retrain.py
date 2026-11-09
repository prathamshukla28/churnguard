"""Tests for retraining: data-change skip, regression gate, champion/challenger."""

from __future__ import annotations

import json

from churnguard import registry
from churnguard.data import dataset_fingerprint
from churnguard.retrain import retrain


def _baseline(cfg, **values) -> None:
    path = cfg.path("artifacts", "baseline_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


def test_skips_when_data_unchanged_since_champion(fresh_cfg):
    sha = dataset_fingerprint(fresh_cfg)
    _baseline(fresh_cfg, roc_auc=0.8, pr_auc=0.5, brier=0.2, data_sha256=sha)
    outcome = retrain(fresh_cfg)
    assert outcome.status == "skipped"
    assert "unchanged" in outcome.reason
    assert not fresh_cfg.path("artifacts", "model_path").exists()


def test_regression_blocks_registration(fresh_cfg):
    _baseline(fresh_cfg, roc_auc=0.999, pr_auc=0.999, brier=0.0)
    outcome = retrain(fresh_cfg, force=True)
    assert outcome.status == "regressed"
    assert registry.models(fresh_cfg) == []
    assert not fresh_cfg.path("artifacts", "model_path").exists()


def test_promotes_and_writes_serving_model_without_champion(fresh_cfg):
    outcome = retrain(fresh_cfg)
    assert outcome.status == "promoted"
    assert registry.production_entry(fresh_cfg)["version"] == outcome.version
    assert fresh_cfg.path("artifacts", "model_path").exists()
    assert fresh_cfg.path("artifacts", "metrics_path").exists()
    # Promotion never rewrites the committed baseline.
    assert not fresh_cfg.path("artifacts", "baseline_path").exists()


def test_stays_staged_when_champion_is_better(fresh_cfg):
    first = retrain(fresh_cfg)
    assert first.status == "promoted"
    # Pretend the incumbent is much better than anything we can train.
    manifest = registry.load_manifest(fresh_cfg)
    manifest["models"][0]["metrics"]["roc_auc"] = 0.999
    registry._save_manifest(manifest, fresh_cfg)

    second = retrain(fresh_cfg, force=True)
    assert second.status == "staged"
    assert registry.production_entry(fresh_cfg)["version"] == first.version


def test_no_promote_flag_registers_staging_only(fresh_cfg):
    outcome = retrain(fresh_cfg, promote=False)
    assert outcome.status == "staged"
    assert registry.production_entry(fresh_cfg) is None
