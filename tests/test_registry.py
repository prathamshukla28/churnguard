"""Tests for the file-based model registry and champion/challenger promotion."""

from __future__ import annotations

import copy

from churnguard import registry
from churnguard.config import Config
from churnguard.train import train


def _registry_cfg(small_cfg, tmp_path) -> Config:
    raw = copy.deepcopy(small_cfg.raw)
    raw["registry"]["dir"] = str(tmp_path / "registry")
    return Config(raw=raw)


def test_register_versions_and_promote_to_production(small_cfg, tmp_path):
    cfg = _registry_cfg(small_cfg, tmp_path)
    bundle = train(cfg).bundle

    v1 = registry.register_model(bundle, {"roc_auc": 0.80, "brier": 0.15}, cfg)
    v2 = registry.register_model(bundle, {"roc_auc": 0.83, "brier": 0.14}, cfg)
    assert (v1, v2) == ("1", "2")
    assert registry.production_entry(cfg) is None  # staging by default

    registry.promote(v1, "production", cfg)
    assert registry.production_entry(cfg)["version"] == "1"

    # Promoting v2 to production archives the incumbent v1.
    registry.promote(v2, "production", cfg)
    assert registry.production_entry(cfg)["version"] == "2"
    assert registry.get_entry("1", cfg)["stage"] == "archived"


def test_promotion_gate_accepts_better_and_rejects_worse(small_cfg, tmp_path):
    cfg = _registry_cfg(small_cfg, tmp_path)
    bundle = train(cfg).bundle

    # No champion yet -> always promote.
    should, _ = registry.evaluate_promotion({"roc_auc": 0.70}, cfg)
    assert should

    v1 = registry.register_model(bundle, {"roc_auc": 0.82}, cfg)
    registry.promote(v1, "production", cfg)

    assert registry.evaluate_promotion({"roc_auc": 0.85}, cfg)[0] is True
    assert registry.evaluate_promotion({"roc_auc": 0.80}, cfg)[0] is False


def test_registered_bundle_is_loadable(small_cfg, tmp_path):
    from churnguard.bundle import ModelBundle

    cfg = _registry_cfg(small_cfg, tmp_path)
    bundle = train(cfg).bundle
    registry.register_model(bundle, {"roc_auc": 0.8}, cfg)
    entry = registry.get_entry("1", cfg)
    saved = tmp_path / "registry" / entry["path"]
    assert saved.exists()
    assert isinstance(ModelBundle.load(saved), ModelBundle)


def test_champion_falls_back_to_committed_baseline(small_cfg, tmp_path):
    """With an empty (gitignored) registry, the committed baseline is the champion."""
    import json

    raw = copy.deepcopy(small_cfg.raw)
    raw["registry"]["dir"] = str(tmp_path / "registry")
    raw["artifacts"]["baseline_path"] = str(tmp_path / "baseline_metrics.json")
    cfg = Config(raw=raw)

    assert registry.champion_metrics(cfg) == (None, "none")
    (tmp_path / "baseline_metrics.json").write_text(
        json.dumps({"roc_auc": 0.84, "pr_auc": 0.65, "brier": 0.14}), encoding="utf-8"
    )
    champ, source = registry.champion_metrics(cfg)
    assert source == "committed baseline" and champ["roc_auc"] == 0.84

    should, reason = registry.evaluate_promotion({"roc_auc": 0.80}, cfg)
    assert should is False and "committed baseline" in reason
    assert registry.evaluate_promotion({"roc_auc": 0.85}, cfg)[0] is True
