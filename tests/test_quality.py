"""Tests for the model-regression gate and the frozen baseline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from churnguard.quality import check_regression, main, promote_baseline, run_check

_ROOT = Path(__file__).resolve().parents[1]
_BASE = {"roc_auc": 0.84, "pr_auc": 0.65, "brier": 0.14}


def test_gate_passes_within_tolerance():
    ok, lines = check_regression({"roc_auc": 0.83, "pr_auc": 0.64, "brier": 0.15}, _BASE, 0.02)
    assert ok
    assert lines[-1].startswith("PASS")


def test_gate_fails_when_auc_drops():
    ok, lines = check_regression({"roc_auc": 0.80, "pr_auc": 0.65, "brier": 0.14}, _BASE, 0.02)
    assert not ok
    assert any(line.startswith("[FAIL] roc_auc") for line in lines)


def test_gate_fails_when_brier_rises():
    ok, _ = check_regression({"roc_auc": 0.84, "pr_auc": 0.65, "brier": 0.20}, _BASE, 0.02)
    assert not ok


def test_gate_fails_when_a_metric_is_missing():
    ok, _ = check_regression({"roc_auc": 0.84, "brier": 0.14}, _BASE, 0.02)
    assert not ok


def test_gate_warns_on_dataset_change():
    ok, lines = check_regression(
        {**_BASE, "data_sha256": "b" * 64}, {**_BASE, "data_sha256": "a" * 64}, 0.02
    )
    assert ok
    assert any(line.startswith("[WARN]") for line in lines)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_check_exit_codes(fresh_cfg):
    metrics = fresh_cfg.path("artifacts", "metrics_path")
    baseline = fresh_cfg.path("artifacts", "baseline_path")
    assert run_check(fresh_cfg) == 2  # nothing yet

    _write(baseline, _BASE)
    assert run_check(fresh_cfg) == 2  # no metrics yet

    _write(metrics, {"roc_auc": 0.70, "pr_auc": 0.50, "brier": 0.20})
    assert run_check(fresh_cfg) == 1  # regression

    _write(metrics, _BASE)
    assert run_check(fresh_cfg) == 0


def test_promote_baseline_freezes_current_metrics(fresh_cfg):
    _write(
        fresh_cfg.path("artifacts", "metrics_path"),
        {**_BASE, "model": "lr", "data_sha256": "c" * 64, "data_source": "real"},
    )
    path = promote_baseline(fresh_cfg)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert {k: frozen[k] for k in _BASE} == _BASE
    assert frozen["data_sha256"] == "c" * 64
    assert "promoted_at" in frozen


def test_cli_script_exits_nonzero_on_regression(fresh_cfg, tmp_path):
    """The exact command CI runs must fail when quality drops."""
    _write(fresh_cfg.path("artifacts", "baseline_path"), _BASE)
    _write(
        fresh_cfg.path("artifacts", "metrics_path"),
        {"roc_auc": 0.79, "pr_auc": 0.60, "brier": 0.14},
    )
    cfg_file = tmp_path / "gate.yaml"
    cfg_file.write_text(yaml.safe_dump(fresh_cfg.raw), encoding="utf-8")

    env = {**os.environ, "CHURNGUARD_CONFIG": str(cfg_file)}
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "check_regression.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout


def test_quality_cli_check_and_promote(fresh_cfg, monkeypatch):
    import churnguard.quality as quality

    monkeypatch.setattr(quality, "load_config", lambda: fresh_cfg)
    _write(fresh_cfg.path("artifacts", "metrics_path"), _BASE)
    assert main(["promote-baseline"]) == 0
    assert main(["check"]) == 0
