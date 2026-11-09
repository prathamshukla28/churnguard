"""Shared pytest fixtures.

The whole suite runs against an isolated config written to a temp directory and
selected through ``$CHURNGUARD_CONFIG`` *before* any ``churnguard`` import, so
tests never read the real dataset, never download anything and never overwrite
the developer's model or committed reports.

The suite trains on the **synthetic** dataset on purpose: it is small, offline
and fast. Its label comes from a hand-written formula, so the scores asserted
here are pipeline smoke checks, not evidence about real churn.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import pytest
import yaml

# Keep MLflow runs out of the test suite.
os.environ.setdefault("CHURNGUARD_TRACKING", "0")

_ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="churnguard_tests_"))


def _fast(raw: dict) -> dict:
    """Shrink a config for fast tests (few rows, no tuning/calibration)."""
    raw = copy.deepcopy(raw)
    raw["data"]["source"] = "synthetic"
    raw["data"]["n_rows"] = 600
    raw["model"]["candidates"] = ["logistic_regression", "gradient_boosting"]
    raw["model"]["cv_folds"] = 2
    raw["model"]["tuning"]["enabled"] = False
    raw["model"]["calibration"]["enabled"] = False
    return raw


def _with_paths(raw: dict, root: Path) -> dict:
    raw = copy.deepcopy(raw)
    raw["data"]["raw_path"] = str(root / "telco_churn.csv")
    raw["data"]["synthetic_path"] = str(root / "synthetic_churn.csv")
    raw["artifacts"]["model_path"] = str(root / "model.joblib")
    raw["artifacts"]["metrics_path"] = str(root / "metrics.json")
    raw["artifacts"]["model_card_path"] = str(root / "model_card.md")
    raw["artifacts"]["figures_dir"] = str(root / "figures")
    raw["artifacts"]["reference_path"] = str(root / "reference_profile.json")
    raw["artifacts"]["baseline_path"] = str(root / "baseline_metrics.json")
    raw["registry"]["dir"] = str(root / "registry")
    return raw


def _session_raw() -> dict:
    with open(_ROOT / "config" / "config.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _fast(raw)
    raw["data"]["n_rows"] = 800
    raw["model"]["candidates"] = ["logistic_regression"]
    raw["model"]["cv_folds"] = 3
    # The served test model is calibrated, like production.
    raw["model"]["calibration"]["enabled"] = True
    return _with_paths(raw, _TMP / "session")


_SESSION_CONFIG = _TMP / "config.yaml"
_SESSION_CONFIG.write_text(yaml.safe_dump(_session_raw()), encoding="utf-8")
os.environ["CHURNGUARD_CONFIG"] = str(_SESSION_CONFIG)

from churnguard import predict  # noqa: E402
from churnguard.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory) -> Config:
    """A fast, uncalibrated synthetic config writing to its own temp dir."""
    tmp = tmp_path_factory.mktemp("churnguard_tmp")
    return Config(raw=_with_paths(_fast(load_config().raw), tmp))


@pytest.fixture()
def fresh_cfg(small_cfg, tmp_path) -> Config:
    """Like ``small_cfg`` but with per-test artifact/registry paths."""
    raw = _with_paths(small_cfg.raw, tmp_path)
    raw["data"]["synthetic_path"] = small_cfg.raw["data"]["synthetic_path"]
    return Config(raw=raw)


@pytest.fixture(scope="session", autouse=True)
def _session_model():
    """Train + evaluate the served test model once into the isolated temp dir."""
    from churnguard.evaluate import evaluate
    from churnguard.train import train_and_save

    cfg = load_config()
    assert str(cfg.path("artifacts", "model_path")).startswith(str(_TMP.resolve())), (
        "tests must never write to the real model path"
    )
    train_and_save(cfg)
    evaluate(cfg)
    predict.load_bundle.cache_clear()
    yield
