"""Shared pytest fixtures."""

from __future__ import annotations

import copy
import os

import pytest

# Keep MLflow runs out of the test suite.
os.environ.setdefault("CHURNGUARD_TRACKING", "0")

from churnguard import predict  # noqa: E402
from churnguard.config import Config, load_config  # noqa: E402


def _fast(raw: dict) -> dict:
    """Shrink a config for fast tests (few rows, no tuning/calibration)."""
    raw = copy.deepcopy(raw)
    raw["data"]["n_rows"] = 600
    raw["model"]["candidates"] = ["logistic_regression", "gradient_boosting"]
    raw["model"]["cv_folds"] = 2
    raw["model"]["tuning"]["enabled"] = False
    raw["model"]["calibration"]["enabled"] = False
    return raw


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory) -> Config:
    """A fast config writing to a temp dir."""
    tmp = tmp_path_factory.mktemp("churnguard_tmp")
    raw = _fast(load_config().raw)
    raw["data"]["raw_path"] = str(tmp / "churn.csv")
    raw["artifacts"]["model_path"] = str(tmp / "model.joblib")
    raw["artifacts"]["metrics_path"] = str(tmp / "metrics.json")
    raw["artifacts"]["model_card_path"] = str(tmp / "model_card.md")
    raw["artifacts"]["figures_dir"] = str(tmp / "figures")
    raw["artifacts"]["reference_path"] = str(tmp / "reference_profile.json")
    raw["artifacts"]["baseline_path"] = str(tmp / "baseline_metrics.json")
    raw["registry"]["dir"] = str(tmp / "registry")
    return Config(raw=raw)


@pytest.fixture(scope="session", autouse=True)
def _ensure_default_model():
    """Guarantee a model exists at the default path so API/predict tests run.

    Trains a small, fast model only if the artifact is missing, so a developer's
    fully-trained model is never overwritten.
    """
    from churnguard.train import train_and_save

    cfg = load_config()
    if not cfg.path("artifacts", "model_path").exists():
        raw = _fast(cfg.raw)
        raw["data"]["n_rows"] = 800
        raw["model"]["candidates"] = ["logistic_regression"]
        train_and_save(Config(raw=raw))
    predict.load_bundle.cache_clear()
    yield
