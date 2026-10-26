"""Shared pytest fixtures."""

from __future__ import annotations

import copy

import pytest

from churnguard import predict
from churnguard.config import Config, load_config


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory) -> Config:
    """A fast config (few rows, one model) writing to a temp dir."""
    tmp = tmp_path_factory.mktemp("churnguard_tmp")
    raw = copy.deepcopy(load_config().raw)
    raw["data"]["n_rows"] = 600
    raw["model"]["candidates"] = ["logistic_regression", "gradient_boosting"]
    raw["model"]["cv_folds"] = 2
    # Absolute paths override PROJECT_ROOT in Config.path().
    raw["data"]["raw_path"] = str(tmp / "churn.csv")
    raw["artifacts"]["model_path"] = str(tmp / "model.joblib")
    raw["artifacts"]["metrics_path"] = str(tmp / "metrics.json")
    raw["artifacts"]["model_card_path"] = str(tmp / "model_card.md")
    raw["artifacts"]["figures_dir"] = str(tmp / "figures")
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
        raw = copy.deepcopy(cfg.raw)
        raw["data"]["n_rows"] = 800
        raw["model"]["candidates"] = ["logistic_regression"]
        raw["model"]["cv_folds"] = 2
        train_and_save(Config(raw=raw))
    predict.load_model.cache_clear()
    yield
