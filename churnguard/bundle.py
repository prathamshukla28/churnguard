"""The serialized model artifact.

A :class:`ModelBundle` is what gets saved to ``models/churnguard.joblib``. It
carries everything inference and explanation need, so the service is a single
self-describing file:

* ``pipeline``       — the serving model (calibrated, if enabled) with ``predict_proba``.
* ``base_pipeline``  — the uncalibrated ``Pipeline(pre, clf)``, used by the SHAP
  explainer (it needs the raw preprocessor + estimator, not the calibration wrapper).
* ``feature_names``  — the raw (post feature-engineering) input columns the model expects.
* ``threshold``      — the business-chosen decision threshold, selected at training
  time on out-of-fold predictions over the training split (never on the test split).
* ``threshold_curve`` — the (grid, cost) curve that threshold was chosen from.
* ``data_source`` / ``data_sha256`` / ``test_index`` — which data trained the model
  and which rows were held out, so ``evaluate`` scores exactly the untouched test
  split of exactly that data without retraining.
* ``version`` / ``trained_at`` / ``model_name`` — provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from . import __version__


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelBundle:
    pipeline: Pipeline
    base_pipeline: Pipeline
    feature_names: list[str]
    model_name: str
    threshold: float = 0.5
    version: str = __version__
    trained_at: str = field(default_factory=_now)
    # For SHAP explanations: a transformed background sample + its column names.
    background: np.ndarray | None = None
    transformed_feature_names: list[str] = field(default_factory=list)
    # How the threshold was chosen, and on what.
    threshold_source: str = "unset"
    threshold_curve: dict[str, list[float]] | None = None
    # Data provenance + the held-out split.
    data_source: str | None = None
    data_sha256: str | None = None
    test_index: list[int] | None = None
    # Model-selection provenance (cross-validated on the training split).
    cv_scores: dict[str, float] | None = None
    best_params: dict[str, Any] | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return obj
