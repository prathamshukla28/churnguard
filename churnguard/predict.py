"""Inference helpers shared by the CLI and the API.

Loads the persisted pipeline once (cached) and turns raw customer records into
churn probabilities, applying the exact same feature engineering used during
training.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import Config, load_config
from .evaluate import DECISION_THRESHOLD
from .features import engineer_features


class ModelNotTrainedError(RuntimeError):
    """Raised when no model artifact exists yet."""


# Cache loaded pipelines by their on-disk path. (A plain lru_cache can't key on
# Config, which holds an unhashable dict.)
_MODEL_CACHE: dict[str, Pipeline] = {}


def load_model(cfg: Config | None = None) -> Pipeline:
    """Load and cache the trained pipeline from disk."""
    cfg = cfg or load_config()
    path = cfg.path("artifacts", "model_path")
    key = str(path)
    if key not in _MODEL_CACHE:
        if not path.exists():
            raise ModelNotTrainedError(
                f"No model at {path}. Run `python -m churnguard.train` first."
            )
        _MODEL_CACHE[key] = joblib.load(path)
    return _MODEL_CACHE[key]


# Expose a cache_clear() hook mirroring functools.lru_cache for tests/retraining.
load_model.cache_clear = _MODEL_CACHE.clear  # type: ignore[attr-defined]


def predict_records(
    records: list[dict], cfg: Config | None = None, threshold: float = DECISION_THRESHOLD
) -> list[dict]:
    """Score a batch of raw customer records.

    Returns one dict per record with the churn probability and the thresholded
    decision. Unknown / missing columns are tolerated by the pipeline's imputers
    and the one-hot encoder's ``handle_unknown="ignore"``.
    """
    cfg = cfg or load_config()
    model = load_model(cfg)

    frame = engineer_features(pd.DataFrame(records))
    # Align to the exact columns the pipeline was fitted on: add any missing
    # columns (the pipeline's imputers fill them) and drop unexpected extras.
    expected = list(model.named_steps["pre"].feature_names_in_)
    frame = frame.reindex(columns=expected)
    proba = model.predict_proba(frame)[:, 1]

    results = []
    for p in proba:
        p = float(p)
        results.append(
            {
                "churn_probability": round(p, 4),
                "will_churn": bool(p >= threshold),
                "risk_band": "high" if p >= 0.66 else "medium" if p >= 0.33 else "low",
            }
        )
    return results
