"""Best-effort MLflow experiment tracking.

If MLflow is installed and tracking is enabled, training and evaluation runs are
logged to a local ``mlruns/`` directory (params, cross-validation scores, test
metrics). Everything here is wrapped so that a missing MLflow, or any logging
error, never breaks the actual pipeline.

Disable with the environment variable ``CHURNGUARD_TRACKING=0`` (the test suite
sets this to keep runs clean).
"""

from __future__ import annotations

import os
from typing import Any

from .config import Config

_EXPERIMENT = "churnguard"


def _enabled() -> bool:
    return os.environ.get("CHURNGUARD_TRACKING", "1") == "1"


def log_training_run(
    cfg: Config,
    model_name: str,
    cv_scores: dict[str, float],
    best_params: dict[str, Any],
) -> None:
    """Log model selection + tuning results. Never raises."""
    if not _enabled():
        return
    try:
        import mlflow

        mlflow.set_experiment(_EXPERIMENT)
        with mlflow.start_run(run_name=f"train-{model_name}"):
            mlflow.log_param("selected_model", model_name)
            mlflow.log_param("seed", cfg.seed)
            mlflow.log_param("tuning", cfg.get("model", "tuning", "enabled"))
            mlflow.log_param("calibration", cfg.get("model", "calibration", "enabled"))
            for key, value in best_params.items():
                mlflow.log_param(key, value)
            for name, score in cv_scores.items():
                mlflow.log_metric(f"cv_auc__{name}", score)
    except Exception:  # noqa: BLE001 - tracking must never break training
        return


def log_evaluation(metrics: dict[str, Any]) -> None:
    """Log flat numeric test metrics. Never raises."""
    if not _enabled():
        return
    try:
        import mlflow

        mlflow.set_experiment(_EXPERIMENT)
        with mlflow.start_run(run_name="evaluate"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, float(value))
    except Exception:  # noqa: BLE001 - tracking must never break evaluation
        return
