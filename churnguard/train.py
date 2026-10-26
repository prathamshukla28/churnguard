"""Model training, selection, tuning and calibration.

Pipeline:
1. Build ``Pipeline`` = preprocessing + classifier for each candidate estimator.
2. Score them with stratified cross-validated ROC-AUC and pick the winner.
3. Optionally run a randomized hyper-parameter search on the winner.
4. Optionally wrap it in ``CalibratedClassifierCV`` so predicted probabilities
   are honest (a model that says "0.7" is right ~70% of the time).
5. Save everything as a :class:`~churnguard.bundle.ModelBundle`.

MLflow logging is best-effort: if MLflow is installed the run (params, CV scores,
chosen model) is logged to a local ``mlruns/`` directory; if not, training is
unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .bundle import ModelBundle
from .config import Config, load_config
from .data import load_dataset
from .features import build_preprocessor, engineer_features, split_xy
from .tracking import log_training_run


def _candidate_estimators(seed: int) -> dict[str, Any]:
    """Factory of the candidate classifiers, keyed by config name."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def _param_distributions() -> dict[str, dict]:
    """Randomized-search spaces, keyed by candidate name (prefixed ``clf__``)."""
    return {
        "logistic_regression": {"clf__C": loguniform(1e-2, 1e2)},
        "random_forest": {
            "clf__n_estimators": randint(200, 600),
            "clf__max_depth": randint(4, 20),
            "clf__min_samples_leaf": randint(5, 40),
        },
        "gradient_boosting": {
            "clf__n_estimators": randint(100, 400),
            "clf__learning_rate": loguniform(1e-2, 3e-1),
            "clf__max_depth": randint(2, 5),
        },
    }


@dataclass
class TrainingResult:
    bundle: ModelBundle
    best_name: str
    cv_scores: dict[str, float]
    best_params: dict[str, Any]
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def model(self) -> Pipeline:
        """The serving pipeline (kept for backwards-compatible callers)."""
        return self.bundle.pipeline


def train(cfg: Config | None = None) -> TrainingResult:
    """Run the full training routine and return the fitted bundle."""
    cfg = cfg or load_config()
    seed = cfg.seed

    frame = engineer_features(load_dataset(cfg))
    X, y = split_xy(frame, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("data", "test_size"), random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor(X_train, cfg)
    candidates = _candidate_estimators(seed)
    wanted = cfg.get("model", "candidates")
    cv = StratifiedKFold(
        n_splits=cfg.get("model", "cv_folds"), shuffle=True, random_state=seed
    )
    scoring = cfg.get("model", "scoring")

    # 1-2. Compare candidates by cross-validated score.
    cv_scores: dict[str, float] = {}
    for name in wanted:
        pipe = Pipeline(steps=[("pre", clone(preprocessor)), ("clf", candidates[name])])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_scores[name] = float(np.mean(scores))
    best_name = max(cv_scores, key=lambda name: cv_scores[name])

    # 3. Optionally tune the winner.
    best_params: dict[str, Any] = {}
    winner = Pipeline(
        steps=[("pre", clone(preprocessor)), ("clf", candidates[best_name])]
    )
    if cfg.get("model", "tuning", "enabled"):
        search = RandomizedSearchCV(
            winner,
            _param_distributions()[best_name],
            n_iter=cfg.get("model", "tuning", "n_iter"),
            scoring=scoring,
            cv=cv,
            random_state=seed,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        base_pipeline = search.best_estimator_
        best_params = {k: _jsonable(v) for k, v in search.best_params_.items()}
        cv_scores[best_name] = float(search.best_score_)
    else:
        base_pipeline = winner.fit(X_train, y_train)

    # 4. Optionally calibrate probabilities (serving model).
    if cfg.get("model", "calibration", "enabled"):
        serving: Pipeline = CalibratedClassifierCV(
            clone(base_pipeline),
            method=cfg.get("model", "calibration", "method"),
            cv=cfg.get("model", "cv_folds"),
        )
        serving.fit(X_train, y_train)
    else:
        serving = base_pipeline

    # Background sample (transformed) for the SHAP explainer at serving time.
    pre = base_pipeline.named_steps["pre"]
    X_train_trans = pre.transform(X_train)
    rng = np.random.default_rng(seed)
    n_bg = min(100, X_train_trans.shape[0])
    bg_idx = rng.choice(X_train_trans.shape[0], size=n_bg, replace=False)

    bundle = ModelBundle(
        pipeline=serving,
        base_pipeline=base_pipeline,
        feature_names=list(X_train.columns),
        model_name=best_name,
        background=np.asarray(X_train_trans)[bg_idx],
        transformed_feature_names=list(pre.get_feature_names_out()),
    )

    log_training_run(cfg, best_name, cv_scores, best_params)

    return TrainingResult(
        bundle=bundle,
        best_name=best_name,
        cv_scores=cv_scores,
        best_params=best_params,
        X_test=X_test,
        y_test=y_test,
    )


def _jsonable(value: Any) -> Any:
    """Make numpy scalars JSON/log friendly."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def train_and_save(cfg: Config | None = None) -> TrainingResult:
    """Train and persist the winning bundle to the configured path."""
    cfg = cfg or load_config()
    result = train(cfg)
    result.bundle.save(cfg.path("artifacts", "model_path"))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    res = train_and_save()
    ranked = sorted(res.cv_scores.items(), key=lambda kv: kv[1], reverse=True)
    print("Cross-validated ROC-AUC:")
    for name, score in ranked:
        marker = "  <- selected" if name == res.best_name else ""
        print(f"  {name:<22} {score:.4f}{marker}")
    if res.best_params:
        print(f"Tuned params: {res.best_params}")
