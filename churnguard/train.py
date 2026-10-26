"""Model training and selection.

Builds a full ``Pipeline`` = preprocessing + classifier for each candidate
estimator, scores them with stratified cross-validation on ROC-AUC, then refits
the winner on the full training split and persists it to disk together with the
train/test split indices needed by :mod:`churnguard.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import Config, load_config
from .data import load_dataset
from .features import build_preprocessor, engineer_features, split_xy


def _candidate_estimators(seed: int) -> dict[str, object]:
    """Factory of the candidate classifiers, keyed by config name."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


@dataclass
class TrainingResult:
    model: Pipeline
    best_name: str
    cv_scores: dict[str, float]
    X_test: pd.DataFrame
    y_test: pd.Series


def train(cfg: Config | None = None) -> TrainingResult:
    """Run the full training routine and return the fitted pipeline."""
    cfg = cfg or load_config()
    seed = cfg.seed

    frame = engineer_features(load_dataset(cfg))
    X, y = split_xy(frame, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.get("data", "test_size"),
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train, cfg)
    candidates = _candidate_estimators(seed)
    wanted = cfg.get("model", "candidates")
    cv = StratifiedKFold(
        n_splits=cfg.get("model", "cv_folds"), shuffle=True, random_state=seed
    )
    scoring = cfg.get("model", "scoring")

    cv_scores: dict[str, float] = {}
    pipelines: dict[str, Pipeline] = {}
    for name in wanted:
        pipe = Pipeline(
            steps=[("pre", preprocessor), ("clf", candidates[name])]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_scores[name] = float(np.mean(scores))
        pipelines[name] = pipe

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = pipelines[best_name].fit(X_train, y_train)

    return TrainingResult(
        model=best_model,
        best_name=best_name,
        cv_scores=cv_scores,
        X_test=X_test,
        y_test=y_test,
    )


def train_and_save(cfg: Config | None = None) -> TrainingResult:
    """Train and persist the winning pipeline to the configured path."""
    cfg = cfg or load_config()
    result = train(cfg)
    model_path = cfg.path("artifacts", "model_path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    res = train_and_save()
    ranked = sorted(res.cv_scores.items(), key=lambda kv: kv[1], reverse=True)
    print("Cross-validated ROC-AUC:")
    for name, score in ranked:
        marker = "  <- selected" if name == res.best_name else ""
        print(f"  {name:<22} {score:.4f}{marker}")
