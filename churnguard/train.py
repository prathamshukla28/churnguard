"""Model training, selection, tuning and calibration.

Pipeline:
1. Build ``Pipeline`` = preprocessing + classifier for each candidate estimator.
2. Score them with stratified cross-validated ROC-AUC and pick the winner.
3. Optionally run a randomized hyper-parameter search on the winner.
4. Optionally wrap it in ``CalibratedClassifierCV`` so predicted probabilities
   are honest (a model that says "0.7" is right ~70% of the time).
5. Choose the cost-minimising decision threshold on **out-of-fold** predictions
   of that same (calibrated) model over the training split. The held-out test
   split is never seen here; ``evaluate`` scores it once with the frozen model
   and threshold.
6. Save everything as a :class:`~churnguard.bundle.ModelBundle`, including which
   data trained it (SHA-256) and which rows were held out.

All model selection, tuning, calibration and threshold choices use the training
split only. (Tuning and threshold selection share the training folds, so the
out-of-fold cost curve is mildly optimistic; the test metrics are not.)

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
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .bundle import ModelBundle
from .config import Config, load_config
from .data import data_source, dataset_fingerprint, load_dataset
from .decision import choose_threshold
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


def load_xy(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """The full engineered feature matrix and target for the configured data."""
    return split_xy(engineer_features(load_dataset(cfg)), cfg)


def split_train_test(
    X: pd.DataFrame, y: pd.Series, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The single, seeded, stratified train/test split used everywhere."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("data", "test_size"), random_state=cfg.seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _serving_estimator(base_pipeline: Pipeline, cfg: Config) -> Any:
    """Unfitted serving model: the tuned pipeline, calibrated if enabled."""
    if cfg.get("model", "calibration", "enabled"):
        return CalibratedClassifierCV(
            clone(base_pipeline),
            method=cfg.get("model", "calibration", "method"),
            cv=cfg.get("model", "cv_folds"),
        )
    return clone(base_pipeline)


def select_threshold_oof(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Config,
    cv: StratifiedKFold,
) -> tuple[float, dict[str, list[float]]]:
    """Choose the decision threshold from out-of-fold training predictions.

    Each training row is scored by a serving-equivalent model (same params, same
    calibration) that never saw it, so the cost curve reflects unseen data
    without touching the test split.
    """
    oof = cross_val_predict(
        _serving_estimator(base_pipeline, cfg),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    threshold, grid, costs = choose_threshold(
        y_train.to_numpy(),
        oof,
        cfg.get("threshold", "cost_fn"),
        cfg.get("threshold", "cost_fp"),
    )
    curve = {
        "grid": [round(float(g), 4) for g in grid],
        "cost": [float(c) for c in costs],
    }
    return threshold, curve


def select_candidate(
    fold_scores: dict[str, np.ndarray], order: list[str], rule: str = "best"
) -> str:
    """Pick a candidate from per-fold CV scores.

    ``best`` takes the highest mean. ``one_standard_error`` takes the first
    candidate in ``order`` (simplest first) whose mean is within one standard
    error of the best mean — preferring simplicity when the data cannot tell
    the candidates apart.
    """
    means = {name: float(np.mean(s)) for name, s in fold_scores.items()}
    best = max(means, key=lambda name: means[name])
    if rule == "best":
        return best
    if rule != "one_standard_error":
        raise ValueError(f"Unknown model.selection_rule {rule!r}")
    scores = np.asarray(fold_scores[best], dtype=float)
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
    floor = means[best] - se
    return next(name for name in order if name in means and means[name] >= floor)


def _single_threaded(model: Any) -> None:
    """Set ``n_jobs=1`` on every fitted sub-estimator that has it.

    Parallel trees speed up training but make single-row serving slower (thread
    start-up dominates), so the saved artifact predicts on one thread.
    """
    stack = [model]
    while stack:
        est = stack.pop()
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
        if isinstance(est, Pipeline):
            stack.extend(step for _, step in est.steps)
        stack.extend(
            getattr(c, "estimator", None) for c in getattr(est, "calibrated_classifiers_", [])
        )
        stack = [e for e in stack if e is not None]


def train(cfg: Config | None = None) -> TrainingResult:
    """Run the full training routine and return the fitted bundle."""
    cfg = cfg or load_config()
    seed = cfg.seed

    X, y = load_xy(cfg)
    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)

    preprocessor = build_preprocessor(X_train, cfg)
    candidates = _candidate_estimators(seed)
    wanted = cfg.get("model", "candidates")
    cv = StratifiedKFold(n_splits=cfg.get("model", "cv_folds"), shuffle=True, random_state=seed)
    scoring = cfg.get("model", "scoring")

    # 1-2. Compare candidates by cross-validated score.
    fold_scores: dict[str, np.ndarray] = {}
    for name in wanted:
        pipe = Pipeline(steps=[("pre", clone(preprocessor)), ("clf", candidates[name])])
        fold_scores[name] = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_scores = {name: float(np.mean(scores)) for name, scores in fold_scores.items()}
    best_name = select_candidate(
        fold_scores, wanted, cfg.get("model", "selection_rule", default="best")
    )

    # 3. Optionally tune the winner.
    best_params: dict[str, Any] = {}
    winner = Pipeline(steps=[("pre", clone(preprocessor)), ("clf", candidates[best_name])])
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
        serving = _serving_estimator(base_pipeline, cfg).fit(X_train, y_train)
    else:
        serving = base_pipeline

    # 5. Business threshold from out-of-fold predictions on the training split.
    threshold, curve = select_threshold_oof(base_pipeline, X_train, y_train, cfg, cv)

    _single_threaded(serving)
    _single_threaded(base_pipeline)

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
        threshold=threshold,
        threshold_source="out-of-fold predictions on the training split",
        threshold_curve=curve,
        data_source=data_source(cfg),
        data_sha256=dataset_fingerprint(cfg),
        test_index=[int(i) for i in X_test.index],
        cv_scores=dict(cv_scores),
        best_params=dict(best_params),
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
    print(f"Decision threshold (out-of-fold, train split): {res.bundle.threshold:.2f}")
