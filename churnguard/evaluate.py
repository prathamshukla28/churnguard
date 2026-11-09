"""Model evaluation: score the frozen model once on the untouched test split.

``evaluate`` **loads the trained artifact** (``models/churnguard.joblib``); it never
retrains and never chooses anything. The decision threshold was already fixed at
training time from out-of-fold predictions on the training split, so every
number reported here (ROC-AUC, PR-AUC, Brier, precision/recall/F1 and the cost
saving at the threshold) is an honest held-out estimate.

It writes:

* ``reports/metrics.json`` — headline metrics with bootstrap 95% intervals, cost
  at the threshold vs. the naive 0.5 cut, and per-segment performance.
* ``reports/figures/*.png`` — ROC, precision-recall, confusion matrix, permutation
  importance, reliability curve, and the cost curve the threshold came from.
* ``reports/model_card.md`` — the model card.
* ``reports/reference_profile.json`` — the drift-monitoring reference.

It does **not** write ``reports/baseline_metrics.json``: that file is the frozen,
committed CI quality floor and changes only via ``make promote-baseline``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .bundle import ModelBundle
from .config import Config, load_config
from .data import dataset_fingerprint
from .decision import THRESHOLD_GRID, expected_cost
from .monitor import build_reference, save_reference
from .predict import ModelNotTrainedError
from .tracking import log_evaluation
from .train import load_xy, split_train_test

_ACCENT = "#4f46e5"
_ACCENT2 = "#0891b2"
_MUTED = "#9ca3af"

# Segments reported in metrics.json / the model card ("where does it fail?").
SEGMENT_COLUMNS = ("Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "tenure_band")
_TENURE_BINS = [-np.inf, 12, 24, 48, np.inf]
_TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
_MIN_SEGMENT_AUC_N = 30


class StaleModelError(RuntimeError):
    """The saved model was trained on different data than is now configured."""


# --------------------------------------------------------------------------- #
# Loading the frozen model and its held-out split
# --------------------------------------------------------------------------- #
def load_trained_bundle(cfg: Config) -> ModelBundle:
    path = cfg.path("artifacts", "model_path")
    if not path.exists():
        raise ModelNotTrainedError(f"No model at {path}. Run `make train` first.")
    return ModelBundle.load(path)


def load_test_split(cfg: Config, bundle: ModelBundle) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild exactly the rows the bundle held out, from exactly its data."""
    if bundle.data_sha256 and bundle.data_sha256 != dataset_fingerprint(cfg):
        raise StaleModelError(
            "The saved model was trained on a different dataset than the one now "
            "configured; retrain (`make train`) before evaluating."
        )
    X, y = load_xy(cfg)
    if bundle.test_index is not None:
        idx = pd.Index(bundle.test_index)
        X_test, y_test = X.loc[idx], y.loc[idx]
    else:  # bundles from before the split was recorded: same seeded split
        _, X_test, _, y_test = split_train_test(X, y, cfg)
    return X_test.reindex(columns=bundle.feature_names), y_test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric: Any, seed: int, n_boot: int = 500
) -> list[float]:
    """Percentile 95% bootstrap interval for ``metric(y, p)``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _tenure_band(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, bins=_TENURE_BINS, labels=_TENURE_LABELS).astype("string")


def segment_metrics(
    X: pd.DataFrame, y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, list[dict[str, Any]]]:
    """Per-segment n, churn rate, ROC-AUC and precision/recall at the threshold."""
    frame = X.copy()
    if "tenure" in frame.columns:
        frame["tenure_band"] = _tenure_band(frame["tenure"])
    y_pred = (y_proba >= threshold).astype(int)
    out: dict[str, list[dict[str, Any]]] = {}
    for col in SEGMENT_COLUMNS:
        if col not in frame.columns:
            continue
        rows = []
        for level, mask in frame.groupby(frame[col].astype("string"), dropna=False).groups.items():
            pos = frame.index.get_indexer(mask)
            yt, yp, pr = y_true[pos], y_pred[pos], y_proba[pos]
            both = len(np.unique(yt)) == 2
            rows.append(
                {
                    "segment": str(level),
                    "n": int(len(pos)),
                    "churn_rate": round(float(yt.mean()), 4),
                    "roc_auc": (
                        round(float(roc_auc_score(yt, pr)), 4)
                        if both and len(pos) >= _MIN_SEGMENT_AUC_N
                        else None
                    ),
                    "flagged_rate": round(float(yp.mean()), 4),
                    "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                    "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
                }
            )
        out[col] = sorted(rows, key=lambda r: -r["n"])
    return out


def compute_metrics(
    bundle: ModelBundle, X_test: pd.DataFrame, y_test: pd.Series, cfg: Config
) -> dict[str, Any]:
    """All held-out metrics for a frozen bundle. Pure: writes nothing."""
    y_true = y_test.to_numpy()
    y_proba = bundle.pipeline.predict_proba(X_test)[:, 1]
    threshold = float(bundle.threshold)
    y_pred = (y_proba >= threshold).astype(int)
    cost_fn = float(cfg.get("threshold", "cost_fn"))
    cost_fp = float(cfg.get("threshold", "cost_fp"))
    cost_thr = expected_cost(y_true, y_proba, threshold, cost_fn, cost_fp)
    cost_half = expected_cost(y_true, y_proba, 0.5, cost_fn, cost_fp)
    prevalence = float(y_true.mean())

    return {
        "model": bundle.model_name,
        "tuned": bool(cfg.get("model", "tuning", "enabled")),
        "calibrated": bool(cfg.get("model", "calibration", "enabled")),
        "data_source": bundle.data_source,
        "data_sha256": bundle.data_sha256,
        "n_test": int(len(y_test)),
        "churn_rate": round(prevalence, 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "roc_auc_ci95": _bootstrap_ci(y_true, y_proba, roc_auc_score, cfg.seed),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "pr_auc_ci95": _bootstrap_ci(y_true, y_proba, average_precision_score, cfg.seed),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        # Brier of always predicting the base rate: the bar calibration must beat.
        "brier_base_rate": round(prevalence * (1 - prevalence), 4),
        "threshold": round(threshold, 3),
        "threshold_source": bundle.threshold_source,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "flagged_rate": round(float(y_pred.mean()), 4),
        "cost_at_threshold": cost_thr,
        "cost_at_0.5": cost_half,
        "cost_reduction_vs_0.5": round(1 - cost_thr / cost_half, 4) if cost_half else None,
        "cv_roc_auc": {k: round(v, 4) for k, v in (bundle.cv_scores or {}).items()},
        "best_params": bundle.best_params or {},
        "segments": segment_metrics(X_test, y_true, y_proba, threshold),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figures_dir(cfg: Config) -> Path:
    path = cfg.path("artifacts", "figures_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: Any, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_roc(y: np.ndarray, p: np.ndarray, auc: float, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color=_ACCENT)
    ax.plot([0, 1], [0, 1], "--", color=_MUTED, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test split)")
    ax.legend(loc="lower right")
    _save(fig, out)


def _plot_pr(y: np.ndarray, p: np.ndarray, ap: float, out: Path) -> None:
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})", color=_ACCENT2)
    ax.axhline(float(np.mean(y)), linestyle="--", color=_MUTED, label="Base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test split)")
    ax.legend(loc="upper right")
    _save(fig, out)


def _plot_confusion(y: np.ndarray, pred: np.ndarray, threshold: float, out: Path) -> None:
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ {threshold:.2f} (test)")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    _save(fig, out)


def _plot_importance(
    model: Any, X: pd.DataFrame, y: pd.Series, seed: int, out: Path
) -> list[tuple[str, float]]:
    result = permutation_importance(model, X, y, n_repeats=8, random_state=seed, scoring="roc_auc")
    order = np.argsort(result.importances_mean)[::-1][:10]
    names = [str(X.columns[i]) for i in order]
    values = [float(result.importances_mean[i]) for i in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names[::-1], values[::-1], color=_ACCENT)
    ax.set_xlabel("Drop in test ROC-AUC when shuffled")
    ax.set_title("Permutation importance (test split)")
    _save(fig, out)
    return list(zip(names, values, strict=True))


def _plot_calibration(y: np.ndarray, p: np.ndarray, brier: float, out: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color=_MUTED, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=_ACCENT, label=f"Model (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability curve (test split)")
    ax.legend(loc="upper left")
    _save(fig, out)


def _plot_cost(
    bundle: ModelBundle,
    y: np.ndarray,
    p: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    out: Path,
) -> None:
    """Cost curves, each scaled to its own maximum so train and test compare.

    The solid line is the out-of-fold training curve the threshold was chosen
    from; the dashed test curve is shown for reference only.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if bundle.threshold_curve:
        grid = np.asarray(bundle.threshold_curve["grid"])
        train_cost = np.asarray(bundle.threshold_curve["cost"])
        ax.plot(
            grid,
            train_cost / max(float(np.max(train_cost)), 1.0),
            color=_ACCENT2,
            label="Train, out-of-fold (used to choose)",
        )
    test_cost = np.asarray([expected_cost(y, p, t, cost_fn, cost_fp) for t in THRESHOLD_GRID])
    ax.plot(
        THRESHOLD_GRID,
        test_cost / max(float(np.max(test_cost)), 1.0),
        color=_MUTED,
        linestyle="--",
        label="Test (reported only, not used)",
    )
    ax.axvline(
        bundle.threshold, color=_ACCENT, linestyle="--", label=f"Chosen = {bundle.threshold:.2f}"
    )
    ax.axvline(0.5, color=_MUTED, linestyle=":", label="Default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (relative to max)")
    ax.set_title(f"Cost vs. threshold (cost_fn={cost_fn:g}, cost_fp={cost_fp:g})")
    ax.legend(fontsize=8)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(cfg: Config | None = None, bundle: ModelBundle | None = None) -> dict[str, Any]:
    """Score the trained model on its untouched test split and write all reports.

    Loads the saved bundle unless one is passed; never retrains, never re-picks
    the threshold, never touches the committed baseline.
    """
    cfg = cfg or load_config()
    bundle = bundle or load_trained_bundle(cfg)
    X_test, y_test = load_test_split(cfg, bundle)
    metrics = compute_metrics(bundle, X_test, y_test, cfg)

    y_true = y_test.to_numpy()
    y_proba = bundle.pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= bundle.threshold).astype(int)
    cost_fn = float(cfg.get("threshold", "cost_fn"))
    cost_fp = float(cfg.get("threshold", "cost_fp"))

    figures = _figures_dir(cfg)
    _plot_roc(y_true, y_proba, metrics["roc_auc"], figures / "roc_curve.png")
    _plot_pr(y_true, y_proba, metrics["pr_auc"], figures / "pr_curve.png")
    _plot_confusion(y_true, y_pred, bundle.threshold, figures / "confusion_matrix.png")
    _plot_calibration(y_true, y_proba, metrics["brier"], figures / "calibration_curve.png")
    _plot_cost(bundle, y_true, y_proba, cost_fn, cost_fp, figures / "cost_curve.png")
    importances = _plot_importance(
        bundle.pipeline, X_test, y_test, cfg.seed, figures / "feature_importance.png"
    )
    metrics["top_drivers"] = [{"feature": n, "importance": round(v, 4)} for n, v in importances[:5]]

    metrics_path = cfg.path("artifacts", "metrics_path")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_model_card(cfg, metrics, bundle)

    # Drift-monitoring reference, captured from the same held-out split.
    reference = build_reference(
        X_test, y_proba, cfg, y=y_true, baseline_roc_auc=float(roc_auc_score(y_true, y_proba))
    )
    save_reference(reference, cfg)

    log_evaluation({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def _segment_table(metrics: dict[str, Any], column: str) -> str:
    rows = metrics.get("segments", {}).get(column, [])
    lines = [
        "| Segment | n | Churn rate | ROC-AUC | Precision | Recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        auc = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        lines.append(
            f"| {column} = {r['segment']} | {r['n']} | {r['churn_rate']:.2f} | {auc} "
            f"| {r['precision']:.2f} | {r['recall']:.2f} |"
        )
    return "\n".join(lines)


def _write_model_card(cfg: Config, metrics: dict[str, Any], bundle: ModelBundle) -> None:
    drivers = "\n".join(
        f"- `{d['feature']}` (drop in test ROC-AUC when shuffled: {d['importance']})"
        for d in metrics["top_drivers"]
    )
    cost_fn, cost_fp = cfg.get("threshold", "cost_fn"), cfg.get("threshold", "cost_fp")
    lo, hi = metrics["roc_auc_ci95"]
    plo, phi = metrics["pr_auc_ci95"]
    reduction = metrics["cost_reduction_vs_0.5"]
    reduction_txt = "n/a" if reduction is None else f"{reduction:.0%}"
    if metrics["data_source"] == "synthetic":
        data_txt = (
            "**SYNTHETIC test data** (label generated from a hand-written formula). "
            "These numbers are circular and must not be quoted as model performance."
        )
    else:
        data_txt = (
            "Public IBM *Telco Customer Churn* sample dataset (7,043 customers, 21 "
            "columns; IBM describes it as a fictional telco). Downloaded from "
            f"`{cfg.get('data', 'real_url')}`, SHA-256 `{metrics['data_sha256']}`."
        )
    base_brier = metrics["brier_base_rate"]
    card = f"""# Model card — ChurnGuard

**Date:** {datetime.now(tz=timezone.utc).date().isoformat()}
**Model:** {metrics["model"]} (tuned={metrics["tuned"]}, calibrated={metrics["calibrated"]})
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not for automated, consequential decisions without a human in the loop.

## Data
{data_txt}

Split: stratified {1 - cfg.get("data", "test_size"):.0%} train / \
{cfg.get("data", "test_size"):.0%} test (seed {cfg.seed}). Model selection, tuning,
calibration and the decision threshold use the training split only; the test split
below is scored exactly once.

## Metrics (held-out test split, n = {metrics["n_test"]}, churn rate {metrics["churn_rate"]:.1%})
| Metric | Value |
| --- | --- |
| ROC-AUC | {metrics["roc_auc"]} (95% bootstrap CI {lo}–{hi}) |
| PR-AUC | {metrics["pr_auc"]} (95% CI {plo}–{phi}; base rate {metrics["churn_rate"]}) |
| Brier score (lower = better) | {metrics["brier"]} (base rate alone: {base_brier}) |
| Precision @ {metrics["threshold"]} | {metrics["precision"]} |
| Recall @ {metrics["threshold"]} | {metrics["recall"]} |
| F1 @ {metrics["threshold"]} | {metrics["f1"]} |
| Share of customers flagged | {metrics["flagged_rate"]:.1%} |

## Decision threshold
**{metrics["threshold"]}**, chosen to minimise expected cost with cost_fn={cost_fn},
cost_fp={cost_fp} on {metrics["threshold_source"]}. On the untouched test split it
gives a cost of {metrics["cost_at_threshold"]:.0f} vs. {metrics["cost_at_0.5"]:.0f} at
the default 0.5 ({reduction_txt} lower). The saving is only as real as the 5:1 cost
ratio, which is an assumption, not a measured business number.

## Where it is weaker (test split, by segment)
{_segment_table(metrics, "Contract")}

{_segment_table(metrics, "tenure_band")}

{_segment_table(metrics, "InternetService")}

Within-segment ROC-AUC is usually below the overall figure because contract type
and tenure, the strongest signals, partly define the segments. Watch the low-churn
segments: with few churners there, the threshold flags few of them (low recall).

## Top drivers (permutation importance on the test split)
{drivers}

These are associations the model uses, not causal effects.

## Explanations
Per-prediction SHAP reasons explain the **uncalibrated base estimator** in its raw
output space (log-odds for linear and gradient-boosted models). The probability
returned by the API comes from the **calibrated** wrapper. Calibration is
monotone per fold, so the direction and ranking of reasons is informative, but
SHAP impacts do not sum to the calibrated probability.

## Monitoring
`churnguard/monitor.py` (CLI and `POST /monitor`) computes PSI data drift,
prediction drift and, when labels are supplied, performance drift against the
reference profile captured at evaluation time. It runs on demand: nothing
schedules it or pages anyone yet.

## Limitations
- One static public snapshot; no timestamps, so no out-of-time validation. Real
  churn data shifts with pricing and competition.
- The threshold depends on an assumed 5:1 cost ratio; retune when real costs are known.
- Drift checks are on-demand only; wire them to a schedule and an alert before relying on them.
- No fairness audit beyond the per-segment table (e.g. `SeniorCitizen`, `gender` are inputs).
"""
    cfg.path("artifacts", "model_card_path").write_text(card, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    m = evaluate()
    print(json.dumps({k: v for k, v in m.items() if k != "segments"}, indent=2))
