"""Model evaluation: metrics, figures, a business threshold and a model card.

For the held-out test split this writes:

* ``reports/metrics.json`` — ROC-AUC, PR-AUC, Brier score (calibration), the
  business-chosen threshold and the precision/recall/F1 at that threshold.
* ``reports/figures/*.png`` — ROC, precision-recall, confusion matrix, top
  permutation-importance drivers, a reliability (calibration) curve, and a
  cost-vs-threshold curve.
* ``reports/model_card.md`` — a short model card.

It also chooses the decision threshold that minimises expected business cost
(a missed churner costs ``cost_fn``; a wasted retention offer costs ``cost_fp``)
and writes it back into the saved model bundle so serving uses it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

from .config import Config, load_config
from .tracking import log_evaluation
from .train import TrainingResult, train

_ACCENT = "#4f46e5"
_ACCENT2 = "#0891b2"


def _figures_dir(cfg: Config):
    path = cfg.path("artifacts", "figures_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, cost_fn: float, cost_fp: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the threshold minimising expected business cost, and the curve."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in grid:
        pred = y_proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        costs.append(cost_fn * fn + cost_fp * fp)
    costs_arr = np.asarray(costs, dtype=float)
    best = float(grid[int(np.argmin(costs_arr))])
    return best, grid, costs_arr


def _plot_roc(y, p, auc, out):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color=_ACCENT)
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_pr(y, p, ap, out):
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})", color=_ACCENT2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_confusion(y, pred, out):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_importance(model, X_test, y_test, seed, out):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=seed, scoring="roc_auc"
    )
    order = np.argsort(result.importances_mean)[::-1][:10]
    names = [X_test.columns[i] for i in order]
    values = [float(result.importances_mean[i]) for i in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names[::-1], values[::-1], color=_ACCENT)
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Top churn drivers (permutation importance)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return list(zip(names, values, strict=True))


def _plot_calibration(y, p, brier, out):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=_ACCENT, label=f"Model (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability curve"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_cost(grid, costs, best, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(grid, costs, color=_ACCENT2)
    ax.axvline(best, color=_ACCENT, linestyle="--", label=f"Chosen = {best:.2f}")
    ax.axvline(0.5, color="#9ca3af", linestyle=":", label="Default = 0.50")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Expected business cost")
    ax.set_title("Cost vs. threshold"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def evaluate(cfg: Config | None = None, result: TrainingResult | None = None) -> dict:
    """Evaluate on the test split, pick a threshold, and write all reports."""
    cfg = cfg or load_config()
    result = result or train(cfg)
    model, X_test, y_test = result.bundle.pipeline, result.X_test, result.y_test
    y_true = y_test.to_numpy()

    y_proba = model.predict_proba(X_test)[:, 1]

    cost_fn = cfg.get("threshold", "cost_fn")
    cost_fp = cfg.get("threshold", "cost_fp")
    threshold, grid, costs = choose_threshold(y_true, y_proba, cost_fn, cost_fp)
    y_pred = (y_proba >= threshold).astype(int)

    # Persist the chosen threshold into the bundle so serving uses it.
    result.bundle.threshold = threshold
    result.bundle.save(cfg.path("artifacts", "model_path"))

    metrics = {
        "model": result.best_name,
        "tuned": bool(cfg.get("model", "tuning", "enabled")),
        "calibrated": bool(cfg.get("model", "calibration", "enabled")),
        "n_test": len(y_test),
        "churn_rate": round(float(y_test.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "threshold": round(threshold, 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "cost_at_threshold": float(costs.min()),
        "cost_at_0.5": float(cost_fn * int(((y_proba < 0.5) & (y_true == 1)).sum())
                             + cost_fp * int(((y_proba >= 0.5) & (y_true == 0)).sum())),
        "cv_roc_auc": {k: round(v, 4) for k, v in result.cv_scores.items()},
        "best_params": result.best_params,
    }

    figures = _figures_dir(cfg)
    _plot_roc(y_true, y_proba, metrics["roc_auc"], figures / "roc_curve.png")
    _plot_pr(y_true, y_proba, metrics["pr_auc"], figures / "pr_curve.png")
    _plot_confusion(y_true, y_pred, figures / "confusion_matrix.png")
    _plot_calibration(y_true, y_proba, metrics["brier"], figures / "calibration_curve.png")
    _plot_cost(grid, costs, threshold, figures / "cost_curve.png")
    importances = _plot_importance(
        model, X_test, y_test, cfg.seed, figures / "feature_importance.png"
    )
    metrics["top_drivers"] = [
        {"feature": n, "importance": round(v, 4)} for n, v in importances[:5]
    ]

    cfg.path("artifacts", "metrics_path").parent.mkdir(parents=True, exist_ok=True)
    cfg.path("artifacts", "metrics_path").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_model_card(cfg, metrics)
    log_evaluation({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def _write_model_card(cfg: Config, metrics: dict) -> None:
    drivers = "\n".join(
        f"- `{d['feature']}` (importance {d['importance']})"
        for d in metrics["top_drivers"]
    )
    saved = cfg.get("threshold", "cost_fn"), cfg.get("threshold", "cost_fp")
    card = f"""# Model card — ChurnGuard

**Date:** {datetime.now(tz=timezone.utc).date().isoformat()}
**Model:** {metrics['model']} (tuned={metrics['tuned']}, calibrated={metrics['calibrated']})
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not for automated, consequential decisions without a human in the loop.

## Data
Telco-style customer dataset ({cfg.get('data', 'n_rows')} rows). See
`churnguard/data.py`; run `make data-real` to train on the genuine Kaggle Telco CSV.

## Metrics (held-out test split, n = {metrics['n_test']})
| Metric | Value |
| --- | --- |
| ROC-AUC | {metrics['roc_auc']} |
| PR-AUC | {metrics['pr_auc']} |
| Brier score (calibration, lower=better) | {metrics['brier']} |
| Precision @ threshold | {metrics['precision']} |
| Recall @ threshold | {metrics['recall']} |
| F1 @ threshold | {metrics['f1']} |

## Decision threshold
Chosen at **{metrics['threshold']}** (not the default 0.5) to minimise expected
business cost with cost_fn={saved[0]}, cost_fp={saved[1]}. This cut expected cost
from {metrics['cost_at_0.5']:.0f} (at 0.5) to {metrics['cost_at_threshold']:.0f}.

## Top churn drivers
{drivers}

## Limitations
- Trained on a single snapshot; no concept-drift monitoring.
- Threshold reflects the configured cost ratio; retune if the business changes.
"""
    cfg.path("artifacts", "model_card_path").write_text(card, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    m = evaluate()
    print(json.dumps(m, indent=2))
