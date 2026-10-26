"""Model evaluation: metrics, figures and a model card.

Produces, for the held-out test split:

* ``reports/metrics.json`` — headline metrics (ROC-AUC, PR-AUC, F1, etc.) plus
  the cross-validation scores from training.
* ``reports/figures/*.png`` — ROC curve, precision-recall curve, confusion
  matrix and the top permutation-importance drivers.
* ``reports/model_card.md`` — a short, human-readable model card.

Matplotlib uses the non-interactive "Agg" backend so this runs head-less in CI
and Docker.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import Config, load_config
from .train import TrainingResult, train

DECISION_THRESHOLD = 0.5


def _figures_dir(cfg: Config):
    path = cfg.path("artifacts", "figures_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_roc(y_true, y_proba, auc, out_path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#4f46e5")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pr(y_true, y_proba, ap, out_path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})", color="#0891b2")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(y_true, y_pred, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_importance(model, X_test, y_test, seed, out_path) -> list[tuple[str, float]]:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=seed, scoring="roc_auc"
    )
    order = np.argsort(result.importances_mean)[::-1][:10]
    names = [X_test.columns[i] for i in order]
    values = [float(result.importances_mean[i]) for i in order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names[::-1], values[::-1], color="#4f46e5")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Top churn drivers (permutation importance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return list(zip(names, values, strict=True))


def evaluate(cfg: Config | None = None, result: TrainingResult | None = None) -> dict:
    """Evaluate the model on the test split and write all reports."""
    cfg = cfg or load_config()
    result = result or train(cfg)
    model, X_test, y_test = result.model, result.X_test, result.y_test

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "model": result.best_name,
        "n_test": len(y_test),
        "churn_rate": round(float(y_test.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "threshold": DECISION_THRESHOLD,
        "cv_roc_auc": {k: round(v, 4) for k, v in result.cv_scores.items()},
    }

    figures = _figures_dir(cfg)
    _plot_roc(y_test, y_proba, metrics["roc_auc"], figures / "roc_curve.png")
    _plot_pr(y_test, y_proba, metrics["pr_auc"], figures / "pr_curve.png")
    _plot_confusion(y_test, y_pred, figures / "confusion_matrix.png")
    importances = _plot_importance(
        model, X_test, y_test, cfg.seed, figures / "feature_importance.png"
    )
    metrics["top_drivers"] = [
        {"feature": n, "importance": round(v, 4)} for n, v in importances[:5]
    ]

    metrics_path = cfg.path("artifacts", "metrics_path")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write_model_card(cfg, metrics)
    return metrics


def _write_model_card(cfg: Config, metrics: dict) -> None:
    drivers = "\n".join(
        f"- `{d['feature']}` (importance {d['importance']})"
        for d in metrics["top_drivers"]
    )
    card = f"""# Model card — ChurnGuard

**Date:** {datetime.now(tz=timezone.utc).date().isoformat()}
**Model:** {metrics['model']}
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not intended for automated, consequential decisions without a
human in the loop.

## Data
Telco-style customer dataset ({cfg.get('data', 'n_rows')} rows). See
`churnguard/data.py` for the schema and the (synthetic) generating process;
swap in the real Kaggle Telco CSV at `data/churn.csv` to retrain on genuine data.

## Metrics (held-out test split, n = {metrics['n_test']})
| Metric | Value |
| --- | --- |
| ROC-AUC | {metrics['roc_auc']} |
| PR-AUC | {metrics['pr_auc']} |
| Precision | {metrics['precision']} |
| Recall | {metrics['recall']} |
| F1 | {metrics['f1']} |

Baseline churn rate in the test split: {metrics['churn_rate']}.

## Top churn drivers
{drivers}

## Limitations
- Trained on a single snapshot; no concept-drift monitoring.
- Decision threshold fixed at {metrics['threshold']}; tune it to the business
  cost of false positives vs. false negatives before production use.
"""
    cfg.path("artifacts", "model_card_path").write_text(card, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    m = evaluate()
    print(json.dumps(m, indent=2))
