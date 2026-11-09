"""Retraining with a data-change check, a regression gate and champion/challenger.

The honest version of "retrain monthly": training is seeded and deterministic,
so retraining on **identical data** reproduces the same model and proves
nothing. A run therefore first fingerprints the dataset (SHA-256) and compares
it with the data the current champion was trained on (registry production entry,
else the committed baseline). If nothing changed it stops with ``skipped``
unless ``force`` is set. A scheduled job only adds value once the data source is
refreshed with new customers and labels.

When the data did change:

1. train a challenger (without overwriting the serving model);
2. score it once on its untouched test split;
3. **regression gate** vs. the committed baseline -> ``regressed`` (no register);
4. register it as ``staging`` in the model registry;
5. promote only if it beats the champion; promotion saves it as the serving
   model and refreshes the reports. The committed baseline is *not* touched —
   freezing a new floor stays an explicit, reviewed ``make promote-baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import registry
from .config import Config, load_config
from .data import dataset_fingerprint
from .evaluate import compute_metrics, evaluate, load_test_split
from .quality import check_regression, load_baseline
from .train import train


@dataclass
class RetrainOutcome:
    status: str  # skipped | regressed | staged | promoted
    reason: str
    version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def retrain(cfg: Config | None = None, force: bool = False, promote: bool = True) -> RetrainOutcome:
    cfg = cfg or load_config()

    data_sha = dataset_fingerprint(cfg)
    champion, source = registry.champion_metrics(cfg)
    if not force and champion is not None and champion.get("data_sha256") == data_sha:
        return RetrainOutcome(
            "skipped",
            f"data unchanged since the champion ({source}, sha256 {data_sha[:12]}…); a "
            "seeded retrain on identical data reproduces the same model. Use --force to "
            "retrain anyway.",
        )

    result = train(cfg)
    X_test, y_test = load_test_split(cfg, result.bundle)
    metrics = compute_metrics(result.bundle, X_test, y_test, cfg)

    baseline = load_baseline(cfg)
    if baseline is not None:
        tol = float(cfg.get("registry", "regression_tolerance", default=0.02))
        ok, lines = check_regression(metrics, baseline, tol)
        if not ok:
            return RetrainOutcome("regressed", "\n".join(lines), metrics=metrics)

    version = registry.register_model(result.bundle, metrics, cfg, stage="staging")
    if not promote:
        return RetrainOutcome("staged", "promotion disabled (--no-promote)", version, metrics)

    should, reason = registry.evaluate_promotion(metrics, cfg)
    if not should:
        return RetrainOutcome("staged", reason, version, metrics)

    registry.promote(version, "production", cfg)
    result.bundle.save(cfg.path("artifacts", "model_path"))
    evaluate(cfg, bundle=result.bundle)
    return RetrainOutcome("promoted", reason, version, metrics)
