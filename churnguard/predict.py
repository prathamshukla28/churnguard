"""Inference helpers shared by the CLI and the API.

Loads the persisted :class:`~churnguard.bundle.ModelBundle` once (cached) and
turns raw customer records into churn probabilities, applying the same feature
engineering used during training and the business-chosen decision threshold.
Optionally attaches per-prediction SHAP explanations.

As a CLI it scores a whole CSV:
    python -m churnguard.predict customers.csv --out scored.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .bundle import ModelBundle
from .config import Config, load_config
from .explain import explain_records
from .features import engineer_features


class ModelNotTrainedError(RuntimeError):
    """Raised when no model artifact exists yet."""


_BUNDLE_CACHE: dict[str, ModelBundle] = {}


def load_bundle(cfg: Config | None = None) -> ModelBundle:
    """Load and cache the trained model bundle from disk."""
    cfg = cfg or load_config()
    path = cfg.path("artifacts", "model_path")
    key = str(path)
    if key not in _BUNDLE_CACHE:
        if not path.exists():
            raise ModelNotTrainedError(
                f"No model at {path}. Run `python -m churnguard.train` first."
            )
        _BUNDLE_CACHE[key] = ModelBundle.load(path)
    return _BUNDLE_CACHE[key]


# Backwards-compatible alias + cache hook used by tests/retraining.
def load_model(cfg: Config | None = None) -> ModelBundle:
    return load_bundle(cfg)


load_model.cache_clear = _BUNDLE_CACHE.clear  # type: ignore[attr-defined]
load_bundle.cache_clear = _BUNDLE_CACHE.clear  # type: ignore[attr-defined]


def _risk_band(p: float) -> str:
    return "high" if p >= 0.66 else "medium" if p >= 0.33 else "low"


def predict_records(
    records: list[dict],
    cfg: Config | None = None,
    threshold: float | None = None,
    explain: bool = False,
) -> list[dict]:
    """Score a batch of raw customer records.

    Returns one dict per record with the churn probability, the thresholded
    decision, a risk band, and (when ``explain``) the top SHAP risk drivers.
    """
    cfg = cfg or load_config()
    bundle = load_bundle(cfg)
    thr = bundle.threshold if threshold is None else threshold

    frame = engineer_features(pd.DataFrame(records))
    frame = frame.reindex(columns=bundle.feature_names)
    proba = bundle.pipeline.predict_proba(frame)[:, 1]

    reasons = explain_records(bundle, frame) if explain else [None] * len(frame)

    results = []
    for p, why in zip(proba, reasons, strict=True):
        p = float(p)
        row = {
            "churn_probability": round(p, 4),
            "will_churn": bool(p >= thr),
            "risk_band": _risk_band(p),
        }
        if explain:
            row["reasons"] = why
        results.append(row)
    return results


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of customers for churn.")
    parser.add_argument("input", help="CSV with customer rows (training schema).")
    parser.add_argument("--out", help="Where to write the scored CSV (default: stdout).")
    parser.add_argument("--explain", action="store_true", help="Add the top risk driver.")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input)
    results = predict_records(frame.to_dict(orient="records"), explain=args.explain)
    scored = frame.copy()
    scored["churn_probability"] = [r["churn_probability"] for r in results]
    scored["will_churn"] = [r["will_churn"] for r in results]
    scored["risk_band"] = [r["risk_band"] for r in results]
    if args.explain:
        scored["top_reason"] = [
            r["reasons"][0]["feature"] if r["reasons"] else "" for r in results
        ]

    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"Wrote {len(scored)} scored rows to {args.out}")
    else:
        scored.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
