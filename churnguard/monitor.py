"""Data, prediction and performance drift monitoring.

A deployed model silently decays as the world shifts underneath it; a model that
is never watched is a liability. This module compares a *live* batch of customers
against a **reference profile** captured at evaluation time and quantifies how far
the inputs, the predicted-risk distribution and (when ground-truth labels arrive)
the model's accuracy have drifted.

Drift is measured with the **Population Stability Index (PSI)** — the standard in
churn and credit scoring — computed per feature against the reference bucketing:

    PSI = Σ (actual%% − expected%%) · ln(actual%% / expected%%)

Rule of thumb: ``PSI < 0.10`` stable, ``0.10–0.25`` moderate drift (investigate),
``> 0.25`` significant drift (retrain). The thresholds live in ``config.yaml``.

Implemented without a monitoring framework on purpose: PSI is a few lines of
NumPy, and a self-contained reference profile (a small JSON committed with the
repo) means monitoring works the moment someone clones it — no server to stand up.
``scripts`` / the ``/monitor`` API endpoint both call :func:`compute_drift`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .config import Config, load_config

_EPS = 1e-6
_PRED_EDGES = np.linspace(0.0, 1.0, 11)  # 10 fixed buckets for the score distribution


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _numeric_edges(values: np.ndarray, n_bins: int = 10) -> list[float]:
    """Quantile bin edges for a numeric feature (deduplicated, open-ended)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if edges.size < 2:  # constant feature
        edges = np.array([values.min() - _EPS, values.max() + _EPS])
    edges[0], edges[-1] = -np.inf, np.inf
    return [float(e) for e in edges]


def _bucket_pct(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Fraction of ``values`` falling in each ``edges`` bucket."""
    counts, _ = np.histogram(values, bins=np.asarray(edges))
    pct = counts / max(counts.sum(), 1)
    return pct


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    """Population Stability Index between two bucketed distributions."""
    e = np.clip(np.asarray(expected_pct, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual_pct, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(psi: float, warn: float, alert: float) -> str:
    return "alert" if psi >= alert else "warn" if psi >= warn else "ok"


def _worst(statuses: list[str]) -> str:
    for level in ("alert", "warn"):
        if level in statuses:
            return level
    return "ok"


# --------------------------------------------------------------------------- #
# Reference profile
# --------------------------------------------------------------------------- #
def build_reference(
    X: pd.DataFrame,
    proba: np.ndarray,
    cfg: Config,
    y: np.ndarray | None = None,
    baseline_roc_auc: float | None = None,
) -> dict[str, Any]:
    """Capture the reference distribution the live stream is later compared to."""
    numeric_cols = [c for c in cfg.get("features", "numeric") if c in X.columns]
    numeric: dict[str, Any] = {}
    for col in numeric_cols:
        vals = pd.to_numeric(X[col], errors="coerce").dropna().to_numpy()
        edges = _numeric_edges(vals)
        numeric[col] = {
            "edges": edges,
            "ref_pct": [float(p) for p in _bucket_pct(vals, edges)],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    categorical: dict[str, Any] = {}
    for col in X.columns:
        if col in numeric_cols:
            continue
        counts = X[col].astype("string").fillna("<NA>").value_counts(normalize=True)
        categorical[col] = {str(k): float(v) for k, v in counts.items()}

    pred_pct = _bucket_pct(np.asarray(proba, dtype=float), list(_PRED_EDGES))
    return {
        "created_at": _now(),
        "n": int(len(X)),
        "numeric": numeric,
        "categorical": categorical,
        "prediction": {"ref_pct": [float(p) for p in pred_pct]},
        "baseline_roc_auc": float(baseline_roc_auc) if baseline_roc_auc is not None else None,
        "churn_rate": float(np.mean(y)) if y is not None else None,
    }


def save_reference(reference: dict[str, Any], cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    path = cfg.path("artifacts", "reference_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return path


def load_reference(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    path = cfg.path("artifacts", "reference_path")
    if not path.exists():
        raise FileNotFoundError(
            f"No reference profile at {path}. Run `python -m churnguard.evaluate` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Drift computation
# --------------------------------------------------------------------------- #
def compute_drift(
    reference: dict[str, Any],
    X: pd.DataFrame,
    proba: np.ndarray,
    cfg: Config | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare a live batch against the reference profile and grade the drift."""
    cfg = cfg or load_config()
    warn = float(cfg.get("monitoring", "psi_warn", default=0.1))
    alert = float(cfg.get("monitoring", "psi_alert", default=0.25))

    features: list[dict[str, Any]] = []
    for col, spec in reference.get("numeric", {}).items():
        if col not in X.columns:
            continue
        vals = pd.to_numeric(X[col], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        cur_pct = _bucket_pct(vals, spec["edges"])
        psi = _psi(np.asarray(spec["ref_pct"]), cur_pct)
        features.append({
            "feature": col,
            "type": "numeric",
            "psi": round(psi, 4),
            "mean_shift": round(float(np.mean(vals)) - spec["mean"], 4),
            "status": _status(psi, warn, alert),
        })

    for col, ref_map in reference.get("categorical", {}).items():
        if col not in X.columns:
            continue
        cur = X[col].astype("string").fillna("<NA>").value_counts(normalize=True)
        cur_map = {str(k): float(v) for k, v in cur.items()}
        cats = list(dict.fromkeys([*ref_map.keys(), *cur_map.keys()]))
        ref_pct = np.array([ref_map.get(c, 0.0) for c in cats])
        cur_pct = np.array([cur_map.get(c, 0.0) for c in cats])
        psi = _psi(ref_pct, cur_pct)
        features.append({
            "feature": col,
            "type": "categorical",
            "psi": round(psi, 4),
            "status": _status(psi, warn, alert),
        })

    features.sort(key=lambda f: f["psi"], reverse=True)

    pred_ref = np.asarray(reference["prediction"]["ref_pct"])
    pred_cur = _bucket_pct(np.asarray(proba, dtype=float), list(_PRED_EDGES))
    pred_psi = _psi(pred_ref, pred_cur)
    prediction_drift = {
        "psi": round(pred_psi, 4),
        "mean_score": round(float(np.mean(proba)), 4),
        "status": _status(pred_psi, warn, alert),
    }

    performance = None
    base_auc = reference.get("baseline_roc_auc")
    if y is not None and base_auc is not None and len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score

        cur_auc = float(roc_auc_score(y, proba))
        drop = base_auc - cur_auc
        pwarn = float(cfg.get("monitoring", "performance_drop_warn", default=0.03))
        palert = float(cfg.get("monitoring", "performance_drop_alert", default=0.05))
        performance = {
            "baseline_roc_auc": round(base_auc, 4),
            "current_roc_auc": round(cur_auc, 4),
            "delta": round(-drop, 4),
            "status": "alert" if drop >= palert else "warn" if drop >= pwarn else "ok",
        }

    statuses = [f["status"] for f in features] + [prediction_drift["status"]]
    if performance:
        statuses.append(performance["status"])

    return {
        "generated_at": _now(),
        "n": int(len(X)),
        "reference_created_at": reference.get("created_at"),
        "overall_status": _worst(statuses),
        "features": features,
        "prediction_drift": prediction_drift,
        "performance": performance,
        "summary": {
            "n_features": len(features),
            "warn": sum(f["status"] == "warn" for f in features),
            "alert": sum(f["status"] == "alert" for f in features),
        },
    }


def monitor_frame(
    frame: pd.DataFrame, cfg: Config | None = None, bundle: ModelBundle | None = None
) -> dict[str, Any]:
    """Convenience: score a raw customer frame and return its drift report.

    If a ``Churn`` label column is present it is used for performance drift.
    """
    from .features import engineer_features
    from .predict import load_bundle

    cfg = cfg or load_config()
    bundle = bundle or load_bundle(cfg)
    reference = load_reference(cfg)

    target = cfg.get("data", "target")
    y = None
    if target in frame.columns:
        y = (frame[target] == "Yes").astype(int).to_numpy()
        frame = frame.drop(columns=[target])

    engineered = engineer_features(frame).reindex(columns=bundle.feature_names)
    proba = bundle.pipeline.predict_proba(engineered)[:, 1]
    return compute_drift(reference, engineered, proba, cfg, y=y)


def _cli(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Check a CSV batch for drift vs. reference.")
    parser.add_argument("input", help="CSV of recent customer rows (training schema).")
    parser.add_argument("--out", help="Write the JSON drift report here (default: stdout).")
    args = parser.parse_args(argv)

    report = monitor_frame(pd.read_csv(args.input))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote drift report to {args.out} (status: {report['overall_status']})")
    else:
        print(text)
    return 0 if report["overall_status"] != "alert" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
