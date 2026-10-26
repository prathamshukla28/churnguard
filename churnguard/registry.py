"""A lightweight, file-based model registry.

Every trained model is a versioned, immutable artifact with a lifecycle stage
(``staging`` → ``production`` → ``archived``). The registry records what was
trained, when, with which metrics, and which version is currently serving — the
provenance a team needs to answer "what is in production and why?" and to roll
back in one call.

It is deliberately file-based (a ``registry.json`` manifest beside the versioned
bundles) rather than a hosted MLflow Model Registry: no server to run, it is
trivially testable, and it version-controls cleanly. The champion/challenger
decision in :func:`evaluate_promotion` is the same logic a scheduled retraining
job (``scripts/retrain.py``) and a CI promotion gate would use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle import ModelBundle
from .config import Config, load_config

STAGES = ("staging", "production", "archived")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _registry_dir(cfg: Config) -> Path:
    path = (cfg.path("registry", "dir") if cfg.get("registry", "dir") else None)
    if path is None:  # pragma: no cover - config always sets it
        raise KeyError("registry.dir not configured")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(cfg: Config) -> Path:
    return _registry_dir(cfg) / "registry.json"


def load_manifest(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    path = _manifest_path(cfg)
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, Any], cfg: Config) -> None:
    _manifest_path(cfg).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def models(cfg: Config | None = None) -> list[dict[str, Any]]:
    """All registered model entries, newest first."""
    return sorted(
        load_manifest(cfg).get("models", []),
        key=lambda m: int(m["version"]),
        reverse=True,
    )


def get_entry(version: str, cfg: Config | None = None) -> dict[str, Any] | None:
    entries = load_manifest(cfg).get("models", [])
    return next((m for m in entries if m["version"] == str(version)), None)


def production_entry(cfg: Config | None = None) -> dict[str, Any] | None:
    """The entry currently in the ``production`` stage, if any."""
    entries = load_manifest(cfg).get("models", [])
    return next((m for m in entries if m["stage"] == "production"), None)


def register_model(
    bundle: ModelBundle,
    metrics: dict[str, Any],
    cfg: Config | None = None,
    stage: str = "staging",
) -> str:
    """Persist ``bundle`` as a new immutable version and record it in the manifest."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    cfg = cfg or load_config()
    manifest = load_manifest(cfg)
    existing = [int(m["version"]) for m in manifest["models"]]
    version = str(max(existing) + 1 if existing else 1)

    version_dir = _registry_dir(cfg) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(version_dir / "model.joblib")
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest["models"].append({
        "version": version,
        "stage": stage,
        "model_name": bundle.model_name,
        "model_version": bundle.version,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "path": str((version_dir / "model.joblib").relative_to(_registry_dir(cfg))),
        "registered_at": _now(),
    })
    _save_manifest(manifest, cfg)
    return version


def promote(version: str, stage: str, cfg: Config | None = None) -> None:
    """Move ``version`` to ``stage``. Promoting to production archives the incumbent."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    cfg = cfg or load_config()
    manifest = load_manifest(cfg)
    target = next((m for m in manifest["models"] if m["version"] == str(version)), None)
    if target is None:
        raise KeyError(f"version {version} not in registry")
    if stage == "production":
        for m in manifest["models"]:
            if m["stage"] == "production" and m["version"] != str(version):
                m["stage"] = "archived"
    target["stage"] = stage
    _save_manifest(manifest, cfg)


def evaluate_promotion(
    candidate_metrics: dict[str, Any], cfg: Config | None = None
) -> tuple[bool, str]:
    """Champion/challenger decision: should the candidate replace production?

    Returns ``(should_promote, human_readable_reason)``. The candidate wins if
    there is no incumbent, or if it beats the incumbent on the configured metric
    by at least ``registry.min_improvement``.
    """
    cfg = cfg or load_config()
    metric = cfg.get("registry", "promotion_metric", default="roc_auc")
    min_improvement = float(cfg.get("registry", "min_improvement", default=0.0))

    champion = production_entry(cfg)
    cand = float(candidate_metrics.get(metric, 0.0))
    if champion is None:
        return True, f"no production model yet; promoting ({metric}={cand:.4f})"

    champ = float(champion["metrics"].get(metric, 0.0))
    delta = cand - champ
    if delta >= min_improvement:
        return True, (
            f"challenger {metric}={cand:.4f} beats champion {champ:.4f} "
            f"(+{delta:.4f} >= {min_improvement})"
        )
    return False, (
        f"challenger {metric}={cand:.4f} does not beat champion {champ:.4f} "
        f"(delta {delta:+.4f} < {min_improvement}); keeping production"
    )


def _cli(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the ChurnGuard model registry.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List registered models.")
    pp = sub.add_parser("promote", help="Promote a version to a stage.")
    pp.add_argument("version")
    pp.add_argument("stage", choices=STAGES)
    args = parser.parse_args(argv)

    if args.cmd == "list":
        prod = production_entry()
        for m in models():
            flag = "  <- production" if m["stage"] == "production" else ""
            auc = m["metrics"].get("roc_auc", "?")
            print(f"  v{m['version']:<3} {m['stage']:<11} {m['model_name']:<20} "
                  f"roc_auc={auc}{flag}")
        if prod is None:
            print("  (no model in production)")
    elif args.cmd == "promote":
        promote(args.version, args.stage)
        print(f"Promoted v{args.version} -> {args.stage}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
