"""Model-quality gate: compare fresh metrics with the frozen, committed baseline.

``reports/baseline_metrics.json`` is the quality floor. It is committed to git
and written **only** by an explicit promotion (``make promote-baseline``), never
by ``evaluate`` — otherwise "current vs. baseline" would always compare a run
with itself and the gate could never fail.

    python -m churnguard.quality check              # CI gate (exit 1 on regression)
    python -m churnguard.quality promote-baseline   # freeze current metrics as the floor

Exit codes for ``check``: 0 = within tolerance, 1 = regression, 2 = file missing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, load_config

# (metric, higher_is_better)
GATED_METRICS: tuple[tuple[str, bool], ...] = (
    ("roc_auc", True),
    ("pr_auc", True),
    ("brier", False),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_regression(
    metrics: dict[str, Any], baseline: dict[str, Any], tolerance: float
) -> tuple[bool, list[str]]:
    """Return ``(ok, report_lines)``.

    Higher-is-better metrics may not fall more than ``tolerance`` below the
    baseline; lower-is-better ones (Brier) may not rise more than ``tolerance``.
    A gated metric present in the baseline but missing from ``metrics`` fails.
    """
    ok = True
    lines: list[str] = []
    for name, higher_better in GATED_METRICS:
        base = baseline.get(name)
        if base is None:
            continue
        cur = metrics.get(name)
        if cur is None:
            ok = False
            lines.append(f"[FAIL] {name}: missing from current metrics")
            continue
        delta = float(cur) - float(base)
        passed = delta >= -tolerance if higher_better else delta <= tolerance
        ok = ok and passed
        flag = "OK  " if passed else "FAIL"
        lines.append(
            f"[{flag}] {name}: {float(cur):.4f} vs baseline {float(base):.4f} "
            f"(delta {delta:+.4f}, tolerance {tolerance})"
        )
    base_sha, cur_sha = baseline.get("data_sha256"), metrics.get("data_sha256")
    if base_sha and cur_sha and base_sha != cur_sha:
        lines.append(
            "[WARN] baseline was captured on a different dataset; the comparison is "
            "across test sets. Re-baseline deliberately once the new data is accepted."
        )
    lines.append(
        "PASS: no regression beyond tolerance."
        if ok
        else f"FAIL: model regressed beyond tolerance ({tolerance})."
    )
    return ok, lines


def run_check(cfg: Config | None = None) -> int:
    """CI entry point: compare ``metrics_path`` with ``baseline_path``."""
    cfg = cfg or load_config()
    metrics_path = cfg.path("artifacts", "metrics_path")
    baseline_path = cfg.path("artifacts", "baseline_path")
    for path, hint in ((baseline_path, "make promote-baseline"), (metrics_path, "make evaluate")):
        if not path.exists():
            print(f"No file at {path}; run `{hint}`.")
            return 2
    tol = float(cfg.get("registry", "regression_tolerance", default=0.02))
    ok, lines = check_regression(_read(metrics_path), _read(baseline_path), tol)
    print("\n".join(lines))
    return 0 if ok else 1


def promote_baseline(cfg: Config | None = None) -> Path:
    """Freeze the current ``metrics_path`` as the committed quality floor."""
    cfg = cfg or load_config()
    metrics_path = cfg.path("artifacts", "metrics_path")
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics at {metrics_path}; run `make evaluate` first.")
    metrics = _read(metrics_path)
    baseline: dict[str, Any] = {name: metrics[name] for name, _ in GATED_METRICS}
    baseline.update(
        {
            "model": metrics.get("model"),
            "data_source": metrics.get("data_source"),
            "data_sha256": metrics.get("data_sha256"),
            "n_test": metrics.get("n_test"),
            "promoted_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "note": (
                "Frozen quality floor. Written only by `make promote-baseline`; CI fails if "
                "a new model regresses beyond registry.regression_tolerance."
            ),
        }
    )
    path = cfg.path("artifacts", "baseline_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return path


def load_baseline(cfg: Config | None = None) -> dict[str, Any] | None:
    path = (cfg or load_config()).path("artifacts", "baseline_path")
    return _read(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChurnGuard model-quality gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Fail (exit 1) if metrics regressed vs. the baseline.")
    sub.add_parser("promote-baseline", help="Freeze current metrics as the new baseline.")
    args = parser.parse_args(argv)
    if args.cmd == "check":
        return run_check()
    path = promote_baseline()
    print(f"Baseline frozen at {path}. Commit it to make it the CI quality floor.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
