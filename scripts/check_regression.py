"""CI model-regression gate.

Compares the freshly evaluated model (``reports/metrics.json``) against the
committed quality floor (``reports/baseline_metrics.json``) and fails the build if
quality has slipped beyond ``registry.regression_tolerance``. This makes model
quality a first-class CI check — a bad retrain is blocked the same way a failing
unit test is, not discovered in production.

    python scripts/check_regression.py

Exit code 0 = within tolerance, 1 = regression detected, 2 = baseline missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    metrics_path = cfg.path("artifacts", "metrics_path")
    baseline_path = cfg.path("artifacts", "baseline_path")

    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}; run evaluate to create one.")
        return 2
    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}; run evaluate first.")
        return 2

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tol = float(cfg.get("registry", "regression_tolerance", default=0.02))

    ok = True
    # Higher-is-better metrics must not fall more than `tol` below baseline.
    for metric in ("roc_auc", "pr_auc"):
        cur, base = metrics.get(metric), baseline.get(metric)
        if cur is None or base is None:
            continue
        delta = cur - base
        flag = "OK " if delta >= -tol else "FAIL"
        ok = ok and delta >= -tol
        print(f"[{flag}] {metric}: {cur:.4f} vs baseline {base:.4f} (delta {delta:+.4f})")
    # Lower-is-better: Brier must not rise more than `tol` above baseline.
    cur, base = metrics.get("brier"), baseline.get("brier")
    if cur is not None and base is not None:
        delta = cur - base
        flag = "OK " if delta <= tol else "FAIL"
        ok = ok and delta <= tol
        print(f"[{flag}] brier: {cur:.4f} vs baseline {base:.4f} (delta {delta:+.4f})")

    print("PASS: no regression beyond tolerance." if ok
          else f"FAIL: model regressed beyond tolerance ({tol}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
