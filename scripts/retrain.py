"""Automated retraining with a champion/challenger promotion gate.

Trains a fresh model on the current data, evaluates it, registers it in the model
registry as a *challenger*, and promotes it to production only if it beats the
incumbent champion on the configured metric (``registry.promotion_metric``). This
is the job a scheduler (cron, Airflow, GitHub Actions `schedule:`) would run to
keep the model fresh without a human promoting every run by hand.

    python scripts/retrain.py            # train, evaluate, auto-promote if better
    python scripts/retrain.py --no-promote   # register as staging only

Exit code is 0 whether or not the challenger is promoted; it is non-zero only on
an actual failure, so a scheduler distinguishes "no improvement" from "broke".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard import registry  # noqa: E402
from churnguard.config import load_config  # noqa: E402
from churnguard.evaluate import evaluate  # noqa: E402
from churnguard.train import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote a model.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Register as staging only; never auto-promote.")
    args = parser.parse_args(argv)

    cfg = load_config()
    print("Training challenger ...")
    result = train(cfg)
    metrics = evaluate(cfg, result=result)
    print(f"Challenger: {result.best_name} | roc_auc={metrics['roc_auc']} "
          f"brier={metrics['brier']}")

    version = registry.register_model(result.bundle, metrics, cfg, stage="staging")
    print(f"Registered as v{version} (staging).")

    if args.no_promote:
        print("--no-promote set; leaving in staging.")
        return 0

    should, reason = registry.evaluate_promotion(metrics, cfg)
    print(f"Promotion decision: {reason}")
    if should:
        registry.promote(version, "production", cfg)
        print(f"Promoted v{version} -> production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
