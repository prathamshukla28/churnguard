"""Retrain with a data-change check, regression gate and champion/challenger.

    python scripts/retrain.py               # skip if data unchanged; else gate + maybe promote
    python scripts/retrain.py --force       # retrain even on identical data
    python scripts/retrain.py --no-promote  # register as staging only

See :mod:`churnguard.retrain` for the full policy. Exit codes: 0 = skipped,
staged or promoted (a scheduler should not page on "no improvement"); 1 = the
challenger regressed beyond tolerance vs. the committed baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard.retrain import retrain  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote a model.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the data is unchanged since the champion.",
    )
    parser.add_argument(
        "--no-promote", action="store_true", help="Register as staging only; never auto-promote."
    )
    args = parser.parse_args(argv)

    outcome = retrain(force=args.force, promote=not args.no_promote)
    print(f"[{outcome.status}] {outcome.reason}")
    if outcome.metrics:
        print(f"Challenger roc_auc={outcome.metrics['roc_auc']} brier={outcome.metrics['brier']}")
    if outcome.status == "promoted":
        print(
            f"Promoted v{outcome.version}. Review the refreshed reports/, then run "
            "`make promote-baseline` and commit to freeze it as the CI floor."
        )
    return 1 if outcome.status == "regressed" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
