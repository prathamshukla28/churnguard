"""CI model-regression gate (thin wrapper around :mod:`churnguard.quality`).

Compares the freshly evaluated model (``reports/metrics.json``) against the
frozen, committed quality floor (``reports/baseline_metrics.json``) and fails the
build if quality slipped beyond ``registry.regression_tolerance``. ``evaluate``
never writes the baseline, so this comparison is real; update the floor only on
purpose with ``make promote-baseline``.

    python scripts/check_regression.py

Exit code 0 = within tolerance, 1 = regression detected, 2 = a file is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard.quality import run_check  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_check())
