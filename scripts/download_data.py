"""CLI: download the real Telco Customer Churn dataset to ``data.raw_path``.

Usage:  python scripts/download_data.py      (or `make data-real`)

The file is verified against ``data.real_sha256`` before it is used. There is no
fallback to synthetic data: real data is the default for training, evaluation
and every published number. (Synthetic data exists only for the unit tests.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard.data import download_real_dataset  # noqa: E402

if __name__ == "__main__":
    try:
        path = download_real_dataset()
    except Exception as exc:  # noqa: BLE001 - CLI: report any failure plainly
        print(f"Could not download the real dataset: {exc}", file=sys.stderr)
        print(
            "Check network access and data.real_url / data.real_sha256 in "
            "config/config.yaml, then retry `make data-real`.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Real Telco dataset downloaded and verified at {path}")
