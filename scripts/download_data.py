"""CLI: download the real Telco Customer Churn dataset to ``data/churn.csv``.

Usage:  python scripts/download_data.py

Falls back with a clear message if there is no network; in that case the
pipeline will generate the realistic synthetic dataset instead.
"""

from __future__ import annotations

import sys

from churnguard.data import download_real_dataset

if __name__ == "__main__":
    try:
        path = download_real_dataset()
    except Exception as exc:  # noqa: BLE001 - CLI: report any failure plainly
        print(f"Could not download the real dataset ({exc}).")
        print("Run `make data` to generate the synthetic dataset instead.")
        sys.exit(1)
    print(f"Real Telco dataset downloaded to {path}")
