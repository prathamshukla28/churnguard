"""CLI: materialise the raw dataset to ``data/churn.csv``.

Usage:  python scripts/make_dataset.py
"""

from __future__ import annotations

from churnguard.data import ensure_dataset

if __name__ == "__main__":
    path = ensure_dataset()
    print(f"Dataset ready at {path}")
