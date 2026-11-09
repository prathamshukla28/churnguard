"""CLI: materialise the configured dataset (``data.source``) on disk.

Usage:  python scripts/make_dataset.py     (or `make data`)

With the default ``source: real`` this downloads and verifies the public Telco
CSV. ``source: synthetic`` (used by the unit tests) generates the stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from churnguard.config import load_config  # noqa: E402
from churnguard.data import data_source, ensure_dataset  # noqa: E402

if __name__ == "__main__":
    path = ensure_dataset()
    print(f"Dataset ({data_source(load_config())}) ready at {path}")
