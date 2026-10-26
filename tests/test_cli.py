"""Test the batch CSV scoring CLI."""

from __future__ import annotations

import pandas as pd

from churnguard.data import generate_dataset
from churnguard.predict import _cli


def test_batch_scoring_cli(tmp_path):
    df = generate_dataset(n_rows=20, seed=5).drop(columns=["Churn"])
    src = tmp_path / "customers.csv"
    out = tmp_path / "scored.csv"
    df.to_csv(src, index=False)

    rc = _cli([str(src), "--out", str(out), "--explain"])
    assert rc == 0

    scored = pd.read_csv(out)
    assert len(scored) == 20
    for col in ("churn_probability", "will_churn", "risk_band", "top_reason"):
        assert col in scored.columns
    assert scored["churn_probability"].between(0, 1).all()
