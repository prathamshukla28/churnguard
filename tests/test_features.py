"""Tests for feature engineering and preprocessing."""

from __future__ import annotations

import pandas as pd

from churnguard.config import load_config
from churnguard.data import generate_dataset
from churnguard.features import build_preprocessor, engineer_features, split_xy


def test_engineered_columns_added():
    df = generate_dataset(n_rows=100, seed=1)
    out = engineer_features(df)
    assert "num_addon_services" in out.columns
    assert "charges_per_tenure" in out.columns
    # Original frame must be untouched (pure function).
    assert "num_addon_services" not in df.columns


def test_addon_count_in_valid_range():
    out = engineer_features(generate_dataset(n_rows=200, seed=2))
    assert out["num_addon_services"].between(0, 6).all()


def test_charges_per_tenure_handles_zero_tenure():
    df = pd.DataFrame(
        {"tenure": [0], "TotalCharges": [0.0]}
        | {c: ["No"] for c in
           ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]}
    )
    out = engineer_features(df)
    assert out["charges_per_tenure"].notna().all()  # no division-by-zero NaN/inf


def test_split_xy_drops_target_and_id():
    cfg = load_config()
    df = engineer_features(generate_dataset(n_rows=100, seed=1))
    X, y = split_xy(df, cfg)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_produces_numeric_matrix():
    cfg = load_config()
    df = engineer_features(generate_dataset(n_rows=150, seed=5))
    X, _ = split_xy(df, cfg)
    pre = build_preprocessor(X, cfg)
    matrix = pre.fit_transform(X)
    assert matrix.shape[0] == len(X)
    # Every output cell must be finite (no NaNs leaking into the model).
    assert pd.notna(matrix).all()
