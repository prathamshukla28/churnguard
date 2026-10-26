"""Feature engineering and the preprocessing pipeline.

Two responsibilities:

1. :func:`engineer_features` — deterministic, row-wise feature derivation that
   is applied identically at training time and at serving time. Keeping it in
   one function guarantees train/serve parity (a classic source of production
   bugs).
2. :func:`build_preprocessor` — a scikit-learn ``ColumnTransformer`` that scales
   numeric columns and one-hot encodes categoricals. It lives *inside* the model
   pipeline, so all fitting (means, variances, categories) happens on the training
   fold only — no data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config

_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure function — returns a new DataFrame.

    Tolerant of partial records (missing columns) so the same function works at
    both training time (full schema) and serving time (whatever the caller sent).
    """
    out = frame.copy()

    # How many optional add-on services the customer actually subscribes to.
    present = [col for col in _ADDON_COLUMNS if col in out.columns]
    out["num_addon_services"] = (
        sum((out[col] == "Yes").astype(int) for col in present) if present else 0
    )

    # Average spend per month of tenure; a proxy for "expensive but new".
    # Guard tenure == 0 to avoid division by zero.
    tenure = out["tenure"] if "tenure" in out.columns else 0
    total = out["TotalCharges"] if "TotalCharges" in out.columns else 0.0
    tenure_safe = pd.Series(tenure, index=out.index).clip(lower=1)
    out["charges_per_tenure"] = (pd.Series(total, index=out.index) / tenure_safe).round(2)

    return out


def split_xy(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and binary target y (0/1)."""
    target = cfg.get("data", "target")
    drop = list(cfg.get("features", "drop") or [])
    y = (frame[target] == "Yes").astype(int)
    X = frame.drop(columns=[target, *drop], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame, cfg: Config) -> ColumnTransformer:
    """Build the column transformer for the given feature frame."""
    numeric = [c for c in cfg.get("features", "numeric") if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
