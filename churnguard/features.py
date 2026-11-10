"""Feature engineering and the preprocessing pipeline.

Three responsibilities:

0. :func:`normalize_schema` — coerce raw values into one canonical shape
   (numeric strings to numbers, ``SeniorCitizen`` 0/1 to ``"Yes"``/``"No"``). It
   runs inside :func:`engineer_features`, so training data and API requests go
   through exactly the same normalisation.
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

import numpy as np
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


_NUMERIC_RAW = ("tenure", "MonthlyCharges", "TotalCharges")
_FLAG_MAP = {
    "1": "Yes",
    "0": "No",
    "1.0": "Yes",
    "0.0": "No",
    "yes": "Yes",
    "no": "No",
    "true": "Yes",
    "false": "No",
}


def _yes_no(value: object) -> object:
    """Map a 0/1/True/False/"Yes"/"No" flag to "Yes"/"No"; missing stays missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if value is pd.NA:
        return np.nan
    return _FLAG_MAP.get(str(value).strip().lower(), np.nan)


def normalize_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise raw Telco values. Pure function — returns a new DataFrame.

    * ``tenure`` / ``MonthlyCharges`` / ``TotalCharges`` become floats; blanks and
      junk become ``NaN`` (imputed later by the pipeline), never a made-up value.
    * A missing ``TotalCharges`` with ``tenure == 0`` is ``0.0``: a customer in
      their first month has not been billed yet (this is exactly the 11 blank
      rows in the real CSV).
    * ``SeniorCitizen`` (0/1 in the raw data, possibly bool/str from API callers)
      becomes ``"Yes"``/``"No"`` like every other binary column.
    """
    out = frame.copy()
    for col in _NUMERIC_RAW:
        if col in out.columns:
            values = out[col]
            if values.dtype == object or pd.api.types.is_string_dtype(values):
                values = values.astype("string").str.strip()
            out[col] = pd.to_numeric(values, errors="coerce").astype(float)
    if "TotalCharges" in out.columns and "tenure" in out.columns:
        unbilled = out["TotalCharges"].isna() & (out["tenure"] == 0)
        out.loc[unbilled, "TotalCharges"] = 0.0
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].map(_yes_no).astype(object)
    return out


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise, then add derived columns. Pure function — returns a new DataFrame.

    Tolerant of partial records (missing columns) so the same function works at
    both training time (full schema) and serving time (whatever the caller sent).
    Missing inputs stay missing (``NaN``) and are imputed by the fitted pipeline.
    """
    out = normalize_schema(frame)

    # How many optional add-on services the customer actually subscribes to.
    present = [col for col in _ADDON_COLUMNS if col in out.columns]
    # Unknown (NaN -> imputed) for any row that sent none of the add-on columns,
    # even when other rows in the same batch did (a batch must not change a score).
    if present:
        addons = out[present]
        count = (addons == "Yes").sum(axis=1)
        unknown = addons.isna().all(axis=1)
        out["num_addon_services"] = count.mask(unknown) if unknown.any() else count
    else:
        out["num_addon_services"] = np.nan

    # Average spend per month of tenure; a proxy for "expensive but new".
    # Guard tenure == 0 to avoid division by zero. If either input is unknown the
    # ratio is unknown too (NaN -> imputed), rather than silently 0.
    tenure = out["tenure"] if "tenure" in out.columns else np.nan
    total = out["TotalCharges"] if "TotalCharges" in out.columns else np.nan
    tenure_safe = pd.Series(tenure, index=out.index, dtype=float).clip(lower=1)
    total_s = pd.Series(total, index=out.index, dtype=float)
    out["charges_per_tenure"] = (total_s / tenure_safe).round(2)

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
