"""Per-prediction explanations via SHAP.

For each scored customer we report the handful of features that pushed their
churn risk up or down the most. The attributions are SHAP values:

* For a **linear** model they are computed in closed form — for a linear model
  the SHAP value of feature *j* is ``coef_j * (x_j - E[x_j])``, which is exactly
  what ``shap.LinearExplainer`` returns, with no approximation.
* For a **tree** model they come from ``shap.TreeExplainer``.

**Which model is explained.** The API's probability comes from the *calibrated*
serving model (``CalibratedClassifierCV``: an average of per-fold isotonic/sigmoid
maps). SHAP cannot see through that wrapper, so the reasons explain the
**uncalibrated base estimator** (``bundle.base_pipeline``) in its raw output
space — log-odds for linear and gradient-boosted models, probability for random
forests. Calibration is monotone per fold, so direction and ranking carry over,
but the impacts do not add up to the returned probability. :func:`explanation_basis`
states this in every explained API response.

SHAP operates in the model's transformed feature space; we map the contributions
back to the original columns (summing the one-hot pieces of each categorical) so
the explanation speaks the business's language ("Contract", not
"cat__Contract_Month-to-month").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .bundle import ModelBundle


def _map_output_to_source(transformed_names: list[str], source_cols: list[str]) -> list[str]:
    """Map each transformed column name back to its original input column."""
    mapping: list[str] = []
    for name in transformed_names:
        rest = name.split("__", 1)[1] if "__" in name else name
        if rest in source_cols:  # numeric column, 1:1
            mapping.append(rest)
            continue
        # One-hot column "<col>_<category>": match the longest source column
        # that is a prefix followed by "_".
        candidates = [c for c in source_cols if rest.startswith(c + "_")]
        mapping.append(max(candidates, key=len) if candidates else rest)
    return mapping


def _shap_values(bundle: ModelBundle, X_trans: np.ndarray) -> np.ndarray:
    """SHAP values in transformed space, shape (n_rows, n_transformed_features)."""
    clf = bundle.base_pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_).ravel()
        bg_mean = (
            bundle.background.mean(axis=0)
            if bundle.background is not None
            else np.zeros(X_trans.shape[1])
        )
        return (X_trans - bg_mean) * coef
    # Non-linear (tree ensembles): use SHAP's TreeExplainer.
    import shap

    explainer = shap.TreeExplainer(clf)
    return _positive_class(np.asarray(explainer.shap_values(X_trans)), X_trans.shape)


def _positive_class(values: np.ndarray, x_shape: tuple[int, ...]) -> np.ndarray:
    """Select class-1 attributions, whatever layout SHAP returned.

    Older SHAP returns ``(classes, n, feat)``; newer returns ``(n, feat, classes)``.
    Decide by matching the known ``(n, feat)`` shape, not by ``shape[0] == 2``,
    which would mistake a 2-row batch for the class axis.
    """
    n_rows, n_feat = x_shape[0], x_shape[1]
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[:2] == (n_rows, n_feat):
        return values[..., 1]
    if values.ndim == 3 and values.shape[1:] == (n_rows, n_feat):
        return values[1]
    raise ValueError(f"Unexpected SHAP output shape {values.shape} for input {x_shape}")


def explanation_basis(bundle: ModelBundle) -> dict[str, str]:
    """Machine-readable statement of what the SHAP reasons are attributions *of*."""
    clf = bundle.base_pipeline.named_steps["clf"]
    if hasattr(clf, "coef_") or type(clf).__name__.startswith("GradientBoosting"):
        space = "log-odds"
    elif type(clf).__name__.startswith("RandomForest"):
        space = "probability"
    else:  # pragma: no cover - only the configured candidates are used
        space = "raw model output"
    calibrated = bundle.pipeline is not bundle.base_pipeline
    return {
        "explained_model": f"uncalibrated base estimator ({bundle.model_name})",
        "impact_units": space,
        "note": (
            "Reasons explain the uncalibrated base model; churn_probability comes from "
            "the calibrated model. Directions and ranking are comparable, but impacts "
            "do not sum to churn_probability."
            if calibrated
            else "Reasons explain the served model; impacts are in its raw output space."
        ),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def explain_records(bundle: ModelBundle, X: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
    """Return the top-k risk drivers for each row of (aligned, engineered) ``X``."""
    pre = bundle.base_pipeline.named_steps["pre"]
    X_trans = np.asarray(pre.transform(X))
    shap_vals = _shap_values(bundle, X_trans)
    source = _map_output_to_source(bundle.transformed_feature_names, bundle.feature_names)

    out: list[list[dict]] = []
    for row in range(X_trans.shape[0]):
        agg: dict[str, float] = {}
        for j, col in enumerate(source):
            agg[col] = agg.get(col, 0.0) + float(shap_vals[row, j])
        top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        reasons = [
            {
                "feature": col,
                "value": _to_jsonable(X.iloc[row].get(col)),
                "impact": round(impact, 4),
                "direction": "increases risk" if impact > 0 else "decreases risk",
            }
            for col, impact in top
        ]
        out.append(reasons)
    return out
