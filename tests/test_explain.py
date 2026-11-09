"""Tests for per-prediction SHAP explanations."""

from __future__ import annotations

import pandas as pd

from churnguard.explain import _map_output_to_source, explain_records
from churnguard.predict import load_bundle

_HIGH_RISK = {
    "tenure": 1,
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "TechSupport": "No",
    "SeniorCitizen": 1,
}


def test_map_output_to_source_handles_numeric_and_onehot():
    src = _map_output_to_source(
        ["num__tenure", "cat__Contract_Month-to-month", "cat__InternetService_DSL"],
        ["tenure", "Contract", "InternetService"],
    )
    assert src == ["tenure", "Contract", "InternetService"]


def test_explain_returns_topk_reasons_with_fields():
    from churnguard.features import engineer_features

    bundle = load_bundle()
    X = engineer_features(pd.DataFrame([_HIGH_RISK])).reindex(columns=bundle.feature_names)
    reasons = explain_records(bundle, X, top_k=3)
    assert len(reasons) == 1
    assert len(reasons[0]) == 3
    for r in reasons[0]:
        assert set(r) == {"feature", "value", "impact", "direction"}
        assert r["direction"] in {"increases risk", "decreases risk"}
    # The strongest driver for this high-risk profile should push risk up.
    assert reasons[0][0]["direction"] == "increases risk"


def test_explanation_basis_names_the_explained_model():
    from churnguard.explain import explanation_basis

    basis = explanation_basis(load_bundle())
    assert basis["explained_model"].startswith("uncalibrated base estimator")
    assert basis["impact_units"] in {"log-odds", "probability"}


def test_positive_class_selection_handles_both_layouts_and_two_rows():
    import numpy as np

    from churnguard.explain import _positive_class

    n, f = 2, 3  # a 2-row batch must not be mistaken for the class axis
    new_layout = np.zeros((n, f, 2))
    new_layout[..., 1] = 1.0
    assert (_positive_class(new_layout, (n, f)) == 1.0).all()
    old_layout = np.zeros((2, n, f))
    old_layout[1] = 1.0
    assert (_positive_class(old_layout, (n, f)) == 1.0).all()
    flat = np.ones((n, f))
    assert _positive_class(flat, (n, f)) is flat


def test_tree_model_explanations(small_cfg):
    """The real-data winner is a random forest: exercise the TreeExplainer path."""
    import copy

    from churnguard.config import Config
    from churnguard.explain import explanation_basis
    from churnguard.features import engineer_features
    from churnguard.train import train

    raw = copy.deepcopy(small_cfg.raw)
    raw["model"]["candidates"] = ["random_forest"]
    bundle = train(Config(raw=raw)).bundle
    X = engineer_features(pd.DataFrame([_HIGH_RISK, _HIGH_RISK | {"tenure": 60}]))
    X = X.reindex(columns=bundle.feature_names)
    reasons = explain_records(bundle, X, top_k=3)
    assert len(reasons) == 2 and all(len(r) == 3 for r in reasons)
    assert explanation_basis(bundle)["impact_units"] == "probability"
