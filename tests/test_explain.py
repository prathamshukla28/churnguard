"""Tests for per-prediction SHAP explanations."""

from __future__ import annotations

import pandas as pd

from churnguard.explain import _map_output_to_source, explain_records
from churnguard.predict import load_bundle

_HIGH_RISK = {
    "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check", "TechSupport": "No", "SeniorCitizen": 1,
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
