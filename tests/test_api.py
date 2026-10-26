"""Tests for the prediction logic and the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from churnguard.api import app
from churnguard.predict import predict_records

client = TestClient(app)

_HIGH_RISK = {
    "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check", "TechSupport": "No",
    "SeniorCitizen": 1,
}
_LOW_RISK = {
    "tenure": 68, "MonthlyCharges": 25.0, "TotalCharges": 1700.0,
    "Contract": "Two year", "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)", "TechSupport": "Yes",
    "SeniorCitizen": 0,
}


def test_predict_records_returns_well_formed_output():
    out = predict_records([_HIGH_RISK, _LOW_RISK])
    assert len(out) == 2
    for row in out:
        assert 0.0 <= row["churn_probability"] <= 1.0
        assert row["risk_band"] in {"low", "medium", "high"}


def test_high_risk_scores_above_low_risk():
    high, low = predict_records([_HIGH_RISK, _LOW_RISK])
    assert high["churn_probability"] > low["churn_probability"]


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_endpoint():
    resp = client.post("/predict", json={"customers": [_HIGH_RISK, _LOW_RISK]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all("churn_probability" in p for p in preds)


def test_predict_endpoint_rejects_empty_batch():
    resp = client.post("/predict", json={"customers": []})
    assert resp.status_code == 422  # pydantic validation error
