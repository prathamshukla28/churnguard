"""Tests for the prediction logic and the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from churnguard.api import app
from churnguard.predict import predict_records

client = TestClient(app)

_HIGH_RISK = {
    "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check", "TechSupport": "No", "SeniorCitizen": 1,
}
_LOW_RISK = {
    "tenure": 68, "MonthlyCharges": 25.0, "TotalCharges": 1700.0,
    "Contract": "Two year", "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)", "TechSupport": "Yes", "SeniorCitizen": 0,
}


def test_predict_records_returns_well_formed_output():
    out = predict_records([_HIGH_RISK, _LOW_RISK])
    assert len(out) == 2
    for row in out:
        assert 0.0 <= row["churn_probability"] <= 1.0
        assert row["risk_band"] in {"low", "medium", "high"}
        assert "reasons" not in row  # not requested


def test_predict_records_with_explanations():
    out = predict_records([_HIGH_RISK], explain=True)
    assert out[0]["reasons"] and len(out[0]["reasons"]) == 3


def test_high_risk_scores_above_low_risk():
    high, low = predict_records([_HIGH_RISK, _LOW_RISK])
    assert high["churn_probability"] > low["churn_probability"]


def test_health_endpoint():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_endpoint_with_reasons():
    resp = client.post("/predict", json={"customers": [_HIGH_RISK], "explain": True})
    assert resp.status_code == 200
    pred = resp.json()["predictions"][0]
    assert pred["reasons"] is not None
    assert "feature" in pred["reasons"][0]


def test_stats_endpoint_counts_predictions():
    before = client.get("/stats").json()["predictions_served"]
    client.post("/predict", json={"customers": [_HIGH_RISK, _LOW_RISK]})
    after = client.get("/stats").json()
    assert after["predictions_served"] >= before + 2
    assert set(after["risk_bands"]) == {"low", "medium", "high"}


def test_predict_endpoint_rejects_empty_batch():
    resp = client.post("/predict", json={"customers": []})
    assert resp.status_code == 422


def test_home_page_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Predict churn risk" in resp.text
