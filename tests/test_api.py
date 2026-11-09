"""Tests for the prediction logic and the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from churnguard import api
from churnguard.api import app
from churnguard.observability import RateLimiter
from churnguard.predict import predict_records

client = TestClient(app)

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
_LOW_RISK = {
    "tenure": 68,
    "MonthlyCharges": 25.0,
    "TotalCharges": 1700.0,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)",
    "TechSupport": "Yes",
    "SeniorCitizen": 0,
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


def test_health_reports_uptime():
    body = client.get("/health").json()
    assert "uptime_seconds" in body


def test_stats_exposes_latency_percentiles():
    client.post("/predict", json={"customers": [_HIGH_RISK]})
    body = client.get("/stats").json()
    assert set(body["latency_ms"]) == {"p50", "p95", "p99"}
    assert body["requests_total"] >= 1


def test_prometheus_exposition():
    client.post("/predict", json={"customers": [_HIGH_RISK]})
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "churnguard_requests_total" in resp.text
    assert resp.headers["content-type"].startswith("text/plain")


def test_monitor_endpoint_returns_drift_report():
    resp = client.post("/monitor", json={"customers": [_HIGH_RISK, _LOW_RISK]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_status"] in {"ok", "warn", "alert"}
    assert "features" in body and "prediction_drift" in body


def test_request_id_header_present():
    resp = client.get("/health")
    assert len(resp.headers.get("X-Request-ID", "")) == 12


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_API_KEY", "s3cret")
    denied = client.post("/predict", json={"customers": [_HIGH_RISK]})
    assert denied.status_code == 401
    allowed = client.post(
        "/predict", json={"customers": [_HIGH_RISK]}, headers={"X-API-Key": "s3cret"}
    )
    assert allowed.status_code == 200


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(api, "limiter", RateLimiter(per_minute=1))
    first = client.post("/predict", json={"customers": [_LOW_RISK]})
    second = client.post("/predict", json={"customers": [_LOW_RISK]})
    assert first.status_code == 200
    assert second.status_code == 429


# --- issue: zero defaults / imputation --------------------------------------
def test_customer_fields_default_to_missing_not_zero():
    from churnguard.api import Customer

    assert Customer().model_dump(exclude_none=True) == {}
    assert Customer(tenure=0).model_dump(exclude_none=True) == {"tenure": 0}


def test_omitted_numeric_fields_reach_the_model_as_missing(monkeypatch):
    seen = {}
    real = api.predict_records

    def spy(records, **kwargs):
        seen["records"] = records
        return real(records, **kwargs)

    monkeypatch.setattr(api, "predict_records", spy)
    partial = {"Contract": "Month-to-month", "InternetService": "Fiber optic"}
    resp = client.post("/predict", json={"customers": [partial]})
    assert resp.status_code == 200
    sent = seen["records"][0]
    for field in ("tenure", "MonthlyCharges", "TotalCharges"):
        assert field not in sent  # imputed by the pipeline, not forced to 0


def test_missing_tenure_is_imputed_not_treated_as_brand_new():
    base = {k: v for k, v in _LOW_RISK.items() if k not in {"tenure", "TotalCharges"}}
    imputed, zero = predict_records([base, {**base, "tenure": 0, "TotalCharges": 0.0}])
    assert imputed["churn_probability"] != zero["churn_probability"]


# --- issue: risk band vs. threshold -----------------------------------------
def test_risk_band_is_consistent_with_will_churn():
    from churnguard.data import generate_dataset

    rows = generate_dataset(n_rows=200, seed=11).drop(columns=["Churn"])
    preds = predict_records(rows.to_dict(orient="records"))
    for p in preds:
        assert p["will_churn"] == (p["risk_band"] != "low"), p
    assert {p["risk_band"] for p in preds} >= {"low"}


def test_predict_response_reports_the_threshold():
    from churnguard.predict import load_bundle

    body = client.post("/predict", json={"customers": [_HIGH_RISK]}).json()
    assert body["threshold"] == round(load_bundle().threshold, 4)
    pred = body["predictions"][0]
    assert pred["will_churn"] == (pred["churn_probability"] >= body["threshold"])


# --- issue: SHAP explains the uncalibrated base model ------------------------
def test_explanation_basis_is_explicit():
    body = client.post("/predict", json={"customers": [_HIGH_RISK], "explain": True}).json()
    basis = body["explanation_basis"]
    assert basis["explained_model"].startswith("uncalibrated base estimator")
    assert basis["impact_units"] == "log-odds"
    assert "calibrated" in basis["note"]
    plain = client.post("/predict", json={"customers": [_HIGH_RISK]}).json()
    assert plain["explanation_basis"] is None


# --- issue: security ----------------------------------------------------------
def test_prometheus_labels_use_route_templates_not_raw_paths():
    for i in range(5):
        assert client.get(f"/scan-{i}-does-not-exist").status_code == 404
    client.request("BREW", "/health")
    text = client.get("/metrics/prometheus").text
    assert "scan-" not in text
    assert 'path="unmatched"' in text
    assert 'method="OTHER"' in text
    assert 'path="/health"' in text


def test_forwarded_for_ignored_without_trusted_proxy(monkeypatch):
    monkeypatch.delenv("CHURNGUARD_TRUST_PROXY", raising=False)
    monkeypatch.setattr(api, "limiter", RateLimiter(per_minute=1))
    a = client.post(
        "/predict", json={"customers": [_LOW_RISK]}, headers={"X-Forwarded-For": "1.1.1.1"}
    )
    b = client.post(
        "/predict", json={"customers": [_LOW_RISK]}, headers={"X-Forwarded-For": "2.2.2.2"}
    )
    assert (a.status_code, b.status_code) == (200, 429)  # spoofing does not help


def test_forwarded_for_used_behind_trusted_proxy(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_TRUST_PROXY", "1")
    monkeypatch.setattr(api, "limiter", RateLimiter(per_minute=1))
    a = client.post(
        "/predict", json={"customers": [_LOW_RISK]}, headers={"X-Forwarded-For": "1.1.1.1"}
    )
    b = client.post(
        "/predict", json={"customers": [_LOW_RISK]}, headers={"X-Forwarded-For": "2.2.2.2"}
    )
    c = client.post(
        "/predict", json={"customers": [_LOW_RISK]}, headers={"X-Forwarded-For": "2.2.2.2"}
    )
    assert (a.status_code, b.status_code, c.status_code) == (200, 200, 429)


def test_ops_endpoints_open_by_default(monkeypatch):
    monkeypatch.delenv("CHURNGUARD_PROTECT_OPS", raising=False)
    monkeypatch.setenv("CHURNGUARD_API_KEY", "s3cret")
    assert client.get("/stats").status_code == 200


def test_ops_endpoints_protected_when_enabled(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_PROTECT_OPS", "1")
    monkeypatch.setenv("CHURNGUARD_API_KEY", "s3cret")
    for path in ("/stats", "/metrics", "/metrics/prometheus"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get(path, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # probes stay open


def test_ops_protection_fails_closed_without_a_key(monkeypatch):
    monkeypatch.setenv("CHURNGUARD_PROTECT_OPS", "1")
    monkeypatch.delenv("CHURNGUARD_API_KEY", raising=False)
    assert client.get("/stats").status_code == 401


def test_metrics_endpoint_serves_evaluation_metrics():
    body = client.get("/metrics").json()
    assert "roc_auc" in body and "threshold" in body
