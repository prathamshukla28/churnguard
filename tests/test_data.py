"""Tests for dataset generation."""

from __future__ import annotations

from churnguard.data import generate_dataset

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def test_schema_and_size():
    df = generate_dataset(n_rows=500, seed=1)
    assert len(df) == 500
    assert set(df.columns) == EXPECTED_COLUMNS


def test_generation_is_deterministic():
    a = generate_dataset(n_rows=300, seed=7)
    b = generate_dataset(n_rows=300, seed=7)
    assert a.equals(b)


def test_churn_rate_is_realistic():
    df = generate_dataset(n_rows=4000, seed=42)
    rate = (df["Churn"] == "Yes").mean()
    # The real Telco churn rate is ~0.27; stay in a sane band.
    assert 0.15 < rate < 0.40


def test_no_addon_services_without_internet():
    df = generate_dataset(n_rows=2000, seed=3)
    no_net = df[df["InternetService"] == "No"]
    assert (no_net["OnlineSecurity"] == "No internet service").all()
