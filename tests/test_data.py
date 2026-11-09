"""Tests for dataset generation (synthetic, tests only) and real-data handling."""

from __future__ import annotations

import copy
import hashlib

import pandas as pd
import pytest

from churnguard.config import Config
from churnguard.data import (
    clean_telco,
    data_source,
    download_real_dataset,
    ensure_dataset,
    generate_dataset,
    load_dataset,
)

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
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


# --- real-data handling -------------------------------------------------------
def _raw_telco_like(n: int = 40) -> pd.DataFrame:
    """A small frame with the real CSV's quirks (as read: TotalCharges is text)."""
    df = generate_dataset(n_rows=n, seed=9)
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[0, ["tenure", "TotalCharges"]] = [0, " "]  # blank for a new customer
    df.loc[1, "TotalCharges"] = "oops"  # junk -> NaN, imputed later
    return df


def test_clean_telco_fixes_known_quirks():
    out = clean_telco(_raw_telco_like())
    assert out["TotalCharges"].dtype == float
    assert out.loc[0, "TotalCharges"] == 0.0
    assert pd.isna(out.loc[1, "TotalCharges"])
    assert set(out["SeniorCitizen"].dropna()) <= {"Yes", "No"}
    assert "customerID" in out.columns  # dropped later as a feature, not here


def test_clean_telco_drops_unlabelled_rows():
    df = _raw_telco_like()
    df.loc[2, "Churn"] = " "
    assert len(clean_telco(df)) == len(df) - 1


def _real_cfg(small_cfg, tmp_path, csv: pd.DataFrame | None = None, sha: str | None = None):
    src = tmp_path / "source.csv"
    (csv if csv is not None else _raw_telco_like()).to_csv(src, index=False)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    raw = copy.deepcopy(small_cfg.raw)
    raw["data"].update(
        {
            "source": "real",
            "raw_path": str(tmp_path / "data" / "telco.csv"),
            "real_url": src.as_uri(),
            "real_sha256": sha if sha is not None else digest,
        }
    )
    return Config(raw=raw)


def test_real_source_downloads_verifies_and_cleans(small_cfg, tmp_path):
    cfg = _real_cfg(small_cfg, tmp_path)
    assert data_source(cfg) == "real"
    path = ensure_dataset(cfg)
    assert path == tmp_path / "data" / "telco.csv"
    frame = load_dataset(cfg)
    assert frame["TotalCharges"].dtype == float and frame.loc[0, "TotalCharges"] == 0.0


def test_checksum_mismatch_aborts_and_leaves_nothing(small_cfg, tmp_path):
    cfg = _real_cfg(small_cfg, tmp_path, sha="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_real_dataset(cfg)
    assert not list((tmp_path / "data").glob("*"))


def test_schema_mismatch_aborts(small_cfg, tmp_path):
    cfg = _real_cfg(small_cfg, tmp_path, csv=pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(ValueError, match="missing columns"):
        download_real_dataset(cfg)


def test_non_https_url_refused(small_cfg, tmp_path):
    raw = copy.deepcopy(small_cfg.raw)
    raw["data"].update(
        {
            "source": "real",
            "real_url": "http://example.com/x.csv",
            "raw_path": str(tmp_path / "x.csv"),
        }
    )
    with pytest.raises(ValueError, match="non-https"):
        download_real_dataset(Config(raw=raw))


def test_unknown_source_rejected(small_cfg):
    raw = copy.deepcopy(small_cfg.raw)
    raw["data"]["source"] = "kaggle"
    with pytest.raises(ValueError):
        data_source(Config(raw=raw))


def test_real_data_is_the_default_config():
    import yaml

    from churnguard.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        assert yaml.safe_load(fh)["data"]["source"] == "real"
