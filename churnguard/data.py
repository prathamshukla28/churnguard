"""Dataset loading (real Telco data) and a synthetic stand-in for unit tests.

**Real data is the default.** ``data.source: real`` trains and evaluates on the
public IBM *Telco Customer Churn* dataset (7,043 customers, 21 columns), which is
downloaded on first use from ``data.real_url`` and verified against
``data.real_sha256``. A failed download is a hard error: there is no silent
fallback to synthetic data.

**Synthetic data is for fast, offline unit tests only.** :func:`generate_dataset`
builds a frame with the Telco schema whose label is drawn from a *hand-written
logistic formula* of a few features. A model trained on it can only rediscover
that formula, so its metrics and "drivers" are circular and say nothing about
real customer churn. Never report numbers from it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config

DATA_SOURCES = ("real", "synthetic")

TELCO_COLUMNS = (
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
)

# Categorical level definitions, mirroring the Telco schema.
_YES_NO = ["Yes", "No"]
_INTERNET = ["DSL", "Fiber optic", "No"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    """Create a reproducible SYNTHETIC frame with the Telco schema (tests only).

    The churn label is drawn from a hand-written logistic formula below, so a
    classifier can learn it -- which makes the generator useful for exercising
    the pipeline quickly and offline, and useless as evidence of anything about
    real churn (the "drivers" are whatever the formula says they are).
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], n_rows)
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(_YES_NO, n_rows, p=[0.48, 0.52])
    dependents = rng.choice(_YES_NO, n_rows, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n_rows)  # months, 0..72
    phone = rng.choice(_YES_NO, n_rows, p=[0.90, 0.10])
    internet = rng.choice(_INTERNET, n_rows, p=[0.34, 0.44, 0.22])
    contract = rng.choice(_CONTRACT, n_rows, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(_YES_NO, n_rows, p=[0.59, 0.41])
    payment = rng.choice(_PAYMENT, n_rows, p=[0.34, 0.23, 0.22, 0.21])

    has_internet = internet != "No"
    multiple_lines = np.where(
        phone == "No",
        "No phone service",
        rng.choice(_YES_NO, n_rows),
    )

    # Add-on services only exist when the customer has internet.
    addons: dict[str, np.ndarray] = {}
    for col in _ADDON_COLUMNS:
        pick = rng.choice(_YES_NO, n_rows, p=[0.40, 0.60])
        addons[col] = np.where(has_internet, pick, "No internet service")

    # Monthly charge is driven by service level; fibre is the pricey tier.
    base = np.where(phone == "Yes", 20.0, 0.0)
    base += np.select(
        [internet == "DSL", internet == "Fiber optic"],
        [35.0, 70.0],
        default=0.0,
    )
    addon_count = sum((addons[c] == "Yes").astype(int) for c in _ADDON_COLUMNS)
    monthly = base + addon_count * 6.0 + rng.normal(0, 4.0, n_rows)
    monthly = np.clip(monthly, 18.25, 120.0).round(2)

    # Total charges roughly tracks tenure * monthly, with a little noise.
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.9, 1.05, n_rows)).round(2)
    total = np.where(tenure == 0, 0.0, total)

    # --- Churn generation: an assumed logistic formula (circular by design) --
    logit = (
        -0.85
        - 0.045 * tenure
        + 0.9 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        + 0.6 * (payment == "Electronic check")
        + 0.012 * (monthly - 65.0)
        + 0.4 * senior
        - 0.5 * (addons["TechSupport"] == "Yes")
        - 0.3 * (partner == "Yes")
        + rng.normal(0, 0.35, n_rows)
    )
    churn = rng.binomial(1, _sigmoid(logit))

    frame = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-CHURN" for i in range(n_rows)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_path(cfg: Config | None = None) -> Path:
    """Where the configured data source lives on disk."""
    cfg = cfg or load_config()
    source = data_source(cfg)
    key = "synthetic_path" if source == "synthetic" else "raw_path"
    return cfg.path("data", key)


def data_source(cfg: Config) -> str:
    source = str(cfg.get("data", "source", default="real"))
    if source not in DATA_SOURCES:
        raise ValueError(f"data.source must be one of {DATA_SOURCES}, got {source!r}")
    return source


def ensure_dataset(cfg: Config | None = None) -> Path:
    """Return the dataset path, materialising it on first use if missing.

    ``real`` downloads (and checksums) the public CSV; ``synthetic`` generates
    the test-only stand-in.
    """
    cfg = cfg or load_config()
    path = dataset_path(cfg)
    if path.exists():
        return path
    if data_source(cfg) == "synthetic":
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(cfg.get("data", "n_rows"), cfg.seed).to_csv(path, index=False)
        return path
    return download_real_dataset(cfg)


def download_real_dataset(cfg: Config | None = None) -> Path:
    """Download the genuine Telco Customer Churn CSV to ``data.raw_path``.

    The file is fetched to a temporary name, its SHA-256 is checked against
    ``data.real_sha256`` (when configured) and its schema against the expected
    columns; only then is it moved into place. Raises on any failure.
    """
    import urllib.request

    cfg = cfg or load_config()
    url = str(cfg.get("data", "real_url"))
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"Refusing to download from a non-https URL: {url}")
    path = cfg.path("data", "raw_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - configured https/file URL
        expected = cfg.get("data", "real_sha256")
        actual = _sha256(tmp)
        if expected and actual != str(expected):
            raise ValueError(f"Checksum mismatch for {url}: expected {expected}, got {actual}")
        head = pd.read_csv(tmp, nrows=5)
        missing = sorted(set(TELCO_COLUMNS) - set(head.columns))
        if missing:
            raise ValueError(f"Downloaded file is missing columns {missing}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def clean_telco(frame: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Fix the known quirks of the raw Telco CSV. Pure function.

    * ``TotalCharges`` is read as text because 11 brand-new customers
      (``tenure == 0``) have a blank ``" "`` value. Those are set to ``0.0`` (they
      have not been billed yet); any other unparsable value stays ``NaN`` for the
      pipeline's imputer rather than being invented.
    * ``SeniorCitizen`` is the only 0/1 flag in a Yes/No schema; it is mapped to
      ``"Yes"``/``"No"`` like every other binary column (see
      :func:`churnguard.features.normalize_schema`).
    * Rows without a valid ``Yes``/``No`` label are dropped.

    ``customerID`` is kept here (it identifies rows) and dropped as a feature by
    ``features.drop`` in the config.
    """
    from .features import normalize_schema

    out = normalize_schema(frame)
    if target in out.columns:
        out[target] = out[target].astype("string").str.strip()
        out = out[out[target].isin(["Yes", "No"])].reset_index(drop=True)
    return out


def load_dataset(cfg: Config | None = None) -> pd.DataFrame:
    """Load the configured dataset as a cleaned DataFrame (fetching it if needed)."""
    cfg = cfg or load_config()
    path = ensure_dataset(cfg)
    frame = pd.read_csv(path, dtype={"TotalCharges": "string"})
    return clean_telco(frame, target=cfg.get("data", "target"))


def dataset_fingerprint(cfg: Config | None = None) -> str:
    """SHA-256 of the dataset file: identifies exactly which data trained a model."""
    return _sha256(ensure_dataset(cfg or load_config()))


if __name__ == "__main__":  # pragma: no cover - manual convenience
    ready = ensure_dataset()
    print(f"Dataset ready at {ready}")
