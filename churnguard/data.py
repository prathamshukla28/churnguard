"""Dataset generation and loading.

The real-world counterpart of this dataset is the well-known *Telco Customer
Churn* dataset (7,043 rows, 21 columns). To keep the project fully reproducible
and runnable with no external download, :func:`generate_dataset` synthesises a
dataset with the *same schema and the same churn drivers* reported in that data:
month-to-month contracts, fibre-optic internet, short tenure, high monthly
charges, electronic-check payment and lack of tech support all push churn up.

To train on the genuine dataset instead, drop the Kaggle CSV at
``data/churn.csv`` (same column names) and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config

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
    """Create a realistic, reproducible churn dataset.

    The churn label is drawn from a logistic model of the features, so the
    relationship is genuine (a classifier can recover it) rather than random
    noise, while still being fully synthetic.
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

    # --- Churn generation: logistic model of the real-world drivers ---------
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


def ensure_dataset(cfg: Config | None = None) -> Path:
    """Return the dataset path, generating it on first use if missing."""
    cfg = cfg or load_config()
    path = cfg.path("data", "raw_path")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = generate_dataset(cfg.get("data", "n_rows"), cfg.seed)
        frame.to_csv(path, index=False)
    return path


def load_dataset(cfg: Config | None = None) -> pd.DataFrame:
    """Load the raw dataset as a DataFrame (generating it if necessary)."""
    cfg = cfg or load_config()
    path = ensure_dataset(cfg)
    frame = pd.read_csv(path)
    # TotalCharges arrives as blank strings for brand-new customers in the real
    # dataset; coerce to numeric and treat blanks as 0 (tenure == 0).
    frame["TotalCharges"] = pd.to_numeric(
        frame["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    return frame


if __name__ == "__main__":  # pragma: no cover - manual convenience
    path = ensure_dataset()
    print(f"Dataset ready at {path}")
