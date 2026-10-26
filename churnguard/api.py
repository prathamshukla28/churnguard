"""FastAPI service exposing the churn model.

Endpoints:
* ``GET  /``         — interactive demo page.
* ``GET  /health``   — liveness/readiness probe (reports whether a model is loaded).
* ``GET  /metrics``  — the metrics the model was evaluated with, if available.
* ``GET  /stats``    — in-memory serving stats (predictions served, score mix).
* ``POST /predict``  — score a batch of customers, optionally with SHAP reasons.

Run locally with:  ``uvicorn churnguard.api:app --reload``
Interactive docs:   http://localhost:8000/docs
"""

from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from ._demo import DEMO_HTML
from .config import load_config
from .predict import ModelNotTrainedError, predict_records

app = FastAPI(
    title="ChurnGuard API",
    version=__version__,
    description="Predict telecom customer churn risk.",
)

# --- lightweight in-memory serving stats (resets on restart) -----------------
_stats_lock = threading.Lock()
_stats = {"predictions_served": 0, "prob_sum": 0.0, "low": 0, "medium": 0, "high": 0}


def _record(predictions: list[dict]) -> None:
    with _stats_lock:
        for p in predictions:
            _stats["predictions_served"] += 1
            _stats["prob_sum"] += p["churn_probability"]
            _stats[p["risk_band"]] += 1


class Customer(BaseModel):
    """A single customer record. Mirrors the training schema.

    Fields are optional so partial records still score (the pipeline imputes
    missing values); extra fields are ignored.
    """

    model_config = {"extra": "allow"}

    tenure: int = Field(0, ge=0, description="Months the customer has stayed.")
    MonthlyCharges: float = Field(0.0, ge=0)
    TotalCharges: float = Field(0.0, ge=0)
    Contract: str | None = Field(None, examples=["Month-to-month"])
    InternetService: str | None = Field(None, examples=["Fiber optic"])
    PaymentMethod: str | None = Field(None, examples=["Electronic check"])


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)
    explain: bool = Field(False, description="Include per-customer SHAP reasons.")


class Reason(BaseModel):
    feature: str
    value: object | None = None
    impact: float
    direction: str


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: str
    reasons: list[Reason] | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Serve the interactive demo page."""
    return DEMO_HTML


@app.get("/health")
def health() -> dict:
    """Report service health and whether a model artifact is present."""
    cfg = load_config()
    model_ready = cfg.path("artifacts", "model_path").exists()
    return {"status": "ok", "version": __version__, "model_ready": model_ready}


@app.get("/metrics")
def metrics() -> dict:
    """Return the stored evaluation metrics, if the model has been evaluated."""
    path = load_config().path("artifacts", "metrics_path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics yet; run evaluation.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/stats")
def stats() -> dict:
    """In-memory serving statistics since the process started."""
    with _stats_lock:
        n = _stats["predictions_served"]
        return {
            "predictions_served": n,
            "mean_churn_probability": round(_stats["prob_sum"] / n, 4) if n else None,
            "risk_bands": {k: _stats[k] for k in ("low", "medium", "high")},
        }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of customers and return per-customer churn risk."""
    try:
        records = [c.model_dump(exclude_none=True) for c in request.customers]
        results = predict_records(records, explain=request.explain)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _record(results)
    return PredictResponse(predictions=[Prediction(**r) for r in results])
