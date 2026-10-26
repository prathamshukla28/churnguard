"""FastAPI service exposing the churn model.

Endpoints:
* ``GET  /``                  — interactive demo page.
* ``GET  /health``            — liveness/readiness probe (model loaded? uptime).
* ``GET  /metrics``           — the stored evaluation metrics, if available.
* ``GET  /metrics/prometheus``— Prometheus text exposition (request/latency/prediction counters).
* ``GET  /stats``             — serving stats (throughput, latency p50/p95/p99, score mix).
* ``POST /predict``           — score a batch of customers, optionally with SHAP reasons.
* ``POST /monitor``           — drift report for a batch vs. the reference profile.

Cross-cutting concerns live in :mod:`churnguard.observability`: every request is
timed, counted and logged as structured JSON with a request id; ``/predict`` and
``/monitor`` are rate-limited per client IP and (when ``CHURNGUARD_API_KEY`` is
set) require an ``X-API-Key`` header.

Run locally with:  ``uvicorn churnguard.api:app --reload``
Interactive docs:   http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from ._demo import DEMO_HTML
from .config import load_config
from .monitor import monitor_frame
from .observability import (
    MetricsRegistry,
    RateLimiter,
    api_key_required,
    configure_logging,
)
from .predict import ModelNotTrainedError, predict_records

logger = configure_logging()
metrics_registry = MetricsRegistry()

_cfg = load_config()
_rate_default = _cfg.get("serving", "rate_limit_per_minute", default=0)
limiter = RateLimiter(int(os.environ.get("CHURNGUARD_RATE_LIMIT", _rate_default)))

app = FastAPI(
    title="ChurnGuard API",
    version=__version__,
    description="Predict telecom customer churn risk.",
)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Assign a request id, time the request, count it, and log it as JSON."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        metrics_registry.record_request(request.method, request.url.path, 500,
                                         time.perf_counter() - start)
        logger.exception("request failed", extra={"request_id": request_id,
                         "method": request.method, "path": request.url.path})
        raise
    latency = time.perf_counter() - start
    metrics_registry.record_request(request.method, request.url.path, status, latency)
    logger.info("request", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "latency_ms": round(latency * 1000, 2),
        "client": request.client.host if request.client else "unknown",
    })
    response.headers["X-Request-ID"] = request_id
    return response


def guard(request: Request) -> None:
    """Rate-limit per client IP and enforce the optional API key."""
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="Rate limit exceeded; try again shortly.")
    required = api_key_required()
    if required is not None and request.headers.get("X-API-Key") != required:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# --- request/response models -------------------------------------------------
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


class MonitorRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=10000)


# --- endpoints ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Serve the interactive demo page."""
    return DEMO_HTML


@app.get("/health")
def health() -> dict:
    """Report service health and whether a model artifact is present."""
    cfg = load_config()
    model_ready = cfg.path("artifacts", "model_path").exists()
    return {
        "status": "ok",
        "version": __version__,
        "model_ready": model_ready,
        "uptime_seconds": metrics_registry.uptime_seconds(),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Return the stored evaluation metrics, if the model has been evaluated."""
    path = load_config().path("artifacts", "metrics_path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics yet; run evaluation.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/metrics/prometheus", response_class=PlainTextResponse, include_in_schema=False)
def prometheus() -> str:
    """Prometheus-scrapeable exposition of live serving metrics."""
    return metrics_registry.prometheus()


@app.get("/stats")
def stats() -> dict:
    """In-memory serving statistics since the process started."""
    return metrics_registry.stats_snapshot()


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(guard)])
def predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of customers and return per-customer churn risk."""
    try:
        records = [c.model_dump(exclude_none=True) for c in request.customers]
        results = predict_records(records, explain=request.explain)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    metrics_registry.record_predictions(results)
    return PredictResponse(predictions=[Prediction(**r) for r in results])


@app.post("/monitor", dependencies=[Depends(guard)])
def monitor(request: MonitorRequest) -> dict:
    """Return a drift report for a batch of customers vs. the reference profile."""
    records = [c.model_dump(exclude_none=True) for c in request.customers]
    try:
        return monitor_frame(pd.DataFrame(records))
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
