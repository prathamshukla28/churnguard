# ChurnGuard — System Design

> A reference for engineers operating or extending ChurnGuard. For *why* each
> choice was made, see the [Architecture Decision Records](adr/). For incident
> response, see the [Runbook](RUNBOOK.md).

## 1. Context & goals

ChurnGuard scores telecom customers by churn risk so a retention team can act
before they leave. It is a **batch-and-online scoring service**, not an automated
actuator: a human (or a downstream campaign tool) decides what to do with a score.

**Functional goals**
- Score one customer or a batch, returning a *calibrated* probability + decision.
- Explain each score (top risk drivers) for trust and actionability.
- Be observable and operable: health, metrics, drift, versioned models, rollback.

**Non-goals**
- Real-time feature computation from event streams (features are request-supplied).
- Automated retention actions (out of scope; ChurnGuard emits scores only).

**Quality attributes (what we optimise for)**
| Attribute | Target / approach |
| --- | --- |
| Correctness | Leak-free pipeline, CV model selection, calibrated probabilities |
| Decision quality | Threshold minimises expected business cost, not accuracy |
| Operability | `/health`, `/stats`, `/metrics/prometheus`, drift monitoring, registry |
| Reproducibility | One seed end-to-end; `requirements.lock`; `make all` |
| Latency | p99 < ~25 ms single-row in-process (see `reports/loadtest.md`) |

## 2. Architecture

```
            config/config.yaml  (single source of truth)
                     │
  data.py ─► features.py ─► train.py ───────────► evaluate.py
 (real or    (engineering +  (CV select →          (threshold choice, metrics,
  synthetic)  ColumnTransformer) tune → calibrate)   figures, reference profile,
                     │                                 baseline floor)
                     ▼
         models/churnguard.joblib  (ModelBundle: model + threshold + SHAP bg)
                     │
        ┌────────────┼───────────────┬───────────────┐
        ▼            ▼                ▼               ▼
   explain.py   predict.py ─► api.py            monitor.py        registry.py
    (SHAP)      (batch CLI)   (REST + obs.)    (PSI drift)      (champion/
                                                                 challenger)
```

### Components

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed, cached config; resolves project-relative paths. |
| `data.py` | Synthetic generator (reproducible) + real-dataset download. |
| `features.py` | `engineer_features` (train/serve parity) + `ColumnTransformer`. |
| `train.py` | CV model selection → randomized tuning → isotonic calibration. |
| `bundle.py` | The serialized artifact: model, threshold, feature schema, SHAP bg. |
| `evaluate.py` | Metrics, figures, cost-based threshold, model card, **reference profile**, **baseline**. |
| `explain.py` | Per-prediction SHAP, mapped back to business columns. |
| `predict.py` | Cached inference shared by the API and the batch CLI. |
| `monitor.py` | PSI-based data/prediction/performance drift vs. the reference profile. |
| `registry.py` | File-based versioned registry + champion/challenger promotion. |
| `observability.py` | Structured logging, metrics registry, rate limiter, API-key auth. |
| `api.py` | FastAPI service wiring all of the above. |

## 3. Data flow

**Training (offline).** `data → engineer_features → train/test split → CV compare
→ tune winner → calibrate → ModelBundle`. Evaluation picks the cost-minimising
threshold, writes it back into the bundle, and captures the reference profile +
baseline floor.

**Serving (online).** `request → Pydantic validation → engineer_features → reindex
to bundle schema → predict_proba → threshold → risk band (+ SHAP)`. The **same**
`engineer_features` runs in both paths — the core guarantee against train/serve skew.

**Monitoring.** A batch of recent customers is scored and compared to the reference
profile; PSI per feature + prediction-distribution drift (+ performance drift if
labels are attached) produce a graded report.

## 4. Key design decisions (summary)

Full rationale in [`docs/adr/`](adr/):
- **ADR-0002** Leak-free scikit-learn `Pipeline` cross-validated as one unit.
- **ADR-0003** Model selection by stratified CV ROC-AUC, then randomized tuning.
- **ADR-0004** Isotonic probability calibration (decisions use probabilities).
- **ADR-0005** Decision threshold minimises expected business cost, not accuracy.
- **ADR-0006** SHAP explanations mapped to business-readable features.
- **ADR-0007** File-based model registry over a hosted MLflow registry.
- **ADR-0008** PSI drift monitoring without a monitoring framework.
- **ADR-0009** Dependency-free observability (JSON logs, Prometheus text, limiter).

## 5. Operability

- **Health:** `GET /health` → model-loaded + uptime (liveness & readiness).
- **Metrics:** `GET /metrics` (model eval metrics), `GET /metrics/prometheus`
  (request/latency/prediction counters, scrapeable).
- **Serving stats:** `GET /stats` → throughput, latency p50/p95/p99, risk-band mix.
- **Drift:** `POST /monitor` or `python -m churnguard.monitor batch.csv`.
- **Model lifecycle:** `scripts/retrain.py` (challenger + auto-promote), `make registry`.
- **CI quality gate:** `scripts/check_regression.py` blocks a model that regresses
  beyond `registry.regression_tolerance` vs. the committed baseline.

### SLOs (illustrative)
| SLO | Target |
| --- | --- |
| Availability | 99.5% monthly |
| Latency p99 | < 50 ms (single-row) |
| Drift freshness | Monitoring batch at least daily |
| Model freshness | Retrain considered monthly or on an alert |

## 6. Failure modes & mitigations

| Failure | Detection | Mitigation |
| --- | --- | --- |
| Model artifact missing | `/health` `model_ready=false`; `/predict` → 503 | Train & deploy; registry rollback |
| Input drift | `/monitor` PSI ≥ alert | Investigate upstream; retrain |
| Performance decay | `/monitor` performance status (needs labels) | Retrain; retune threshold |
| Bad retrain | CI regression gate | Build fails; production unchanged |
| Abuse / overload | 429s in logs/metrics | Rate limiter; scale out |
| Unauthorized access | 401s | `CHURNGUARD_API_KEY` + network policy |

## 7. Scaling

The API is **stateless** (the only in-process state is best-effort metrics), so it
scales horizontally behind a load balancer; the model loads once per process and is
cached. Throughput is CPU-bound on preprocessing + `predict_proba`; batch requests
amortise per-request overhead. Next steps for scale: a feature store for parity at
volume, async batch scoring on a schedule, and a hosted registry with auto-promotion.

## 8. Future work

- Concept-drift alerting wired to the monitoring batch (page on sustained alert).
- Online threshold re-tuning as the real cost ratio changes.
- A/B test the retention actions the scores trigger to measure realised ROI.
- DVC for data versioning; a small dashboard over batch scores.
