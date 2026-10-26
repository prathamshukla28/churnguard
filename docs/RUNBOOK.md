# ChurnGuard — Operations Runbook

On-call reference for the ChurnGuard scoring service. Pair with [DESIGN.md](DESIGN.md)
(architecture) and [adr/](adr/) (why things are the way they are).

## Service at a glance

| | |
| --- | --- |
| Serve | `make serve` → `uvicorn churnguard.api:app` (`:8000`) |
| Health | `GET /health` — `{status, model_ready, uptime_seconds}` |
| Live metrics | `GET /metrics/prometheus` · `GET /stats` |
| Logs | structured JSON on stdout (`request_id`, `path`, `status`, `latency_ms`) |
| State | **stateless** except best-effort in-memory metrics |
| Artifact | `models/churnguard.joblib` (the `ModelBundle`) |

## Quick diagnosis

```bash
curl -s localhost:8000/health            # is a model loaded? how long up?
curl -s localhost:8000/stats             # throughput + latency percentiles
curl -s localhost:8000/metrics/prometheus | grep churnguard_requests_total
```
Logs are one JSON object per request — filter by `"status": 5` or by `request_id`
(returned to the caller in the `X-Request-ID` response header, so a user-reported
failure can be traced to its exact log line).

---

## Alerts & responses

### A1 — `model_ready: false` / `/predict` returns 503
**Meaning:** no model artifact at the configured path.
1. Confirm: `curl -s localhost:8000/health`.
2. If a trained model exists elsewhere, deploy it to `models/churnguard.joblib`.
3. Otherwise roll back to the last good registry version:
   ```bash
   make registry                       # find the production / last-good version
   cp models/registry/v<N>/model.joblib models/churnguard.joblib
   ```
4. Restart the service. Verify `model_ready: true`.

### A2 — Data drift (PSI ≥ alert)
**Meaning:** the live population no longer matches training (`/monitor` → `alert`).
1. Pull a recent batch and run `python -m churnguard.monitor recent.csv --out drift.json`.
2. Read `features[]` sorted by PSI — which inputs moved? Check for an **upstream
   data bug** first (unit change, new category, broken join) before assuming real drift.
3. If genuine drift: retrain (A4). If a data bug: fix the source; scores since the
   bug began are suspect.

### A3 — Performance decay (performance status ≥ warn)
**Meaning:** with ground-truth labels attached, live ROC-AUC dropped vs. baseline.
1. Confirm the label join is correct (stale/partial labels mimic decay).
2. Retrain (A4) and re-evaluate; compare to `reports/baseline_metrics.json`.
3. Re-tune the threshold if the business cost ratio has changed.

### A4 — Retrain & promote
```bash
python scripts/retrain.py        # trains a challenger, evaluates, auto-promotes if better
make registry                    # confirm the new production version
```
The challenger is promoted only if it beats the champion on `registry.promotion_metric`.
CI runs `scripts/check_regression.py`, so a regressing model is blocked before deploy.

**Rollback:** `python -m churnguard.registry promote <OLD_VERSION> production`, then
copy that version's `model.joblib` to the serving path and restart.

### A5 — Elevated latency (p99 breach)
1. `GET /stats` for current p50/p95/p99; confirm against the SLO in DESIGN.md §5.
2. Check CPU saturation — serving is CPU-bound on preprocessing + `predict_proba`.
3. Short term: scale out (the service is stateless). Encourage **batched** requests
   (many customers per call) to amortise per-request overhead.

### A6 — Spike of 429s
**Meaning:** clients are hitting the per-IP rate limit.
- Legitimate load: raise `CHURNGUARD_RATE_LIMIT` (req/min) and scale out.
- Abuse: keep the limit; block upstream at the gateway.

### A7 — Spike of 401s
**Meaning:** API-key auth is on (`CHURNGUARD_API_KEY` set) and callers lack the key.
- Confirm clients send `X-API-Key`. Rotate the key via the environment + redeploy if leaked.

---

## Routine operations

- **Full rebuild:** `make all` (data → train → evaluate → lint → typecheck → test).
- **Regenerate monitoring baseline:** `make evaluate` (rewrites `reports/reference_profile.json`
  and `reports/baseline_metrics.json`). Do this intentionally — it moves the goalposts.
- **Benchmark serving:** `make loadtest` → `reports/loadtest.md`.
- **Inspect models:** `make registry`.

## Configuration (env overrides)
| Variable | Effect |
| --- | --- |
| `CHURNGUARD_API_KEY` | Require `X-API-Key` on `/predict` and `/monitor` (unset = open). |
| `CHURNGUARD_RATE_LIMIT` | Requests/min per client IP (overrides config; `0` = off). |
| `CHURNGUARD_TRACKING` | `0` disables MLflow logging. |
| `LOG_LEVEL` | Log verbosity (default `INFO`). |
