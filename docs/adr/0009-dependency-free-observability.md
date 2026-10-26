# ADR-0009: Dependency-free observability & hardening

**Status:** Accepted

## Context
A service that only "works on my machine" is not production-ready. Operators need
structured logs, scrapeable metrics, latency visibility, and basic protection
(auth, rate limiting). Each concern has a popular library (`prometheus-client`,
`slowapi`, `python-json-logger`), but each adds a dependency, and the needs here are
modest and standard enough to meet without them.

## Decision
Implement observability in one standard-library module (`churnguard/observability.py`):
- **Structured JSON logs** with a per-request `request_id` (also returned as the
  `X-Request-ID` header), method, path, status and latency — parseable by any aggregator.
- A thread-safe **metrics registry** exposing a **Prometheus text exposition** at
  `/metrics/prometheus` plus latency p50/p95/p99 at `/stats`.
- A fixed-window **rate limiter** per client IP, and optional **API-key** auth gated
  by the `CHURNGUARD_API_KEY` env var (open when unset, so local/demo needs no secret).

## Consequences
- Zero new runtime dependencies; the metrics are genuinely Prometheus-scrapeable.
- Metrics are in-process (reset on restart) — correct for per-instance scraping, not
  a substitute for a central TSDB across replicas.
- The rate limiter is per-process; multi-replica deployments enforce global limits at
  the gateway. Both are honest about their scope rather than pretending to be more.
- `reports/loadtest.md` records the serving-path latency these metrics describe.
