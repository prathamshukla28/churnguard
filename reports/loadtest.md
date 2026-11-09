# ChurnGuard serving benchmark

In-process benchmark via Starlette TestClient (no network). One machine.

| Setting | Value |
| --- | --- |
| Requests | 500 |
| Batch size | 1 customer(s)/request |
| Explanations | False |

| Metric | Value |
| --- | --- |
| Throughput | **18 req/s** (18 customers/s) |
| Latency mean | 56.84 ms |
| Latency p50 | 51.67 ms |
| Latency p95 | 97.16 ms |
| Latency p99 | 138.25 ms |

> Benchmarks the serving path on one host; for distributed load testing point
> Locust or k6 at a deployed instance.
