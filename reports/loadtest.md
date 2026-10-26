# ChurnGuard serving benchmark

In-process benchmark via Starlette TestClient (no network). One machine.

| Setting | Value |
| --- | --- |
| Requests | 1000 |
| Batch size | 1 customer(s)/request |
| Explanations | False |

| Metric | Value |
| --- | --- |
| Throughput | **51 req/s** (51 customers/s) |
| Latency mean | 19.59 ms |
| Latency p50 | 19.32 ms |
| Latency p95 | 20.90 ms |
| Latency p99 | 22.39 ms |

> Benchmarks the serving path on one host; for distributed load testing point
> Locust or k6 at a deployed instance.
