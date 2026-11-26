# ChurnGuard serving benchmark

In-process benchmark via Starlette TestClient (no network), sequential requests,
one machine: Darwin 25.2.0 (arm64), 10 logical CPUs, Python 3.12.10; load average (1/5/15 min) at start: 7.2 / 8.3 / 7.2.

| Setting | Value |
| --- | --- |
| Requests | 500 |
| Batch size | 1 customer(s)/request |
| Explanations | False |

| Metric | Value |
| --- | --- |
| Throughput | **42 req/s** (42 customers/s) |
| Latency mean | 23.94 ms |
| Latency p50 | 23.70 ms |
| Latency p95 | 25.88 ms |
| Latency p99 | 28.67 ms |

> Benchmarks the serving path on one host; for distributed load testing point
> Locust or k6 at a deployed instance.
