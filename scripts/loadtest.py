"""A small, dependency-free serving benchmark.

Fires N prediction requests at the app in-process (via Starlette's TestClient, so
it measures serving/model latency without network noise) and reports throughput
and latency percentiles. Writes a short Markdown report to ``reports/loadtest.md``.

    python scripts/loadtest.py --requests 500 --batch 1

Honest about what it is: this benchmarks the serving path on one machine, not a
distributed load test. For real load testing point Locust or k6 at a deployed
instance — the numbers here are a reproducible lower bound on per-request cost.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from fastapi.testclient import TestClient  # noqa: E402

from churnguard.api import app  # noqa: E402
from churnguard.config import load_config  # noqa: E402

_CUSTOMER = {
    "tenure": 3, "MonthlyCharges": 89.0, "TotalCharges": 267.0,
    "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check", "TechSupport": "No", "SeniorCitizen": 1,
}


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the serving path.")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1, help="Customers per request.")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    # Benchmark the serving path itself: silence per-request logs and lift the
    # rate limit (which exists to protect production, not to throttle a benchmark).
    import logging

    from churnguard import api
    from churnguard.observability import RateLimiter

    logging.getLogger("churnguard").setLevel(logging.WARNING)
    api.limiter = RateLimiter(0)

    client = TestClient(app)
    payload = {"customers": [_CUSTOMER] * args.batch, "explain": args.explain}
    client.post("/predict", json=payload)  # warm up (model load)

    latencies: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(args.requests):
        t = time.perf_counter()
        resp = client.post("/predict", json=payload)
        latencies.append((time.perf_counter() - t) * 1000)
        resp.raise_for_status()
    wall = time.perf_counter() - wall_start

    rps = args.requests / wall
    report = f"""# ChurnGuard serving benchmark

In-process benchmark via Starlette TestClient (no network). One machine.

| Setting | Value |
| --- | --- |
| Requests | {args.requests} |
| Batch size | {args.batch} customer(s)/request |
| Explanations | {args.explain} |

| Metric | Value |
| --- | --- |
| Throughput | **{rps:,.0f} req/s** ({rps * args.batch:,.0f} customers/s) |
| Latency mean | {statistics.mean(latencies):.2f} ms |
| Latency p50 | {_pct(latencies, 0.50):.2f} ms |
| Latency p95 | {_pct(latencies, 0.95):.2f} ms |
| Latency p99 | {_pct(latencies, 0.99):.2f} ms |

> Benchmarks the serving path on one host; for distributed load testing point
> Locust or k6 at a deployed instance.
"""
    out = load_config().path("artifacts", "metrics_path").parent / "loadtest.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
