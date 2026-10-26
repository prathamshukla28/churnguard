"""Tests for the metrics registry and the rate limiter."""

from __future__ import annotations

from churnguard.observability import MetricsRegistry, RateLimiter


def test_rate_limiter_blocks_over_the_window():
    rl = RateLimiter(per_minute=3)
    assert [rl.allow("1.2.3.4") for _ in range(3)] == [True, True, True]
    assert rl.allow("1.2.3.4") is False
    # A different client has its own budget.
    assert rl.allow("9.9.9.9") is True


def test_rate_limiter_disabled_allows_everything():
    rl = RateLimiter(per_minute=0)
    assert all(rl.allow("x") for _ in range(200))


def test_metrics_registry_stats_and_prometheus():
    m = MetricsRegistry()
    m.record_request("POST", "/predict", 200, 0.012)
    m.record_request("GET", "/health", 200, 0.001)
    m.record_predictions([
        {"churn_probability": 0.91, "risk_band": "high"},
        {"churn_probability": 0.10, "risk_band": "low"},
    ])

    snap = m.stats_snapshot()
    assert snap["predictions_served"] == 2
    assert snap["requests_total"] == 2
    assert snap["risk_bands"]["high"] == 1
    assert set(snap["latency_ms"]) == {"p50", "p95", "p99"}

    text = m.prometheus()
    assert "churnguard_requests_total" in text
    assert 'path="/predict"' in text
    assert 'churnguard_predictions_total{band="high"} 1' in text
    assert "churnguard_uptime_seconds" in text
