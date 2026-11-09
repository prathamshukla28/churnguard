"""Tests for the metrics registry, rate limiter, client IP and auth helpers."""

from __future__ import annotations

from churnguard import observability
from churnguard.observability import (
    MetricsRegistry,
    RateLimiter,
    api_key_valid,
    client_ip,
    env_flag,
    method_label,
    route_label,
)


class _Clock:
    def __init__(self, t: float = 1_000_020.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_rate_limiter_blocks_over_the_window():
    rl = RateLimiter(per_minute=3)
    assert [rl.allow("1.2.3.4") for _ in range(3)] == [True, True, True]
    assert rl.allow("1.2.3.4") is False
    # A different client has its own budget.
    assert rl.allow("9.9.9.9") is True


def test_rate_limiter_disabled_allows_everything():
    rl = RateLimiter(per_minute=0)
    assert all(rl.allow("x") for _ in range(200))


def test_rate_limiter_evicts_stale_windows():
    clock = _Clock()
    rl = RateLimiter(per_minute=5, clock=clock)
    for i in range(500):
        rl.allow(f"10.0.{i // 256}.{i % 256}")
    assert len(rl) == 500
    clock.t += 60  # next window: every old entry is dead
    rl.allow("1.1.1.1")
    assert len(rl) == 1


def test_rate_limiter_caps_tracked_clients():
    rl = RateLimiter(per_minute=5, max_keys=100, clock=_Clock())
    for i in range(1000):
        rl.allow(f"client-{i}")
    assert len(rl) == 100


def test_rate_limiter_resets_budget_next_window():
    clock = _Clock()
    rl = RateLimiter(per_minute=1, clock=clock)
    assert rl.allow("a") is True
    assert rl.allow("a") is False
    clock.t += 60
    assert rl.allow("a") is True


def test_client_ip_ignores_forwarded_for_unless_trusted():
    headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.9"}
    assert client_ip("10.0.0.1", headers) == "10.0.0.1"
    assert client_ip("10.0.0.1", headers, trust_forwarded_for=True) == "203.0.113.9"
    # Two trusted proxies: the client is two entries from the right.
    assert client_ip("10.0.0.1", headers, trust_forwarded_for=True, trusted_hops=2) == "6.6.6.6"
    # Asking for more hops than present never walks past the header.
    assert client_ip("10.0.0.1", headers, trust_forwarded_for=True, trusted_hops=9) == "6.6.6.6"
    assert client_ip(None, {}, trust_forwarded_for=True) == "unknown"


def test_api_key_compare_is_constant_time(monkeypatch):
    calls = []
    real = observability.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(observability.hmac, "compare_digest", spy)
    assert api_key_valid("s3cret", "s3cret") is True
    assert api_key_valid("nope", "s3cret") is False
    assert api_key_valid(None, "s3cret") is False
    assert len(calls) == 2


def test_env_flag(monkeypatch):
    monkeypatch.delenv("CG_FLAG", raising=False)
    assert env_flag("CG_FLAG", True) is True
    monkeypatch.setenv("CG_FLAG", "0")
    assert env_flag("CG_FLAG", True) is False
    monkeypatch.setenv("CG_FLAG", "yes")
    assert env_flag("CG_FLAG", False) is True


def test_labels_are_bounded():
    class _Route:
        path = "/predict"

    assert route_label({"route": _Route()}) == "/predict"
    assert route_label({}) == "unmatched"
    assert method_label("get") == "GET"
    assert method_label("FOOBAR") == "OTHER"


def test_metrics_registry_stats_and_prometheus():
    m = MetricsRegistry()
    m.record_request("POST", "/predict", 200, 0.012)
    m.record_request("GET", "/health", 200, 0.001)
    m.record_predictions(
        [
            {"churn_probability": 0.91, "risk_band": "high"},
            {"churn_probability": 0.10, "risk_band": "low"},
        ]
    )

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
