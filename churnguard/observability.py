"""Production observability and request hardening for the API.

Kept out of ``api.py`` so the service wiring stays readable. Provides:

* :class:`MetricsRegistry` — thread-safe request/latency/prediction counters with
  a Prometheus text exposition and percentile latencies (p50/p95/p99).
* :func:`configure_logging` — one-line structured **JSON** logs with a request id,
  the shape an aggregator (Loki, CloudWatch, Datadog) can parse directly.
* :class:`RateLimiter` — a dependency-free fixed-window per-client limiter whose
  memory is bounded (stale windows are evicted; a hard cap on tracked clients).
* :func:`client_ip` — the rate-limit key; honours ``X-Forwarded-For`` **only** when
  trusted-proxy mode is explicitly enabled, otherwise any client could spoof it.
* :func:`api_key_required` / :func:`api_key_valid` — optional ``X-API-Key`` auth,
  enabled by setting ``CHURNGUARD_API_KEY`` (open when unset, so local and demo
  use needs no secret), compared in constant time.
* :func:`route_label` / :func:`method_label` — bounded Prometheus label values
  (the matched route *template*, never the raw path), so a scan of random URLs
  cannot create unbounded time series.

All of it is standard-library only; no agent or exporter to deploy.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from typing import Any

_logger = logging.getLogger("churnguard")


class _JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "latency_ms", "client"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    """Attach a JSON handler once; honour ``LOG_LEVEL`` (default INFO)."""
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        _logger.addHandler(handler)
        _logger.propagate = False
    _logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return _logger


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


class MetricsRegistry:
    """Thread-safe in-memory metrics for the serving process."""

    def __init__(self, max_latencies: int = 2000) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum = 0.0
        self._latencies: deque[float] = deque(maxlen=max_latencies)
        self.predictions_served = 0
        self._prob_sum = 0.0
        self._bands: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

    def record_request(self, method: str, path: str, status: int, latency_s: float) -> None:
        with self._lock:
            self._requests[(method, path, status)] += 1
            self._latency_sum += latency_s
            self._latencies.append(latency_s)

    def record_predictions(self, predictions: list[dict]) -> None:
        with self._lock:
            for p in predictions:
                self.predictions_served += 1
                self._prob_sum += p["churn_probability"]
                band = p.get("risk_band")
                if band in self._bands:
                    self._bands[band] += 1

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._started, 1)

    def stats_snapshot(self) -> dict:
        with self._lock:
            n = self.predictions_served
            lat = sorted(self._latencies)
            return {
                "uptime_seconds": self.uptime_seconds(),
                "predictions_served": n,
                "mean_churn_probability": round(self._prob_sum / n, 4) if n else None,
                "risk_bands": dict(self._bands),
                "requests_total": sum(self._requests.values()),
                "latency_ms": {
                    "p50": round(_percentile(lat, 0.50) * 1000, 2),
                    "p95": round(_percentile(lat, 0.95) * 1000, 2),
                    "p99": round(_percentile(lat, 0.99) * 1000, 2),
                },
            }

    def prometheus(self) -> str:
        """Prometheus text exposition (version 0.0.4)."""
        with self._lock:
            lat = sorted(self._latencies)
            lines = [
                "# HELP churnguard_requests_total Total HTTP requests served.",
                "# TYPE churnguard_requests_total counter",
            ]
            for (method, path, status), count in sorted(self._requests.items()):
                lines.append(
                    f'churnguard_requests_total{{method="{method}",path="{path}",'
                    f'status="{status}"}} {count}'
                )
            lines += [
                "# HELP churnguard_request_latency_seconds Request latency.",
                "# TYPE churnguard_request_latency_seconds summary",
                f"churnguard_request_latency_seconds_sum {self._latency_sum:.6f}",
                f"churnguard_request_latency_seconds_count {len(self._latencies)}",
            ]
            for q in (0.5, 0.95, 0.99):
                lines.append(
                    f'churnguard_request_latency_seconds{{quantile="{q}"}} '
                    f"{_percentile(lat, q):.6f}"
                )
            lines += [
                "# HELP churnguard_predictions_total Predictions served, by risk band.",
                "# TYPE churnguard_predictions_total counter",
            ]
            for band, count in self._bands.items():
                lines.append(f'churnguard_predictions_total{{band="{band}"}} {count}')
            lines += [
                "# HELP churnguard_uptime_seconds Seconds since process start.",
                "# TYPE churnguard_uptime_seconds gauge",
                f"churnguard_uptime_seconds {self.uptime_seconds()}",
            ]
        return "\n".join(lines) + "\n"


class RateLimiter:
    """Fixed-window per-key rate limiter (requests per 60s). ``per_minute=0`` = off.

    Memory is bounded two ways: entries from finished windows are swept whenever
    a new window starts, and at most ``max_keys`` clients are tracked at once
    (the oldest entry is dropped to admit a new one).
    """

    def __init__(
        self,
        per_minute: int,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.per_minute = int(per_minute)
        self.max_keys = max(1, int(max_keys))
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)
        self._current_window: int | None = None

    def __len__(self) -> int:
        return len(self._windows)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = int(self._clock())
        window = now - (now % 60)
        with self._lock:
            if window != self._current_window:
                # New window: every older entry is dead weight.
                self._windows = {k: v for k, v in self._windows.items() if v[0] == window}
                self._current_window = window
            start, count = self._windows.get(key, (window, 0))
            if start != window:
                start, count = window, 0
            if key not in self._windows and len(self._windows) >= self.max_keys:
                self._windows.pop(next(iter(self._windows)))
            count += 1
            self._windows[key] = (start, count)
            return count <= self.per_minute


def client_ip(
    peer: str | None,
    headers: Mapping[str, str],
    trust_forwarded_for: bool = False,
    trusted_hops: int = 1,
) -> str:
    """The client address to key rate limits (and logs) on.

    Without trusted-proxy mode this is always the TCP peer: ``X-Forwarded-For`` is
    attacker-controlled. With it, each trusted proxy appends the address it saw,
    so the real client is ``trusted_hops`` entries from the right; anything to
    the left of that was supplied by the client and is ignored.
    """
    fallback = peer or "unknown"
    if not trust_forwarded_for:
        return fallback
    raw = headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in raw.split(",") if h.strip()]
    if not hops:
        return fallback
    return hops[-min(max(1, trusted_hops), len(hops))]


def api_key_required() -> str | None:
    """The configured API key, or ``None`` when auth is disabled."""
    key = os.environ.get("CHURNGUARD_API_KEY", "").strip()
    return key or None


def api_key_valid(provided: str | None, required: str) -> bool:
    """Constant-time comparison so response timing does not leak the key."""
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), required.encode("utf-8"))


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment override (1/true/yes/on), else ``default``."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


_KNOWN_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})


def method_label(method: str) -> str:
    """Bound the ``method`` label: arbitrary verbs collapse to ``OTHER``."""
    upper = method.upper()
    return upper if upper in _KNOWN_METHODS else "OTHER"


def route_label(scope: Mapping[str, Any]) -> str:
    """The matched route template (e.g. ``/predict``), or ``unmatched``.

    Never the raw request path, which a client controls (``/a1``, ``/a2``, ...).
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else "unmatched"
