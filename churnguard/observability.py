"""Production observability and request hardening for the API.

Kept out of ``api.py`` so the service wiring stays readable. Provides:

* :class:`MetricsRegistry` — thread-safe request/latency/prediction counters with
  a Prometheus text exposition and percentile latencies (p50/p95/p99).
* :func:`configure_logging` — one-line structured **JSON** logs with a request id,
  the shape an aggregator (Loki, CloudWatch, Datadog) can parse directly.
* :class:`RateLimiter` — a dependency-free fixed-window per-client limiter.
* :func:`require_api_key` — optional ``X-API-Key`` auth, enabled by setting the
  ``CHURNGUARD_API_KEY`` environment variable (open when it is unset, so local and
  demo use needs no secret).

All of it is standard-library only; no agent or exporter to deploy.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque

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
    """Fixed-window per-key rate limiter (requests per 60s). ``per_minute=0`` = off."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = int(per_minute)
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = int(time.time())
        window = now - (now % 60)
        with self._lock:
            start, count = self._windows.get(key, (window, 0))
            if start != window:
                start, count = window, 0
            count += 1
            self._windows[key] = (start, count)
            return count <= self.per_minute


def api_key_required() -> str | None:
    """The configured API key, or ``None`` when auth is disabled."""
    key = os.environ.get("CHURNGUARD_API_KEY", "").strip()
    return key or None
