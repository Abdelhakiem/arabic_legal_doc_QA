"""Small, privacy-safe Prometheus metrics for the API and RAGAS monitor."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Mapping

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    push_to_gateway,
)


@dataclass
class HTTPMetrics:
    """Per-app registry; avoids duplicate collectors in tests/reloads."""

    registry: CollectorRegistry
    requests: Counter
    latency: Histogram

    def observe(self, method: str, route: str, status: int, elapsed: float) -> None:
        labels = (method, route, str(status))
        self.requests.labels(*labels).inc()
        self.latency.labels(method, route).observe(elapsed)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def create_http_metrics() -> HTTPMetrics:
    registry = CollectorRegistry()
    requests = Counter(
        "http_requests_total",
        "HTTP requests served by the RAG API.",
        ("method", "route", "status"),
        registry=registry,
    )
    latency = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds.",
        ("method", "route"),
        registry=registry,
    )
    return HTTPMetrics(registry, requests, latency)


def push_ragas_metrics(
    gateway_url: str | None,
    *,
    scores: Mapping[str, Mapping[str, float]] | None = None,
    deltas: Mapping[str, float] | None = None,
    faithfulness_threshold: float = 0.8,
    drift_threshold: float = 0.1,
    success: bool,
    alerts: Mapping[str, bool] | None = None,
) -> None:
    """Push only bounded-cardinality aggregates, never question/user labels."""

    if not gateway_url:
        return

    registry = CollectorRegistry()
    last_attempt = Gauge(
        "ragas_monitor_last_attempt_timestamp_seconds",
        "Unix timestamp of the latest RAGAS monitor attempt.",
        registry=registry,
    )
    last_success = Gauge(
        "ragas_monitor_last_success_timestamp_seconds",
        "Unix timestamp of the latest successful RAGAS monitor run.",
        registry=registry,
    )
    run_success = Gauge(
        "ragas_monitor_run_success",
        "Whether the latest monitor run completed successfully (1/0).",
        registry=registry,
    )
    threshold = Gauge(
        "ragas_faithfulness_threshold",
        "Configured faithfulness alert threshold.",
        registry=registry,
    )
    drift_limit = Gauge(
        "ragas_faithfulness_drift_threshold",
        "Configured negative faithfulness drift threshold.",
        registry=registry,
    )
    quality = Gauge(
        "ragas_quality_score",
        "Mean RAGAS score for the most recent monitoring run.",
        ("metric", "language"),
        registry=registry,
    )
    metric_delta = Gauge(
        "ragas_metric_delta",
        "Change from the previous run on the same question set and corpus.",
        ("metric", "language"),
        registry=registry,
    )
    alert_gauge = Gauge(
        "ragas_quality_alert",
        "Whether a quality alert condition is present in the latest run.",
        ("alert",),
        registry=registry,
    )

    timestamp = time()
    last_attempt.set(timestamp)
    run_success.set(float(success))
    threshold.set(faithfulness_threshold)
    drift_limit.set(drift_threshold)
    if success:
        last_success.set(timestamp)
    if scores:
        for metric, languages in scores.items():
            for language, value in languages.items():
                quality.labels(metric=metric, language=language).set(value)
    if deltas:
        for metric, value in deltas.items():
            metric_delta.labels(metric=metric, language="all").set(value)
    for alert_name in ("faithfulness_below_threshold", "faithfulness_drift"):
        alert_gauge.labels(alert=alert_name).set(
            float(bool((alerts or {}).get(alert_name, False)))
        )

    # A fixed job/grouping key replaces the previous snapshot; Prometheus then
    # scrapes it over time to make session-to-session trends queryable.
    push_to_gateway(
        gateway_url,
        job="ragas-monitor",
        grouping_key={"service": "arabic-legal-qa"},
        registry=registry,
    )


__all__ = ["HTTPMetrics", "create_http_metrics", "push_ragas_metrics"]
