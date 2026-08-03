"""Prometheus metrics for the demo recommender API."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_fastapi_instrumentator import Instrumentator

# ── Request-level histograms ──────────────────────────────────────────

recommendation_request_duration_seconds = Histogram(
    "recommendation_request_duration_seconds",
    "Full recommendation request duration (wall-clock)",
)

content_retrieval_duration_seconds = Histogram(
    "content_retrieval_duration_seconds",
    "Content-based retrieval step duration",
)

collaborative_retrieval_duration_seconds = Histogram(
    "collaborative_retrieval_duration_seconds",
    "Collaborative (cosine-similarity) retrieval step duration",
)

business_rules_duration_seconds = Histogram(
    "business_rules_duration_seconds",
    "Business-rules post-processing duration",
)

# ── Counters ──────────────────────────────────────────────────────────

recommendation_fallback_triggered_total = Counter(
    "recommendation_fallback_triggered_total",
    "How many times popular-fallback was used",
)

recommendation_candidates_total = Counter(
    "recommendation_candidates_total",
    "Candidates produced by each source",
    ["source"],
)

recommendation_user_type_total = Counter(
    "recommendation_user_type_total",
    "User type distribution (warm / cold)",
    ["type"],
)

business_rules_filtered_total = Counter(
    "business_rules_filtered_total",
    "Candidates removed by business rules",
)

# ── Gauges (model metadata) ───────────────────────────────────────────

model_users_count = Gauge(
    "model_users_count",
    "Number of users in the model",
    multiprocess_mode="max",
)

model_items_count = Gauge(
    "model_items_count",
    "Number of items in the model",
    multiprocess_mode="max",
)

model_interactions_count = Gauge(
    "model_interactions_count",
    "Number of interactions in the training set",
    multiprocess_mode="max",
)


# ── Setup ─────────────────────────────────────────────────────────────

def setup_metrics(app) -> None:
    """Register FastAPI instrumentator and expose /metrics endpoint."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    )
    instrumentator.instrument(app)

    @app.get("/metrics", include_in_schema=True)
    def _metrics() -> Response:
        registry = REGISTRY
        if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )


# ── Helpers ───────────────────────────────────────────────────────────

@contextmanager
def measure_time(histogram: Histogram) -> Generator[None, None, None]:
    """Context manager that records elapsed seconds into *histogram*."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)
