"""Pushgateway-backed pipeline metrics for offline evaluation runs.

Mirrors src/monitoring/pipeline_metrics.py from go_through_the_forest
(production).  Used by eval scripts to push precision/recall/NDCG/etc.
into Prometheus so they can be tracked over time without scraping the
live API.
"""

from __future__ import annotations

import os
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "127.0.0.1:9091")
PUSHGATEWAY_TIMEOUT_SECONDS = float(os.getenv("PUSHGATEWAY_TIMEOUT_SECONDS", "5"))

_PUSH_ENABLED: bool | None = None

_REGISTRY = CollectorRegistry()
_REGISTERED: dict[str, Gauge] = {}


def _push_enabled() -> bool:
    global _PUSH_ENABLED
    if _PUSH_ENABLED is None:
        _PUSH_ENABLED = os.getenv("PROMETHEUS_PUSH_ENABLED", "1").strip() not in (
            "0",
            "false",
            "",
        )
    return _PUSH_ENABLED


def _get_or_create_gauge(
    name: str,
    label_names: list[str],
    description: str,
) -> Gauge:
    key = name
    if key in _REGISTERED:
        return _REGISTERED[key]
    g = Gauge(name, description, label_names, registry=_REGISTRY)
    _REGISTERED[key] = g
    return g


def push_pipeline_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
    description: str = "",
    job: str = "pipeline",
) -> None:
    """Push a single gauge metric to the Pushgateway."""
    if not _push_enabled():
        return

    try:
        label_names = list(labels.keys()) if labels else []
        g = _get_or_create_gauge(name, label_names, description)
        if labels:
            g.labels(**labels).set(value)
        else:
            g.set(value)
        push_to_gateway(
            PUSHGATEWAY_URL,
            job=job,
            registry=_REGISTRY,
            timeout=PUSHGATEWAY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(f"PUSH_METRIC_ERROR: {name}={value} labels={labels}: {exc}")


def push_pipeline_metrics_batch(
    metrics: list[dict[str, Any]],
    job: str = "pipeline",
) -> None:
    """Push a batch of gauge metrics in a single Pushgateway call."""
    if not _push_enabled():
        return

    try:
        for m in metrics:
            name = m["name"]
            value = float(m["value"])
            labels = m.get("labels")
            description = m.get("description", "")
            label_names = list(labels.keys()) if labels else []
            g = _get_or_create_gauge(name, label_names, description)
            if labels:
                g.labels(**labels).set(value)
            else:
                g.set(value)
        push_to_gateway(
            PUSHGATEWAY_URL,
            job=job,
            registry=_REGISTRY,
            timeout=PUSHGATEWAY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(f"PUSH_METRICS_BATCH_ERROR: {exc}")
