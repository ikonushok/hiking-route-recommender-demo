"""Publish the demo's offline measurements as a cumulative Pushgateway snapshot."""

import os
from prometheus_client import CollectorRegistry, push_to_gateway
from prometheus_client.core import GaugeMetricFamily


class _Measurements:
    def __init__(self):
        self.values = {}

    def record(self, rows):
        for row in rows:
            labels = row.get("labels") or {}
            keys = tuple(sorted(labels))
            family = self.values.setdefault(row["name"], {
                "help": row.get("description", ""), "keys": keys, "samples": {},
            })
            if family["keys"] != keys:
                raise ValueError(f"Inconsistent labels for {row['name']}")
            family["samples"][tuple(str(labels[key]) for key in keys)] = float(row["value"])

    def collect(self):
        for name, family in self.values.items():
            metric = GaugeMetricFamily(name, family["help"], labels=family["keys"])
            for labels, value in family["samples"].items():
                metric.add_metric(labels, value)
            yield metric


_measurements = _Measurements()
_registry = CollectorRegistry()
_registry.register(_measurements)


def push_pipeline_metric(name, value, labels=None, description="", job="pipeline"):
    """Publish one observation through the same snapshot used for batches."""
    push_pipeline_metrics_batch([
        dict(name=name, value=value, labels=labels, description=description)
    ], job=job)


def push_pipeline_metrics_batch(metrics, job="pipeline"):
    """Keep previously published series when updating this process's snapshot."""
    enabled = os.environ.get("PROMETHEUS_PUSH_ENABLED", "1").strip()
    if enabled in {"", "0", "false"}:
        return
    try:
        _measurements.record(metrics)
        push_to_gateway(
            os.environ.get("PUSHGATEWAY_URL", "127.0.0.1:9091"),
            registry=_registry, job=job,
            timeout=float(os.environ.get("PUSHGATEWAY_TIMEOUT_SECONDS", "5")),
        )
    except Exception as error:
        print(f"Offline metric publication failed: {error}")
