"""Public-library instrumentation with the demo dashboard's metric names."""

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


_HISTOGRAMS = {
    "recommendation_request_duration_seconds": "Recommendation request time",
    "content_retrieval_duration_seconds": "Content candidate retrieval time",
    "collaborative_retrieval_duration_seconds": "Collaborative candidate retrieval time",
    "business_rules_duration_seconds": "Candidate filtering time",
}
_COUNTERS = {
    "recommendation_fallback_triggered_total": ("Popularity fallback uses", ()),
    "recommendation_candidates_total": ("Retrieved candidates", ("source",)),
    "recommendation_user_type_total": ("Requests by user history", ("type",)),
    "business_rules_filtered_total": ("Filtered candidates", ()),
}
for _name, _help in _HISTOGRAMS.items():
    globals()[_name] = Histogram(_name, _help)
for _name, (_help, _labels) in _COUNTERS.items():
    globals()[_name] = Counter(_name, _help, labelnames=_labels)
for _entity in ("users", "items", "interactions"):
    _name = f"model_{_entity}_count"
    globals()[_name] = Gauge(_name, f"Loaded {_entity}", multiprocess_mode="max")


def setup_metrics(app):
    """Delegate HTTP instrumentation and multiprocess exposition to Instrumentator."""
    Instrumentator(
        excluded_handlers=["/health", "/metrics"],
        should_ignore_untemplated=True, should_group_status_codes=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


def measure_time(histogram):
    """Use Prometheus's timer, including observations when the body raises."""
    return histogram.time()
