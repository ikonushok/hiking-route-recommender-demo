import pytest
from prometheus_client import CollectorRegistry, Histogram, generate_latest
from hiking_recommender import pipeline_metrics as offline
from hiking_recommender.monitoring import measure_time
from hiking_recommender.api import app
from fastapi.testclient import TestClient


def test_api_metrics_remain_available():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/recommendations", json={"user_id": "user_001", "top_k": 3}).status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        for name in ("recommendation_request_duration_seconds_count",
                     "recommendation_candidates_total", "model_users_count", "http_requests_total"):
            assert name in metrics.text


def test_timer_records_failed_body():
    registry = CollectorRegistry()
    histogram = Histogram("test_seconds", "test", registry=registry)
    with pytest.raises(RuntimeError), measure_time(histogram):
        raise RuntimeError("body failed")
    assert registry.get_sample_value("test_seconds_count") == 1


def test_offline_snapshot_updates_without_losing_other_series(monkeypatch):
    measurements = offline._Measurements()
    registry = CollectorRegistry()
    registry.register(measurements)
    monkeypatch.setattr(offline, "_measurements", measurements)
    monkeypatch.setattr(offline, "_registry", registry)
    monkeypatch.setenv("PROMETHEUS_PUSH_ENABLED", "1")
    snapshots = []
    monkeypatch.setattr(offline, "push_to_gateway", lambda *args, **kwargs:
                        snapshots.append(generate_latest(kwargs["registry"]).decode()))
    offline.push_pipeline_metric("score", .5, {"model": "content"})
    offline.push_pipeline_metrics_batch([
        {"name": "score", "value": .8, "labels": {"model": "content"}},
        {"name": "score", "value": .6, "labels": {"model": "hybrid"}},
    ])
    assert 'score{model="content"} 0.8' in snapshots[-1]
    assert 'score{model="hybrid"} 0.6' in snapshots[-1]
    monkeypatch.setenv("PROMETHEUS_PUSH_ENABLED", "0")
    offline.push_pipeline_metric("score", 9, {"model": "content"})
    assert len(snapshots) == 2


def test_gateway_failure_does_not_break_caller(monkeypatch, capsys):
    monkeypatch.setenv("PROMETHEUS_PUSH_ENABLED", "1")
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(offline, "push_to_gateway", unavailable)
    offline.push_pipeline_metric("offline_test", 1)
    assert "unavailable" in capsys.readouterr().out
