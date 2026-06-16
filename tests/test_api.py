from fastapi.testclient import TestClient

from hiking_recommender.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["users"] > 0
    assert payload["routes"] > 0


def test_recommendations_warm_user_contract() -> None:
    response = client.post(
        "/recommendations",
        json={
            "user_id": "user_001",
            "region": "north",
            "top_k": 5,
            "max_difficulty": "moderate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user_001"
    assert 0 < len(payload["recommendations"]) <= 5
    assert [item["rank"] for item in payload["recommendations"]] == list(
        range(1, len(payload["recommendations"]) + 1)
    )
    route_ids = [item["route_id"] for item in payload["recommendations"]]
    assert len(route_ids) == len(set(route_ids))
    assert all(isinstance(item["score"], float) for item in payload["recommendations"])


def test_recommendations_unknown_user_uses_popularity_fallback() -> None:
    response = client.post(
        "/recommendations",
        json={
            "user_id": "user_999",
            "region": "north",
            "top_k": 5,
            "max_difficulty": "moderate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user_999"
    assert 0 < len(payload["recommendations"]) <= 5


def test_recommendations_invalid_difficulty_returns_400() -> None:
    response = client.post(
        "/recommendations",
        json={
            "user_id": "user_001",
            "top_k": 5,
            "max_difficulty": "extreme",
        },
    )

    assert response.status_code == 400
