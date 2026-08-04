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


def test_web_ui_index_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Recommender Lab" in response.text
    assert "Engine Active" in response.text


def test_web_ui_search_endpoint_returns_recommendations_partial() -> None:
    response = client.post("/search", data={"user_id": "user_001", "top_k": "5"})

    assert response.status_code == 200
    assert "Recommendations for" in response.text
    assert "user_001" in response.text
    assert "route_" in response.text
    assert "km" in response.text
    assert "hours" in response.text
    assert "m gain" in response.text
    assert "RUB" not in response.text


def test_web_ui_upload_endpoint_reports_validation_errors() -> None:
    response = client.post(
        "/upload",
        files={
            "items_csv": ("items.csv", b"route_id\nroute_001\n", "text/csv"),
            "interactions_csv": ("interactions.csv", b"user_id\nuser_001\n", "text/csv"),
        },
    )

    assert response.status_code == 200
    assert "Error" in response.text
    assert "missing columns" in response.text


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


def test_web_ui_upload_endpoint_accepts_synthetic_data() -> None:
    with open("data/synthetic_routes.csv", "rb") as routes_file:
        routes_content = routes_file.read()
    with open("data/synthetic_interactions.csv", "rb") as interactions_file:
        interactions_content = interactions_file.read()

    response = client.post(
        "/upload",
        files={
            "items_csv": ("synthetic_routes.csv", routes_content, "text/csv"),
            "interactions_csv": ("synthetic_interactions.csv", interactions_content, "text/csv"),
        },
    )

    assert response.status_code == 200
    assert "Data uploaded & models trained!" in response.text
    assert "items" in response.text
