import pandas as pd

from hiking_recommender.data_loader import DemoDataset
from hiking_recommender.features import build_features, build_route_features, build_seen_routes, build_user_item_matrix


def test_build_route_features_adds_numeric_and_encoded_columns() -> None:
    routes = pd.DataFrame(
        [
            {
                "route_id": "route_001",
                "region": "north",
                "length_km": 5.0,
                "duration_hours": 2.0,
                "elevation_gain_m": 250,
                "difficulty": "moderate",
                "popularity": 0.7,
                "season": "summer",
            },
            {
                "route_id": "route_002",
                "region": "south",
                "length_km": 10.0,
                "duration_hours": 4.0,
                "elevation_gain_m": 900,
                "difficulty": "hard",
                "popularity": 0.4,
                "season": "winter",
            },
        ]
    )

    features = build_route_features(routes)

    assert list(features["route_id"]) == ["route_001", "route_002"]
    assert features.loc[0, "elevation_per_km"] == 50.0
    assert features.loc[1, "difficulty_level"] == 3
    assert {"region_north", "season_summer", "difficulty_hard"} <= set(features.columns)


def test_build_user_item_matrix_aggregates_repeated_implicit_events() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001", "interaction_weight": 1},
            {"user_id": "user_001", "route_id": "route_001", "interaction_weight": 3},
            {"user_id": "user_001", "route_id": "route_002", "interaction_weight": 5},
            {"user_id": "user_002", "route_id": "route_002", "interaction_weight": 8},
        ]
    )

    matrix = build_user_item_matrix(interactions)

    assert matrix.loc["user_001", "route_001"] == 4.0
    assert matrix.loc["user_001", "route_002"] == 5.0
    assert matrix.loc["user_002", "route_001"] == 0.0
    assert list(matrix.index) == ["user_001", "user_002"]
    assert list(matrix.columns) == ["route_001", "route_002"]


def test_build_features_uses_train_interactions_only() -> None:
    dataset = DemoDataset(
        users=pd.DataFrame(),
        routes=pd.DataFrame(
            [
                {
                    "route_id": "route_001",
                    "region": "north",
                    "length_km": 5.0,
                    "duration_hours": 2.0,
                    "elevation_gain_m": 250,
                    "difficulty": "moderate",
                    "popularity": 0.7,
                    "season": "summer",
                },
                {
                    "route_id": "route_002",
                    "region": "south",
                    "length_km": 10.0,
                    "duration_hours": 4.0,
                    "elevation_gain_m": 900,
                    "difficulty": "hard",
                    "popularity": 0.4,
                    "season": "winter",
                },
            ]
        ),
        interactions=pd.DataFrame(),
        train_interactions=pd.DataFrame(
            [
                {"user_id": "user_001", "route_id": "route_001", "interaction_weight": 1},
            ]
        ),
        test_interactions=pd.DataFrame(
            [
                {"user_id": "user_001", "route_id": "route_002", "interaction_weight": 8},
            ]
        ),
    )

    features = build_features(dataset)

    assert set(features) == {"route_features", "user_item_matrix"}
    assert list(features["user_item_matrix"].columns) == ["route_001"]
    assert "route_002" not in features["user_item_matrix"].columns


def test_build_seen_routes_returns_public_route_ids() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001"},
            {"user_id": "user_001", "route_id": "route_002"},
        ]
    )

    assert build_seen_routes(interactions) == {"user_001": {"route_001", "route_002"}}
