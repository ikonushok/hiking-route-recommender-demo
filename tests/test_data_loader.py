from pathlib import Path

import pandas as pd
import pytest

from hiking_recommender.data_loader import EVENT_WEIGHTS, load_demo_dataset


def test_load_demo_dataset_validates_schema_and_references(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)

    dataset = load_demo_dataset(tmp_path)

    assert set(dataset.users.columns) == {
        "user_id",
        "preferred_difficulty",
        "preferred_region",
        "preferred_season",
        "preferred_tags",
        "activity_level",
    }
    assert set(dataset.routes.columns) == {
        "route_id",
        "region",
        "length_km",
        "duration_hours",
        "elevation_gain_m",
        "difficulty",
        "popularity",
        "season",
        "route_tags",
    }
    assert set(dataset.interactions["interaction_type"]) <= set(EVENT_WEIGHTS)
    assert dataset.train_interactions["timestamp"].dt.tz is not None


def test_load_demo_dataset_rejects_unknown_route_reference(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    interactions_path = tmp_path / "synthetic_interactions_train.csv"
    interactions = pd.read_csv(interactions_path)
    interactions.loc[0, "route_id"] = "route_999"
    interactions.to_csv(interactions_path, index=False)

    with pytest.raises(ValueError, match="unknown route_id"):
        load_demo_dataset(tmp_path)


def test_load_demo_dataset_rejects_inconsistent_event_weight(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    interactions_path = tmp_path / "synthetic_interactions.csv"
    interactions = pd.read_csv(interactions_path)
    interactions.loc[0, "interaction_weight"] = 99
    interactions.to_csv(interactions_path, index=False)

    with pytest.raises(ValueError, match="inconsistent interaction weights"):
        load_demo_dataset(tmp_path)


def _write_minimal_dataset(root: Path) -> None:
    users = pd.DataFrame(
        [
            {
                "user_id": "user_001",
                "preferred_difficulty": "easy",
                "preferred_region": "north",
                "preferred_season": "summer",
                "preferred_tags": "forest|lake",
                "activity_level": "low",
            },
            {
                "user_id": "user_002",
                "preferred_difficulty": "moderate",
                "preferred_region": "south",
                "preferred_season": "spring",
                "preferred_tags": "mountain|viewpoint",
                "activity_level": "high",
            },
        ]
    )
    routes = pd.DataFrame(
        [
            {
                "route_id": "route_001",
                "region": "north",
                "length_km": 4.0,
                "duration_hours": 1.5,
                "elevation_gain_m": 120,
                "difficulty": "easy",
                "popularity": 0.8,
                "season": "summer",
                "route_tags": "forest|lake",
            },
            {
                "route_id": "route_002",
                "region": "south",
                "length_km": 9.0,
                "duration_hours": 3.5,
                "elevation_gain_m": 520,
                "difficulty": "moderate",
                "popularity": 0.6,
                "season": "spring",
                "route_tags": "mountain|viewpoint",
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            {
                "user_id": "user_001",
                "route_id": "route_001",
                "interaction_type": "view",
                "timestamp": "2025-01-01T00:00:00Z",
                "interaction_weight": 1,
            },
            {
                "user_id": "user_002",
                "route_id": "route_002",
                "interaction_type": "checkin",
                "timestamp": "2025-01-02T00:00:00Z",
                "interaction_weight": 8,
            },
        ]
    )

    users.to_csv(root / "synthetic_users.csv", index=False)
    routes.to_csv(root / "synthetic_routes.csv", index=False)
    interactions.to_csv(root / "synthetic_interactions.csv", index=False)
    interactions.head(1).to_csv(root / "synthetic_interactions_train.csv", index=False)
    interactions.tail(1).to_csv(root / "synthetic_interactions_test.csv", index=False)
