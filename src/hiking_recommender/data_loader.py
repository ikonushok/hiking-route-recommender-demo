"""Data loading and validation for the synthetic hiking recommender demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

USER_COLUMNS = {
    "user_id",
    "preferred_difficulty",
    "preferred_region",
    "preferred_season",
    "preferred_tags",
    "activity_level",
}
ROUTE_COLUMNS = {
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
INTERACTION_COLUMNS = {
    "user_id",
    "route_id",
    "interaction_type",
    "timestamp",
    "interaction_weight",
}
EVENT_WEIGHTS = {
    "view": 1,
    "like": 3,
    "visit": 5,
    "checkin": 8,
}


@dataclass(frozen=True)
class DemoDataset:
    """Validated synthetic demo tables."""

    users: pd.DataFrame
    routes: pd.DataFrame
    interactions: pd.DataFrame
    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame


def load_demo_dataset(data_dir: str | Path = "data") -> DemoDataset:
    """Load and validate all synthetic CSV files from the demo data directory."""

    root = Path(data_dir)
    users = _load_csv(root / "synthetic_users.csv", USER_COLUMNS)
    routes = _load_csv(root / "synthetic_routes.csv", ROUTE_COLUMNS)
    interactions = _load_interactions(root / "synthetic_interactions.csv")
    train_interactions = _load_interactions(root / "synthetic_interactions_train.csv")
    test_interactions = _load_interactions(root / "synthetic_interactions_test.csv")

    _validate_users(users)
    _validate_routes(routes)
    _validate_interactions(interactions, users, routes, "interactions")
    _validate_interactions(train_interactions, users, routes, "train_interactions")
    _validate_interactions(test_interactions, users, routes, "test_interactions")

    return DemoDataset(
        users=users,
        routes=routes,
        interactions=interactions,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
    )


def _load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required synthetic data file is missing: {path}")

    frame = pd.read_csv(path)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = frame.loc[:, list(required_columns)].copy()
    if frame.isna().any().any():
        raise ValueError(f"{path} contains null values")
    return frame


def _load_interactions(path: Path) -> pd.DataFrame:
    frame = _load_csv(path, INTERACTION_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["interaction_weight"] = frame["interaction_weight"].astype(int)
    return frame


def _validate_users(users: pd.DataFrame) -> None:
    _validate_unique_id(users, "user_id", "users")
    if not users["user_id"].str.match(r"^user_\d{3}$").all():
        raise ValueError("users contain invalid public synthetic user_id values")


def _validate_routes(routes: pd.DataFrame) -> None:
    _validate_unique_id(routes, "route_id", "routes")
    if not routes["route_id"].str.match(r"^route_\d{3}$").all():
        raise ValueError("routes contain invalid public synthetic route_id values")

    numeric_columns = ["length_km", "duration_hours", "elevation_gain_m", "popularity"]
    for column in numeric_columns:
        routes[column] = pd.to_numeric(routes[column], errors="raise")
    if (routes["length_km"] <= 0).any() or (routes["duration_hours"] <= 0).any():
        raise ValueError("routes contain non-positive length or duration")
    if ((routes["popularity"] < 0) | (routes["popularity"] > 1)).any():
        raise ValueError("route popularity must be in [0, 1]")


def _validate_interactions(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    routes: pd.DataFrame,
    table_name: str,
) -> None:
    unknown_users = set(interactions["user_id"]) - set(users["user_id"])
    unknown_routes = set(interactions["route_id"]) - set(routes["route_id"])
    if unknown_users:
        raise ValueError(f"{table_name} contains unknown user_id values")
    if unknown_routes:
        raise ValueError(f"{table_name} contains unknown route_id values")

    unexpected_events = set(interactions["interaction_type"]) - set(EVENT_WEIGHTS)
    if unexpected_events:
        raise ValueError(f"{table_name} contains unsupported interaction types")

    expected_weights = interactions["interaction_type"].map(EVENT_WEIGHTS)
    if not interactions["interaction_weight"].eq(expected_weights).all():
        raise ValueError(f"{table_name} contains inconsistent interaction weights")


def _validate_unique_id(frame: pd.DataFrame, column: str, table_name: str) -> None:
    if frame[column].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate {column} values")
