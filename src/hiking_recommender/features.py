"""Feature builders for the synthetic hiking route recommender demo."""

from __future__ import annotations

import pandas as pd

from hiking_recommender.data_loader import DemoDataset

DIFFICULTY_LEVEL = {
    "easy": 1,
    "moderate": 2,
    "hard": 3,
}


def build_route_features(routes: pd.DataFrame) -> pd.DataFrame:
    """Build numeric route-level features for retrieval models."""

    features = routes.copy()
    features["elevation_per_km"] = features["elevation_gain_m"] / features["length_km"]
    features["difficulty_level"] = features["difficulty"].map(DIFFICULTY_LEVEL)
    if features["difficulty_level"].isna().any():
        raise ValueError("routes contain unsupported difficulty values")

    encoded = pd.get_dummies(
        features[["region", "season", "difficulty"]],
        prefix=["region", "season", "difficulty"],
        dtype=float,
    )
    numeric = features[
        [
            "route_id",
            "length_km",
            "duration_hours",
            "elevation_gain_m",
            "elevation_per_km",
            "difficulty_level",
            "popularity",
        ]
    ].copy()
    return pd.concat([numeric, encoded], axis=1)


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build a user-route matrix from implicit feedback weights."""

    matrix = interactions.pivot_table(
        index="user_id",
        columns="route_id",
        values="interaction_weight",
        aggfunc="sum",
        fill_value=0,
    )
    return matrix.sort_index().sort_index(axis=1).astype(float)


def build_seen_routes(interactions: pd.DataFrame) -> dict[str, set[str]]:
    """Return route ids already observed for each user."""

    seen: dict[str, set[str]] = {}
    for user_id, rows in interactions.groupby("user_id"):
        seen[str(user_id)] = set(rows["route_id"].astype(str))
    return seen


def build_features(dataset: DemoDataset) -> dict[str, pd.DataFrame]:
    """Build the minimal feature bundle used by the MVP recommenders."""

    return {
        "route_features": build_route_features(dataset.routes),
        "user_item_matrix": build_user_item_matrix(dataset.train_interactions),
    }
