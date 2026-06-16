"""Content-based retriever for the synthetic hiking demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hiking_recommender.candidates import RecommendationCandidate
from hiking_recommender.data_loader import DemoDataset
from hiking_recommender.features import build_route_features, build_seen_routes


class ContentBasedRecommender:
    """Recommend routes similar to a user's historical route profile."""

    source = "content"

    def __init__(self) -> None:
        self._features: pd.DataFrame | None = None
        self._route_ids: list[str] = []
        self._route_regions: dict[str, str] = {}
        self._seen_routes: dict[str, set[str]] = {}
        self._user_profiles: dict[str, np.ndarray] = {}
        self._route_matrix: np.ndarray | None = None

    def fit(self, dataset: DemoDataset) -> "ContentBasedRecommender":
        route_features = build_route_features(dataset.routes).set_index("route_id").sort_index()
        route_matrix = route_features.to_numpy(dtype=float)
        route_matrix = _standardize(route_matrix)
        route_matrix = _normalize_rows(route_matrix)

        route_index = {route_id: index for index, route_id in enumerate(route_features.index.astype(str))}
        user_profiles: dict[str, np.ndarray] = {}
        for user_id, rows in dataset.train_interactions.groupby("user_id"):
            vectors = []
            weights = []
            for row in rows.itertuples(index=False):
                route_id = str(row.route_id)
                if route_id not in route_index:
                    continue
                vectors.append(route_matrix[route_index[route_id]])
                weights.append(float(row.interaction_weight))
            if vectors:
                profile = np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights, dtype=float))
                user_profiles[str(user_id)] = _normalize_vector(profile)

        self._features = route_features
        self._route_ids = [str(route_id) for route_id in route_features.index]
        self._route_regions = dict(zip(dataset.routes["route_id"].astype(str), dataset.routes["region"].astype(str)))
        self._seen_routes = build_seen_routes(dataset.train_interactions)
        self._user_profiles = user_profiles
        self._route_matrix = route_matrix
        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        region: str | None = None,
        exclude_seen: bool = True,
    ) -> list[RecommendationCandidate]:
        if top_k <= 0:
            return []
        if self._route_matrix is None:
            raise RuntimeError("ContentBasedRecommender must be fitted before recommend()")
        if user_id not in self._user_profiles:
            return []

        profile = self._user_profiles[user_id]
        scores = self._route_matrix @ profile
        seen = self._seen_routes.get(user_id, set()) if exclude_seen else set()

        ranked: list[tuple[str, float]] = []
        for route_id, score in zip(self._route_ids, scores):
            if route_id in seen:
                continue
            if region and self._route_regions.get(route_id) != region:
                continue
            ranked.append((route_id, float(score)))

        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [
            RecommendationCandidate(route_id=route_id, rank=rank, score=score, source=self.source)
            for rank, (route_id, score) in enumerate(ranked[:top_k], start=1)
        ]


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / safe_std


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms > 0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
