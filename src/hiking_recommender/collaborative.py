"""Item-item collaborative retriever for the synthetic hiking demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hiking_recommender.candidates import RecommendationCandidate
from hiking_recommender.data_loader import DemoDataset
from hiking_recommender.features import build_seen_routes, build_user_item_matrix


class CollaborativeRecommender:
    """Recommend routes using item-item cosine similarity over implicit feedback."""

    source = "collaborative"

    def __init__(self) -> None:
        self._matrix: pd.DataFrame | None = None
        self._similarity: np.ndarray | None = None
        self._route_ids: list[str] = []
        self._route_regions: dict[str, str] = {}
        self._seen_routes: dict[str, set[str]] = {}

    def fit(self, dataset: DemoDataset) -> "CollaborativeRecommender":
        matrix = build_user_item_matrix(dataset.train_interactions)
        values = matrix.to_numpy(dtype=float)
        item_vectors = values.T
        norms = np.linalg.norm(item_vectors, axis=1)
        denominator = np.outer(norms, norms)

        similarity = np.divide(
            item_vectors @ item_vectors.T,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 0,
        )
        np.fill_diagonal(similarity, 0.0)

        self._matrix = matrix
        self._similarity = similarity
        self._route_ids = [str(route_id) for route_id in matrix.columns]
        self._route_regions = dict(zip(dataset.routes["route_id"].astype(str), dataset.routes["region"].astype(str)))
        self._seen_routes = build_seen_routes(dataset.train_interactions)
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
        if self._matrix is None or self._similarity is None:
            raise RuntimeError("CollaborativeRecommender must be fitted before recommend()")
        if user_id not in self._matrix.index:
            return []

        user_vector = self._matrix.loc[user_id].to_numpy(dtype=float)
        scores = self._similarity @ user_vector
        seen = self._seen_routes.get(user_id, set()) if exclude_seen else set()

        ranked: list[tuple[str, float]] = []
        for route_id, score in zip(self._route_ids, scores):
            if score <= 0:
                continue
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
