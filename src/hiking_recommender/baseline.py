"""Popularity baseline recommender for the synthetic hiking demo."""

from __future__ import annotations

import pandas as pd

from hiking_recommender.candidates import RecommendationCandidate
from hiking_recommender.data_loader import DemoDataset
from hiking_recommender.features import build_seen_routes


class PopularityRecommender:
    """Recommend popular routes with region filtering and seen-route exclusion."""

    source = "popularity"

    def __init__(self) -> None:
        self._routes: pd.DataFrame | None = None
        self._seen_routes: dict[str, set[str]] = {}

    def fit(self, dataset: DemoDataset) -> "PopularityRecommender":
        weighted_popularity = (
            dataset.train_interactions.groupby("route_id", as_index=False)["interaction_weight"]
            .sum()
            .rename(columns={"interaction_weight": "behavior_score"})
        )
        routes = dataset.routes.merge(weighted_popularity, on="route_id", how="left")
        routes["behavior_score"] = routes["behavior_score"].fillna(0.0).astype(float)
        routes["popularity"] = routes["popularity"].astype(float)
        max_behavior_score = max(float(routes["behavior_score"].max()), 1.0)
        routes["score"] = (routes["behavior_score"] / max_behavior_score) * 0.8 + routes["popularity"] * 0.2

        self._routes = routes.sort_values(
            by=["score", "popularity", "route_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
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
        if self._routes is None:
            raise RuntimeError("PopularityRecommender must be fitted before recommend()")

        candidates = self._routes
        if region:
            candidates = candidates[candidates["region"] == region]
        if exclude_seen:
            seen = self._seen_routes.get(user_id, set())
            if seen:
                candidates = candidates[~candidates["route_id"].isin(seen)]

        recommendations: list[RecommendationCandidate] = []
        for rank, row in enumerate(candidates.head(top_k).itertuples(index=False), start=1):
            recommendations.append(
                RecommendationCandidate(
                    route_id=str(row.route_id),
                    rank=rank,
                    score=float(row.score),
                    source=self.source,
                )
            )
        return recommendations
