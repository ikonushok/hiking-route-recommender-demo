"""FastAPI app for serving synthetic hiking route recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, HTTPException

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.business_rules import apply_business_rules
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import DemoDataset, load_demo_dataset
from hiking_recommender.features import build_seen_routes
from hiking_recommender.merger import merge_candidates
from hiking_recommender.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)

DEFAULT_CANDIDATE_MULTIPLIER = 3


@dataclass(frozen=True)
class RecommenderRuntime:
    """In-memory runtime state for the small synthetic demo API."""

    dataset: DemoDataset
    popularity: PopularityRecommender
    collaborative: CollaborativeRecommender
    content: ContentBasedRecommender
    seen_routes: dict[str, set[str]]
    route_difficulty: dict[str, str]


def build_runtime() -> RecommenderRuntime:
    """Load synthetic data and fit MVP recommenders."""

    dataset = load_demo_dataset()
    return RecommenderRuntime(
        dataset=dataset,
        popularity=PopularityRecommender().fit(dataset),
        collaborative=CollaborativeRecommender().fit(dataset),
        content=ContentBasedRecommender().fit(dataset),
        seen_routes=build_seen_routes(dataset.train_interactions),
        route_difficulty=dict(
            zip(dataset.routes["route_id"].astype(str), dataset.routes["difficulty"].astype(str))
        ),
    )


runtime = build_runtime()
app = FastAPI(
    title="Hiking Route Recommender Demo",
    version="0.1.0",
    description="Synthetic commercial-style recommender-system demo API.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic runtime health and loaded synthetic artifact counts."""

    return HealthResponse(
        status="ok",
        users=len(runtime.dataset.users),
        routes=len(runtime.dataset.routes),
        train_interactions=len(runtime.dataset.train_interactions),
        test_interactions=len(runtime.dataset.test_interactions),
    )


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """Return hybrid recommendations with post-retrieval business rules."""

    region = _normalize_optional_text(request.region)
    max_difficulty = _normalize_optional_text(request.max_difficulty)
    candidate_limit = max(request.top_k * DEFAULT_CANDIDATE_MULTIPLIER, request.top_k)

    try:
        popularity_candidates = runtime.popularity.recommend(
            request.user_id,
            top_k=candidate_limit,
            region=region,
        )
        collaborative_candidates = runtime.collaborative.recommend(
            request.user_id,
            top_k=candidate_limit,
            region=region,
        )
        content_candidates = runtime.content.recommend(
            request.user_id,
            top_k=candidate_limit,
            region=region,
        )
        merged = merge_candidates(
            [collaborative_candidates, content_candidates, popularity_candidates],
            top_k=candidate_limit,
        )
        fallback = merge_candidates([popularity_candidates], top_k=candidate_limit)
        final_candidates = apply_business_rules(
            merged,
            routes=runtime.dataset.routes,
            top_k=request.top_k,
            region=region,
            max_difficulty=max_difficulty,
            seen_route_ids=runtime.seen_routes.get(request.user_id, set()),
            fallback_candidates=fallback,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=[
            RecommendationItem(
                route_id=candidate.route_id,
                rank=candidate.final_rank,
                score=candidate.merged_score,
                difficulty=runtime.route_difficulty[candidate.route_id],
                sources=list(candidate.sources),
            )
            for candidate in final_candidates
        ],
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
