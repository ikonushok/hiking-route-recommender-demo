"""Public API schemas for the synthetic recommender demo."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request contract for route recommendations."""

    user_id: str = Field(..., min_length=1, examples=["user_001"])
    region: str | None = Field(default=None, examples=["north"])
    top_k: int = Field(default=10, ge=1, le=50)
    max_difficulty: str | None = Field(default=None, examples=["moderate"])


class RecommendationItem(BaseModel):
    """Single recommendation item returned by the API."""

    route_id: str
    rank: int
    score: float
    difficulty: str
    sources: list[str]


class RecommendationResponse(BaseModel):
    """Response contract for route recommendations."""

    user_id: str
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Health response with loaded synthetic artifact counts."""

    status: str
    users: int
    routes: int
    train_interactions: int
    test_interactions: int
