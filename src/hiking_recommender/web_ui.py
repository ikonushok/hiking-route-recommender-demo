"""Interactive web UI for the recommender demo (HTMX + Jinja2 + Tailwind).

Provides a self-service demo experience:
  1. Upload two CSVs (items + interactions)
  2. See dataset health metrics
  3. Get personalized recommendations for any user
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import (
    EVENT_WEIGHTS,
    INTERACTION_COLUMNS,
    ROUTE_COLUMNS,
    USER_COLUMNS,
    DemoDataset,
)
from hiking_recommender.features import build_seen_routes
from hiking_recommender.merger import merge_candidates

TIER_LABELS = {"easy": "budget", "moderate": "moderate", "hard": "premium"}

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _validate_upload(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> DemoDataset:
    """Validate uploaded CSVs against engine schema and build a DemoDataset."""

    items_missing = ROUTE_COLUMNS - set(items_df.columns)
    if items_missing:
        raise ValueError(f"Items CSV missing columns: {', '.join(sorted(items_missing))}")

    interactions_missing = INTERACTION_COLUMNS - set(interactions_df.columns)
    if interactions_missing:
        raise ValueError(f"Interactions CSV missing columns: {', '.join(sorted(interactions_missing))}")

    items_df = items_df[list(ROUTE_COLUMNS)].copy()
    interactions_df = interactions_df[list(INTERACTION_COLUMNS)].copy()

    if items_df["route_id"].duplicated().any():
        raise ValueError("Items CSV contains duplicate route_id values")

    items_df["length_km"] = pd.to_numeric(items_df["length_km"], errors="raise")
    items_df["duration_hours"] = pd.to_numeric(items_df["duration_hours"], errors="raise")
    items_df["elevation_gain_m"] = pd.to_numeric(items_df["elevation_gain_m"], errors="raise")
    items_df["popularity"] = pd.to_numeric(items_df["popularity"], errors="raise")

    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"], utc=True, errors="raise")
    interactions_df["interaction_weight"] = interactions_df["interaction_weight"].astype(int)

    unexpected_events = set(interactions_df["interaction_type"]) - set(EVENT_WEIGHTS)
    if unexpected_events:
        raise ValueError(f"Unsupported interaction types: {', '.join(sorted(unexpected_events))}")

    # Simple users table from interactions (engine expects a users DF, but only user_id is used by recommenders)
    users_df = interactions_df[["user_id"]].drop_duplicates().reset_index(drop=True)
    users_df = pd.concat([users_df] * 1, ignore_index=True)

    # Build minimal users table with required columns (only user_id is actually used downstream)
    for col in USER_COLUMNS:
        if col not in users_df.columns:
            users_df[col] = ""

    # Temporal split: last 20% per user goes to test
    interactions_df = interactions_df.sort_values(["user_id", "timestamp"])
    test_idx = interactions_df.groupby("user_id").cumcount(ascending=False) < (
        interactions_df.groupby("user_id")["route_id"].transform("count") * 0.2
    )
    train_df = interactions_df[~test_idx].copy()
    test_df = interactions_df[test_idx].copy()

    return DemoDataset(
        users=users_df,
        routes=items_df,
        interactions=interactions_df,
        train_interactions=train_df,
        test_interactions=test_df,
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main demo page with upload form and recommendation search."""

    from hiking_recommender.api import runtime

    health = {
        "users": len(runtime.dataset.users),
        "routes": len(runtime.dataset.routes),
        "train_interactions": len(runtime.dataset.train_interactions),
        "test_interactions": len(runtime.dataset.test_interactions),
        "has_custom_data": False,
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "health": health,
            "recommendations": None,
            "error": None,
        },
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_data(
    request: Request,
    items_csv: UploadFile = File(...),
    interactions_csv: UploadFile = File(...),
):
    """Upload custom CSVs, re-fit models, redirect to index."""

    try:
        items_content = await items_csv.read()
        interactions_content = await interactions_csv.read()

        items_df = pd.read_csv(io.BytesIO(items_content))
        interactions_df = pd.read_csv(io.BytesIO(interactions_content))

        dataset = _validate_upload(items_df, interactions_df)

        # Re-fit all models with the new data
        import hiking_recommender.api as api_module

        api_module.runtime = build_runtime_from_dataset(dataset)

        health = {
            "users": len(dataset.users),
            "routes": len(dataset.routes),
            "train_interactions": len(dataset.train_interactions),
            "test_interactions": len(dataset.test_interactions),
            "has_custom_data": True,
        }

        return templates.TemplateResponse(
            request=request,
            name="partials/_upload_success.html",
            context={"health": health},
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/_error.html",
            context={"error": str(e)},
        )


def build_runtime_from_dataset(dataset: DemoDataset):
    """Build recommender runtime from a given dataset (for custom uploads)."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Runtime:
        dataset: DemoDataset
        popularity: PopularityRecommender
        collaborative: CollaborativeRecommender
        content: ContentBasedRecommender
        seen_routes: dict[str, set[str]]
        route_difficulty: dict[str, str]

    return Runtime(
        dataset=dataset,
        popularity=PopularityRecommender().fit(dataset),
        collaborative=CollaborativeRecommender().fit(dataset),
        content=ContentBasedRecommender().fit(dataset),
        seen_routes=build_seen_routes(dataset.train_interactions),
        route_difficulty=dict(
            zip(dataset.routes["route_id"].astype(str), dataset.routes["difficulty"].astype(str))
        ),
    )


@router.post("/search", response_class=HTMLResponse)
async def search_recommendations(request: Request):
    """Get recommendations for a user_id and render as product cards."""

    from hiking_recommender.api import runtime

    form = await request.form()
    user_id = str(form.get("user_id", "")).strip()
    top_k = int(form.get("top_k", 10))

    if not user_id:
        return templates.TemplateResponse(
            request=request,
            name="partials/_error.html",
            context={"error": "Please enter a user_id"},
        )

    try:
        candidate_limit = max(top_k * 3, top_k)
        pop_cands = runtime.popularity.recommend(user_id, top_k=candidate_limit)
        collab_cands = runtime.collaborative.recommend(user_id, top_k=candidate_limit)
        content_cands = runtime.content.recommend(user_id, top_k=candidate_limit)

        merged = merge_candidates([collab_cands, content_cands, pop_cands], top_k=top_k)

        # Enrich with item metadata
        recommendations = []
        for m in merged:
            item_info = runtime.dataset.routes[runtime.dataset.routes["route_id"] == m.route_id]
            if item_info.empty:
                continue
            row = item_info.iloc[0]
            price = int(row["length_km"] * 1000)
            tier = TIER_LABELS.get(str(row["difficulty"]), str(row["difficulty"]))
            recommendations.append(
                {
                    "route_id": m.route_id,
                    "rank": m.final_rank,
                    "score": m.merged_score,
                    "category": str(row["region"]),
                    "price": price,
                    "rating": float(row["duration_hours"]),
                    "tier": tier,
                    "tags": str(row["route_tags"]),
                    "sources": list(m.sources),
                }
            )

        return templates.TemplateResponse(
            request=request,
            name="partials/_recommendations.html",
            context={
                "user_id": user_id,
                "recommendations": recommendations,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/_error.html",
            context={"error": str(e)},
        )
