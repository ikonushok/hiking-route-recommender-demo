import pytest
import pandas as pd

from hiking_recommender.evaluation import build_relevance, diversity_at_cutoff, evaluate_recommendations, novelty_at_cutoff


def test_build_relevance_groups_test_routes_by_user() -> None:
    test_interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001"},
            {"user_id": "user_001", "route_id": "route_002"},
            {"user_id": "user_002", "route_id": "route_003"},
        ]
    )

    assert build_relevance(test_interactions) == {
        "user_001": {"route_001", "route_002"},
        "user_002": {"route_003"},
    }


def test_evaluate_recommendations_computes_topk_metrics() -> None:
    test_interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001"},
            {"user_id": "user_001", "route_id": "route_002"},
            {"user_id": "user_002", "route_id": "route_003"},
        ]
    )
    recommendations = {
        "user_001": ["route_001", "route_999"],
        "user_002": ["route_004", "route_003"],
    }

    metrics = evaluate_recommendations(
        recommendations,
        test_interactions,
        all_route_ids=["route_001", "route_002", "route_003", "route_004"],
        cutoff=2,
    )

    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.75
    assert round(metrics["map@2"], 4) == 0.5
    assert round(metrics["coverage@2"], 4) == 0.75


def test_evaluate_recommendations_adds_novelty_and_diversity_when_inputs_provided() -> None:
    test_interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001"},
        ]
    )
    train_interactions = pd.DataFrame(
        [
            {"user_id": "user_001", "route_id": "route_001"},
            {"user_id": "user_002", "route_id": "route_001"},
            {"user_id": "user_003", "route_id": "route_002"},
        ]
    )
    routes = pd.DataFrame(
        [
            {
                "route_id": "route_001",
                "region": "north",
                "difficulty": "easy",
                "season": "summer",
                "route_tags": "forest|lake",
            },
            {
                "route_id": "route_002",
                "region": "south",
                "difficulty": "hard",
                "season": "winter",
                "route_tags": "mountain|viewpoint",
            },
        ]
    )

    metrics = evaluate_recommendations(
        {"user_001": ["route_001", "route_002"]},
        test_interactions,
        all_route_ids=["route_001", "route_002"],
        cutoff=2,
        train_interactions=train_interactions,
        routes=routes,
    )

    assert 0.0 <= metrics["novelty@2"] <= 1.0
    assert metrics["diversity@2"] == 1.0


def test_novelty_at_cutoff_scores_less_popular_routes_higher() -> None:
    train_interactions = pd.DataFrame(
        [
            {"route_id": "route_001"},
            {"route_id": "route_001"},
            {"route_id": "route_001"},
            {"route_id": "route_002"},
        ]
    )

    metrics = {
        "popular": novelty_at_cutoff({"user_001": ["route_001"]}, train_interactions, cutoff=1),
        "less_popular": novelty_at_cutoff({"user_001": ["route_002"]}, train_interactions, cutoff=1),
    }

    assert metrics["less_popular"] > metrics["popular"]


def test_diversity_at_cutoff_requires_route_metadata_columns() -> None:
    routes = pd.DataFrame([{"route_id": "route_001", "region": "north"}])

    with pytest.raises(ValueError, match="missing required columns"):
        diversity_at_cutoff({"user_001": ["route_001"]}, routes, cutoff=1)
