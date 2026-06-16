import pandas as pd

from hiking_recommender.evaluation import build_relevance, evaluate_recommendations


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
