from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.candidates import RecommendationCandidate
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import load_demo_dataset
from hiking_recommender.merger import merge_candidates


def test_retrievers_emit_non_duplicate_candidates() -> None:
    dataset = load_demo_dataset()
    user_id = "user_001"
    region = "north"

    candidate_lists = [
        PopularityRecommender().fit(dataset).recommend(user_id, top_k=20, region=region),
        CollaborativeRecommender().fit(dataset).recommend(user_id, top_k=20, region=region),
        ContentBasedRecommender().fit(dataset).recommend(user_id, top_k=20, region=region),
    ]

    assert any(candidate_lists)
    for candidates in candidate_lists:
        route_ids = [candidate.route_id for candidate in candidates]
        assert len(route_ids) == len(set(route_ids))
        assert [candidate.rank for candidate in candidates] == list(range(1, len(candidates) + 1))


def test_merge_candidates_boosts_overlap_and_deduplicates() -> None:
    collaborative = [
        RecommendationCandidate("route_001", 1, 0.9, "collaborative"),
        RecommendationCandidate("route_002", 2, 0.8, "collaborative"),
    ]
    content = [
        RecommendationCandidate("route_002", 1, 0.7, "content"),
        RecommendationCandidate("route_003", 2, 0.6, "content"),
    ]

    merged = merge_candidates([collaborative, content], top_k=3, rank_constant=10)

    assert [candidate.route_id for candidate in merged] == ["route_002", "route_001", "route_003"]
    assert merged[0].sources == ("collaborative", "content")
    assert merged[0].n_sources == 2
    assert len({candidate.route_id for candidate in merged}) == len(merged)
    assert [candidate.final_rank for candidate in merged] == [1, 2, 3]


def test_merge_candidates_uses_stable_tie_break() -> None:
    merged = merge_candidates(
        [
            [
                RecommendationCandidate("route_002", 1, 1.0, "unknown"),
                RecommendationCandidate("route_001", 1, 1.0, "unknown"),
            ]
        ],
        top_k=2,
        rank_constant=10,
    )

    assert [candidate.route_id for candidate in merged] == ["route_001", "route_002"]
