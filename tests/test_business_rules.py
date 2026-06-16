import pandas as pd

from hiking_recommender.business_rules import apply_business_rules
from hiking_recommender.candidates import MergedCandidate


def candidate(route_id: str, rank: int, score: float = 1.0) -> MergedCandidate:
    return MergedCandidate(
        route_id=route_id,
        final_rank=rank,
        merged_score=score,
        sources=("content",),
        n_sources=1,
        best_source_rank=rank,
    )


def routes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"route_id": "route_001", "region": "north", "difficulty": "easy"},
            {"route_id": "route_002", "region": "north", "difficulty": "moderate"},
            {"route_id": "route_003", "region": "north", "difficulty": "hard"},
            {"route_id": "route_004", "region": "south", "difficulty": "easy"},
            {"route_id": "route_005", "region": "north", "difficulty": "easy"},
        ]
    )


def test_business_rules_apply_region_difficulty_seen_and_rerank() -> None:
    result = apply_business_rules(
        [
            candidate("route_004", 1),
            candidate("route_003", 2),
            candidate("route_001", 3),
            candidate("route_002", 4),
        ],
        routes=routes(),
        top_k=10,
        region="north",
        max_difficulty="moderate",
        seen_route_ids={"route_001"},
    )

    assert [item.route_id for item in result] == ["route_002"]
    assert [item.final_rank for item in result] == [1]


def test_business_rules_fallback_preserves_hard_filters_and_deduplicates() -> None:
    result = apply_business_rules(
        [candidate("route_001", 1)],
        routes=routes(),
        top_k=3,
        region="north",
        max_difficulty="moderate",
        fallback_candidates=[
            candidate("route_001", 1),
            candidate("route_003", 2),
            candidate("route_004", 3),
            candidate("route_005", 4),
            candidate("route_002", 5),
        ],
    )

    assert [item.route_id for item in result] == ["route_001", "route_005", "route_002"]
    assert [item.final_rank for item in result] == [1, 2, 3]
    assert len({item.route_id for item in result}) == len(result)


def test_business_rules_reject_unknown_max_difficulty() -> None:
    try:
        apply_business_rules(
            [candidate("route_001", 1)],
            routes=routes(),
            max_difficulty="extreme",
        )
    except ValueError as error:
        assert "Unsupported max_difficulty" in str(error)
    else:
        raise AssertionError("Expected unsupported max_difficulty to fail")
