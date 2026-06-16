"""Rank-based candidate merger for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict

from hiking_recommender.candidates import MergedCandidate, RecommendationCandidate

DEFAULT_SOURCE_WEIGHTS = {
    "collaborative": 1.0,
    "content": 0.8,
    "popularity": 0.5,
}


def merge_candidates(
    candidate_lists: list[list[RecommendationCandidate]],
    top_k: int = 10,
    source_weights: dict[str, float] | None = None,
    rank_constant: float = 60.0,
) -> list[MergedCandidate]:
    """Merge candidate lists using weighted reciprocal-rank contributions."""

    if top_k <= 0:
        return []
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    weights = source_weights or DEFAULT_SOURCE_WEIGHTS
    contributions: dict[str, float] = defaultdict(float)
    sources_by_route: dict[str, set[str]] = defaultdict(set)
    best_rank_by_route: dict[str, int] = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate.rank <= 0:
                raise ValueError("candidate rank must be positive")
            source_weight = weights.get(candidate.source, 1.0)
            contributions[candidate.route_id] += source_weight / (candidate.rank + rank_constant)
            sources_by_route[candidate.route_id].add(candidate.source)
            best_rank_by_route[candidate.route_id] = min(
                candidate.rank,
                best_rank_by_route.get(candidate.route_id, candidate.rank),
            )

    ranked = sorted(
        contributions,
        key=lambda route_id: (
            -contributions[route_id],
            -len(sources_by_route[route_id]),
            best_rank_by_route[route_id],
            route_id,
        ),
    )

    return [
        MergedCandidate(
            route_id=route_id,
            final_rank=final_rank,
            merged_score=float(contributions[route_id]),
            sources=tuple(sorted(sources_by_route[route_id])),
            n_sources=len(sources_by_route[route_id]),
            best_source_rank=best_rank_by_route[route_id],
        )
        for final_rank, route_id in enumerate(ranked[:top_k], start=1)
    ]
