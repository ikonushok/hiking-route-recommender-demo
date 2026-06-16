"""Shared candidate schemas for MVP retrieval and merge stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationCandidate:
    """Candidate emitted by a single retrieval source."""

    route_id: str
    rank: int
    score: float
    source: str


@dataclass(frozen=True)
class MergedCandidate:
    """Candidate after rank-based hybrid merging."""

    route_id: str
    final_rank: int
    merged_score: float
    sources: tuple[str, ...]
    n_sources: int
    best_source_rank: int
