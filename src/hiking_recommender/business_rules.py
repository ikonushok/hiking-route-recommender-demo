"""Post-retrieval business rules for demo recommendations."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from hiking_recommender.candidates import MergedCandidate

DIFFICULTY_ORDER = {
    "easy": 1,
    "moderate": 2,
    "hard": 3,
}


def apply_business_rules(
    candidates: list[MergedCandidate],
    routes: pd.DataFrame,
    top_k: int = 10,
    region: str | None = None,
    max_difficulty: str | None = None,
    seen_route_ids: set[str] | None = None,
    fallback_candidates: list[MergedCandidate] | None = None,
) -> list[MergedCandidate]:
    """Apply hard filters, deduplicate, and optionally fill from safe fallback candidates."""

    if top_k <= 0:
        return []

    route_metadata = _build_route_metadata(routes)
    seen = seen_route_ids or set()
    selected: list[MergedCandidate] = []
    selected_ids: set[str] = set()

    for candidate in candidates:
        if _passes_filters(candidate.route_id, route_metadata, region, max_difficulty, seen):
            _append_unique(selected, selected_ids, candidate, top_k)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k and fallback_candidates:
        for candidate in fallback_candidates:
            if candidate.route_id in selected_ids:
                continue
            if not _passes_filters(candidate.route_id, route_metadata, region, max_difficulty, seen):
                continue
            _append_unique(selected, selected_ids, candidate, top_k)
            if len(selected) >= top_k:
                break

    return _rerank(selected)


def _build_route_metadata(routes: pd.DataFrame) -> dict[str, dict[str, str]]:
    required_columns = {"route_id", "region", "difficulty"}
    missing_columns = required_columns - set(routes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"routes metadata is missing required columns: {missing}")

    return {
        str(row.route_id): {
            "region": str(row.region),
            "difficulty": str(row.difficulty),
        }
        for row in routes.itertuples(index=False)
    }


def _passes_filters(
    route_id: str,
    route_metadata: dict[str, dict[str, str]],
    region: str | None,
    max_difficulty: str | None,
    seen_route_ids: set[str],
) -> bool:
    metadata = route_metadata.get(route_id)
    if metadata is None:
        return False
    if route_id in seen_route_ids:
        return False
    if region and metadata["region"] != region:
        return False
    if max_difficulty:
        if max_difficulty not in DIFFICULTY_ORDER:
            raise ValueError(f"Unsupported max_difficulty: {max_difficulty}")
        route_difficulty = metadata["difficulty"]
        if route_difficulty not in DIFFICULTY_ORDER:
            raise ValueError(f"Unsupported route difficulty: {route_difficulty}")
        if DIFFICULTY_ORDER[route_difficulty] > DIFFICULTY_ORDER[max_difficulty]:
            return False
    return True


def _append_unique(
    selected: list[MergedCandidate],
    selected_ids: set[str],
    candidate: MergedCandidate,
    top_k: int,
) -> None:
    if len(selected) >= top_k or candidate.route_id in selected_ids:
        return
    selected.append(candidate)
    selected_ids.add(candidate.route_id)


def _rerank(candidates: list[MergedCandidate]) -> list[MergedCandidate]:
    return [
        replace(candidate, final_rank=rank)
        for rank, candidate in enumerate(candidates, start=1)
    ]
