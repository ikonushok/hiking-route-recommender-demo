"""Offline top-K evaluation for the synthetic hiking recommender demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def build_relevance(test_interactions: pd.DataFrame) -> dict[str, set[str]]:
    """Build held-out relevant routes per user from synthetic test interactions."""

    relevance: dict[str, set[str]] = {}
    for user_id, rows in test_interactions.groupby("user_id"):
        relevance[str(user_id)] = set(rows["route_id"].astype(str))
    return relevance


def evaluate_recommendations(
    recommendations_by_user: Mapping[str, Sequence[str]],
    test_interactions: pd.DataFrame,
    all_route_ids: Sequence[str],
    cutoff: int = 10,
) -> dict[str, float]:
    """Evaluate recommendation lists against held-out synthetic interactions."""

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    relevance_by_user = build_relevance(test_interactions)
    precision_values: list[float] = []
    recall_values: list[float] = []
    average_precision_values: list[float] = []
    ndcg_values: list[float] = []
    all_route_id_set = set(all_route_ids)
    recommended_route_ids: set[str] = set()

    for user_id, relevant_route_ids in sorted(relevance_by_user.items()):
        recommended = list(recommendations_by_user.get(user_id, []))[:cutoff]
        recommended_route_ids.update(route_id for route_id in recommended if route_id in all_route_id_set)
        hits = [route_id in relevant_route_ids for route_id in recommended]

        precision_values.append(_precision_at_cutoff(hits, cutoff))
        recall_values.append(_recall_at_cutoff(hits, relevant_route_ids))
        average_precision_values.append(_average_precision_at_cutoff(hits, relevant_route_ids))
        ndcg_values.append(_ndcg_at_cutoff(hits, relevant_route_ids, cutoff))

    total_routes = len(all_route_id_set)
    coverage = len(recommended_route_ids) / total_routes if total_routes else 0.0

    return {
        f"precision@{cutoff}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall@{cutoff}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"map@{cutoff}": float(np.mean(average_precision_values)) if average_precision_values else 0.0,
        f"ndcg@{cutoff}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        f"coverage@{cutoff}": float(coverage),
    }


def _precision_at_cutoff(hits: Sequence[bool], cutoff: int) -> float:
    return sum(hits) / cutoff


def _recall_at_cutoff(hits: Sequence[bool], relevant_route_ids: set[str]) -> float:
    if not relevant_route_ids:
        return 0.0
    return sum(hits) / len(relevant_route_ids)


def _average_precision_at_cutoff(hits: Sequence[bool], relevant_route_ids: set[str]) -> float:
    if not relevant_route_ids:
        return 0.0

    precision_sum = 0.0
    hit_count = 0
    for position, is_hit in enumerate(hits, start=1):
        if is_hit:
            hit_count += 1
            precision_sum += hit_count / position
    return precision_sum / min(len(relevant_route_ids), len(hits) or 1)


def _ndcg_at_cutoff(hits: Sequence[bool], relevant_route_ids: set[str], cutoff: int) -> float:
    if not relevant_route_ids:
        return 0.0

    dcg = sum((1.0 / np.log2(position + 1)) for position, is_hit in enumerate(hits, start=1) if is_hit)
    ideal_hits = min(len(relevant_route_ids), cutoff)
    ideal_dcg = sum(1.0 / np.log2(position + 1) for position in range(1, ideal_hits + 1))
    return float(dcg / ideal_dcg) if ideal_dcg else 0.0
