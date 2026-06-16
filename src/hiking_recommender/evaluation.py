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
    train_interactions: pd.DataFrame | None = None,
    routes: pd.DataFrame | None = None,
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

    metrics = {
        f"precision@{cutoff}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall@{cutoff}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"map@{cutoff}": float(np.mean(average_precision_values)) if average_precision_values else 0.0,
        f"ndcg@{cutoff}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        f"coverage@{cutoff}": float(coverage),
    }
    if train_interactions is not None:
        metrics[f"novelty@{cutoff}"] = novelty_at_cutoff(
            recommendations_by_user,
            train_interactions,
            cutoff,
        )
    if routes is not None:
        metrics[f"diversity@{cutoff}"] = diversity_at_cutoff(
            recommendations_by_user,
            routes,
            cutoff,
        )
    return metrics


def novelty_at_cutoff(
    recommendations_by_user: Mapping[str, Sequence[str]],
    train_interactions: pd.DataFrame,
    cutoff: int = 10,
) -> float:
    """Compute normalized self-information novelty from train interaction popularity."""

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if train_interactions.empty:
        return 0.0

    route_counts = train_interactions["route_id"].astype(str).value_counts().to_dict()
    total_interactions = int(sum(route_counts.values()))
    if total_interactions <= 1:
        return 0.0

    max_novelty = float(-np.log2(1.0 / total_interactions))
    if max_novelty == 0:
        return 0.0

    novelty_values: list[float] = []
    for recommendations in recommendations_by_user.values():
        route_ids = [str(route_id) for route_id in list(recommendations)[:cutoff]]
        if not route_ids:
            continue
        route_novelties = []
        for route_id in route_ids:
            probability = route_counts.get(route_id, 1) / total_interactions
            route_novelties.append(float(-np.log2(probability) / max_novelty))
        novelty_values.append(float(np.mean(route_novelties)))

    return float(np.mean(novelty_values)) if novelty_values else 0.0


def diversity_at_cutoff(
    recommendations_by_user: Mapping[str, Sequence[str]],
    routes: pd.DataFrame,
    cutoff: int = 10,
) -> float:
    """Compute mean intra-list diversity from synthetic route metadata."""

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    metadata = _build_route_metadata(routes)
    diversity_values: list[float] = []
    for recommendations in recommendations_by_user.values():
        route_ids = [
            str(route_id)
            for route_id in list(recommendations)[:cutoff]
            if str(route_id) in metadata
        ]
        if len(route_ids) < 2:
            continue
        pairwise_values = []
        for left_index, left_route_id in enumerate(route_ids):
            for right_route_id in route_ids[left_index + 1 :]:
                pairwise_values.append(
                    _route_dissimilarity(
                        metadata[left_route_id],
                        metadata[right_route_id],
                    )
                )
        diversity_values.append(float(np.mean(pairwise_values)))

    return float(np.mean(diversity_values)) if diversity_values else 0.0


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


def _build_route_metadata(routes: pd.DataFrame) -> dict[str, dict[str, str]]:
    required_columns = {"route_id", "region", "difficulty", "season", "route_tags"}
    missing_columns = required_columns - set(routes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"routes metadata is missing required columns: {missing}")

    return {
        str(row.route_id): {
            "region": str(row.region),
            "difficulty": str(row.difficulty),
            "season": str(row.season),
            "route_tags": str(row.route_tags),
        }
        for row in routes.itertuples(index=False)
    }


def _route_dissimilarity(left: dict[str, str], right: dict[str, str]) -> float:
    categorical_scores = [
        0.0 if left["region"] == right["region"] else 1.0,
        0.0 if left["difficulty"] == right["difficulty"] else 1.0,
        0.0 if left["season"] == right["season"] else 1.0,
    ]
    left_tags = _split_tags(left["route_tags"])
    right_tags = _split_tags(right["route_tags"])
    tag_union = left_tags | right_tags
    tag_dissimilarity = 0.0
    if tag_union:
        tag_dissimilarity = 1.0 - len(left_tags & right_tags) / len(tag_union)
    return float(np.mean([*categorical_scores, tag_dissimilarity]))


def _split_tags(value: str) -> set[str]:
    return {tag for tag in value.split("|") if tag}
