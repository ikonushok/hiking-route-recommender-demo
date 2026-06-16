"""Run a small hybrid retrieval and merge smoke check."""

from __future__ import annotations

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.business_rules import apply_business_rules
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import load_demo_dataset
from hiking_recommender.features import build_seen_routes
from hiking_recommender.merger import merge_candidates


def main() -> None:
    dataset = load_demo_dataset()
    user_id = "user_001"
    region = "north"

    popularity_candidates = PopularityRecommender().fit(dataset).recommend(user_id, top_k=30, region=region)
    collaborative_candidates = CollaborativeRecommender().fit(dataset).recommend(user_id, top_k=30, region=region)
    content_candidates = ContentBasedRecommender().fit(dataset).recommend(user_id, top_k=30, region=region)

    merged = merge_candidates(
        [collaborative_candidates, content_candidates, popularity_candidates],
        top_k=30,
    )
    recommendations = apply_business_rules(
        merged,
        routes=dataset.routes,
        top_k=10,
        region=region,
        max_difficulty="moderate",
        seen_route_ids=build_seen_routes(dataset.train_interactions).get(user_id, set()),
        fallback_candidates=merge_candidates([popularity_candidates], top_k=30),
    )
    if not recommendations:
        raise RuntimeError("Hybrid merger returned no recommendations")

    route_ids = [candidate.route_id for candidate in recommendations]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("Hybrid merger returned duplicate route_id values")

    print("Hybrid retrieval smoke passed")
    for candidate in recommendations:
        sources = ",".join(candidate.sources)
        print(
            f"rank={candidate.final_rank} route_id={candidate.route_id} "
            f"score={candidate.merged_score:.5f} sources={sources}"
        )


if __name__ == "__main__":
    main()
