"""Run a small popularity baseline smoke check."""

from __future__ import annotations

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.data_loader import load_demo_dataset


def main() -> None:
    dataset = load_demo_dataset()
    recommender = PopularityRecommender().fit(dataset)
    recommendations = recommender.recommend(user_id="user_001", region="north", top_k=10)

    if not recommendations:
        raise RuntimeError("Popularity baseline returned no recommendations")

    route_ids = [candidate.route_id for candidate in recommendations]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("Popularity baseline returned duplicate route_id values")

    print("Popularity baseline smoke passed")
    for candidate in recommendations:
        print(
            f"rank={candidate.rank} route_id={candidate.route_id} "
            f"score={candidate.score:.4f} source={candidate.source}"
        )


if __name__ == "__main__":
    main()
