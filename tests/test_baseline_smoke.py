from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.data_loader import load_demo_dataset
from hiking_recommender.features import build_features


def test_load_features_and_popularity_baseline_smoke() -> None:
    dataset = load_demo_dataset()
    features = build_features(dataset)

    assert not features["route_features"].empty
    assert not features["user_item_matrix"].empty

    recommendations = PopularityRecommender().fit(dataset).recommend(
        user_id="user_001",
        region="north",
        top_k=10,
    )

    assert recommendations
    assert len(recommendations) <= 10
    assert [candidate.rank for candidate in recommendations] == list(range(1, len(recommendations) + 1))
    route_ids = [candidate.route_id for candidate in recommendations]
    assert len(route_ids) == len(set(route_ids))
