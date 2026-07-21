"""Smoke test: run the e-commerce data through the engine with zero code changes.

Demonstrates that the recommender engine is fully domain-agnostic:
  - Same code that recommends hiking routes now recommends products
  - Same models (popularity, collaborative, content-based, hybrid)
  - Same evaluation metrics (precision, recall, NDCG, coverage, novelty, diversity)
  - Zero changes to engine source code
"""

from __future__ import annotations

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import load_demo_dataset
from hiking_recommender.evaluation import evaluate_recommendations
from hiking_recommender.merger import merge_candidates

TIER_LABELS = {"easy": "budget", "moderate": "moderate", "hard": "premium"}


def main() -> None:
    # Load e-commerce data instead of hiking data
    ds = load_demo_dataset("ecommerce_data")
    print(f"=== E-COMMERCE RECOMMENDER DEMO ===")
    print(f"Dataset: {len(ds.users)} customers | {len(ds.routes)} products | {len(ds.train_interactions)} train interactions")
    print()

    # Fit all models (same code as hiking, zero changes)
    pop = PopularityRecommender().fit(ds)
    collab = CollaborativeRecommender().fit(ds)
    content = ContentBasedRecommender().fit(ds)
    print("Models fitted: popularity, collaborative, content-based")
    print()

    # Show recommendations for sample users
    for user in ["user_042", "user_123", "user_250"]:
        p = pop.recommend(user, top_k=15)
        c = collab.recommend(user, top_k=15)
        ct = content.recommend(user, top_k=15)
        merged = merge_candidates([c, ct, p], top_k=5)

        print(f"Top-5 recommendations for {user}:")
        for m in merged:
            route_info = ds.routes[ds.routes["route_id"] == m.route_id].iloc[0]
            price = int(route_info["length_km"] * 1000)
            tier = TIER_LABELS.get(route_info["difficulty"], route_info["difficulty"])
            print(
                f"  #{m.final_rank} {m.route_id} | {route_info['region']:12s} | "
                f"{price:>6d} RUB | rating={route_info['duration_hours']} | "
                f"tier={tier:8s} | score={m.merged_score:.4f} | "
                f"sources={list(m.sources)}"
            )
        print()

    # Offline evaluation (same metrics as hiking demo)
    print("=== OFFLINE EVALUATION (e-commerce data) ===")
    all_route_ids = list(ds.routes["route_id"].astype(str))

    models = {
        "popularity": pop,
        "collaborative": collab,
        "content": content,
    }

    for name, model in models.items():
        recs_by_user: dict[str, list[str]] = {}
        for user_id in ds.users["user_id"].astype(str):
            candidates = model.recommend(user_id, top_k=10)
            recs_by_user[user_id] = [c.route_id for c in candidates]

        metrics = evaluate_recommendations(
            recs_by_user,
            ds.test_interactions,
            all_route_ids,
            cutoff=10,
            train_interactions=ds.train_interactions,
            routes=ds.routes,
        )
        print(f"\n{name}:")
        for metric, value in sorted(metrics.items()):
            print(f"  {metric:20s} {value:.4f}")

    # Hybrid evaluation
    recs_by_user: dict[str, list[str]] = {}
    for user_id in ds.users["user_id"].astype(str):
        p = pop.recommend(user_id, top_k=15)
        c = collab.recommend(user_id, top_k=15)
        ct = content.recommend(user_id, top_k=15)
        merged = merge_candidates([c, ct, p], top_k=10)
        recs_by_user[user_id] = [m.route_id for m in merged]

    hybrid_metrics = evaluate_recommendations(
        recs_by_user,
        ds.test_interactions,
        all_route_ids,
        cutoff=10,
        train_interactions=ds.train_interactions,
        routes=ds.routes,
    )
    print("\nhybrid:")
    for metric, value in sorted(hybrid_metrics.items()):
        print(f"  {metric:20s} {value:.4f}")

    print("\n" + "=" * 60)
    print("CONCLUSION: Engine works with e-commerce data.")
    print("Zero changes to engine source code.")
    print("Same models, same metrics, different domain.")
    print("=" * 60)


if __name__ == "__main__":
    main()
