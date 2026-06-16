"""Run offline evaluation for MVP recommenders on the synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hiking_recommender.baseline import PopularityRecommender
from hiking_recommender.business_rules import apply_business_rules
from hiking_recommender.collaborative import CollaborativeRecommender
from hiking_recommender.content_based import ContentBasedRecommender
from hiking_recommender.data_loader import load_demo_dataset
from hiking_recommender.evaluation import build_relevance, evaluate_recommendations
from hiking_recommender.features import build_seen_routes
from hiking_recommender.merger import merge_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"


def main() -> None:
    cutoff = 10
    candidate_limit = 30
    dataset = load_demo_dataset()
    user_ids = sorted(build_relevance(dataset.test_interactions))
    all_route_ids = list(dataset.routes["route_id"].astype(str))
    seen_routes = build_seen_routes(dataset.train_interactions)

    popularity = PopularityRecommender().fit(dataset)
    collaborative = CollaborativeRecommender().fit(dataset)
    content = ContentBasedRecommender().fit(dataset)

    recommendation_sets = {
        "popularity": {
            user_id: [candidate.route_id for candidate in popularity.recommend(user_id, top_k=cutoff)]
            for user_id in user_ids
        },
        "collaborative": {
            user_id: [candidate.route_id for candidate in collaborative.recommend(user_id, top_k=cutoff)]
            for user_id in user_ids
        },
        "content": {
            user_id: [candidate.route_id for candidate in content.recommend(user_id, top_k=cutoff)]
            for user_id in user_ids
        },
    }
    recommendation_sets["hybrid"] = {
        user_id: [
            candidate.route_id
            for candidate in merge_candidates(
                [
                    collaborative.recommend(user_id, top_k=candidate_limit),
                    content.recommend(user_id, top_k=candidate_limit),
                    popularity.recommend(user_id, top_k=candidate_limit),
                ],
                top_k=cutoff,
            )
        ]
        for user_id in user_ids
    }
    recommendation_sets["hybrid_with_rules"] = {
        user_id: [
            candidate.route_id
            for candidate in apply_business_rules(
                merge_candidates(
                    [
                        collaborative.recommend(user_id, top_k=candidate_limit),
                        content.recommend(user_id, top_k=candidate_limit),
                        popularity.recommend(user_id, top_k=candidate_limit),
                    ],
                    top_k=candidate_limit,
                ),
                routes=dataset.routes,
                top_k=cutoff,
                max_difficulty="moderate",
                seen_route_ids=seen_routes.get(user_id, set()),
                fallback_candidates=merge_candidates(
                    [popularity.recommend(user_id, top_k=candidate_limit)],
                    top_k=candidate_limit,
                ),
            )
        ]
        for user_id in user_ids
    }

    metrics_rows = []
    for model_name, recommendations_by_user in recommendation_sets.items():
        metrics = evaluate_recommendations(
            recommendations_by_user,
            dataset.test_interactions,
            all_route_ids,
            cutoff=cutoff,
            train_interactions=dataset.train_interactions,
            routes=dataset.routes,
        )
        metrics_rows.append({"model": model_name, **metrics})

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("model")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "evaluation_metrics.csv"
    report_path = DOCS_DIR / "evaluation_report.md"
    metrics_frame.to_csv(metrics_path, index=False)
    report_path.write_text(_format_report(metrics_frame, cutoff), encoding="utf-8")

    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote report to {report_path}")


def _format_report(metrics_frame: pd.DataFrame, cutoff: int) -> str:
    metric_columns = [column for column in metrics_frame.columns if column != "model"]
    header = "| Model | " + " | ".join(metric_columns) + " |"
    separator = "|---|" + "|".join("---:" for _ in metric_columns) + "|"
    rows = []
    for _, row in metrics_frame.iterrows():
        values = [str(row["model"])]
        values.extend(f"{float(row[column]):.4f}" for column in metric_columns)
        rows.append("| " + " | ".join(values) + " |")

    best_precision = _best_model(metrics_frame, f"precision@{cutoff}")
    best_recall = _best_model(metrics_frame, f"recall@{cutoff}")
    best_coverage = _best_model(metrics_frame, f"coverage@{cutoff}")
    best_novelty = _best_model(metrics_frame, f"novelty@{cutoff}")
    best_diversity = _best_model(metrics_frame, f"diversity@{cutoff}")

    return "\n".join(
        [
            "# Offline Evaluation Report",
            "",
            "Synthetic offline metrics on held-out synthetic interactions for the hiking route recommender demo.",
            "",
            "These numbers are demo validation metrics, not production quality or business impact claims.",
            "",
            "## Evaluation setup",
            "",
            "- Data source: reproducible synthetic CSV files in `data/`.",
            "- Train/test split: time-aware holdout from synthetic interaction timestamps.",
            "- Models compared: popularity, collaborative, content, hybrid, hybrid with business rules.",
            "- Serving parity: the API uses the same retrieval, merge and business-rule modules as this evaluation.",
            "",
            f"Cutoff: `{cutoff}`",
            "",
            "## Metrics",
            "",
            f"- `precision@{cutoff}`: share of top-K recommendations that appear in held-out interactions.",
            f"- `recall@{cutoff}`: share of held-out relevant routes recovered in top-K.",
            f"- `map@{cutoff}` and `ndcg@{cutoff}`: ranking-sensitive quality metrics.",
            f"- `coverage@{cutoff}`: share of catalog routes surfaced across evaluated users.",
            f"- `novelty@{cutoff}`: normalized inverse popularity from train interactions; higher means less popularity-biased.",
            f"- `diversity@{cutoff}`: mean intra-list dissimilarity from synthetic route metadata; higher means broader lists.",
            "",
            "## Results",
            "",
            header,
            separator,
            *rows,
            "",
            "## Reading the results",
            "",
            f"- Best `precision@{cutoff}`: `{best_precision}`.",
            f"- Best `recall@{cutoff}`: `{best_recall}`.",
            f"- Best `coverage@{cutoff}`: `{best_coverage}`.",
            f"- Best `novelty@{cutoff}`: `{best_novelty}`.",
            f"- Best `diversity@{cutoff}`: `{best_diversity}`.",
            "- Business rules can reduce recall or coverage because they apply hard product constraints after retrieval.",
            "- Hybrid results should be read as a candidate-generation demo, not as proof of online lift.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/run_offline_evaluation.py",
            "```",
            "",
        ]
    )


def _best_model(metrics_frame: pd.DataFrame, column: str) -> str:
    if column not in metrics_frame.columns or metrics_frame.empty:
        return "n/a"
    row = metrics_frame.sort_values(by=[column, "model"], ascending=[False, True]).iloc[0]
    return str(row["model"])


if __name__ == "__main__":
    main()
