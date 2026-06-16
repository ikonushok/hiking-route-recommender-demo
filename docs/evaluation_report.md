# Offline Evaluation Report

Synthetic offline metrics on held-out synthetic interactions for the hiking route recommender demo.

These numbers are demo validation metrics, not production quality or business impact claims.

## Evaluation setup

- Data source: reproducible synthetic CSV files in `data/`.
- Train/test split: time-aware holdout from synthetic interaction timestamps.
- Models compared: popularity, collaborative, content, hybrid, hybrid with business rules.
- Serving parity: the API uses the same retrieval, merge and business-rule modules as this evaluation.

Cutoff: `10`

## Metrics

- `precision@10`: share of top-K recommendations that appear in held-out interactions.
- `recall@10`: share of held-out relevant routes recovered in top-K.
- `map@10` and `ndcg@10`: ranking-sensitive quality metrics.
- `coverage@10`: share of catalog routes surfaced across evaluated users.
- `novelty@10`: normalized inverse popularity from train interactions; higher means less popularity-biased.
- `diversity@10`: mean intra-list dissimilarity from synthetic route metadata; higher means broader lists.

## Results

| Model | precision@10 | recall@10 | map@10 | ndcg@10 | coverage@10 | novelty@10 | diversity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| collaborative | 0.0790 | 0.1484 | 0.0572 | 0.1249 | 0.9167 | 0.5545 | 0.6684 |
| content | 0.1035 | 0.1898 | 0.0740 | 0.1584 | 1.0000 | 0.5761 | 0.5120 |
| hybrid | 0.1055 | 0.1932 | 0.0741 | 0.1605 | 0.9500 | 0.5549 | 0.6043 |
| hybrid_with_rules | 0.1000 | 0.1852 | 0.0741 | 0.1567 | 0.8000 | 0.5525 | 0.5818 |
| popularity | 0.0735 | 0.1375 | 0.0498 | 0.1130 | 0.1833 | 0.5379 | 0.6890 |

## Reading the results

- Best `precision@10`: `hybrid`.
- Best `recall@10`: `hybrid`.
- Best `coverage@10`: `content`.
- Best `novelty@10`: `content`.
- Best `diversity@10`: `popularity`.
- Business rules can reduce recall or coverage because they apply hard product constraints after retrieval.
- Hybrid results should be read as a candidate-generation demo, not as proof of online lift.

## Reproduce

```bash
python scripts/run_offline_evaluation.py
```
