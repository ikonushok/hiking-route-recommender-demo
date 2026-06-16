# Отчёт по offline evaluation

Синтетические offline-метрики на отложенных synthetic interactions.

Эти числа являются demo validation metrics, а не утверждениями о production quality или business impact.

Cutoff: `10`

| Модель | precision@10 | recall@10 | map@10 | ndcg@10 | coverage@10 |
|---|---:|---:|---:|---:|---:|
| collaborative | 0.0790 | 0.1484 | 0.0572 | 0.1249 | 0.9167 |
| content | 0.1035 | 0.1898 | 0.0740 | 0.1584 | 1.0000 |
| hybrid | 0.1055 | 0.1932 | 0.0741 | 0.1605 | 0.9500 |
| hybrid_with_rules | 0.1000 | 0.1852 | 0.0741 | 0.1567 | 0.8000 |
| popularity | 0.0735 | 0.1375 | 0.0498 | 0.1130 | 0.1833 |
