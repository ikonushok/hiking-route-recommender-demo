# Стабильный baseline P0

Статус: стабильная базовая версия для портфолио и позиционирования под freelance-заказы.

Дата фиксации: 2026-06-16.

## Область фиксации

P0 включает полный синтетический пайплайн рекомендательной системы:

```text
synthetic data
  -> data loading
  -> feature engineering
  -> popularity baseline
  -> collaborative retrieval
  -> content-based retrieval
  -> rank-based candidate merger
  -> business rules
  -> offline evaluation
  -> FastAPI serving
```

Baseline считается стабильным, когда эти компоненты работают вместе без изменения публичных данных, retrieval-логики, merger-логики, business rules, evaluation и API-контрактов.

## Что входит

- Воспроизводимая генерация синтетических users, routes и implicit interactions.
- Валидация CSV-схем и проверка консистентности синтетических ID.
- Retrieval-источники: popularity, item-item collaborative и content-based.
- Объединение кандидатов с удалением дубликатов по `route_id`.
- Business rules для `region`, `difficulty`, исключения уже просмотренных маршрутов и fallback fill.
- Offline top-K evaluation на отложенных синтетических interactions.
- FastAPI-serving через `GET /health` и `POST /recommendations`.

## Замороженные P0-контракты

- Публичные ID используют форматы `user_000` и `route_000`.
- Типы событий: `view`, `like`, `visit`, `checkin`.
- Route features используют текущую синтетическую схему маршрутов.
- Candidate outputs возвращают публичные `route_id`, а не внутренние индексы матрицы.
- Финальные рекомендации сохраняют hard filters и не содержат дубликатов `route_id`.
- API-контракт остаётся `POST /recommendations` с ранжированным payload рекомендаций.

## Не входит

- Learning-to-rank stage не входит в P0.
- Новые runtime-зависимости не входят в P0.
- Реальные записи, названия маршрутов, регионы, client data и production code не допускаются.
- Синтетические offline-метрики не являются утверждениями о business impact.

## Проверка baseline

Для будущих изменений сначала использовать самую узкую релевантную проверку. Перед представлением P0 как стабильного baseline запускать полный набор:

```bash
python -m pytest
python scripts/run_baseline_smoke.py
python scripts/run_hybrid_smoke.py
python scripts/run_offline_evaluation.py
```

Текущие синтетические offline-метрики хранятся в `docs/evaluation_report.md` и `outputs/evaluation_metrics.csv`. P1-метрики `novelty@K` и `diversity@K` добавлены как диагностические сигналы popularity bias и разнообразия выдачи; они не являются production-impact claims.

## Точка расширения

Будущую ranking-логику нужно добавлять как отдельное post-P0 расширение, не меняя P0-контракты неявно. Первый прагматичный шаг перед learning-to-rank — прозрачный deterministic ranker.
