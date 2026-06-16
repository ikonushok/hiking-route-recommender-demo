# Hiking Route Recommender Demo

Синтетическое commercial-style demo рекомендательной системы для каталога hiking / tourism routes.

Проект показывает практический pipeline каталожных рекомендаций:

```text
synthetic data -> data loading -> feature engineering -> baseline -> retrieval -> merge -> business rules -> evaluation -> API
```

## Граница безопасности

Репозиторий использует полностью синтетические данные.

В нём нет:

- client data;
- production code;
- proprietary database schema;
- internal business metrics;
- customer-specific business logic.

## Что показывает demo

MVP намеренно небольшой, но покрывает полный практический цикл рекомендательной системы:

- воспроизводимая генерация synthetic data;
- валидированная загрузка CSV;
- feature engineering для маршрутов;
- user-route implicit-feedback matrix;
- popularity baseline с region filter, seen-route exclusion и cold-start fallback behavior;
- item-item collaborative retriever;
- content-based retriever;
- rank-based candidate merger;
- post-retrieval business rules для region, difficulty, seen-route exclusion и fallback fill;
- offline top-K evaluation на отложенных synthetic interactions;
- FastAPI endpoint для online-style recommendation serving.

## Статус baseline

P0 зафиксирован как стабильный baseline этого demo. Он включает synthetic data generation, feature engineering, popularity/collaborative/content retrieval, candidate merging, business rules, offline evaluation и FastAPI serving.

Будущую ranking-логику нужно рассматривать как post-P0 расширение, а не как скрытую замену стабильного baseline.

Замороженный scope P0, контракты, команды проверки и границы расширения описаны в `docs/p0_baseline.md`.

## Быстрый старт

Установить локальный package:

```bash
python -m pip install -e ".[dev]"
```

Создать или обновить synthetic dataset:

```bash
python scripts/generate_synthetic_data.py
```

Запустить baseline smoke check:

```bash
python scripts/run_baseline_smoke.py
```

Запустить hybrid retrieval smoke check:

```bash
python scripts/run_hybrid_smoke.py
```

Запустить offline evaluation:

```bash
python scripts/run_offline_evaluation.py
```

Запустить tests:

```bash
python -m pytest
```

## Evaluation artifacts

Offline evaluation записывает synthetic metrics в:

- `outputs/evaluation_metrics.csv`;
- `docs/evaluation_report.md`.

Метрики полезны для проверки demo pipeline. Они не являются утверждениями о production quality или business impact.

## Notebook demo

Открыть end-to-end notebook:

```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

Notebook показывает data loading, feature engineering, retrieval sources, candidate merging, business rules, offline metrics и пример API payload.
Также notebook добавляет локальный каталог `src/` в `sys.path`, поэтому может запускаться из Jupyter kernel без установки package в editable mode.

## API demo

Запустить API:

```bash
uvicorn hiking_recommender.api:app --reload
```

Пример API request:

```bash
curl -sS -X POST http://127.0.0.1:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","region":"north","top_k":5,"max_difficulty":"moderate"}'
```

Пример response:

```json
{
  "user_id": "user_001",
  "recommendations": [
    {
      "route_id": "route_095",
      "rank": 1,
      "score": 0.0377,
      "difficulty": "easy",
      "sources": ["collaborative", "content", "popularity"]
    }
  ]
}
```

## Пример baseline output

```text
Popularity baseline smoke passed
rank=1 route_id=route_095 score=0.8730 source=popularity
rank=2 route_id=route_054 score=0.6554 source=popularity
```

Точные `route_id` и scores могут измениться при изменении конфигурации synthetic generator.

## Архитектура

MVP сохраняет разделение ответственности:

- `data_loader.py` валидирует публичные synthetic CSV contracts.
- `features.py` строит переиспользуемые route и interaction features.
- `baseline.py` предоставляет самый простой надёжный recommendation source.
- `collaborative.py` реализует item-item retrieval поверх implicit feedback.
- `content_based.py` строит route-profile retrieval по item features.
- `merger.py` дедуплицирует и ранжирует candidates из нескольких sources.
- `business_rules.py` применяет hard product filters после retrieval и перед serving.
- `evaluation.py` считает offline top-K metrics на synthetic test interactions.
- `api.py` обслуживает тот же MVP pipeline через `GET /health` и `POST /recommendations`.

Pipeline diagram и границы модулей описаны в `docs/architecture.md`.

ALS намеренно не входит в первый MVP. Начальная collaborative model использует item-item cosine similarity поверх implicit feedback matrix, чтобы candidate merger можно было собрать без тяжёлых dependencies.
