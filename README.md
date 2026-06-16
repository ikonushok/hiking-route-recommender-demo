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
- offline top-K evaluation на отложенных synthetic interactions, включая precision/recall/MAP/NDCG/coverage/novelty/diversity;
- FastAPI endpoint для online-style recommendation serving.

## Краткая логика пайплайна

В production-like системе входом могли бы быть SQLite / raw tables. В этой demo-версии вход зафиксирован как воспроизводимые synthetic CSV files.

```text
data/synthetic_users.csv
data/synthetic_routes.csv
data/synthetic_interactions.csv
    ↓
data_loader.py
    ↓
validated synthetic users/routes/interactions datasets
    ↓
features.py
    ↓
route features + user-route implicit-feedback matrix + seen-route maps
    ↓
baseline.py / collaborative.py / content_based.py
    ↓
retrieval candidates from popularity, item-item collaborative and content-based sources
    ↓
merger.py
    ↓
deduplicated hybrid candidate list with merged scores and sources
    ↓
business_rules.py
    ├─ region filter
    ├─ difficulty filter
    ├─ seen-route exclusion
    └─ fallback fill
    ↓
evaluation.py / api.py
    ↓
offline metrics / hybrid API response
    ↓
итоговая выдача рекомендаций
```

## Архитектура проекта

```text
project/                                      # Корень demo-проекта рекомендательной системы
├─ README.md                                  # Основной README: обзор, запуск, pipeline, API и ссылки на docs
├─ pyproject.toml                             # Package metadata, pytest settings и dev-зависимости
├─ requirements.txt                           # Список runtime Python-зависимостей
├─ LICENSE                                    # Лицензия проекта
│
├─ data/                                      # Воспроизводимые synthetic CSV datasets
│  ├─ synthetic_users.csv                     # Синтетические пользователи
│  ├─ synthetic_routes.csv                    # Синтетический каталог маршрутов
│  ├─ synthetic_interactions.csv              # Полный synthetic implicit-feedback dataset
│  ├─ synthetic_interactions_train.csv        # Train split для retrieval/evaluation сценариев
│  └─ synthetic_interactions_test.csv         # Test split для offline evaluation
│
├─ scripts/                                   # CLI-скрипты для генерации, smoke checks и evaluation
│  ├─ generate_synthetic_data.py              # Генерация synthetic users/routes/interactions
│  ├─ run_baseline_smoke.py                   # Smoke check popularity baseline
│  ├─ run_hybrid_smoke.py                     # Smoke check hybrid retrieval + merger + business rules
│  └─ run_offline_evaluation.py               # Запуск offline top-K evaluation и запись artifacts
│
├─ src/hiking_recommender/                    # Основной Python package
│  ├─ data_loader.py                          # Загрузка CSV и валидация public synthetic contracts
│  ├─ schemas.py                              # Общие схемы и dataclass-модели рекомендаций
│  ├─ features.py                             # Feature engineering и implicit-feedback matrix
│  ├─ baseline.py                             # Popularity baseline и fallback candidates
│  ├─ collaborative.py                        # Item-item collaborative retrieval
│  ├─ content_based.py                        # Content-based retrieval по route features
│  ├─ candidates.py                           # Общие структуры retrieval candidates
│  ├─ merger.py                               # Candidate merger, deduplication и score aggregation
│  ├─ business_rules.py                       # Region/difficulty/seen filters и fallback fill
│  ├─ evaluation.py                           # Offline precision/recall/MAP/NDCG/coverage/novelty/diversity metrics
│  └─ api.py                                  # FastAPI app: `GET /health`, `POST /recommendations`
│
├─ tests/                                     # Contract, smoke и regression tests для P0
│  ├─ test_api.py                             # API health/recommendations contract checks
│  ├─ test_baseline_smoke.py                  # Baseline smoke behavior
│  ├─ test_business_rules.py                  # Business rules и fallback edge cases
│  ├─ test_data_loader.py                     # Synthetic schema, references и event-weight checks
│  ├─ test_evaluation.py                      # Offline metrics behavior
│  ├─ test_features.py                        # Route features, matrix aggregation и train-only checks
│  └─ test_hybrid_retrieval.py                # Hybrid retrieval, merge и duplicate checks
│
├─ docs/                                      # Документация baseline, архитектуры, evaluation и commercial handoff
│  ├─ p0_baseline.md                          # Замороженный P0 scope, contracts и validation
│  ├─ architecture.md                         # Pipeline и module boundaries
│  ├─ data_readiness_checklist.md             # Checklist для оценки готовности каталожных данных
│  ├─ commercial_use_cases.md                 # Переносимость demo на другие каталожные домены
│  └─ evaluation_report.md                    # Текущие synthetic offline metrics
│
├─ outputs/                                   # Evaluation artifacts
│  └─ evaluation_metrics.csv                  # CSV с offline metrics
│
└─ notebooks/                                 # Notebook demo для ручного walkthrough
   └─ 01_pipeline_demo.ipynb                  # End-to-end demo pipeline
```

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

Метрики полезны для проверки demo pipeline: ranking quality (`precision`, `recall`, `MAP`, `NDCG`), catalog reach (`coverage`) и P1-сигналы popularity bias / list variety (`novelty`, `diversity`). Они не являются утверждениями о production quality или business impact.

## Notebook demo

Открыть end-to-end notebook:

```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

Notebook показывает data loading, feature engineering, retrieval sources, candidate merging, business rules, offline metrics включая novelty/diversity и пример API payload.
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

Client-facing checklist и переносимость demo описаны в `docs/data_readiness_checklist.md` и `docs/commercial_use_cases.md`.

ALS намеренно не входит в первый MVP. Начальная collaborative model использует item-item cosine similarity поверх implicit feedback matrix, чтобы candidate merger можно было собрать без тяжёлых dependencies.
