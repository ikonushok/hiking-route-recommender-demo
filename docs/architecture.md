# Архитектура

Проект — синтетическое commercial-style demo рекомендательной системы для каталога hiking routes.

Репозиторий не содержит client data, production code, proprietary database schema, internal metrics или customer-specific business logic.

## Статус P0 baseline

Текущий pipeline зафиксирован как стабильный P0 baseline. Будущую ranking-логику или learning-to-rank нужно добавлять как отдельное post-P0 расширение с сохранением публичных данных, candidate-схемы, business rules, evaluation и API-контрактов, описанных здесь.

Область baseline и checklist проверки описаны в `docs/p0_baseline.md`.

## Пайплайн

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

## Границы модулей

- `scripts/generate_synthetic_data.py` создаёт воспроизводимые синтетические CSV-файлы.
- `data_loader.py` валидирует CSV-схемы, ссылки между ID, типы событий и веса interactions.
- `features.py` строит route features, seen-route maps и user-route implicit-feedback matrix.
- `baseline.py` выдаёт popularity recommendations и cold-start fallback candidates.
- `collaborative.py` реализует item-item cosine retrieval поверх implicit feedback.
- `content_based.py` рекомендует routes, похожие на user route-feature profile.
- `merger.py` объединяет retrieval sources через weighted reciprocal-rank contributions.
- `business_rules.py` применяет hard post-retrieval filters и safe fallback fill.
- `evaluation.py` считает offline top-K metrics на отложенных синтетических interactions.
- `api.py` обслуживает `GET /health` и `POST /recommendations`.

## Защищённые контракты

- Публичные synthetic IDs используют форматы `user_000` и `route_000`.
- Типы событий: `view`, `like`, `visit`, `checkin`.
- API response возвращает публичные `route_id`, а не внутренние индексы матрицы.
- Business rules не обходят hard filters во время fallback.
- Offline metrics являются demo validation metrics, а не production или business-impact claims.
