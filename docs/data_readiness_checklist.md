# Data Readiness Checklist

Этот checklist помогает быстро оценить, готов ли каталог к recommender-system MVP.

Проект использует только synthetic data и не содержит client data, production code, proprietary database schema, internal metrics или customer-specific business logic.

## Minimum viable data

| Вопрос | Зачем | Demo equivalent |
|---|---|---|
| Есть ли стабильный `user_id`? | Нужна персонализация и user history. | `synthetic_users.csv:user_id` |
| Есть ли стабильный `item_id`? | Нужна привязка событий к каталогу. | `synthetic_routes.csv:route_id` |
| Есть ли implicit feedback events? | Нужен behavioral signal для baseline/retrieval. | `view`, `like`, `visit`, `checkin` |
| Есть ли `timestamp` события? | Нужен time-aware train/test split. | `synthetic_interactions.csv:timestamp` |
| Есть ли item features? | Нужен content-based и hybrid fallback. | length, duration, elevation, difficulty, region, season, tags |
| Есть ли cold-start сценарии? | Нужны fallback rules и неперсональные candidates. | popularity baseline |

## Data quality checks

| Check | Why it matters |
|---|---|
| `user_id` и `route_id` уникальны в справочниках. | Иначе joins и matrix mapping будут нестабильны. |
| Все interaction `user_id` существуют в users. | Иначе user history теряется или ломает training. |
| Все interaction `route_id` существуют в catalog. | Иначе retrievers могут вернуть неизвестные items. |
| Event weights соответствуют event type. | Иначе implicit-feedback semantics становятся невалидными. |
| Train/test split строится по времени. | Иначе offline metrics получают leakage из будущего. |
| Категории нормализованы. | Иначе фильтры и one-hot features расходятся между train/inference. |
| Null values обработаны явно. | Иначе model behavior зависит от неявных pandas/numpy defaults. |

## MVP recommendation path

```text
validated catalog + interactions
  -> feature engineering
  -> popularity baseline
  -> collaborative/content retrieval
  -> candidate merger
  -> business rules
  -> offline evaluation
  -> API serving
```

## Readiness decision

- **Ready for MVP**: есть stable IDs, interactions, timestamps, item features и понятные cold-start rules.
- **Needs data cleanup**: есть IDs и interactions, но нарушены joins, null handling или category consistency.
- **Not ready for personalization**: нет user history или events; начинать с popularity/content baseline.

## Validation command

Для demo-проекта минимальная проверка:

```bash
python -m pytest
```

Для пересчёта synthetic offline metrics:

```bash
python scripts/run_offline_evaluation.py
```
