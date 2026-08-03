# Коммерческие сценарии применения

Этот документ показывает, как synthetic hiking recommender demo переносится на другие каталожные продукты.

Репозиторий остаётся demo-проектом: все данные синтетические, production code и customer-specific business logic не включены.

## Где применим этот подход

| Домен | Объекты каталога | Типовые события | Полезные business rules |
|---|---|---|---|
| Туристический каталог | Маршруты, туры, достопримечательности | view, like, visit, booking intent | регион, сезон, сложность, доступность |
| E-commerce | Товары, наборы, комплекты | view, cart, purchase, return | наличие, маржинальность, категория, ценовой сегмент |
| Образовательная платформа | Курсы, уроки, треки | view, start, complete, save | уровень, язык, prerequisites |
| Контентный сайт | Статьи, видео, подборки | view, dwell, like, share | тема, свежесть, safety filters |
| B2B-каталог | Поставщики, услуги, SKU | view, shortlist, inquiry | география, сертификация, SLA tier |
| Marketplace | Объявления, исполнители, предложения | view, favorite, contact, order | локация, вместимость, policy constraints |

## Что показывает demo

- Baseline-first подход перед добавлением более сложного ML.
- Разделение retrieval, candidate merge, business rules и serving.
- Offline top-K evaluation на отложенных interactions.
- Cold-start fallback через popularity candidates.
- Публичный API-контракт, который возвращает synthetic `route_id`, rank, score, difficulty и sources.

## Вопросы для первичного обсуждения с клиентом

| Вопрос | Зачем спрашивать |
|---|---|
| Какие действия пользователя считаются положительным сигналом? | Определяет веса implicit feedback. |
| Какие поля каталога надёжны и регулярно заполнены? | Определяет content-based features и фильтры. |
| Какие фильтры являются hard business rules? | Предотвращает рекомендации, которые нельзя показывать. |
| Как часто меняется каталог? | Влияет на стратегию retraining и cache. |
| Что ожидается в cold-start сценариях? | Определяет fallback для новых пользователей и объектов. |
| Какая offline-метрика важнее на первом этапе? | Связывает Precision/Recall/NDCG/Coverage с продуктовыми целями. |

## Рекомендуемый MVP scope

1. Проверить data contracts и event semantics.
2. Построить popularity baseline и content fallback.
3. Добавить collaborative retrieval, если user-item history достаточно плотная.
4. Объединить candidates с детерминированной source diagnostics.
5. Применить hard business rules после retrieval.
6. Оценить offline metrics до интеграции API.

## Ограничения

- Offline metrics — это validation signals, а не production business impact.
- Synthetic route data не представляет реальный каталог.
- Ranking/LTR намеренно остаётся post-P0 расширением и должен добавляться без скрытого изменения публичных API-контрактов.
