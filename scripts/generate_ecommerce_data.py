"""Generate a reproducible synthetic e-commerce dataset for the recommender demo.

Maps e-commerce concepts onto the engine's existing schema:
  route_id        -> product_id (format: route_XXX to pass validation)
  region          -> category (electronics, clothing, home, sports, books)
  length_km       -> price_rub / 1000
  duration_hours  -> avg_rating (1.0-5.0)
  elevation_gain  -> review_count
  difficulty      -> price_tier (budget=easy, moderate=moderate, premium=hard)
  popularity      -> sales_popularity (0-1)
  season          -> target_audience (men/women/kids/unisex)
  route_tags      -> product_tags (waterproof|lightweight|eco|gift|new|sale)

Users become customers with preferences. Interactions become:
  view   -> view
  like   -> wishlist
  visit  -> cart_add
  checkin-> purchase

This proves the engine is domain-agnostic: zero code changes, only new data.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ecommerce_data"

N_CUSTOMERS = 500
N_PRODUCTS = 300
MIN_INTERACTIONS_PER_USER = 10
MAX_INTERACTIONS_PER_USER = 45
MIN_TRAIN_INTERACTIONS_PER_PRODUCT = 5
RANDOM_SEED = 777
TEST_FRACTION_PER_USER = 0.2
EXPLORATION_RATE = 0.15

CATEGORIES = ["electronics", "clothing", "home", "sports", "books", "beauty"]
# Engine expects "easy/moderate/hard" for difficulty field.
# Mapping: budget=easy, moderate=moderate, premium=hard
PRICE_TIERS = ["budget", "moderate", "premium"]
TIER_TO_ENGINE = {"budget": "easy", "moderate": "moderate", "premium": "hard"}
AUDIENCE_LEVELS = ["low", "medium", "high"]  # spending levels
TARGET_AUDIENCES = ["men", "women", "kids", "unisex"]
PRODUCT_TAGS = [
    "waterproof", "lightweight", "eco", "gift", "new", "sale",
    "premium", "compact", "durable", "trending",
]

INTERACTION_WEIGHTS = {
    "view": 1,
    "like": 3,
    "visit": 5,
    "checkin": 8,
}

TIER_BY_SPENDING = {
    "low": ["budget", "budget", "moderate"],
    "medium": ["budget", "moderate", "moderate", "premium"],
    "high": ["moderate", "premium", "premium"],
}

TAG_PRIORS_BY_TIER = {
    "budget": ["sale", "compact", "lightweight", "eco", "durable"],
    "moderate": ["gift", "new", "durable", "waterproof", "compact"],
    "premium": ["premium", "gift", "new", "trending", "waterproof"],
}

EVENT_DISTRIBUTION = [
    ("view", 0.55),
    ("like", 0.25),
    ("visit", 0.15),
    ("checkin", 0.05),
]

USER_FIELDS = [
    "user_id",
    "preferred_difficulty",  # preferred price tier
    "preferred_region",      # preferred category
    "preferred_season",      # preferred audience
    "preferred_tags",
    "activity_level",        # spending level
]

ROUTE_FIELDS = [
    "route_id",
    "region",              # category
    "length_km",           # price_rub / 1000
    "duration_hours",      # avg_rating
    "elevation_gain_m",    # review_count
    "difficulty",          # price_tier
    "popularity",          # sales_popularity
    "season",              # target_audience
    "route_tags",          # product_tags
]

INTERACTION_FIELDS = ["user_id", "route_id", "interaction_type", "timestamp", "interaction_weight"]


def weighted_choice(options: list[tuple[str, float]], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for value, probability in options:
        cumulative += probability
        if threshold <= cumulative:
            return value
    return options[-1][0]


def choose_tags(price_tier: str, rng: random.Random) -> list[str]:
    tag_count = rng.choice([2, 2, 3])
    primary_pool = TAG_PRIORS_BY_TIER[price_tier]
    tags = set(rng.sample(primary_pool, k=min(tag_count, len(primary_pool))))
    if rng.random() < 0.30:
        tags.add(rng.choice(PRODUCT_TAGS))
    return sorted(tags)


def build_users(rng: random.Random) -> list[dict[str, str]]:
    users: list[dict[str, str]] = []
    for index in range(1, N_CUSTOMERS + 1):
        spending_level = rng.choices(
            AUDIENCE_LEVELS,
            weights=[0.30, 0.50, 0.20],
            k=1,
        )[0]
        preferred_tier = rng.choice(TIER_BY_SPENDING[spending_level])
        preferred_tags = sorted(rng.sample(TAG_PRIORS_BY_TIER[preferred_tier], k=2))
        users.append(
            {
                "user_id": f"user_{index:03d}",
                "preferred_difficulty": TIER_TO_ENGINE[preferred_tier],  # map to engine format
                "preferred_region": rng.choice(CATEGORIES),
                "preferred_season": rng.choice(TARGET_AUDIENCES),
                "preferred_tags": "|".join(preferred_tags),
                "activity_level": spending_level,
            }
        )
    return users


def build_routes(rng: random.Random) -> list[dict[str, str | float]]:
    routes: list[dict[str, str | float]] = []
    for index in range(1, N_PRODUCTS + 1):
        price_tier = rng.choices(
            PRICE_TIERS,
            weights=[0.35, 0.45, 0.20],
            k=1,
        )[0]

        # price_tier -> price range (stored as length_km = price/1000)
        if price_tier == "budget":
            price_rub = rng.randint(290, 1990)
            review_count = rng.randint(5, 120)
        elif price_tier == "moderate":
            price_rub = rng.randint(1990, 6990)
            review_count = rng.randint(15, 350)
        else:
            price_rub = rng.randint(6990, 49990)
            review_count = rng.randint(10, 200)

        avg_rating = round(rng.uniform(3.2, 4.9), 1)  # stored as duration_hours
        popularity = min(0.98, max(0.05, rng.betavariate(2.0, 3.5)))

        routes.append(
            {
                "route_id": f"route_{index:03d}",  # engine expects route_XXX format
                "region": rng.choice(CATEGORIES),
                "length_km": round(price_rub / 1000, 1),  # price in thousands
                "duration_hours": avg_rating,               # rating
                "elevation_gain_m": review_count,           # review count
                "difficulty": TIER_TO_ENGINE[price_tier],   # map to engine format
                "popularity": round(popularity, 3),
                "season": rng.choice(TARGET_AUDIENCES),
                "route_tags": "|".join(choose_tags(price_tier, rng)),
            }
        )
    return routes


def split_tags(value: str) -> set[str]:
    return {tag for tag in value.split("|") if tag}


def compatibility_score(user: dict[str, str], route: dict[str, str | float]) -> float:
    user_tags = split_tags(user["preferred_tags"])
    route_tags = split_tags(str(route["route_tags"]))
    tag_overlap = len(user_tags & route_tags) / max(len(user_tags), 1)

    score = float(route["popularity"]) * 0.30
    if user["preferred_region"] == route["region"]:
        score += 0.20
    if user["preferred_difficulty"] == route["difficulty"]:
        score += 0.20
    if user["preferred_season"] == route["season"]:
        score += 0.12
    score += tag_overlap * 0.18
    if user["activity_level"] == "high" and route["difficulty"] in {"moderate", "premium"}:
        score += 0.08
    if user["activity_level"] == "low" and route["difficulty"] == "budget":
        score += 0.08
    return score + random_noise_key(user["user_id"], str(route["route_id"]))


def random_noise_key(user_id: str, route_id: str) -> float:
    seed = f"{user_id}:{route_id}:{RANDOM_SEED}"
    local_rng = random.Random(seed)
    return local_rng.uniform(0.0, 0.08)


def select_routes_for_user(
    user: dict[str, str],
    routes: list[dict[str, str | float]],
    n_interactions: int,
    rng: random.Random,
) -> list[dict[str, str | float]]:
    scored_routes = sorted(
        routes,
        key=lambda route: compatibility_score(user, route),
        reverse=True,
    )
    preferred_count = round(n_interactions * (1.0 - EXPLORATION_RATE))
    exploration_count = n_interactions - preferred_count

    preferred_pool = scored_routes[: min(120, len(scored_routes))]
    exploration_pool = scored_routes[min(120, len(scored_routes)):] or scored_routes

    sampled = rng.sample(preferred_pool, k=min(preferred_count, len(preferred_pool)))
    sampled_ids = {str(route["route_id"]) for route in sampled}
    available_exploration = [route for route in exploration_pool if str(route["route_id"]) not in sampled_ids]
    if len(available_exploration) < exploration_count:
        available_exploration = [route for route in scored_routes if str(route["route_id"]) not in sampled_ids]
    sampled.extend(rng.sample(available_exploration, k=exploration_count))
    return sampled


def event_type_for_score(route_score: float, rng: random.Random) -> str:
    event_distribution = list(EVENT_DISTRIBUTION)
    if route_score >= 0.65:
        event_distribution = [
            ("view", 0.30),
            ("like", 0.30),
            ("visit", 0.25),
            ("checkin", 0.15),
        ]
    elif route_score <= 0.35:
        event_distribution = [
            ("view", 0.80),
            ("like", 0.12),
            ("visit", 0.06),
            ("checkin", 0.02),
        ]
    return weighted_choice(event_distribution, rng)


def build_interaction(
    user: dict[str, str],
    route: dict[str, str | float],
    event_at: datetime,
    rng: random.Random,
) -> dict[str, str | int]:
    route_score = compatibility_score(user, route)
    interaction_type = event_type_for_score(route_score, rng)
    return {
        "user_id": user["user_id"],
        "route_id": route["route_id"],
        "interaction_type": interaction_type,
        "timestamp": event_at.isoformat().replace("+00:00", "Z"),
        "interaction_weight": INTERACTION_WEIGHTS[interaction_type],
    }


def build_interactions(
    users: list[dict[str, str]],
    routes: list[dict[str, str | float]],
    rng: random.Random,
) -> list[dict[str, str | int]]:
    interactions: list[dict[str, str | int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    start_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for user in users:
        n_interactions = rng.randint(MIN_INTERACTIONS_PER_USER, MAX_INTERACTIONS_PER_USER)
        sampled_routes = select_routes_for_user(user, routes, n_interactions, rng)

        for route in sampled_routes:
            event_at = start_at + timedelta(
                days=rng.randint(0, 364),
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )
            interactions.append(build_interaction(user, route, event_at, rng))
            seen_pairs.add((user["user_id"], str(route["route_id"])))

    interactions.extend(build_coverage_interactions(users, routes, interactions, seen_pairs, rng, start_at))
    interactions.sort(key=lambda row: (str(row["user_id"]), str(row["timestamp"]), str(row["route_id"])))
    return interactions


def build_coverage_interactions(
    users: list[dict[str, str]],
    routes: list[dict[str, str | float]],
    interactions: list[dict[str, str | int]],
    seen_pairs: set[tuple[str, str]],
    rng: random.Random,
    start_at: datetime,
) -> list[dict[str, str | int]]:
    route_counts = Counter(str(row["route_id"]) for row in interactions)
    route_by_id = {str(route["route_id"]): route for route in routes}
    supplemental: list[dict[str, str | int]] = []

    for route_id in sorted(route_by_id):
        while route_counts[route_id] < MIN_TRAIN_INTERACTIONS_PER_PRODUCT + 2:
            route = route_by_id[route_id]
            compatible_users = sorted(
                users,
                key=lambda user: compatibility_score(user, route),
                reverse=True,
            )
            user = next(
                candidate
                for candidate in compatible_users
                if (candidate["user_id"], route_id) not in seen_pairs
            )
            event_at = start_at + timedelta(
                days=rng.randint(0, 300),
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )
            supplemental.append(build_interaction(user, route, event_at, rng))
            seen_pairs.add((user["user_id"], route_id))
            route_counts[route_id] += 1

    return supplemental


def temporal_train_test_split(
    interactions: list[dict[str, str | int]],
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    by_user: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)

    train: list[dict[str, str | int]] = []
    test: list[dict[str, str | int]] = []
    for user_id in sorted(by_user):
        user_rows = sorted(by_user[user_id], key=lambda row: (str(row["timestamp"]), str(row["route_id"])))
        n_test = max(1, round(len(user_rows) * TEST_FRACTION_PER_USER))
        if len(user_rows) - n_test < 2:
            n_test = max(1, len(user_rows) - 2)
        train.extend(user_rows[:-n_test])
        test.extend(user_rows[-n_test:])

    train.sort(key=lambda row: (str(row["user_id"]), str(row["timestamp"]), str(row["route_id"])))
    test.sort(key=lambda row: (str(row["user_id"]), str(row["timestamp"]), str(row["route_id"])))
    return train, test


def write_csv(path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    users = build_users(rng)
    routes = build_routes(rng)
    interactions = build_interactions(users, routes, rng)
    train_interactions, test_interactions = temporal_train_test_split(interactions)

    write_csv(DATA_DIR / "synthetic_users.csv", users, USER_FIELDS)
    write_csv(DATA_DIR / "synthetic_routes.csv", routes, ROUTE_FIELDS)
    write_csv(DATA_DIR / "synthetic_interactions.csv", interactions, INTERACTION_FIELDS)
    write_csv(DATA_DIR / "synthetic_interactions_train.csv", train_interactions, INTERACTION_FIELDS)
    write_csv(DATA_DIR / "synthetic_interactions_test.csv", test_interactions, INTERACTION_FIELDS)

    print(f"E-commerce data generated in {DATA_DIR}")
    print(f"  {len(users)} customers")
    print(f"  {len(routes)} products")
    print(f"  {len(interactions)} total interactions")
    print(f"  {len(train_interactions)} train interactions")
    print(f"  {len(test_interactions)} test interactions")

    # Print sample for verification
    print(f"\n--- Sample products ---")
    for r in routes[:3]:
        tier_reverse = {"easy": "budget", "moderate": "moderate", "hard": "premium"}
        tier_name = tier_reverse.get(r["difficulty"], r["difficulty"])
        print(f"  {r['route_id']} | category={r['region']} | price={r['length_km']}k RUB | "
              f"rating={r['duration_hours']} | tier={tier_name} | tags={r['route_tags']}")

    print("\n--- Sample interactions ---")
    for i in interactions[:5]:
        event_map = {"view": "view", "like": "wishlist", "visit": "cart_add", "checkin": "purchase"}
        print(f"  {i['user_id']} -> {i['route_id']} | {event_map[i['interaction_type']]} | {i['timestamp']}")


if __name__ == "__main__":
    main()
