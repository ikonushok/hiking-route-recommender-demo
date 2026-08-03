#!/usr/bin/env python3
"""Нагрузочное тестирование POST /recommendations.

Запуск:
  venv/bin/python -m src.tests.load_test --duration 30 --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = os.getenv("LOAD_TEST_TARGET", "http://127.0.0.1:8001")

# warm users (существуют в demo-датасете: user_001..user_200)
WARM_USERS = [f"user_{i:03d}" for i in range(1, 201)]
# cold users (нет в demo-датасете)
COLD_USERS = [f"user_{i}" for i in range(900000, 900100)]


def make_request(session: requests.Session, user_id: str, top_k: int = 10) -> dict:
    payload = {"user_id": user_id, "top_k": top_k}
    start = time.perf_counter()
    try:
        resp = session.post(f"{BASE_URL}/recommendations", json=payload, timeout=30)
        elapsed = time.perf_counter() - start
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "status": 0,
            "elapsed": elapsed,
            "error": str(e)[:100],
        }


def run_load_test(duration_sec: int, concurrency: int) -> list[dict]:
    session = requests.Session()
    results: list[dict] = []
    end_time = time.time() + duration_sec
    req_counter = 0

    print(f"\n{'='*60}")
    print(f"  LOAD TEST: {duration_sec}s, concurrency={concurrency}")
    print(f"  Target: {BASE_URL}/recommendations")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: list = []
        while time.time() < end_time:
            # 70% warm, 30% cold
            if random.random() < 0.7:
                uid = random.choice(WARM_USERS)
            else:
                uid = random.choice(COLD_USERS)
            futures.append(pool.submit(make_request, session, uid))
            req_counter += 1

            # batch futures to avoid memory explosion
            if len(futures) >= concurrency * 4:
                for f in as_completed(futures):
                    results.append(f.result())
                futures = []

        # collect remaining
        for f in as_completed(futures):
            results.append(f.result())

    return results


def push_metrics(results: list[dict], duration_sec: int, concurrency: int) -> None:
    try:
        from hiking_recommender.pipeline_metrics import push_pipeline_metrics_batch

        total = len(results)
        ok = sum(1 for r in results if r["ok"])
        fail = total - ok
        latencies = [r["elapsed"] for r in results if r["ok"]]
        fail_latencies = [r["elapsed"] for r in results if not r["ok"]]

        metrics: list[dict] = [
            {"name": "loadtest", "value": total, "labels": {"metric": "total_requests", "concurrency": str(concurrency)}},
            {"name": "loadtest", "value": ok, "labels": {"metric": "ok_count", "concurrency": str(concurrency)}},
            {"name": "loadtest", "value": fail, "labels": {"metric": "fail_count", "concurrency": str(concurrency)}},
            {"name": "loadtest", "value": round(total / duration_sec, 1) if duration_sec else 0, "labels": {"metric": "rps", "concurrency": str(concurrency)}},
        ]

        if latencies:
            latencies_sorted = sorted(latencies)
            metrics.extend([
                {"name": "loadtest", "value": latencies_sorted[len(latencies_sorted) // 2], "labels": {"metric": "latency_p50", "concurrency": str(concurrency)}},
                {"name": "loadtest", "value": latencies_sorted[int(len(latencies_sorted) * 0.95)], "labels": {"metric": "latency_p95", "concurrency": str(concurrency)}},
                {"name": "loadtest", "value": latencies_sorted[int(len(latencies_sorted) * 0.99)], "labels": {"metric": "latency_p99", "concurrency": str(concurrency)}},
            ])

        if fail_latencies:
            metrics.append({"name": "loadtest", "value": statistics.mean(fail_latencies), "labels": {"metric": "fail_latency_mean", "concurrency": str(concurrency)}})

        push_pipeline_metrics_batch(metrics, job="loadtest")
        gw = os.getenv("PUSHGATEWAY_URL", "127.0.0.1:9091")
        print(f"\n  Metrics pushed to pushgateway ({gw})")
    except Exception as exc:
        print(f"\n  Push metrics skipped: {exc}")


def print_report(results: list[dict], duration_sec: int) -> None:
    total = len(results)
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    latencies = [r["elapsed"] for r in ok]

    print(f"\n{'='*60}")
    print(f"  ОТЧЁТ НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
    print(f"{'='*60}")
    print(f"  Запросов отправлено:  {total}")
    print(f"  Успешных (200):       {len(ok)} ({len(ok)/total*100:.1f}%)")
    print(f"  Ошибок:               {len(fail)} ({len(fail)/total*100:.1f}%)")
    print(f"  RPS:                  {total/duration_sec:.1f}")

    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean = statistics.mean(latencies)
        print(f"\n  LATENCY (секунды):")
        print(f"    mean:  {mean:.3f}")
        print(f"    p50:   {p50:.3f}")
        print(f"    p95:   {p95:.3f}")
        print(f"    p99:   {p99:.3f}")
        print(f"    min:   {min(latencies):.3f}")
        print(f"    max:   {max(latencies):.3f}")

    if fail:
        errors_by_code: dict[str, int] = {}
        for r in fail:
            key = str(r["status"]) if r["status"] else r["error"][:50]
            errors_by_code[key] = errors_by_code.get(key, 0) + 1
        print(f"\n  ОШИБКИ:")
        for code, count in sorted(errors_by_code.items(), key=lambda x: -x[1]):
            print(f"    {code}: {count}")

    print(f"\n{'='*60}")

    # Save report
    report_path = Path("outputs/load_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "total_requests": total,
        "ok_count": len(ok),
        "fail_count": len(fail),
        "rps": round(total / duration_sec, 1),
        "duration_sec": duration_sec,
        "latency": {
            "mean": round(statistics.mean(latencies), 4) if latencies else None,
            "p50": round(latencies[len(latencies) // 2], 4) if latencies else None,
            "p95": round(latencies[int(len(latencies) * 0.95)], 4) if latencies else None,
            "p99": round(latencies[int(len(latencies) * 0.99)], 4) if latencies else None,
            "min": round(min(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Отчёт: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Нагрузочное тестирование API")
    parser.add_argument("--duration", type=int, default=30, help="Длительность в секундах (default: 30)")
    parser.add_argument("--concurrency", type=int, default=10, help="Параллельных запросов (default: 10)")
    args = parser.parse_args()

    results = run_load_test(args.duration, args.concurrency)
    print_report(results, args.duration)
    push_metrics(results, args.duration, args.concurrency)
