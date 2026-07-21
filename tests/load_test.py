"""Exercise the demo API with independent workers: python tests/load_test.py."""

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

BASE_URL = os.environ.get("LOAD_TEST_TARGET", "http://127.0.0.1:8001")
WARM_USERS = tuple(f"user_{number:03d}" for number in range(1, 201))
COLD_USERS = tuple(f"user_{number}" for number in range(900000, 900100))


def make_request(session, user_id, top_k=10):
    started = time.perf_counter()
    status, error = 0, None
    try:
        response = session.post(
            BASE_URL + "/recommendations", timeout=30,
            json=dict(user_id=user_id, top_k=top_k),
        )
        status = response.status_code
    except requests.RequestException as failure:
        error = str(failure)[:100]
    return dict(status=status, error=error, ok=(status == 200),
                elapsed=time.perf_counter() - started)


def run_load_test(duration_sec, concurrency):
    """Each worker owns its HTTP session and submits one request at a time."""
    if duration_sec <= 0 or concurrency <= 0:
        raise ValueError("Duration and concurrency must be positive")
    deadline = time.perf_counter() + duration_sec

    def worker(_):
        observations = []
        with requests.Session() as session:
            while time.perf_counter() < deadline:
                population = WARM_USERS if random.random() < .7 else COLD_USERS
                observations.append(make_request(session, random.choice(population)))
        return observations

    print(f"Load target {BASE_URL}: {concurrency} workers for {duration_sec}s")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return [row for group in executor.map(worker, range(concurrency)) for row in group]


def _summary(results, duration_sec):
    samples = sorted(row["elapsed"] for row in results if row["ok"])
    quantiles = {f"p{percent}": samples[min(len(samples) - 1, len(samples) * percent // 100)]
                 if samples else None for percent in (50, 95, 99)}
    latency = dict(mean=statistics.fmean(samples) if samples else None,
                   min=min(samples) if samples else None, max=max(samples) if samples else None,
                   **quantiles)
    successes = sum(row["ok"] for row in results)
    return dict(total_requests=len(results), ok_count=successes,
                fail_count=len(results) - successes,
                rps=round(len(results) / duration_sec, 1) if duration_sec else 0,
                duration_sec=duration_sec,
                latency={name: round(value, 4) if value is not None else None
                         for name, value in latency.items()})


def push_metrics(results, duration_sec, concurrency):
    from hiking_recommender.pipeline_metrics import push_pipeline_metrics_batch
    report = _summary(results, duration_sec)
    values = {name: report[name] for name in ("total_requests", "ok_count", "fail_count", "rps")}
    values.update({f"latency_{name}": report["latency"][name]
                   for name in ("p50", "p95", "p99") if report["latency"][name] is not None})
    failures = [row["elapsed"] for row in results if not row["ok"]]
    if failures:
        values["fail_latency_mean"] = statistics.fmean(failures)
    push_pipeline_metrics_batch([
        dict(name="loadtest", value=value,
             labels=dict(concurrency=str(concurrency), metric=name))
        for name, value in values.items()
    ], job="loadtest")


def print_report(results, duration_sec):
    report = _summary(results, duration_sec)
    destination = Path("outputs") / "load_test_report.json"
    destination.parent.mkdir(exist_ok=True, parents=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    errors = Counter(str(row["status"] or row["error"]) for row in results if not row["ok"])
    if errors:
        print("Failed requests:", dict(errors))
    print("Report:", destination)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Demo API load check")
    cli.add_argument("--duration", default=30, type=int)
    cli.add_argument("--concurrency", default=10, type=int)
    options = cli.parse_args()
    records = run_load_test(options.duration, options.concurrency)
    print_report(records, options.duration)
    push_metrics(records, options.duration, options.concurrency)
