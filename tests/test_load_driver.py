import importlib.util
import json
import time
from pathlib import Path
import requests

spec = importlib.util.spec_from_file_location("demo_load_driver", Path(__file__).with_name("load_test.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_request_failure_is_reported():
    class FailedSession:
        def post(self, *args, **kwargs):
            raise requests.Timeout("timed out")
    row = driver.make_request(FailedSession(), "user_001")
    assert not row["ok"] and row["status"] == 0 and row["error"] == "timed out"
    assert row["elapsed"] >= 0


def test_worker_sessions_close_and_report_schema_survives(monkeypatch, tmp_path):
    sessions = []
    class Session:
        def __enter__(self):
            sessions.append(self)
            self.closed = False
            return self
        def __exit__(self, *args):
            self.closed = True
        def post(self, *args, **kwargs):
            time.sleep(.005)
            return type("Reply", (), {"status_code": 200})()
    monkeypatch.setattr(driver.requests, "Session", Session)
    records = driver.run_load_test(.03, 2)
    assert records and all(row["ok"] for row in records)
    assert len(sessions) == 2 and all(session.closed for session in sessions)
    monkeypatch.chdir(tmp_path)
    driver.print_report(records, .03)
    report = json.loads(Path("outputs/load_test_report.json").read_text())
    assert report["ok_count"] == len(records) and report["fail_count"] == 0
    assert set(report["latency"]) == {"mean", "p50", "p95", "p99", "min", "max"}
    driver.print_report([], 0)
    assert json.loads(Path("outputs/load_test_report.json").read_text())["rps"] == 0
