import json
import os
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

from bot import status
from bot.dashboard_pages import render_page
from bot.process_lease import LEASE_FILENAME, producer_lease


def test_dashboard_with_missing_and_damaged_status(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "ROOT", str(tmp_path))
    data = tmp_path / "workspace" / "autobot"
    data.mkdir(parents=True)
    (data / "bot.pid").write_text("0")

    missing = status.build_status()
    overview = render_page("/", "en", missing)
    assert "Stopped</dd>" in overview
    assert "No metrics received</p>" in overview
    assert render_page("/alerts", "en", missing).count("Unavailable</dd>") == 2

    (data / "metrics.json").write_text("{")
    (data / "alerts.log").mkdir()
    damaged = status.build_status()
    assert "Metrics could not be read</p>" in render_page("/queue", "en", damaged)
    assert "parse_error" in render_page("/diagnostics", "en", damaged)
    assert damaged["logs"]["critical_alerts_recent"] is None


def test_dashboard_with_live_process_and_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "ROOT", str(tmp_path))
    data = tmp_path / "workspace" / "autobot"
    data.mkdir(parents=True)
    (data / "bot.pid").write_text(str(os.getpid()))
    (data / "dead_letter.jsonl").write_text("")
    now = datetime.now(timezone.utc)
    (data / "alerts.log").write_text(
        f"ALERT ts={now.isoformat(timespec='milliseconds')[:-6]}Z severity=CRITICAL kind=threshold\n"
        f"ALERT ts={(now - timedelta(hours=1)).isoformat()} severity=CRITICAL kind=old\n"
    )

    with producer_lease(str(data / LEASE_FILENAME)) as launch_id:
        (data / "metrics.json").write_text(json.dumps({
            "producer_pid": os.getpid(), "producer_launch_id": launch_id,
            "ts": "2026-11-23T00:00:00", "queue_pending": 0, "new_samples": 3,
            "failed_item_count": 2, "control_actions": ["<paused>"],
        }))
        current = status.build_status()
    assert "Running</dd>" in render_page("/", "en", current)
    queue = render_page("/queue", "en", current)
    assert 'class="">0</dd>' in queue
    assert 'class="warn">2</dd>' in queue
    assert "&lt;paused&gt;" in queue
    alerts = render_page("/alerts", "en", current)
    assert 'class="">0</dd>' in alerts
    assert 'class="warn">1</dd>' in alerts


def test_status_collection_failure_returns_recoverable_pages(monkeypatch, caplog):
    from io import BytesIO

    from bot import dashboard

    def failed_status(verbose=False):
        raise OSError("private failure details")

    def get(path):
        handler = dashboard.Handler.__new__(dashboard.Handler)
        handler.rfile = BytesIO(f"GET {path} HTTP/1.0\r\n\r\n".encode())
        handler.wfile = BytesIO()
        handler.handle_one_request()
        headers, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
        return int(headers.split()[1]), body.decode()

    monkeypatch.setattr(dashboard, "build_status", failed_status)
    for path in dashboard.PAGES:
        code, html = get(path + "?lang=ja")
        assert code == 503
        assert 'lang="ja"' in html
        assert "状態を取得できません" in html
        assert '<div id="panel"></div>' in html
        assert "private failure details" not in html
    code, body = get("/api/status")
    assert code == 503
    assert json.loads(body) == {"error": "status_unavailable"}
    assert "Autobot status collection failed" in caplog.text

    monkeypatch.setattr(dashboard, "build_status", lambda verbose=False: {"process": {"running": True}})
    code, html = get("/")
    assert code == 200
    assert "Running</dd>" in html
    assert json.loads(get("/api/status")[1]) == {"process": {"running": True}}


def test_alert_window_uses_only_live_producer_metrics(tmp_path, monkeypatch):
    from bot.autonomous_bot import AutonomousLearningBot

    monkeypatch.setattr(status, "ROOT", str(tmp_path))
    data = tmp_path / "workspace" / "autobot"
    data.mkdir(parents=True)
    (data / "bot.pid").write_text(str(os.getpid()))
    (data / "alerts.log").write_text(
        f"ALERT ts={(datetime.utcnow() - timedelta(minutes=30)).isoformat()} severity=CRITICAL\n"
    )
    metrics = data / "metrics.json"

    bot = AutonomousLearningBot.__new__(AutonomousLearningBot)
    bot.config = SimpleNamespace(critical_alert_window_minutes=45)
    bot.policy = SimpleNamespace(max_pages_per_cycle=1)
    bot.state = SimpleNamespace(visited_urls=set(), processed_files=set(), modality_counts={})
    bot.alert_suppressed_total = 0
    bot.metrics_path = str(metrics)
    with producer_lease(str(data / LEASE_FILENAME)) as launch_id:
        bot._producer_launch_id = launch_id
        bot._write_metrics({})
        live = status.build_status()
        stale = json.loads(metrics.read_text())
        stale["producer_launch_id"] = "previous-launch"
        metrics.write_text(json.dumps(stale))
        assert status.build_status()["logs"]["critical_alert_window_source"] == "fallback"

    assert live["logs"]["critical_alert_window_minutes"] == 45
    assert live["logs"]["critical_alerts_recent"] == 1
    assert live["logs"]["critical_alert_window_source"] == "runtime"
    assert "Fallback window." not in render_page("/alerts", "en", live)

    for producer, window, pid, expected, source, count in [
        (os.getpid() + 1, 45, os.getpid(), 20, "fallback", 0),
        (os.getpid(), 45, 0, 20, "fallback", 0),
        (os.getpid(), -1, os.getpid(), 20, "fallback", 0),
    ]:
        (data / "bot.pid").write_text(str(pid))
        metrics.write_text(json.dumps({
            "producer_pid": producer, "critical_alert_window_minutes": window,
        }))
        snapshot = status.build_status()
        assert snapshot["logs"]["critical_alert_window_minutes"] == expected
        assert snapshot["logs"]["critical_alert_window_source"] == source
        assert snapshot["logs"]["critical_alerts_recent"] == count
        assert ("Fallback window." in render_page("/alerts", "en", snapshot)) == (source == "fallback")
