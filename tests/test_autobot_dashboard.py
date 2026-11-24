import json
import os
from datetime import datetime, timedelta, timezone

from bot import status
from bot.dashboard_pages import render_page


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
    (data / "metrics.json").write_text(json.dumps({
        "ts": "2026-11-23T00:00:00", "queue_pending": 0, "new_samples": 3,
        "failed_item_count": 2, "control_actions": ["<paused>"],
    }))
    (data / "dead_letter.jsonl").write_text("")
    now = datetime.now(timezone.utc)
    (data / "alerts.log").write_text(
        f"ALERT ts={now.isoformat(timespec='milliseconds')[:-6]}Z severity=CRITICAL kind=threshold\n"
        f"ALERT ts={(now - timedelta(hours=1)).isoformat()} severity=CRITICAL kind=old\n"
    )

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
