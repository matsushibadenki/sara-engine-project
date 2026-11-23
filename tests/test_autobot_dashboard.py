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
