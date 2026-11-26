"""Exercise alert status through two isolated Autobot processes."""

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import BytesIO

from bot import dashboard, status


_LAUNCH = """
import sys
import time
from dataclasses import replace
from pathlib import Path

from sara_engine.utils import project_paths as paths

root, minutes, gate_file = sys.argv[1:]
paths.PROJECT_ROOT = root
paths.DATA_DIR = str(Path(root) / "data")
paths.RAW_DATA_DIR = str(Path(paths.DATA_DIR) / "raw")
paths.PROCESSED_DATA_DIR = str(Path(paths.DATA_DIR) / "processed")
paths.INTERIM_DATA_DIR = str(Path(paths.DATA_DIR) / "interim")
paths.WORKSPACE_DIR = str(Path(root) / "workspace")
paths.MODELS_DIR = str(Path(root) / "models")
paths.ALLOWED_OUTPUT_ROOTS = (
    paths.RAW_DATA_DIR, paths.PROCESSED_DATA_DIR, paths.INTERIM_DATA_DIR,
    paths.WORKSPACE_DIR, paths.MODELS_DIR,
)

from bot.autonomous_bot import AutonomousLearningBot, DEFAULT_CONFIG, _install_signal_handlers

config = replace(
    DEFAULT_CONFIG, offline_mode=True, collector_plugins_enabled=False,
    critical_alert_window_minutes=int(minutes), crawl_interval_sec=60,
    weekly_digest_path=paths.workspace_path("autobot", "weekly_digest.json"),
    audit_snapshot_path=paths.workspace_path("autobot", "audit_snapshot.json"),
)
bot = AutonomousLearningBot(config)
_install_signal_handlers(bot)
write_metrics = bot._write_metrics

def gated_write(snapshot):
    deadline = time.monotonic() + 15
    while bot.running and not Path(gate_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("First metrics fixture gate timed out")
        time.sleep(0.01)
    write_metrics(snapshot)

bot._write_metrics = gated_write
bot.run()
"""


@contextmanager
def _autobot(root, minutes, gate):
    with subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", _LAUNCH, str(root), str(minutes), str(gate)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            assert process.returncode == 0, output


def _wait_for(process, read, expected):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process.poll() is None:
        if read() == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Autobot did not publish expected value {expected!r}")


def _alert_page():
    handler = dashboard.Handler.__new__(dashboard.Handler)
    handler.rfile = BytesIO(b"GET /alerts?lang=ja HTTP/1.0\r\n\r\n")
    handler.wfile = BytesIO()
    handler.handle_one_request()
    headers, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
    assert headers.split()[1] == b"200"
    return body.decode()


def test_alert_window_switches_after_real_autobot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "ROOT", str(tmp_path))
    data = tmp_path / "workspace" / "autobot"
    data.mkdir(parents=True)
    pid_file = data / "bot.pid"
    metrics = data / "metrics.json"
    now = datetime.utcnow()
    alert_lines = "".join(
        f"ALERT ts={(now - timedelta(minutes=age)).isoformat()} severity=CRITICAL kind=fixture\n"
        for age in (60, 30, 5)
    )
    (data / "alerts.log").write_text(alert_lines)

    def metrics_pid():
        return json.loads(metrics.read_text())["producer_pid"] if metrics.exists() else None

    def check_window(source, minutes, count):
        logs = status.build_status()["logs"]
        assert logs["critical_alert_window_source"] == source
        assert logs["critical_alert_window_minutes"] == minutes
        assert logs["critical_alerts_recent"] == count
        page = _alert_page()
        assert ("代替の集計期間" in page) == (source == "fallback")
        assert f'集計期間（分）</dt><dd class="">{minutes}</dd>' in page
        assert f'直近の重大アラート</dt><dd class="warn">{count}</dd>' in page

    first_gate = data / "first-metrics"
    with _autobot(tmp_path, 90, first_gate) as first:
        _wait_for(first, lambda: status.build_status()["process"]["pid"], first.pid)
        assert not metrics.exists()
        check_window("fallback", 20, 1)
        first_gate.touch()
        _wait_for(first, metrics_pid, first.pid)
        check_window("runtime", 90, 3)

    prior_metrics = metrics.read_bytes()
    check_window("fallback", 20, 1)

    second_gate = data / "second-metrics"
    with _autobot(tmp_path, 45, second_gate) as second:
        assert second.pid != first.pid
        _wait_for(second, lambda: status.build_status()["process"]["pid"], second.pid)
        assert pid_file.read_text() == str(second.pid)
        assert metrics.read_bytes() == prior_metrics
        check_window("fallback", 20, 1)
        second_gate.touch()
        _wait_for(second, metrics_pid, second.pid)
        check_window("runtime", 45, 2)
