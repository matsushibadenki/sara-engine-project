"""Exercise alert status through two isolated Autobot processes."""

import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import BytesIO

import pytest

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
def _autobot(root, minutes, gate, *, force_stop=False):
    with subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", _LAUNCH, str(root), str(minutes), str(gate)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGKILL if force_stop else signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            expected = -signal.SIGKILL if force_stop else 0
            assert process.returncode == expected, output


def _wait_for(process, read, expected):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process.poll() is None:
        if read() == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Autobot did not publish expected value {expected!r}")


def _page(path):
    handler = dashboard.Handler.__new__(dashboard.Handler)
    handler.rfile = BytesIO(f"GET {path}?lang=ja HTTP/1.0\r\n\r\n".encode())
    handler.wfile = BytesIO()
    handler.handle_one_request()
    headers, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
    assert headers.split()[1] == b"200"
    return body.decode()


@pytest.mark.parametrize("force_stop", [False, True], ids=["graceful", "sigkill"])
def test_alert_window_switches_after_real_autobot_restart(tmp_path, monkeypatch, force_stop):
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
        page = _page("/alerts")
        assert ("代替の集計期間" in page) == (source == "fallback")
        assert f'集計期間（分）</dt><dd class="">{minutes}</dd>' in page
        assert f'直近の重大アラート</dt><dd class="warn">{count}</dd>' in page

    def check_metrics(current, reason="no_live_producer"):
        source = status.build_status()["metrics"]
        assert source["current"] is current
        assert source["reason"] == ("verified_live_producer" if current else reason)
        timestamp = source["data"]["ts"]
        overview = _page("/")
        queue = _page("/queue")
        assert (timestamp in overview) is current
        if not current:
            assert 'メトリクス更新時刻</dt><dd class="">取得不可' in overview
        assert queue.count("取得不可</dd>") == (0 if current else 4)
        assert ("前回のメトリクス" in queue) == (not current)
        diagnostics = _page("/diagnostics")
        assert f'<dd class="">{timestamp}</dd>' in diagnostics
        assert f'<dt>発行PID</dt><dd class="">{source["data"]["producer_pid"]}</dd>' in diagnostics
        assert f'<dt>起動ID</dt><dd class="">{source["data"]["producer_launch_id"]}</dd>' in diagnostics
        assert 'id="metrics-reason"' in diagnostics
        label = "現行メトリクス" if current else "前回のメトリクス（診断用）"
        assert f">{label}</h2>" in diagnostics
        checks = diagnostics.split('id="identity-checks"', 1)[1].split("</dl>", 1)[0]
        if current:
            assert checks.count(">一致</dd>") == 2
        elif reason == "launch_id_mismatch":
            assert '<dt>PIDの照合</dt><dd class="ok">一致</dd>' in checks
            assert '<dt>起動IDの照合</dt><dd class="warn">不一致</dd>' in checks
        elif reason == "no_live_producer":
            assert checks.count(">照合不可</dd>") == 2

    first_gate = data / "first-metrics"
    with _autobot(tmp_path, 90, first_gate, force_stop=force_stop) as first:
        _wait_for(first, lambda: status.build_status()["process"]["pid"], first.pid)
        assert not metrics.exists()
        assert "稼働中</dd>" in _page("/")
        check_window("fallback", 20, 1)
        first_gate.touch()
        _wait_for(first, metrics_pid, first.pid)
        check_window("runtime", 90, 3)
        check_metrics(True)
        if force_stop:
            pre_stop_metrics = metrics.read_bytes()

    prior_metrics = metrics.read_bytes()
    if force_stop:
        assert pid_file.read_text() == str(first.pid)
        assert prior_metrics == pre_stop_metrics
        assert status.build_status()["process"]["running"] is False
    check_window("fallback", 20, 1)
    check_metrics(False)

    if force_stop:
        # Model PID reuse with a live unrelated process and matching old PID fields.
        with subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"]) as unrelated:
            try:
                reused_metrics = json.loads(prior_metrics)
                reused_metrics["producer_pid"] = unrelated.pid
                metrics.write_text(json.dumps(reused_metrics))
                pid_file.write_text(str(unrelated.pid))
                assert status.build_status()["process"]["running"] is False
                assert "停止中</dd>" in _page("/")
                check_window("fallback", 20, 1)
                check_metrics(False)
                prior_metrics = metrics.read_bytes()
            finally:
                unrelated.terminate()

    second_gate = data / "second-metrics"
    with _autobot(tmp_path, 45, second_gate) as second:
        assert second.pid != first.pid
        _wait_for(second, lambda: status.build_status()["process"]["pid"], second.pid)
        assert pid_file.read_text() == str(second.pid)
        assert metrics.read_bytes() == prior_metrics
        assert "稼働中</dd>" in _page("/")
        check_window("fallback", 20, 1)
        check_metrics(False, "pid_mismatch")
        old_snapshot = json.loads(metrics.read_text())
        old_snapshot["producer_pid"] = second.pid
        metrics.write_text(json.dumps(old_snapshot))
        check_metrics(False, "launch_id_mismatch")
        live = status.build_status()["process"]
        assert live["launch_id"] != old_snapshot["producer_launch_id"]
        active = _page("/diagnostics").split('id="active-producer"', 1)[1].split("</section>", 1)[0]
        assert f'<dd class="">{second.pid}</dd>' in active
        assert f'<dd class="">{live["launch_id"]}</dd>' in active
        matching = dict(old_snapshot, producer_launch_id=live["launch_id"])
        cases = [
            (dict(matching, producer_pid=first.pid), "pid_mismatch", "不一致", "一致"),
            ({k: v for k, v in matching.items() if k != "producer_pid"},
             "identity_missing", "照合不可", "一致"),
            ({k: v for k, v in matching.items() if k != "producer_launch_id"},
             "identity_missing", "一致", "照合不可"),
            ({k: v for k, v in matching.items() if k not in ("producer_pid", "producer_launch_id")},
             "identity_missing", "照合不可", "照合不可"),
        ]
        for snapshot, reason, pid_result, launch_result in cases:
            metrics.write_text(json.dumps(snapshot))
            verdict = status.build_status()["metrics"]
            assert verdict["current"] is False
            assert verdict["reason"] == reason
            diagnostics = _page("/diagnostics")
            checks = diagnostics.split('id="identity-checks"', 1)[1].split("</dl>", 1)[0]
            assert '<dt>PIDの照合</dt><dd class="' in checks
            assert f'>{pid_result}</dd>' in checks.split("起動IDの照合", 1)[0]
            assert f'>{launch_result}</dd>' in checks.split("起動IDの照合", 1)[1]
            assert ">前回のメトリクス（診断用）</h2>" in diagnostics

        second_gate.touch()
        _wait_for(second, lambda: status.build_status()["metrics"]["current"], True)
        check_window("runtime", 45, 2)
        check_metrics(True)
