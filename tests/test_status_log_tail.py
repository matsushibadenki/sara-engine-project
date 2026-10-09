"""Verify bounded status log reads preserve output and alert semantics."""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bot.status import _critical_alerts_recent, _tail_lines
from sara_engine.utils.project_paths import workspace_path


@pytest.fixture
def log_path():
    root = workspace_path('optimization', 'stage10', 'tests')
    Path(root).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as directory:
        yield Path(directory) / 'events.log'


@pytest.mark.parametrize('limit', [-3, -1, 0, 1, 3, 100])
def test_tail_preserves_slice_semantics(log_path, limit):
    log_path.write_bytes('English\n日本語\n简体中文\n\nlast'.encode() + b'\xff')
    with log_path.open(encoding='utf-8', errors='ignore') as handle:
        expected = [line.rstrip('\n') for line in handle.readlines()[-limit:]]
    assert _tail_lines(str(log_path), limit) == expected


def test_missing_and_empty_logs(log_path):
    assert _tail_lines(str(log_path), 10) == []
    assert _critical_alerts_recent(str(log_path), 60) == 0
    log_path.write_text('')
    assert _tail_lines(str(log_path), 10) == []
    assert _critical_alerts_recent(str(log_path), 60) == 0


def test_alert_count_uses_last_3000_lines(log_path):
    timestamp = (datetime.utcnow() + timedelta(days=1)).isoformat()
    log_path.write_text((f'ts={timestamp} severity=CRITICAL 日本語\n') * 3100)
    assert _critical_alerts_recent(str(log_path), 60) == 3000


def test_alert_count_preserves_cutoff_and_malformed_records(log_path):
    recent = (datetime.utcnow() + timedelta(days=1)).isoformat()
    old = (datetime.utcnow() - timedelta(days=1)).isoformat()
    log_path.write_text('\n'.join([
        f'ts={recent} severity=CRITICAL',
        f'ts={old} severity=CRITICAL',
        'severity=CRITICAL',
        'ts=invalid severity=CRITICAL',
        f'ts={recent} severity=INFO',
        f'ts={recent} severity=CRITICAL',
    ]))
    assert _critical_alerts_recent(str(log_path), 60) == 1
