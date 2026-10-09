"""Verify log trimming preserves bytes below the cap and decoded lines above it."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bot.autonomous_bot import AutonomousLearningBot
from sara_engine.utils.project_paths import workspace_path


@pytest.fixture
def log_path():
    root = workspace_path('optimization', 'stage13', 'tests')
    Path(root).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as directory:
        yield Path(directory) / 'log.txt'


@pytest.mark.parametrize('raw,limit', [
    (b'', 3), (b'a\nb\n', 2), (b'a\nb', 2),
    (b'a\r\nb\r\n', 2), (b'a\xff\nb\n', 2),
    (b'a\nb\nc\n', 0), (b'a\nb\nc\n', -1),
])
def test_retention_does_not_rewrite_within_limit_or_disabled(log_path, raw, limit):
    log_path.write_bytes(raw)
    before = log_path.stat()
    AutonomousLearningBot._trim_file_lines(None, str(log_path), limit)
    assert log_path.read_bytes() == raw
    after = log_path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


@pytest.mark.parametrize('raw,limit', [
    (b'old\nnew\nlast', 1),
    (b'old\r\n\r\nnew\r\n', 2),
    ('old\nEnglish\n日本語\n简体中文'.encode(), 3),
    (b'old\ninvalid\xff\nlast\n', 2),
    (b'old\n\n\n', 2),
])
def test_retention_preserves_decoded_tail_and_line_boundaries(log_path, raw, limit):
    log_path.write_bytes(raw)
    with log_path.open(encoding='utf-8', errors='ignore') as handle:
        expected = ''.join(handle.readlines()[-limit:]).encode('utf-8')
    AutonomousLearningBot._trim_file_lines(None, str(log_path), limit)
    assert log_path.read_bytes() == expected


def test_missing_log_remains_missing(log_path):
    AutonomousLearningBot._trim_file_lines(None, str(log_path), 2)
    assert not log_path.exists()


@pytest.mark.parametrize('raw', [b'\n' * 10, b'abcdefghij', b'a\r\nb\r\n', b'a\xff\nb'])
def test_byte_bound_avoids_content_reads_without_changing_file(log_path, raw):
    from unittest.mock import patch

    log_path.write_bytes(raw)
    before = log_path.stat()
    with patch('builtins.open', side_effect=AssertionError('Small files need no content read')):
        AutonomousLearningBot._trim_file_lines(None, str(log_path), 10)
    assert log_path.read_bytes() == raw
    assert log_path.stat().st_mtime_ns == before.st_mtime_ns


def test_external_append_crossing_byte_bound_still_trims(log_path):
    log_path.write_bytes(b'\n' * 10)
    AutonomousLearningBot._trim_file_lines(None, str(log_path), 10)
    with log_path.open('ab') as handle:
        handle.write(b'latest')
    AutonomousLearningBot._trim_file_lines(None, str(log_path), 10)
    assert log_path.read_bytes() == b'\n' * 9 + b'latest'
