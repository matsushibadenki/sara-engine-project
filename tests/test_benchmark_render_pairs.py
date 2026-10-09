"""Preserve physical-line limits and best-effort render-pair metrics."""
import io
from unittest.mock import patch

import pytest

from bot.benchmark_suite import _count_recent_render_pairs


@pytest.mark.parametrize('text', [
    '', '\n \t\n', 'English\n日本語\n简体中文',
    'data\n' * 200 + '\n' * 200,
    '\n' * 200 + 'data\n' * 199,
    'older\n' * 210 + 'latest\n' * 100 + '\n' * 100,
])
def test_count_matches_last_200_physical_lines(text):
    class StreamingFile(io.StringIO):
        def readlines(self, *args):
            raise AssertionError('Count must not retain the entire log')

    expected = sum(1 for line in io.StringIO(text).readlines()[-200:] if line.strip())
    with patch('bot.benchmark_suite.os.path.exists', return_value=True), patch(
            'builtins.open', return_value=StreamingFile(text)) as opened:
        assert _count_recent_render_pairs('unused') == expected
    opened.assert_called_once_with('unused', 'r', encoding='utf-8', errors='ignore')


def test_count_preserves_missing_and_unreadable_fallbacks():
    with patch('bot.benchmark_suite.os.path.exists', return_value=False):
        assert _count_recent_render_pairs('unused') == 0
    with patch('bot.benchmark_suite.os.path.exists', return_value=True), patch(
            'builtins.open', side_effect=OSError('unreadable')):
        assert _count_recent_render_pairs('unused') == 0
