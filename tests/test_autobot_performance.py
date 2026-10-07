"""Keep bot waiting interruptible and replay memory bounded."""
import threading
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bot.autonomous_bot import AutonomousLearningBot, _install_signal_handlers


def make_bot():
    bot = AutonomousLearningBot.__new__(AutonomousLearningBot)
    bot.running = True
    bot._stop_event = threading.Event()
    bot.config = SimpleNamespace(crawl_interval_sec=120, idle_sleep_sec=1)
    return bot


def test_idle_waits_once_and_preserves_delay():
    bot = make_bot()
    bot._stop_event = Mock()
    bot._wait_for_next_cycle(3.9)
    bot._stop_event.wait.assert_called_once_with(117)
    bot.config.idle_sleep_sec = 2
    bot._stop_event.reset_mock()
    bot._wait_for_next_cycle(125)
    bot._stop_event.wait.assert_called_once_with(2)


def test_stop_interrupts_long_wait():
    bot = make_bot()
    entered = threading.Event()

    def wait():
        entered.set()
        bot._wait_for_next_cycle(0)

    worker = threading.Thread(target=wait, daemon=True)
    worker.start()
    assert entered.wait(1)
    bot.request_stop()
    worker.join(1)
    assert not worker.is_alive()
    assert not bot.running
    assert bot._stop_event.is_set()


def test_stopped_bot_does_not_wait():
    bot = make_bot()
    bot.request_stop()
    bot._stop_event = Mock()
    bot._wait_for_next_cycle(0)
    bot._stop_event.wait.assert_not_called()


def test_signal_handler_requests_stop():
    bot = make_bot()
    with patch('bot.autonomous_bot.signal.signal') as register:
        _install_signal_handlers(bot)
    handler = register.call_args_list[0].args[1]
    handler(15, None)
    assert bot.shutdown_reason == 'signal_15'
    assert not bot.running
    assert bot._stop_event.is_set()


def test_replay_streams_only_last_5000_lines_without_readlines():
    class StreamingFile(io.StringIO):
        def readlines(self, *args):
            raise AssertionError('Replay must not materialize the entire file')

    rows = [dict(record_text=f'Record {i}: multilingual 日本語 简体中文 content',
                 source=str(i), meta=dict(quality=.9)) for i in range(5010)]
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    bot = make_bot()
    bot.records_jsonl = 'unused'
    bot.state = SimpleNamespace(replay_cursor=4999)
    with patch('bot.autonomous_bot.os.path.exists', return_value=True), patch(
            'builtins.open', return_value=StreamingFile(text)):
        result = bot._select_replay_records(3, .5)
    assert [item['source'] for item in result] == ['5009', '10', '11']
    assert bot.state.replay_cursor == 2


def test_replay_preserves_filters_and_empty_inputs():
    text = '\n'.join(['invalid json', '[]',
        json.dumps(dict(meta=[], record_text='Invalid metadata content')),
        json.dumps(dict(meta=dict(quality=.1), record_text='Low quality content')),
        json.dumps(dict(meta=dict(quality=.9), record_text='short')),
        json.dumps(dict(meta=dict(quality=.8), record_text='Valid content 日本語 简体中文', source='valid'))])
    bot = make_bot()
    bot.records_jsonl = 'unused'
    bot.state = SimpleNamespace(replay_cursor=0)
    with patch('bot.autonomous_bot.os.path.exists', return_value=True), patch(
            'builtins.open', return_value=io.StringIO(text)):
        result = bot._select_replay_records(1, .5)
    assert len(result) == 1 and result[0]['source'] == 'valid'
    assert bot._select_replay_records(0, .5) == []
    with patch('bot.autonomous_bot.os.path.exists', return_value=False):
        assert bot._select_replay_records(1, .5) == []
