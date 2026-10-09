"""Verify durable batch enqueue and bounded persistence call counts."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.training_queue import TrainingQueue
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


@pytest.fixture
def queue_path():
    directory = ensure_output_directory(workspace_path('optimization', 'stage5', 'tests'))
    with tempfile.TemporaryDirectory(dir=directory) as root:
        yield str(Path(root) / 'queue.json')


def test_batch_materials_loads_saves_and_syncs_once(queue_path):
    queue = TrainingQueue(queue_path)
    queue.enqueue({'source': 'existing'})
    materials = [dict(material_hash=str(i), curriculum_stage='repair', priority=.1,
                      source='English 日本語 简体中文') for i in range(100)]
    with patch.object(queue, '_load', wraps=queue._load) as load, patch.object(
            queue, '_save', wraps=queue._save) as save, patch(
            'bot.io_utils.os.fsync', wraps=os.fsync) as sync:
        assert queue.enqueue_learning_materials(materials) == 100
    load.assert_called_once()
    save.assert_called_once()
    sync.assert_called_once()
    rows = queue._load()
    assert len(rows) == 101
    assert rows[0]['source'] == 'existing'
    assert [row['material_hash'] for row in rows[1:]] == [str(i) for i in range(100)]
    assert all(row['priority'] == 1.0 for row in rows[1:])
    assert all(row['created_at'] for row in rows)


def test_empty_or_filtered_batch_does_not_touch_queue(queue_path):
    queue = TrainingQueue(queue_path)
    with patch.object(queue, '_load') as load, patch.object(queue, '_save') as save:
        queue.enqueue_many([])
        assert queue.enqueue_learning_materials([]) == 0
        assert queue.enqueue_learning_materials([dict(curriculum_stage='replay', priority=.74)]) == 0
    load.assert_not_called()
    save.assert_not_called()


def test_single_enqueue_and_explicit_defaults_remain_compatible(queue_path):
    queue = TrainingQueue(queue_path)
    first = dict(source='first', priority=.9, created_at='2026-01-01')
    second = dict(source='second')
    assert queue.enqueue(first) is None
    assert queue.enqueue_many([second]) is None
    assert first['created_at'] == '2026-01-01'
    assert second['priority'] == 0.0 and second['created_at']
    assert queue._load() == [first, second]


def test_failed_batch_preserves_existing_file_and_cleans_temporary_file(queue_path):
    queue = TrainingQueue(queue_path)
    queue.enqueue(dict(source='existing'))
    previous = Path(queue_path).read_bytes()
    with patch('bot.io_utils.os.replace', side_effect=OSError('Injected replace failure')):
        with pytest.raises(OSError):
            queue.enqueue_many([dict(source='new')])
    assert Path(queue_path).read_bytes() == previous
    assert list(Path(queue_path).parent.glob('.tmp_json_*')) == []


def test_invalid_material_does_not_partially_commit_batch(queue_path):
    queue = TrainingQueue(queue_path)
    previous = Path(queue_path).read_bytes()
    with pytest.raises(ValueError):
        queue.enqueue_learning_materials([dict(priority=.8), dict(priority='invalid')])
    assert Path(queue_path).read_bytes() == previous


def test_corruption_recovery_still_precedes_batch_commit(queue_path):
    queue = TrainingQueue(queue_path)
    Path(queue_path).write_text('{broken')
    queue.enqueue_many([dict(source='restored')])
    assert queue._load()[0]['source'] == 'restored'
    backups = list(Path(queue_path).parent.glob('queue.json.corrupt.*.json'))
    assert len(backups) == 1
    assert backups[0].read_text() == '{broken'


def test_material_filter_and_drain_priority_remain_compatible(queue_path):
    queue = TrainingQueue(queue_path)
    assert queue.enqueue_learning_materials([
        dict(material_hash='repair', curriculum_stage='repair', priority=.2),
        dict(material_hash='skip', curriculum_stage='replay', priority=.74),
        dict(material_hash='replay', curriculum_stage='replay', priority=.75),
        dict(material_hash='medium', curriculum_stage='medium', priority=.3)]) == 3
    result = queue.drain(10)
    assert [row['material_hash'] for row in result] == ['repair', 'replay', 'medium']
    assert queue._load() == []


@pytest.mark.parametrize('curriculum', [False, True])
def test_empty_drain_reads_once_without_save_or_sync(queue_path, curriculum):
    queue = TrainingQueue(queue_path)
    previous = Path(queue_path).stat()
    with patch.object(queue, '_load', wraps=queue._load) as load, patch.object(
            queue, '_save', wraps=queue._save) as save, patch('bot.io_utils.os.fsync') as sync:
        result = queue.drain_curriculum(10, .3, .4, .3) if curriculum else queue.drain(10)
    assert result == []
    load.assert_called_once()
    save.assert_not_called()
    sync.assert_not_called()
    current = Path(queue_path).stat()
    assert (current.st_ino, current.st_mtime_ns) == (previous.st_ino, previous.st_mtime_ns)


@pytest.mark.parametrize('raw', ['[ ]', '["invalid row"]', '{broken'])
@pytest.mark.parametrize('curriculum', [False, True])
def test_empty_drain_preserves_normalization_and_recovery(queue_path, raw, curriculum):
    queue = TrainingQueue(queue_path)
    Path(queue_path).write_text(raw)
    result = queue.drain_curriculum(10, .3, .4, .3) if curriculum else queue.drain(10)
    assert result == []
    assert Path(queue_path).read_text() == '[]'
    if raw == '{broken':
        backups = list(Path(queue_path).parent.glob('queue.json.corrupt.*.json'))
        assert len(backups) == 1 and backups[0].read_text() == raw


def test_empty_curriculum_drain_still_rejects_invalid_ratios(queue_path):
    queue = TrainingQueue(queue_path)
    with pytest.raises(ValueError):
        queue.drain_curriculum(10, 'invalid', .4, .3)


def test_nonempty_drain_remains_durable(queue_path):
    queue = TrainingQueue(queue_path)
    queue.enqueue({'source': 'English 日本語 简体中文'})
    with patch('bot.io_utils.os.fsync', wraps=os.fsync) as sync:
        result = queue.drain_curriculum(1, .3, .4, .3)
    assert len(result) == 1
    sync.assert_called_once()
    assert Path(queue_path).read_text() == '[]'
