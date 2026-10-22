"""Post-failure checks only; never launch allocation workers or measurements."""
from pathlib import Path

import pytest

from scripts.eval import event_allocation as runner


def test_failure_receipt_preserves_registered_limit_and_consumed_inputs():
    failure = runner.read('failure')
    assert failure['status'] == 'failed_result_serialization_budget'
    assert failure['registered_result_byte_limit'] == runner.read('selection')['budgets']['result_bytes'] == 8388608
    for suffix in ('selection', 'registration', 'execution'):
        assert failure['pins'][suffix + '_sha256'] == runner.digest(runner.path(suffix))
    assert not failure['all_allocation_gates_passed']


def test_unsaved_measurements_are_unknown_not_fabricated_results():
    failure = runner.read('failure')
    assert failure['worker_progress_reported'] == 144
    assert failure['saved_worker_measurements'] == failure['fresh_replication_executions'] == 0
    assert failure['serialized_result_bytes'] is None
    assert failure['measured_allocation_statistics'] is None
    assert not failure['semantic_replay_verified']
    assert not failure['snn_resource_advantage_established']
    assert not any(runner.path(s).exists() for s in ('result', 'replication', 'replay', 'replay_execution'))


def test_execution_cannot_be_repeated_after_failed_result_save(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Measurement restarted after consumed execution')
    monkeypatch.setattr(runner, 'evaluate', forbidden)
    with pytest.raises(FileExistsError):
        runner.main('execute')


def test_original_pins_negatives_and_seals_are_preserved():
    runner.verify()
    assert len(runner.read('selection')['preserved']) == 368
    assert runner.read('failure')['actual_learning_writes'] == 0
    assert not runner.read('failure')['test_opening_authorized']
    # No generated output escapes managed fixture paths.
    assert Path(runner.path('failure')).parent == Path(runner.path('selection')).parent
