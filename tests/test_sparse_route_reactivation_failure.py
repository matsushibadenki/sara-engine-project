"""Post-result regression checks; never rerun a consumed comparison."""
import json

from scripts.eval import sparse_route_reactivation as runner


def test_saved_learning_equality_does_not_erase_scientific_failure():
    result = runner.read('result')
    assert result['execution_integrity_passed']
    assert not result['component_passed']
    assert len(result['cases']) == 20
    for case in result['cases']:
        for arm in case['arms'].values():
            assert arm['route_updates'] == 9216
            assert arm['actual_learning_field_changes'] == 36864


def test_every_changed_case_fails_original_recovery_gate():
    cases = runner.read('result')['cases']
    changed = [c for c in cases if c['condition'] == 'changed']
    stationary = [c for c in cases if c['condition'] == 'stationary']
    assert len(changed) == len(stationary) == 10
    assert all(not c['component_gates']['recovery_within_256'] for c in changed)
    assert all(all(c['component_gates'].values()) for c in stationary)
    assert all(c['arms']['error_triggered']['exploration_starts'] == [] for c in stationary)


def test_initial_pool_type_is_not_preserved_by_json():
    saved = runner.read('result')['cases'][0]['arms']['permanent']['initial_pool']
    assert type(saved) is list and len(saved) == 9
    runtime_shape = tuple(saved)
    assert runtime_shape != saved
    assert json.loads(json.dumps(runtime_shape)) == saved
    # This isolates a comparator defect; it does not prove full numeric replay.


def test_consumed_failed_replay_is_retained_without_success_receipt():
    assert runner.path('replay_execution').exists()
    assert not runner.path('replay').exists()
    failure = runner.read('failure')
    assert failure['full_replay_verified'] is False
    assert failure['new_study_run_authorized'] is False
    assert failure['numeric_trace_equality'] is None
    assert failure['result_sha256'] == runner.digest(runner.path('result'))
    assert failure['replay_execution_sha256'] == runner.digest(runner.path('replay_execution'))
    runner.verify()
