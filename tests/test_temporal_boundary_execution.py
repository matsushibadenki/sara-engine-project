"""Use tiny non-registered identities; never score the frozen experiment."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sara_engine.evaluation.temporal_boundary_learning import load_protocol
from sara_engine.evaluation.temporal_boundary_execution import (
    execute_once, feedback, gates_for_family, new_readout, restore_readout,
    run_arm, run_phase, save_exclusive, shuffled_labels,
)
from sara_engine.learning.local_outcome import LocalOutcomeConfig
from sara_engine.utils.project_paths import ensure_output_directory, project_path, workspace_path


@pytest.fixture
def protocol():
    value = deepcopy(load_protocol())
    value['training_seeds'] = [991001]
    value['evaluation_seeds'] = [991002]
    value['pairs_per_family'] = {'training': 3, 'evaluation': 2}
    return value


@pytest.fixture
def managed_tmp():
    root = ensure_output_directory(workspace_path('tests'))
    with TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def test_registered_capacity_and_configurable_upper_bound(protocol):
    model = new_readout(protocol)
    assert model.config.max_routes == 26673
    assert LocalOutcomeConfig().max_routes == 256
    assert LocalOutcomeConfig(max_routes=65536).max_routes == 65536
    with pytest.raises(ValueError):
        LocalOutcomeConfig(max_routes=65537)


def test_exact_delayed_feedback_rejects_wrong_time_and_receipts(protocol):
    model = new_readout(protocol)
    receipt = model.predict(((26672, 1.0),), time=5)
    before = model.snapshot()
    for time in (4, 5, 8, 10, True):
        with pytest.raises(ValueError):
            feedback(model, receipt, 1, time, protocol)
        assert model.snapshot() == before
    with pytest.raises(ValueError):
        feedback(model, replace(receipt), 1, 9, protocol)
    assert model.snapshot() == before
    assert feedback(model, receipt, 1, 9, protocol).route_visits == 1
    with pytest.raises(ValueError):
        feedback(model, receipt, 1, 9, protocol)


def test_sha_shuffle_balanced_deterministic_and_independent():
    labels, bound = shuffled_labels(991001, 'middle_order', 200)
    assert labels == shuffled_labels(991001, 'middle_order', 200)[0]
    assert sum(labels) == 100
    assert labels != shuffled_labels(991002, 'middle_order', 200)[0]
    assert labels != shuffled_labels(991001, 'gap_order', 200)[0]
    assert bound > len(labels)
    with pytest.raises(ValueError):
        shuffled_labels(1, 'gap_order', 3)


@pytest.mark.parametrize('family', ['middle_order', 'gap_order'])
def test_all_arms_frozen_disk_replay_and_work_accounting(protocol, family, managed_tmp):
    for arm in protocol['arms']:
        result, weights = run_arm(protocol, arm, family, 991001, 991002)
        path = managed_tmp / (arm + '.json')
        save_exclusive(path, weights)
        model = restore_readout(protocol, json.loads(path.read_text()))
        before = dict(model._weights)
        replay = run_phase(protocol, model, arm, family, 991002, training=False)
        assert replay['prediction_sha256'] == result['evaluation']['prediction_sha256']
        assert model._weights == before
        assert model._pending is None
        assert result['training']['work']['episodes'] == 6
        assert replay['work']['episodes'] == 4
        assert replay['work']['update_route_visits'] == 0
        for phase in (result['training'], result['evaluation'], replay):
            work = phase['work']
            assert work['validation_event_visits'] == work['observation_visits'] == work['canonical_event_visits']
            assert work['prediction_route_visits'] == work['feature_sort_items']
            assert work['wire_value_visits'] == 2 * work['observation_visits']
            assert work['prediction_receipts'] == work['episodes']
            assert phase['peak_total_state_bound_bytes'] < 1048576
            assert phase['cpu_process_seconds_including_instrumentation_and_input_generation'] >= 0
        assert result['training']['work']['update_route_visits'] == result['training']['work']['prediction_route_visits']
        if arm in ('histogram', 'first_last_count', 'gap' if family == 'middle_order' else 'ordinal'):
            assert replay['opposite_pair_predictions_equal']


@pytest.mark.parametrize('weights', [{'01': 0.2}, {'-1': 0.2}, {'26673': 0.2}, {'2': True}, {'2': float('nan')}, {'2': 1.1}])
def test_corrupt_checkpoint_is_rejected(protocol, weights):
    with pytest.raises(ValueError):
        restore_readout(protocol, weights)


@pytest.mark.parametrize('cap', ['max_state_bytes_per_arm', 'max_prediction_route_visits', 'max_update_route_visits'])
def test_resource_failure_before_learning(protocol, cap):
    protocol['budgets'][cap] = 1
    model = new_readout(protocol)
    before = model.snapshot()
    with pytest.raises(ValueError, match='budget'):
        run_phase(protocol, model, 'joint', 'middle_order', 991001, training=True)
    assert model.snapshot() == before


def test_quality_failure_does_not_become_success():
    arms = {name: {'evaluation': {'accuracy': 0.5, 'brier': 0.25, 'opposite_pair_predictions_equal': True}}
            for name in ('histogram', 'first_last_count', 'ordinal', 'gap', 'joint', 'joint_shuffled')}
    gates = gates_for_family(arms, 'middle_order')
    assert not gates['quality'] and not gates['mechanism']
    assert gates['blind_pair_equality']
    assert not gates['strong_reference_superiority']


def test_completed_attempt_cannot_repeat(managed_tmp):
    output, prefix, checkpoint = (managed_tmp / name for name in ('result.json', 'attempt', 'checkpoint.json'))
    calls = []

    def work():
        calls.append(1)
        return {'passed': False}

    assert execute_once(output, prefix, checkpoint, {}, work) == {'passed': False}
    assert json.loads(Path(str(prefix) + '_completed.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        execute_once(output, prefix, checkpoint, {}, work)
    assert calls == [1]


def test_failed_attempt_stays_consumed(managed_tmp):
    output, prefix, checkpoint = (managed_tmp / name for name in ('result.json', 'attempt', 'checkpoint.json'))

    def fail():
        raise RuntimeError('Deliberate test failure')

    with pytest.raises(RuntimeError):
        execute_once(output, prefix, checkpoint, {}, fail)
    assert json.loads(Path(str(prefix) + '_failed.json').read_text())['error_type'] == 'RuntimeError'
    with pytest.raises(FileExistsError):
        execute_once(output, prefix, checkpoint, {}, lambda: {})


def test_concurrent_attempt_has_one_winner(managed_tmp):
    output, prefix, checkpoint = (managed_tmp / name for name in ('result.json', 'attempt', 'checkpoint.json'))
    calls = []

    def attempt(_):
        try:
            execute_once(output, prefix, checkpoint, {}, lambda: calls.append(1) or {'passed': True})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(4))) == 1
    assert calls == [1]


def test_output_outside_managed_directories_is_rejected():
    with pytest.raises(ValueError, match='Output path'):
        save_exclusive('/unmanaged/temporal-result.json', {})


def test_complete_runner_with_test_only_identities(protocol, managed_tmp, monkeypatch):
    spec = importlib.util.spec_from_file_location('temporal_runner_test', project_path('scripts/eval/temporal_boundary_learning.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Real preflight is read-only. No registered training/evaluation is run.
    _, metadata = runner.preflight()
    monkeypatch.setattr(runner, 'load_protocol', lambda: protocol)
    checkpoint = managed_tmp / 'checkpoint.json'
    report = runner.experiment(protocol, metadata, checkpoint)
    assert report['execution_integrity_passed']
    assert not report['component_learning_passed']
    assert len(json.loads(checkpoint.read_text())['states']) == 2
    assert all(row['gates']['saved_checkpoint_replay'] for row in report['runs'])
    assert all(arm['replay']['work']['episodes'] == 4 for row in report['runs'] for arm in row['arms'].values())
