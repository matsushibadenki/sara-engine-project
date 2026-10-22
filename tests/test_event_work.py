"""Accounting conservation, causal receipt boundaries and fair scalar control."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sara_engine.evaluation.event_work import (
    ARMS, UNITS, Backend, Lifecycle, Meter, make_tape, owned_size, run_case,
    validate_lifecycle, verify_charges,
)
from sara_engine.utils.project_paths import project_path, processed_data_path


def protocol():
    return json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_selection_v1.json')).read_text())


def events(family='middle_order', member=0):
    source = json.loads(Path(processed_data_path('benchmark_fixtures', 'temporal_capability_boundary_v1.json')).read_text())
    return tuple(zip(source['templates'][family][member], source['gap_templates'][family][member]))


@pytest.mark.parametrize('scenario', ['on_time', 'late', 'missing', 'mixed'])
def test_real_delivery_expiry_and_stale_timer_work_is_charged(scenario):
    result = run_case(events(), scenario, 16, protocol())
    assert result['gates_passed']
    for arm in ARMS:
        own = result['physical'][arm]
        assert own['expiry.push'] == own['expiry.pop'] == 6
        assert own['backend.step'] == 6
        assert own['feedback.delivered'] + own['feedback.cancelled'] + own['expiry.expired'] == 6
        assert own['expiry.stale'] == own['feedback.delivered'] + own['feedback.cancelled']
        assert own['feedback.pending_insert'] == own['feedback.pending_delete'] == 6
        assert result['actual_learning_writes'][arm] == 0
    assert result['predictions_scored'] == 0


def test_long_idle_time_does_not_hide_expiry_or_force_scalar_polling():
    short = run_case(events(), 'mixed', 16, protocol())
    long = run_case(events(), 'mixed', 1000000000, protocol())
    for owner in short['physical']:
        assert {k: v for k, v in short['physical'][owner].items() if k != 'wait.elapsed'} == {
            k: v for k, v in long['physical'][owner].items() if k != 'wait.elapsed'}
    assert short['semantic_trace_sha256'] == long['semantic_trace_sha256']
    for arm in ARMS:
        assert long['physical'][arm]['wait.elapsed'] == 1000000000
        assert long['physical'][arm]['expiry.pop'] > 0
        assert long['physical'][arm]['backend.step'] == len(events())


def test_observer_is_physically_shared_and_charged_once_to_each_arm():
    result = run_case(events(), 'on_time', 16, protocol())
    physical, charged = result['physical'], result['charged']
    assert physical['shared']['observer.ordinal'] == 6
    assert physical['neuron']['observer.ordinal'] == physical['scalar']['observer.ordinal'] == 0
    assert charged['neuron']['observer.ordinal'] == charged['scalar']['observer.ordinal'] == 6
    verify_charges(physical, charged)
    for factor in (0, 2):
        corrupt = deepcopy(charged)
        corrupt['scalar']['observer.ordinal'] *= factor
        with pytest.raises(ValueError, match='Shared charge'):
            verify_charges(physical, corrupt)


def test_journal_rejects_missing_cost_and_invalid_ownership():
    meter = Meter(4)
    meter.add('shared', 'observer.ordinal')
    meter.totals['shared']['observer.ordinal'] = 0
    with pytest.raises(ValueError, match='journal mismatch'):
        meter.reconcile()
    with pytest.raises(ValueError, match='wrong owner'):
        meter.add('scalar', 'observer.ordinal')
    with pytest.raises(ValueError):
        meter.add('shared', 'observer.ordinal', -1)
    with pytest.raises(ValueError):
        meter.add('shared', 'observer.ordinal', True)


def test_scalar_recurrence_independently_matches_existing_neuron():
    meter = Meter(4096)
    a, b = [Backend(arm, meter, 0.6, 2) for arm in ARMS]
    states = []
    for route, _ in events():
        left, right = a.step(route), b.step(route)
        assert left == right
        states.append(left)
    assert any(state[2] for state in states)
    assert any(state[1] == 1 for state in states)
    assert meter.totals['scalar']['backend.refractory'] > 0
    assert meter.totals['neuron']['backend.step'] == meter.totals['scalar']['backend.step']


def test_boundary_result_is_valid_and_later_duplicate_is_rejected():
    meter = Meter(100)
    life = Lifecycle('scalar', meter, protocol()['budgets'], 4)
    life.create(0, 0, ())
    life.advance(4, False)
    life.notify(0, 4, 'result')
    life.advance(4, True)
    life.advance(5, False)
    life.notify(0, 5, 'result')
    assert [row[1] for row in life.trace] == ['created', 'delivered', 'stale_timer', 'rejected_result']
    assert meter.totals['scalar']['expiry.stale'] == 1


def test_same_time_cancellation_wins_before_result_and_expiry():
    meter = Meter(100)
    life = Lifecycle('scalar', meter, protocol()['budgets'], 4)
    life.create(0, 0, ())
    life.advance(4, False)
    life.notify(0, 4, 'cancel')
    life.notify(0, 4, 'result')
    life.advance(4, True)
    assert [row[1] for row in life.trace] == ['created', 'cancelled', 'rejected_result', 'stale_timer']


def test_incorrect_scalar_backend_fails_semantic_gate(monkeypatch):
    original = Backend.step
    def changed(self, route):
        potential, remaining, spike = original(self, route)
        return (potential, remaining, not spike) if self.arm == 'scalar' else (potential, remaining, spike)
    monkeypatch.setattr(Backend, 'step', changed)
    with pytest.raises(ValueError, match='semantic trace'):
        run_case(events(), 'on_time', 16, protocol())


def test_unrecorded_shared_operation_fails_independent_conservation(monkeypatch):
    original = Meter.add
    def omitted(self, owner, metric, amount=1):
        if metric != 'observer.ordinal':
            original(self, owner, metric, amount)
    monkeypatch.setattr(Meter, 'add', omitted)
    with pytest.raises(ValueError, match='observer work conservation'):
        run_case(events(), 'on_time', 16, protocol())


def test_missing_result_expires_without_any_later_input():
    meter = Meter(100)
    life = Lifecycle('scalar', meter, protocol()['budgets'], 4)
    life.create(0, 0, ())
    callback_times = []
    life.sample = lambda: callback_times.append(life.clock)
    life.advance(1000000000, True)
    assert life.trace[-1] == (4, 'expired', 0)
    assert not life.pending and not life.timers
    assert meter.totals['scalar']['expiry.pop'] == 1
    assert meter.totals['scalar']['backend.step'] == 0
    assert callback_times == [4]
    assert meter.totals['scalar']['wait.timer_dispatch'] == 1
    assert meter.totals['scalar']['wait.elapsed'] == 1000000000


def test_noncausal_delivery_and_clock_regression_are_rejected():
    life = Lifecycle('scalar', Meter(100), protocol()['budgets'], 4)
    life.create(0, 0, ())
    with pytest.raises(ValueError, match='Noncausal'):
        life.notify(0, 0, 'result')
    life.advance(2, False)
    with pytest.raises(ValueError, match='Clock'):
        life.advance(1, True)
    with pytest.raises(ValueError, match='notification'):
        life.notify(0, 3, 'result')


@pytest.mark.parametrize('budget', ['pending_per_arm', 'timers_per_arm', 'journal_entries_per_case', 'sampled_owned_bytes'])
def test_bounds_fail_instead_of_dropping_work(budget):
    p = protocol()
    p['budgets'][budget] = 1
    with pytest.raises(ValueError, match='budget'):
        run_case(events(), 'missing', 16, p)


def test_journal_cap_fails_before_counter_mutation():
    meter = Meter(1)
    meter.add('shared', 'input.decode')
    with pytest.raises(ValueError, match='budget'):
        meter.add('shared', 'input.decode')
    assert meter.totals['shared']['input.decode'] == 1


def test_sampled_memory_deduplicates_shared_objects_and_visits_deques():
    from collections import deque
    value = [bytearray(200)]
    shared, visits = owned_size((value, value))
    separate, _ = owned_size(([bytearray(200)], [bytearray(200)]))
    assert shared < separate and visits > 0
    assert owned_size(deque(value))[0] > sys.getsizeof(deque(value))


def test_independent_oracle_catches_duplicate_terminal_and_future_input():
    e = events('first_order')
    trace = []
    time = 0
    for i, (_, gap) in enumerate(e):
        time += gap
        trace.append((time, 'created', i, (((), e), (0.0, 0, False))))
        trace.append((time + 4, 'delivered', i))
    with pytest.raises(ValueError, match='noncausal'):
        validate_lifecycle(trace, e, 'on_time')
    trace.append((4, 'expired', 0))
    with pytest.raises(ValueError, match='exactly once'):
        validate_lifecycle(trace, e, 'on_time')


def test_results_are_opaque_and_transport_is_bounded():
    tape = make_tape(events(), 'mixed')
    assert all(payload is None for _, _, _, kind, payload in tape if kind != 'input')
    assert {kind for _, _, _, kind, _ in tape} == {'input', 'result', 'cancel'}
    assert len(tape) <= 3 * len(events()) + 1
    assert all(key.split('.')[0] in ('input', 'observer', 'backend', 'probe', 'feedback', 'expiry', 'wait', 'teardown', 'harness') for key in UNITS)


def test_frozen_execution_and_fresh_process_replay():
    result_path = Path(processed_data_path('benchmark_fixtures', 'event_work_result_v1.json'))
    if not result_path.exists():
        pytest.skip('Finite accounting execution not consumed')
    command = [sys.executable, project_path('scripts/eval/event_work.py')]
    replay = subprocess.run(command + ['replay'], cwd=project_path(), capture_output=True, text=True)
    assert replay.returncode == 0, replay.stderr
    result = json.loads(result_path.read_text())
    assert result['case_count'] == 48 and result['all_accounting_gates_passed']
    assert result['cpu_seconds'] is None and result['energy_joules'] is None
    assert not result['snn_resource_advantage_established']
    repeated = subprocess.run(command + ['execute'], cwd=project_path(), capture_output=True, text=True)
    assert repeated.returncode != 0 and 'FileExistsError' in repeated.stderr
