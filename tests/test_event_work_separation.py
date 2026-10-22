"""Measurement isolation, causal equivalence and honest transient ownership."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sara_engine.evaluation import event_execution as core
from sara_engine.evaluation import event_work_separation as audit
from sara_engine.evaluation import event_work as prior
from sara_engine.utils.project_paths import processed_data_path, project_path


def fixture(suffix):
    return json.loads(Path(processed_data_path('benchmark_fixtures', suffix + '_v1.json')).read_text())


def protocol():
    return fixture('event_work_separation_selection')


EVENTS = ((0, 0), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1))


@pytest.mark.parametrize('scenario', prior.SCENARIOS)
def test_every_boundary_and_receipt_matches_frozen_reference(scenario):
    golden = next(row for row in fixture('event_work_result')['cases']
                  if (row['family'], row['member'], row['scenario'], row['horizon']) == ('middle_order', 0, scenario, 16))
    result = audit.run_case(EVENTS, scenario, 16, protocol(), golden)
    assert result['gates_passed'] and result['prior_receipt_hash_matched']
    assert result['bare_matches_diagnostic'] and result['reference_matches_every_boundary']
    for arm in core.ARMS:
        assert result['service_volumes'][arm]['expiry.pop'] == 6
        assert result['charged_service_volumes'][arm]['observer.ordinal'] == 6
        assert result['sampled_owned_bytes'][arm] > result['sampled_owned_bytes']['shared']


def test_core_imports_and_state_have_no_diagnostic_dependency():
    tree = ast.parse(inspect.getsource(core))
    dependencies = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert dependencies == {'collections', 'sara_engine.evaluation.temporal_information',
                            'sara_engine.evaluation.temporal_capability_boundary', 'sara_engine.neuro.neuron'}
    assert {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} == {'math'}
    forbidden = {'meter', 'journal', 'trace', 'sample', 'callback', 'owned_size', 'fingerprint'}
    assert not forbidden & {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    engine = core.Executor()
    for obj in (engine, engine.shared, *engine.backends.values(), *engine.queues.values()):
        assert not hasattr(obj, '__dict__')
        assert not forbidden & set(obj.__slots__)


def test_receipt_only_path_runs_with_diagnostics_poisoned(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostic entered receipt-only path')
    for obj, name in ((audit, 'memory_sample'), (audit, 'inventory'), (prior, 'owned_size'),
                      (prior, 'fingerprint'), (prior.Meter, 'add')):
        monkeypatch.setattr(obj, name, forbidden)
    _, operations = audit.schedule(EVENTS, 'mixed', 16)
    traces, final, closed = audit.bare_run(operations)
    assert traces['neuron'] == traces['scalar']
    assert not final['closed'] and closed['closed']


def test_graph_probe_runs_outside_mutating_stack_and_is_pure(monkeypatch):
    original = audit.memory_sample
    calls = []
    def check(executor, batch):
        assert not any(frame.function in ('apply', 'step', 'observe', 'advance', 'notify', 'create')
                       for frame in inspect.stack())
        before = audit.snapshot(executor)
        result = original(executor, batch)
        assert before == audit.snapshot(executor)
        calls.append(1)
        return result
    monkeypatch.setattr(audit, 'memory_sample', check)
    result = audit.run_case(EVENTS, 'mixed', 16, protocol())
    assert len(calls) == result['boundary_calls'] + 1


def test_shared_observer_really_executes_once_and_backend_only_on_arrival(monkeypatch):
    counts = {'observer': 0, 'neuron': 0, 'scalar': 0}
    observe, step = core.SharedInput.observe, core.Backend.step
    def watched_observe(self, *args):
        counts['observer'] += 1
        return observe(self, *args)
    def watched_step(self, *args):
        counts[self.arm] += 1
        return step(self, *args)
    monkeypatch.setattr(core.SharedInput, 'observe', watched_observe)
    monkeypatch.setattr(core.Backend, 'step', watched_step)
    _, operations = audit.schedule(EVENTS, 'mixed', 1000000000)
    audit.bare_run(operations)
    assert counts == {'observer': 6, 'neuron': 6, 'scalar': 6}


def test_internal_backend_corruption_fails_even_if_returned_receipt_matches(monkeypatch):
    step = core.Backend.step
    def corrupt(self, route):
        result = step(self, route)
        if self.arm == 'neuron':
            self.cells[route].branches[0].is_active = True
        return result
    monkeypatch.setattr(core.Backend, 'step', corrupt)
    with pytest.raises(ValueError, match='Boundary state mismatch'):
        audit.run_case(EVENTS, 'on_time', 16, protocol())


def test_dropped_receipt_and_corrupted_prior_hash_fail(monkeypatch):
    advance = core.ReceiptQueue.advance
    def drop(self, *args):
        return advance(self, *args)[1:]
    monkeypatch.setattr(core.ReceiptQueue, 'advance', drop)
    with pytest.raises(ValueError, match='Boundary receipt mismatch'):
        audit.run_case(EVENTS, 'missing', 16, protocol())
    monkeypatch.undo()
    golden = deepcopy(fixture('event_work_result')['cases'][0])
    golden['semantic_trace_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Prior saved receipt trace'):
        audit.run_case(EVENTS, 'on_time', 16, protocol(), golden)


def test_deadline_result_and_cancel_tie_keep_stale_timer():
    queue = core.ReceiptQueue()
    queue.create(0, 0, ())
    assert queue.advance(4, False) == []
    assert queue.notify(0, 4, 'cancel') == [(4, 'cancelled', 0)]
    assert queue.notify(0, 4, 'result') == [(4, 'rejected_result', 0)]
    assert queue.advance(4, True) == [(4, 'stale_timer', 0)]
    other = core.ReceiptQueue()
    other.create(0, 0, ())
    other.advance(4, False)
    assert other.notify(0, 4, 'result') == [(4, 'delivered', 0)]
    assert other.advance(5, True) == [(4, 'stale_timer', 0)]
    assert other.notify(0, 5, 'result') == [(5, 'rejected_result', 0)]


def test_functional_batch_cap_and_causal_expiry_times():
    queue = core.ReceiptQueue()
    for index in range(8):
        queue.create(index, 0, ())
    with pytest.raises(ValueError, match='budget'):
        queue.create(8, 0, ())
    rows = queue.advance(1000000000, True)
    assert len(rows) == 8 and all(row[0] == 4 for row in rows)
    assert not queue.pending and not queue.timers
    assert not hasattr(queue, 'trace') and queue.clock == 1000000000


def test_returned_batch_is_counted_and_diagnostic_buffers_are_not_owned():
    engine = core.Executor()
    batch = engine.apply(('input', (0, 0, 0, 0)))
    assert engine.queues['neuron'].pending[0][1][0] is engine.queues['scalar'].pending[0][1][0]
    assert batch['neuron'][0][3][0] is engine.shared.packet
    sampled, _ = audit.memory_sample(engine, batch)
    empty, _ = audit.memory_sample(engine, {arm: [] for arm in core.ARMS})
    assert all(sampled[arm] > empty[arm] for arm in (*core.ARMS, 'physical'))
    retained_trace = list(batch['neuron']) * 1000
    again, _ = audit.memory_sample(engine, batch)
    assert sampled == again and len(retained_trace) == 1000
    engine.apply(('advance', (4, True)))
    engine.close()
    after, _ = audit.memory_sample(engine, batch)
    released, _ = audit.memory_sample(engine, {arm: [] for arm in core.ARMS})
    assert after['physical'] > released['physical']  # Returned packets survive teardown until consumed.


def test_scope_does_not_mislabel_structural_caps_as_transient_byte_peak():
    scope = audit.MEMORY_SCOPE
    assert scope['total_transient_peak_bytes'] is None
    assert all(site['peak_bytes'] is None for site in scope['between_boundary_sites'])
    assert 'frames' in scope['unknown'] and 'walker' in scope['harness']
    engine = core.Executor()
    with pytest.raises(ValueError, match='occupancy'):
        audit.inventory(engine, {arm: [(0, 'expired', i) for i in range(9)] for arm in core.ARMS})


def test_missing_prior_service_charge_is_rejected():
    golden = deepcopy(next(row for row in fixture('event_work_result')['cases']
                           if (row['family'], row['member'], row['scenario'], row['horizon']) == ('middle_order', 0, 'mixed', 16)))
    golden['physical']['shared']['observer.ordinal'] = 0
    with pytest.raises(ValueError, match='Prior service volume mismatch'):
        audit.run_case(EVENTS, 'mixed', 16, protocol(), golden)


def test_seventh_input_is_rejected_without_changing_state():
    _, operations = audit.schedule(EVENTS, 'mixed', 16)
    engine = core.Executor()
    for operation in operations:
        engine.apply(operation)
    before = audit.snapshot(engine)
    with pytest.raises(ValueError, match='capacity'):
        engine.apply(('input', (6, 16, 0, 11)))
    assert before == audit.snapshot(engine)


def test_idle_extension_keeps_receipts_occupancy_and_service_work():
    short = audit.run_case(EVENTS, 'mixed', 16, protocol())
    long = audit.run_case(EVENTS, 'mixed', 1000000000, protocol())
    for key in ('semantic_trace_sha256', 'nonclock_final_state_sha256', 'service_volumes', 'peak_occupancy', 'boundary_calls'):
        assert short[key] == long[key]
    assert short['final_state_sha256'] != long['final_state_sha256']  # The logical clock remains observable.


@pytest.mark.parametrize('bad', [('advance', (-1, True)), ('advance', (1, 1)),
                                ('input', (0, 0, 2, 0)), ('input', (0, 0, 0, 1)),
                                ('input', (0, 1, 0, 0)), ('other', ())])
def test_invalid_operations_are_rejected_before_valid_run(bad):
    engine = core.Executor()
    before = audit.snapshot(engine)
    with pytest.raises(ValueError):
        engine.apply(bad)
    assert audit.snapshot(engine) == before


def test_teardown_requires_drained_queue_and_closed_executor_rejects_calls():
    engine = core.Executor()
    engine.apply(('input', (0, 0, 0, 0)))
    with pytest.raises(ValueError, match='undrained'):
        engine.close()
    engine.apply(('advance', (4, True)))
    engine.close()
    with pytest.raises(ValueError, match='closed'):
        engine.apply(('advance', (5, True)))


def test_preservation_and_consume_once_runner():
    selection = protocol()
    assert len(selection['preserved']) == 363
    assert selection['fixed_original_task'] == fixture('event_work_selection')['fixed_original_task']
    from scripts.eval import event_work_separation as runner
    for name, digest in selection['preserved'].items():
        assert runner.digest(project_path(name)) == digest
    if not runner.path('result').exists():
        pytest.skip('Finite execution is not consumed before code registration')
    runner.verify()
    result = fixture('event_work_separation_result')
    assert result['case_count'] == 48 and result['all_separation_gates_passed']
    assert result['actual_learning_writes'] == result['predictions_scored'] == 0
    assert result['total_transient_peak_bytes'] is None
    assert not result['runtime_promotion_authorized'] and not result['test_opening_authorized']
    proc = subprocess.run([sys.executable, 'scripts/eval/event_work_separation.py', 'execute'],
                          cwd=project_path(), capture_output=True, text=True)
    assert proc.returncode != 0 and 'FileExistsError' in proc.stderr
