"""Isolated allocation scope, semantic equivalence and retained evidence gates."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import sys
import tracemalloc

import pytest

from sara_engine.evaluation import event_allocation as audit
from sara_engine.evaluation.event_allocation_execution import SingleArmExecutor
from sara_engine.evaluation.event_execution import SharedInput, Backend
from sara_engine.evaluation.event_work_separation import schedule
from sara_engine.utils.project_paths import processed_data_path, project_path


EVENTS = ((0, 0), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1))


def fixture(name):
    return json.loads(Path(processed_data_path('benchmark_fixtures', name + '_v1.json')).read_text())


def golden(scenario='mixed', horizon=16):
    return next(row for row in fixture('event_work_separation_result')['cases']
                if (row['family'], row['member'], row['scenario'], row['horizon']) == ('middle_order', 0, scenario, horizon))


@pytest.mark.parametrize('mode', ['paired', 'neuron', 'scalar'])
@pytest.mark.parametrize('scenario', ['on_time', 'late', 'missing', 'mixed'])
def test_traced_calls_match_every_original_boundary_and_receipt(mode, scenario):
    row = audit.run_case(EVENTS, golden(scenario), mode)
    assert row['all_semantic_gates_passed'] and row['shared_input_calls'] == 6
    assert row['receipt_sha256'] == golden(scenario)['semantic_trace_sha256']
    assert len(row['records']) == row['recorded_calls'] <= 66
    assert row['records'][0]['stage'] == 'initialize' and row['records'][-1]['stage'] == 'close'
    for arm, counts in row['service_receipts'].items():
        assert counts['created'] == counts['delivered'] + counts['cancelled'] + counts['expired'] == 6
        assert counts['stale_timer'] + counts['expired'] == 6
        assert arm in ('neuron', 'scalar')
    for record in row['records']:
        allocation, owned = record['allocation'], record['ownership']
        assert allocation['new_peak_bytes'] >= allocation['new_live_at_return_bytes'] >= 0
        assert allocation['peak_excess_over_return_bytes'] == allocation['new_peak_bytes'] - allocation['new_live_at_return_bytes']
        assert owned['union_bytes'] == owned['executor_only_bytes'] + owned['output_only_bytes'] + owned['shared_reachable_bytes']
    assert not tracemalloc.is_tracing()


def test_known_temporary_retained_and_preexisting_witnesses():
    result = audit.witnesses()
    assert result['passed'] and result['buffer_bytes'] == 65536
    rows = result['measurements']
    assert rows['ephemeral']['peak_excess_over_return_bytes'] >= 65536
    assert rows['retained']['new_live_at_return_bytes'] >= 65536
    assert rows['preexisting']['new_live_at_return_bytes'] < 65536
    assert rows['empty']['new_peak_bytes'] >= 0


@pytest.mark.parametrize('gc_enabled', [True, False])
def test_trace_and_gc_cleanup_on_failed_call(gc_enabled):
    was_enabled = gc.isenabled()
    gc.enable() if gc_enabled else gc.disable()
    def fail():
        raise RuntimeError('Synthetic failure')
    try:
        with pytest.raises(RuntimeError, match='Synthetic failure'):
            audit.measure(fail)
        assert gc.isenabled() == gc_enabled and not tracemalloc.is_tracing()
    finally:
        gc.enable() if was_enabled else gc.disable()


def test_preexisting_tracer_is_rejected_without_disabling_it():
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match='untraced process'):
            audit.measure(audit.empty)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_peak_cap_is_enforced_and_tracer_stops():
    with pytest.raises(ValueError, match='peak outside budget'):
        audit.measure(bytearray, (2 * 1048576,))
    assert not tracemalloc.is_tracing()


def test_alias_partition_does_not_double_charge_retained_packets():
    state = {'packet': ((0, 0), (1, 1))}
    same = audit.ownership(state, state)
    assert same['executor_only_bytes'] == same['output_only_bytes'] == 0
    assert same['shared_reachable_bytes'] == same['union_bytes']
    output = {'packet': state['packet'], 'buffer': bytearray(65536)}
    parts = audit.ownership(state, output)
    assert parts['output_only_bytes'] >= 65536 and parts['shared_reachable_bytes'] > 0
    assert parts['union_bytes'] == parts['executor_graph_bytes'] + parts['output_graph_bytes'] - parts['shared_reachable_bytes']
    assert audit.ownership(state, None)['output_graph_bytes'] == 0


def test_graph_analysis_is_rejected_inside_traced_window():
    with pytest.raises(ValueError, match='after tracing stops'):
        audit.measure(audit.ownership, ({}, {}))
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize('arm', ['neuron', 'scalar'])
def test_each_standalone_arm_really_executes_shared_observer(monkeypatch, arm):
    counts = {'shared': 0, 'backend': 0}
    observe, step = SharedInput.observe, Backend.step
    def counted_observe(self, *args):
        counts['shared'] += 1
        return observe(self, *args)
    def counted_step(self, *args):
        counts['backend'] += 1
        return step(self, *args)
    monkeypatch.setattr(SharedInput, 'observe', counted_observe)
    monkeypatch.setattr(Backend, 'step', counted_step)
    core = SingleArmExecutor(arm)
    _, operations = schedule(EVENTS, 'mixed', 1000000000)
    for op in operations:
        core.apply(op)
    core.close()
    assert counts == {'shared': 6, 'backend': 6}
    assert set(core.backends) == set(core.queues) == {arm}


def test_idle_tail_keeps_semantics_without_a_peak_ranking_gate():
    short = audit.run_case(EVENTS, golden(), 'scalar')
    long = audit.run_case(EVENTS, golden(horizon=1000000000), 'scalar')
    for key in ('receipt_sha256', 'recorded_calls', 'shared_input_calls', 'service_receipts'):
        assert short[key] == long[key]
    assert short['final_state_sha256'] != long['final_state_sha256']


def test_bad_prior_state_hash_stops_before_tracing(monkeypatch):
    bad = deepcopy(golden())
    bad['state_trace_sha256'] = '0' * 64
    def forbidden(*args):
        pytest.fail('Tracing entered after a bad reference')
    monkeypatch.setattr(audit, 'measure', forbidden)
    with pytest.raises(ValueError, match='reference identity'):
        audit.run_case(EVENTS, bad, 'scalar')


def test_adapter_corruption_is_detected(monkeypatch):
    original = SingleArmExecutor.apply
    def corrupt(self, op):
        batch = original(self, op)
        if op[0] == 'input':
            self.shared.ordinal.event_count += 1
        return batch
    monkeypatch.setattr(SingleArmExecutor, 'apply', corrupt)
    with pytest.raises(ValueError, match='Traced boundary state'):
        audit.run_case(EVENTS, golden(), 'scalar')


def test_live_snapshot_owners_conserve_and_unknown_sites_remain_explicit():
    _, allocation = audit.measure(audit.retained, (4096,))
    assert allocation['post_query_live_by_owner']['measurement']['bytes'] >= 4096
    sites = allocation['post_query_live_sites']
    assert sum(item['bytes'] for item in sites.values()) == sum(item['bytes'] for item in allocation['post_query_live_by_owner'].values())
    assert audit.allocation_site('/foreign/unknown.py', 3) == ('other', 'unknown.py:3')


def test_semantic_replication_does_not_hide_ownership_changes():
    result = {'workers': [{'mode': 'scalar', 'records': [{'stage': 'input', 'allocation': {'new_peak_bytes': 10},
                                                       'ownership': {'union_bytes': 100}}]}],
              'witnesses': {'passed': True, 'measurements': {'empty': 0}}}
    other = deepcopy(result)
    other['workers'][0]['records'][0]['allocation']['new_peak_bytes'] = 12
    assert audit.semantic_result(result) == audit.semantic_result(other)
    other['workers'][0]['records'][0]['ownership']['union_bytes'] += 1
    assert audit.semantic_result(result) != audit.semantic_result(other)


def test_preserved_artifacts_fixed_conditions_and_consumed_execution():
    from scripts.eval import event_allocation as runner
    selection = fixture('event_allocation_selection')
    assert len(selection['preserved']) == 368
    assert selection['fixed_original_task'] == fixture('event_work_selection')['fixed_original_task']
    for name, digest in selection['preserved'].items():
        assert runner.digest(project_path(name)) == digest
    if not runner.path('result').exists():
        pytest.skip('Finite diagnostic remains unconsumed before registration')
    runner.verify()
    result = runner.read('result')
    assert result['worker_count'] == 144 and result['case_count'] == 48
    assert result['actual_learning_writes'] == result['predictions_scored'] == 0
    assert result['total_transient_peak_bytes'] is None and result['rss_bytes'] is None
    assert not result['runtime_promotion_authorized'] and not result['test_opening_authorized']
    proc = subprocess.run([sys.executable, 'scripts/eval/event_allocation.py', 'execute'],
                          cwd=project_path(), capture_output=True, text=True)
    assert proc.returncode != 0 and 'FileExistsError' in proc.stderr
