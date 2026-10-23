"""Schema, size proofs and pre-launch refusal without tracing or workers."""
import ast
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest

from sara_engine.evaluation import allocation_report as codec
from sara_engine.evaluation.allocation_report_preflight import constructed_payload, roster


@pytest.fixture(scope='module')
def payload():
    return constructed_payload()


def test_complete_max_call_report_round_trips_without_losing_unknowns(payload):
    raw = codec.encode(payload)
    assert codec.decode(raw) == payload
    assert len(raw) < codec.LIMIT
    table = json.loads(raw)
    assert len(table['sites']) == 3
    assert [2, 'unknown/未知.py:7'] in table['sites']
    assert sum(len(w[-1]) for w in table['workers']) == codec.WORKERS * codec.CALLS


def test_dominating_byte_proof_uses_full_registered_capacity():
    proof = codec.byte_proof(roster())
    assert proof['saturation_witness_bytes'] <= proof['proved_upper_bound_bytes'] < 8388608
    assert proof['conditional_on_domain_validation']
    assert proof['remaining_budget_bytes'] == 8388608 - proof['proved_upper_bound_bytes']


def test_report_modules_cannot_import_tracing_or_execution_components():
    from sara_engine.evaluation import allocation_report_preflight
    for module in (codec, allocation_report_preflight):
        tree = ast.parse(inspect.getsource(module))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(any(word in name for word in ('tracemalloc', 'subprocess', 'event_execution', 'event_allocation', 'neuron')) for name in imports)


@pytest.mark.parametrize('value', [-1, True, 1.5, None, codec.U64 + 1])
def test_bad_integer_domains_are_rejected(payload, value):
    bad = deepcopy(payload)
    bad['workers'][0]['records'][0]['allocation'][codec.SCALARS[3]] = value
    with pytest.raises(ValueError):
        codec.encode(bad)


def test_max_u64_and_escaped_name_boundary_are_supported(payload):
    sample = deepcopy(payload)
    item = sample['workers'][0]['records'][0]['allocation']
    item[codec.SCALARS[3]] = codec.U64
    old = item['post_query_live_sites'].pop('unknown/未知.py:7')
    item['post_query_live_sites']['x'*254] = old
    assert codec.decode(codec.encode(sample)) == sample
    item['post_query_live_sites']['x'*255] = item['post_query_live_sites'].pop('x'*254)
    with pytest.raises(ValueError, match='site name'):
        codec.encode(sample)


def test_unicode_capacity_counts_json_escaping(payload):
    sample = deepcopy(payload)
    item = sample['workers'][0]['records'][0]['allocation']
    old = item['post_query_live_sites'].pop('unknown/未知.py:7')
    item['post_query_live_sites']['漢'*43] = old
    with pytest.raises(ValueError, match='site name'):
        codec.encode(sample)


@pytest.mark.parametrize('kind', ['owner', 'graph', 'peak', 'service', 'identity', 'calls', 'extra', 'bool_identity', 'witness'])
def test_conservation_and_fixed_roster_corruption_are_rejected(payload, kind):
    bad = deepcopy(payload)
    worker = bad['workers'][0]
    record = worker['records'][0]
    if kind == 'owner':
        record['allocation']['post_query_live_by_owner']['other']['bytes'] += 1
    elif kind == 'graph':
        record['ownership']['union_bytes'] += 1
    elif kind == 'peak':
        record['allocation'][codec.SCALARS[2]] += 1
    elif kind == 'service':
        worker['service_receipts']['scalar']['created'] = 0
    elif kind == 'identity':
        worker['mode'] = 'scalar'
    elif kind == 'calls':
        worker['records'].pop()
    elif kind == 'extra':
        record['allocation']['unreported_cost'] = 1
    elif kind == 'bool_identity':
        worker['member'] = False
    else:
        bad['witnesses']['measurements']['ephemeral'].update(zip(codec.SCALARS[:3], (0, 0, 0)))
    with pytest.raises(ValueError):
        codec.encode(bad)


def test_registry_overflow_is_rejected_without_clipping(payload):
    bad = deepcopy(payload)
    item = bad['workers'][0]['records'][0]['allocation']
    item['post_query_live_sites'] = {'unknown'+str(i): {'owner': 'other', 'bytes': 1, 'blocks': 1} for i in range(codec.SITES+1)}
    item['post_query_live_by_owner'] = {o: {'bytes': codec.SITES+1 if o == 'other' else 0,
                                         'blocks': codec.SITES+1 if o == 'other' else 0} for o in codec.OWNERS}
    with pytest.raises(ValueError, match='registry capacity'):
        codec.encode(bad)


def test_incidence_overflow_is_rejected_without_omitting_sites(payload):
    bad = deepcopy(payload)
    for worker in bad['workers']:
        for record in worker['records']:
            item = record['allocation']
            item['post_query_live_sites'] = {'other'+str(i): {'owner': 'other', 'bytes': 1, 'blocks': 1} for i in range(8)}
            item['post_query_live_by_owner'] = {o: {'bytes': 8 if o == 'other' else 0,
                                                 'blocks': 8 if o == 'other' else 0} for o in codec.OWNERS}
    with pytest.raises(ValueError, match='incidence capacity'):
        codec.encode(bad)


@pytest.mark.parametrize('kind', ['duplicate', 'index', 'unused_registry', 'columns', 'truncated', 'oversized'])
def test_corrupt_saved_representations_are_rejected(payload, kind):
    raw = codec.encode(payload)
    table = json.loads(raw)
    if kind == 'duplicate':
        raw = raw.replace(b'"schema":', b'"schema":"bad","schema":', 1)
    elif kind == 'index':
        table['workers'][0][-1][0][-1][0][0] = codec.SITES
        raw = codec.canonical(table)
    elif kind == 'unused_registry':
        table['sites'].append([2, 'unreferenced.py:1'])
        raw = codec.canonical(table)
    elif kind == 'columns':
        table['columns']['record'].reverse()
        raw = codec.canonical(table)
    elif kind == 'truncated':
        raw = raw[:-20]
    else:
        raw = b' '*(codec.LIMIT + 1)
    with pytest.raises(ValueError):
        codec.decode(raw)


@pytest.mark.parametrize('kind', ['missing', 'hash', 'registration', 'execution', 'synthetic', 'incomplete'])
def test_invalid_saved_results_launch_zero_callbacks(payload, tmp_path, kind):
    fixture = deepcopy(payload)
    fixture['metadata']['synthetic'] = kind == 'synthetic'
    raw = codec.encode(fixture)
    if kind == 'incomplete':
        raw = raw[:-10]
    path = tmp_path/'toy_result.json'
    if kind != 'missing':
        path.write_bytes(raw)
    calls = []
    with pytest.raises((ValueError, FileNotFoundError)):
        codec.saved_replication_gate(path, 'f'*64 if kind == 'hash' else sha256(raw).hexdigest(),
                                    'f'*64 if kind == 'registration' else '0'*64,
                                    'f'*64 if kind == 'execution' else '0'*64, calls.append)
    assert calls == []


def test_valid_toy_file_is_checked_before_inert_callback(payload, tmp_path):
    fixture = deepcopy(payload)
    fixture['metadata']['synthetic'] = False  # Test-only provenance stub in workspace.
    raw = codec.encode(fixture)
    path = tmp_path/'toy_result.json'
    path.write_bytes(raw)
    calls = []
    codec.saved_replication_gate(path, sha256(raw).hexdigest(), '0'*64, '0'*64, calls.append)
    assert calls == [fixture]


def test_preserved_failure_and_static_execution_receipt():
    from scripts.eval import allocation_report as runner
    selection = runner.read('selection')
    assert len(selection['preserved']) == 372
    old = json.loads(Path(runner.path('selection').parent/'event_allocation_selection_v1.json').read_text())
    assert selection['fixed_original_task'] == old['fixed_original_task']
    for name, expected in selection['preserved'].items():
        from sara_engine.utils.project_paths import project_path
        assert runner.digest(project_path(name)) == expected
    if not runner.path('result').exists():
        pytest.skip('Retained static preflight is not consumed before code registration')
    runner.verify()
    result = runner.read('result')
    assert result['all_report_preflight_gates_passed']
    assert result['allocation_measurements'] == result['worker_launches'] == 0
    with pytest.raises(FileExistsError):
        runner.main('execute')
