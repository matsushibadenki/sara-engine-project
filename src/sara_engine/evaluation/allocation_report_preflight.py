"""Static report proof using frozen inputs and constructed, unmeasured numbers."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

from sara_engine.evaluation import allocation_report as codec
from sara_engine.utils.project_paths import project_path, processed_data_path


def roster():
    source = json.loads(Path(processed_data_path('benchmark_fixtures', 'temporal_capability_boundary_v1.json')).read_text())
    prior = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_separation_result_v1.json')).read_text())
    rows = []
    for old in prior['cases']:
        row = {k: old[k] for k in ('family', 'member', 'scenario', 'horizon', 'input_sha256')}
        row['input_events'] = len(source['templates'][row['family']][row['member']])
        rows.append(row)
    codec.roster_check(rows)
    return rows


def constructed_payload(rows=None):
    """A schema test fixture, never an allocation observation or recovered run."""
    rows = roster() if rows is None else rows
    metric = dict(zip(codec.SCALARS, (0, 1, 1, 17)))
    metric['post_query_live_by_owner'] = {o: {'bytes': i+1, 'blocks': i+1} for i, o in enumerate(codec.OWNERS)}
    metric['post_query_live_sites'] = {name: {'owner': o, 'bytes': i+1, 'blocks': i+1} for i, (o, name) in enumerate(zip(
        codec.OWNERS, ('src/sara_engine/evaluation/event_execution.py:1', 'measurement.py:1', 'unknown/未知.py:7')))}
    graph = dict(zip(codec.GRAPH, (11, 7, 5, 23, 16, 12)))
    workers = []
    for case in rows:
        for mode in codec.MODES:
            arms = ('neuron', 'scalar') if mode == 'paired' else (mode,)
            # Exercise the registered maximum representation, not an execution.
            stages = ['initialize'] + ['input']*case['input_events'] + ['advance']*(codec.CALLS-case['input_events']-2) + ['close']
            workers.append({**{k: case[k] for k in ('family', 'member', 'scenario', 'horizon', 'input_sha256')},
                'mode': mode, **{k: '0'*64 for k in codec.HASHES if k != 'input_sha256'},
                'shared_input_calls': case['input_events'],
                'service_receipts': {arm: dict(zip(codec.COUNTS, codec.expected_counts(case))) for arm in arms},
                'recorded_calls': codec.CALLS, 'all_semantic_gates_passed': True,
                'records': [{'stage': stage, 'allocation': deepcopy(metric), 'ownership': dict(graph)} for stage in stages]})
    measurements = {k: deepcopy(metric) for k in ('empty', 'ephemeral', 'retained', 'preexisting')}
    measurements['ephemeral'].update(zip(codec.SCALARS[:3], (0, 65536, 65536)))
    measurements['retained'].update(zip(codec.SCALARS[:3], (65536, 65536, 0)))
    return {'metadata': {'synthetic': True, 'registration_sha256': '0'*64, 'execution_sha256': '0'*64,
                         'prior_failure_sha256': sha256(Path(processed_data_path('benchmark_fixtures', 'event_allocation_failure_v1.json')).read_bytes()).hexdigest(),
                         'runtime': ['3.10.20', 'CPython', 'arm64', 'little']},
            'roster': rows, 'workers': workers,
            'witnesses': {'buffer_bytes': 65536, 'passed': True, 'measurements': measurements}}


def evaluate(selection):
    expected = {'report_bytes': codec.LIMIT, 'workers': codec.WORKERS, 'calls_per_worker': codec.CALLS,
                'sites': codec.SITES, 'site_json_bytes': codec.SITE_BYTES, 'site_incidences': codec.INCIDENCES,
                'unsigned_integer_max': codec.U64}
    if selection['limits'] != expected:
        raise ValueError('Registered report limits changed')
    rows = roster()
    payload = constructed_payload(rows)
    raw = codec.encode(payload)
    if codec.decode(raw) != payload:
        raise ValueError('Constructed report round-trip mismatch')
    proof = codec.byte_proof(rows)
    inventory = []
    for name in selection['source_inventory']:
        content = Path(project_path(name)).read_bytes()
        inventory.append({'path': name, 'sha256': sha256(content).hexdigest(),
                          'lines': len(content.splitlines()), 'site_example_json_bytes': len(codec.canonical(name + ':1'))-1})
    if any(item['site_example_json_bytes'] > codec.SITE_BYTES for item in inventory):
        raise ValueError('Frozen source name cannot fit the report registry')
    return {'schema': 'sara-allocation-report-preflight-result-v1', 'all_report_preflight_gates_passed': True,
            'byte_proof': proof, 'constructed_report_bytes': len(raw), 'constructed_report_sha256': sha256(raw).hexdigest(),
            'constructed_records': codec.WORKERS*codec.CALLS, 'constructed_report_is_measurement': False,
            'lossless_round_trip': True, 'unknown_sites_preserved': True, 'source_inventory': inventory,
            'prior_artifacts_preserved': len(selection['preserved']), 'allocation_measurements': 0, 'worker_launches': 0,
            'actual_learning_writes': 0, 'predictions_scored': 0,
            'cpu_seconds': None, 'energy_joules': None, 'rss_bytes': None, 'total_transient_peak_bytes': None,
            'future_site_capacity_certified': False, 'runtime_promotion_authorized': False,
            'snn_resource_advantage_established': False, 'test_opening_authorized': False,
            'decision': 'bounded_report_representation_verified_no_measurement_authorized'}
