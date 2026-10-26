#!/usr/bin/env python3
"""Separate fixed-condition v2 execution; never resumes the old runner."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.reactivation_codec import loads
from sara_engine.evaluation.route_numeric_v2 import (
    PROTOCOL_SHA256, canonical, compare, payload, record, validate_record,
)
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory

BUDGETS = {'phases': 2, 'cases_per_phase': 20, 'arms_per_case': 5, 'report_bytes': 2097152,
           'auxiliary_bytes': 1048576, 'all_artifacts_bytes': 6291456}


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'route_numeric_v2_'+suffix+'_v1.json'))


def old_path(name):
    return Path(processed_data_path('benchmark_fixtures', name))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read_file(file):
    with Path(file).open('rb') as stream:
        raw = stream.read(BUDGETS['report_bytes']+1)
    return loads(raw)


def read(suffix):
    return read_file(path(suffix))


def save(suffix, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    # Every emitted object is already a strict primitive tree; no tuple coercion.
    canonical(value)
    files = list(path('selection').parent.glob('route_numeric_v2_*_v1.json'))
    auxiliary = suffix not in ('primary', 'replication')
    used = sum(f.stat().st_size for f in files)
    aux = sum(f.stat().st_size for f in files if f != path('primary') and f != path('replication'))
    if len(raw) > (BUDGETS['auxiliary_bytes'] if auxiliary else BUDGETS['report_bytes']) or used+len(raw) > BUDGETS['all_artifacts_bytes']:
        raise ValueError('Registered report/total budget exceeded')
    if auxiliary and aux+len(raw) > BUDGETS['auxiliary_bytes']:
        raise ValueError('Registered aggregate auxiliary budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def runtime():
    return {'implementation': platform.python_implementation(), 'python': platform.python_version(),
            'version': sys.version, 'machine': platform.machine(), 'executable': sys.executable}


def verify():
    if digest(path('selection')) != PROTOCOL_SHA256 or read('selection')['budgets'] != BUDGETS:
        raise ValueError('Fixed v2 protocol/budget changed')
    selection, registration = read('selection'), read('registration')
    for pins in (selection['preserved'], registration['sources']):
        for name, expected in pins.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned source/evidence changed: '+name)
    original = read_file(old_path('sparse_route_reactivation_selection_v1.json'))
    spec = {key: value for key, value in original.items()
            if key not in {'schema', 'registered_at_utc', 'authorization', 'preserved', 'stop'}}
    if canonical(spec) != canonical(selection['frozen_scientific_spec']):
        raise ValueError('Original scientific conditions differ')
    if canonical(registration['runtime']) != canonical(runtime()) or runtime()['python'] != '3.10.22' or runtime()['implementation'] != 'CPython':
        raise ValueError('Use the exact registered CPython runtime')
    if old_path('sparse_route_reactivation_replay_v1.json').exists():
        raise ValueError('Old consumed failed replay was reclassified')
    failure = read_file(old_path('sparse_route_reactivation_failure_v1.json'))
    if failure['full_replay_verified'] is not False or failure['numeric_trace_equality'] is not None:
        raise ValueError('Old replay failure changed')
    preflight = read_file(old_path('reactivation_codec_result_v1.json'))
    if preflight['codec_preflight_passed'] is not True or not all(v is True for v in preflight['checks'].values()):
        raise ValueError('Validated comparator preflight required')


def marker(phase):
    value = read(phase+'_execution')
    if set(value) != {'started_at_utc', 'pid', 'registration_sha256', 'selection_sha256'}:
        raise ValueError('Unexpected phase marker fields')
    if value['registration_sha256'] != digest(path('registration')) or value['selection_sha256'] != PROTOCOL_SHA256:
        raise ValueError('Phase marker binding changed')
    if type(value['pid']) is not int or value['pid'] <= 0:
        raise ValueError('Expected actual positive process identity')
    if datetime.fromisoformat(value['started_at_utc']).utcoffset() is None:
        raise ValueError('Timezone-aware phase timestamp required')
    return value


def context(phase):
    marker(phase)
    return {'phase': phase, 'registration_sha256': digest(path('registration')),
            'execution_sha256': digest(path(phase+'_execution')), 'runtime': runtime(),
            'preserved_count': len(read('selection')['preserved']),
            'historical_report_sha256': digest(old_path(read('selection')['reference_report']))}


def baseline():
    return read_file(old_path(read('selection')['reference_report']))


def validated(phase):
    return validate_record(read(phase), baseline(), context(phase))


def run_study():
    # Lazy import: missing/invalid saved-result gates run before any learner path.
    from sara_engine.evaluation.sparse_route_reactivation import run_case, summarize
    spec = read('selection')['frozen_scientific_spec']
    cases = []
    for task in spec['tasks']:
        for seed in spec['seeds']:
            for condition in spec['conditions']:
                case = run_case(seed, task, condition)
                cases.append(case)
                print(json.dumps({'task': task, 'seed': seed, 'condition': condition,
                                  'failed_gates': [k for k, v in case['component_gates'].items() if not v]}), flush=True)
    if len(cases) != 20 or any(len(case['arms']) != 5 for case in cases):
        raise ValueError('Fixed case/arm roster differs')
    return payload(cases, summarize(cases), baseline(), digest(path('registration')), len(read('selection')['preserved']))


def proof():
    primary, replica = validated('primary'), validated('replication')
    if marker('primary')['pid'] == marker('replication')['pid']:
        raise ValueError('Independent fresh process required')
    if not compare(primary, replica, baseline()):
        raise ValueError('Full numerical payload comparison differs')
    return {'schema': 'sara-route-numeric-reproduction-proof-v2', 'numerical_reproduction_passed': True,
            'primary_replication_exact_match': True, 'historical_values_exact_match': True,
            'independent_processes': True, 'primary_sha256': digest(path('primary')),
            'replication_sha256': digest(path('replication')),
            'selection_sha256': PROTOCOL_SHA256, 'registration_sha256': digest(path('registration')),
            'payload_canonical_sha256': primary['payload_canonical_sha256'],
            'cases_per_phase': 20, 'arms_per_case': 5,
            'actual_route_updates_per_arm_case': 9216, 'actual_learning_field_changes_per_arm_case': 36864,
            'component_passed': primary['payload']['component_passed'],
            'old_failed_replay_reclassified': False, 'old_discarded_trace_equality': None,
            'conditions_relaxed': False, 'test_opened': False, 'resource_measurements': 0,
            'new_real_data_reads': 0, 'r2_independent_usefulness_established': False,
            'promotion_authorized': False}


def main(mode):
    if mode == 'register':
        names = ['scripts/eval/route_numeric_v2.py', 'src/sara_engine/evaluation/route_numeric_v2.py',
                 'tests/test_route_numeric_v2.py', str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                             'runtime': runtime(), 'sources': {name: digest(project_path(name)) for name in names}})
        verify()
        print('Separate fixed-condition numerical reproduction v2 registered')
        return
    verify()
    if mode == 'verify':
        expected = proof()
        if canonical(read('result')) != canonical(expected):
            raise ValueError('Immutable numerical proof differs')
        print(json.dumps(expected, sort_keys=True))
        return
    phase = 'primary' if mode == 'execute' else 'replication'
    if path('primary_failure').exists() or path('replication_failure').exists():
        raise ValueError('Consumed failed v2 attempt cannot restart')
    if phase == 'replication':
        validated('primary')
        if marker('primary')['pid'] == os.getpid():
            raise ValueError('Replication requires a fresh process')
    save(phase+'_execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                             'registration_sha256': digest(path('registration')), 'selection_sha256': PROTOCOL_SHA256})
    try:
        observed = run_study()
        report = record(observed, baseline(), context(phase))
        save(phase, report)
        verify()
        validated(phase)
        if phase == 'replication':
            save('result', proof())
    except Exception as exc:
        save(phase+'_failure', {'status': 'failed_consumed_v2_phase', 'phase': phase,
                               'error_type': type(exc).__name__, 'error': str(exc),
                               'execution_sha256': digest(path(phase+'_execution')),
                               'report_retained': path(phase).exists(), 'retry_authorized': False})
        raise
    print(json.dumps({'phase': phase, 'historical_values_exact_match': report['historical_values_exact_match'],
                      'component_passed': report['payload']['component_passed']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replicate', 'verify'))
    main(parser.parse_args().mode)
