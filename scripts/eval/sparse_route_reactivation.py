#!/usr/bin/env python3
"""One frozen reactivation component run and deterministic replay."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.sparse_route_reactivation import ARMS, PROTOCOL_SHA256, run_case, summarize
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_bytes())


def save(suffix, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if len(raw) > 2097152:
        raise ValueError('Registered artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def verify():
    if digest(path('selection')) != PROTOCOL_SHA256:
        raise ValueError('Fixed protocol changed')
    if read('selection')['arms'] != list(ARMS):
        raise ValueError('Fixed comparison roster changed')
    for pins in (read('selection')['preserved'], read('registration')['sources']):
        for name, expected in pins.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned source/evidence changed: '+name)
    if read('registration')['python'] != platform.python_version():
        raise ValueError('Use registered Python patch release')


def execute():
    selection = read('selection')
    cases = []
    for task in selection['tasks']:
        for seed in selection['seeds']:
            for condition in selection['conditions']:
                case = run_case(seed, task, condition)
                cases.append(case)
                print(json.dumps({'task': task, 'seed': seed, 'condition': condition,
                                  'failed_gates': [k for k, v in case['component_gates'].items() if not v]}), flush=True)
    if len(cases) != 20:
        raise AssertionError('Finite case roster changed')
    return {'schema': 'sara-route-reactivation-result-v1', 'cases': cases,
            'selection_sha256': PROTOCOL_SHA256, 'registration_sha256': digest(path('registration')),
            'preserved_artifacts': len(selection['preserved']), **summarize(cases)}


def main(mode):
    if mode == 'register':
        names = ['src/sara_engine/evaluation/'+n+'.py' for n in (
            'sparse_route_reactivation', 'sparse_route_budget', 'sparse_factor_routing', 'sparse_predictive_memory')]
        names += ['scripts/eval/sparse_route_reactivation.py', 'tests/test_sparse_route_reactivation.py']
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                             'python': platform.python_version(),
                             'sources': {n: digest(project_path(n)) for n in names}})
        verify()
        print('Nonstationary component code frozen')
        return
    verify()
    if mode == 'replay' and path('replay').exists():
        expected = {'exact_match': True, 'result_sha256': digest(path('result')), 'new_real_data_reads': 0}
        if read('replay') != expected:
            raise ValueError('Immutable replay receipt differs')
        print('Consumed component replay receipt verified; no new study run')
        return
    if mode == 'replay' and not path('result').exists():
        raise ValueError('Retained primary result required')
    save('execution' if mode == 'execute' else 'replay_execution', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'registration_sha256': digest(path('registration'))})
    if read('execution')['registration_sha256'] != digest(path('registration')):
        raise ValueError('Consumed execution binding differs')
    result = execute()
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Exact deterministic replay differs')
        save('replay', {'exact_match': True, 'result_sha256': digest(path('result')), 'new_real_data_reads': 0})
    print(json.dumps({k: result[k] for k in ('execution_integrity_passed', 'component_passed', 'preserved_artifacts')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
