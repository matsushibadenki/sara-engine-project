#!/usr/bin/env python3
"""Exclusive finite factorial phases; preserves every registered historical file."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_memory_factorial_v4 import run_case, summarize
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory

PREFIX = 'route_memory_factorial_v4_'
SELECTION_SHA = '8b196ab28022353fc60a7ee4e5bc304c07b226b305fcf0c8249667a933413a1e'
SOURCES = ('src/sara_engine/evaluation/route_memory_factorial_v4.py',
           'scripts/eval/route_memory_factorial_v4.py', 'tests/test_route_memory_factorial_v4.py',
           'src/sara_engine/evaluation/sparse_route_reactivation.py',
           'src/sara_engine/evaluation/sparse_route_budget.py',
           'src/sara_engine/evaluation/sparse_factor_routing.py',
           'src/sara_engine/evaluation/sparse_predictive_memory.py',
           'src/sara_engine/utils/project_paths.py')


def path(name):
    return Path(processed_data_path('benchmark_fixtures', PREFIX + name + '_v1.json'))


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def read(name):
    return json.loads(path(name).read_text())


def save(name, data):
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    used = sum(p.stat().st_size for p in path('selection').parent.glob(PREFIX + '*_v1.json'))
    if len(raw) > 8388608 or used + len(raw) > 33554432:
        raise ValueError('Registered artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(name)))).open('xb') as f:
        f.write(raw)


def verify():
    if digest(path('selection')) != SELECTION_SHA:
        raise ValueError('Protocol changed')
    selection = read('selection')
    for name, expected in selection['preserved'].items():
        if digest(project_path(name)) != expected:
            raise ValueError('Historical file changed: ' + name)
    if Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_replay_v1.json')).exists():
        raise ValueError('Old failed replay must remain failed')
    if path('registration').exists():
        registration = read('registration')
        if registration['python'] != platform.python_version():
            raise ValueError('Registered Python changed')
        for name, expected in {**registration['preserved'], **registration['sources']}.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Frozen source changed: ' + name)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('register', 'primary', 'replication', 'verify'))
    args = parser.parse_args()
    selection = verify()
    if args.phase == 'register':
        save('registration', {'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                             'sources': {s: digest(project_path(s)) for s in SOURCES},
                             'preserved': {**json.loads(Path(processed_data_path('benchmark_fixtures', 'route_count_mixture_selection_v1.json')).read_text())['preserved'], **selection['preserved']}})
        return
    if not path('registration').exists():
        raise ValueError('Registration required')
    if args.phase == 'verify':
        proof = read('result')
        for name in ('primary', 'replication'):
            if digest(path(name)) != proof[name + '_sha256']:
                raise ValueError('Report hash mismatch')
        if path('primary').read_bytes() != path('replication').read_bytes():
            raise ValueError('Numeric replay mismatch')
        print('Read-only verification passed; historical pins and seals preserved')
        return
    if args.phase == 'replication' and not path('primary').exists():
        raise ValueError('Primary required first')
    save(args.phase + '_execution', {'phase': args.phase, 'consumed': True,
                                    'registration_sha256': digest(path('registration'))})
    historical = json.loads(Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_result_v1.json')).read_text())
    cases = []
    for seed in selection['seeds']:
        for task in selection['tasks']:
            for condition in selection['conditions']:
                reference = next(c for c in historical['cases'] if (c['seed'], c['task'], c['condition']) == (seed, task, condition))
                cases.append(run_case(seed, task, condition, reference['arms']['error_triggered']))
    report = {'schema': 'sara-route-memory-factorial-v4-result-v1',
              'registration_sha256': digest(path('registration')), 'cases': cases, 'summary': summarize(cases)}
    if sum(len(c['snapshots']) for case in cases for c in case['cells'].values()) != 320:
        raise AssertionError('Fixed snapshot count differs')
    save(args.phase, report)
    verify()
    if args.phase == 'replication':
        if path('primary').read_bytes() != path('replication').read_bytes():
            save('failure', {'reason': 'Exact numeric replay mismatch', 'old_failure_preserved': True})
            raise AssertionError('Replay mismatch; no retry')
        save('result', {'primary_sha256': digest(path('primary')), 'replication_sha256': digest(path('replication')),
                        'exact_replay_equal': True, 'preserved_files': len(read('registration')['preserved']),
                        'models_per_phase': 80, 'snapshots_per_phase': 320,
                        'actual_route_updates_per_cell_case': 9216, 'actual_learning_field_changes_per_cell_case': 36864,
                        'promotion_authorized': False, 'test_opened': False})
    print(args.phase + ' completed')


if __name__ == '__main__':
    main()
