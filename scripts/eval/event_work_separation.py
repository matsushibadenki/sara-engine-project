#!/usr/bin/env python3
"""Freeze, execute once and replay the synthetic instrumentation-separation contract."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.event_work_separation import evaluate
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'event_work_separation_' + suffix + '_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_text())


def save(suffix, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if suffix == 'result' and len(raw.encode()) > read('selection')['budgets']['result_bytes']:
        raise ValueError('Result byte budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        stream.write(raw)


def verify():
    registration = read('registration')
    if registration['python'] != platform.python_version() or sys.version_info[:2] != (3, 10):
        raise ValueError('Use the registered Python runtime for owned-memory replay')
    for collection in (read('selection')['preserved'], registration['files']):
        for name, expected in collection.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned evidence/code changed: ' + name)


def main(mode):
    if mode == 'register':
        if sys.version_info[:2] != (3, 10):
            raise ValueError('Python 3.10 required')
        old = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_registration_v1.json')).read_text())
        names = ['scripts/eval/event_work_separation.py', 'src/sara_engine/evaluation/event_work_separation.py',
                 'src/sara_engine/evaluation/event_execution.py', 'tests/test_event_work_separation.py', 'src/sara_engine/neuro/neuron.py',
                 'src/sara_engine/evaluation/temporal_information.py',
                 'src/sara_engine/evaluation/temporal_capability_boundary.py',
                 'data/processed/benchmark_fixtures/temporal_capability_boundary_v1.json',
                 str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
             'python': platform.python_version(),
             'scope': 'Contract selected before implementation; code and tests frozen before the one-shot finite 48-case, three-pass boundary execution. Only known synthetic input templates and unlabeled receipt-ready schedules. No new learner experiment.',
             'files': {**old['files'], **{name: digest(project_path(name)) for name in names}}})
        verify()
        print('Separation contract code registered before finite execution')
        return
    verify()
    if mode == 'execute':
        save('execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'registration_sha256': digest(path('registration')), 'selection_sha256': digest(path('selection'))})
    execution = read('execution')
    for suffix in ('registration', 'selection'):
        if digest(path(suffix)) != execution[suffix + '_sha256']:
            raise ValueError('Consumed audit input changed')
    selection = read('selection')
    source = Path(processed_data_path('benchmark_fixtures', 'temporal_capability_boundary_v1.json'))
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != selection['input']['sha256']:
        raise ValueError('Existing synthetic input changed')
    previous = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_result_v1.json')).read_text())
    result = evaluate(selection, json.loads(raw), previous)
    result['preserved_artifacts'] = len(selection['preserved'])
    result['python'] = platform.python_version()
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Fresh-process separation replay differs')
        receipt = {'exact_match': True, 'result_sha256': digest(path('result')),
                   'preserved_artifacts': len(selection['preserved']), 'case_count': result['case_count']}
        if path('replay').exists():
            if receipt != read('replay'):
                raise ValueError('Immutable replay receipt differs')
        else:
            save('replay', receipt)
    print(json.dumps({key: result[key] for key in ('case_count', 'all_separation_gates_passed', 'decision', 'actual_learning_writes', 'preserved_artifacts')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
