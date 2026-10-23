#!/usr/bin/env python3
"""Freeze and replay a static report preflight; never launch allocation workers."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.allocation_report_preflight import evaluate
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'allocation_report_' + suffix + '_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_text())


def save(suffix, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if len(raw.encode()) > read('selection')['limits']['report_bytes']:
        raise ValueError('Preflight artifact exceeds fixed report budget')
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        stream.write(raw)


def verify():
    registration = read('registration')
    if registration['python'] != platform.python_version() or sys.version_info[:2] != (3, 10):
        raise ValueError('Use the registered preflight Python runtime')
    for collection in (read('selection')['preserved'], registration['files']):
        for name, expected in collection.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned evidence/code changed: ' + name)
    # The failed run stays consumed; the preflight cannot recover its lost data.
    for suffix in ('result', 'replication', 'replay', 'replay_execution'):
        if Path(processed_data_path('benchmark_fixtures', 'event_allocation_' + suffix + '_v1.json')).exists():
            raise ValueError('Failed allocation run was unexpectedly resumed')


def main(mode):
    if mode == 'register':
        if sys.version_info[:2] != (3, 10):
            raise ValueError('Python 3.10 required')
        old = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_allocation_registration_v1.json')).read_text())
        names = ['scripts/eval/allocation_report.py', 'src/sara_engine/evaluation/allocation_report.py',
                 'src/sara_engine/evaluation/allocation_report_preflight.py', 'tests/test_allocation_report.py',
                 'tests/test_event_allocation_failure.py', str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
             'python': platform.python_version(),
             'scope': 'Report domain selected before implementation. Construction checks are known engineering inputs; code/tests frozen before retained static preflight. Zero tracing, worker processes or new measurement.',
             'files': {**old['files'], **{name: digest(project_path(name)) for name in names}}})
        verify()
        print('Report preflight code registered')
        return
    verify()
    if mode == 'replay' and path('replay').exists():
        if read('replay')['result_sha256'] != digest(path('result')):
            raise ValueError('Immutable preflight replay receipt differs')
        print('Consumed preflight replay receipt verified without new work')
        return
    marker = 'execution' if mode == 'execute' else 'replay_execution'
    if mode == 'replay' and not path('result').exists():
        raise ValueError('A saved preflight result is required before replay')
    save(marker, {'started_at_utc': datetime.now(timezone.utc).isoformat(),
         'registration_sha256': digest(path('registration')), 'selection_sha256': digest(path('selection'))})
    for suffix in ('registration', 'selection'):
        if read('execution')[suffix + '_sha256'] != digest(path(suffix)):
            raise ValueError('Consumed static contract changed')
    result = evaluate(read('selection'))
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Static report preflight replay differs')
        save('replay', {'exact_match': True, 'result_sha256': digest(path('result')),
                       'prior_artifacts_preserved': len(read('selection')['preserved']),
                       'allocation_measurements': 0, 'worker_launches': 0})
    print(json.dumps({k: result[k] for k in ('decision', 'constructed_report_bytes', 'prior_artifacts_preserved', 'allocation_measurements', 'worker_launches')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
