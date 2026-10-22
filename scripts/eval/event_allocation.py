#!/usr/bin/env python3
"""Register, execute and replicate the bounded isolated allocation diagnostic."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.event_allocation import evaluate, cases, run_case, semantic_result
from sara_engine.evaluation.event_work import fingerprint
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'event_allocation_' + suffix + '_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_text())


def save(suffix, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if len(raw.encode()) > read('selection')['budgets']['result_bytes']:
        raise ValueError('Artifact byte budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        stream.write(raw)


def runtime():
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'machine': platform.machine(), 'byteorder': sys.byteorder}


def verify(worker=False):
    registration = read('registration')
    if registration['runtime'] != runtime() or sys.version_info[:2] != (3, 10):
        raise ValueError('Use the registered CPython runtime')
    if not worker:
        for collection in (read('selection')['preserved'], registration['files']):
            for name, expected in collection.items():
                if digest(project_path(name)) != expected:
                    raise ValueError('Pinned evidence/code changed: ' + name)


def inputs():
    selection = read('selection')
    raw = Path(project_path(selection['input']['path'])).read_bytes()
    if sha256(raw).hexdigest() != selection['input']['sha256']:
        raise ValueError('Existing synthetic input changed')
    previous = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_separation_result_v1.json')).read_text())
    return selection, json.loads(raw), previous


def replication_receipt(original, replica):
    if semantic_result(original) != semantic_result(replica):
        raise ValueError('Fresh-process semantic/ownership replication differs')
    equal = total = 0
    ranges = {}
    for mode in ('paired', 'neuron', 'scalar'):
        a = [r['allocation'] for w in original['workers'] if w['mode'] == mode for r in w['records']]
        b = [r['allocation'] for w in replica['workers'] if w['mode'] == mode for r in w['records']]
        equal += sum(x == y for x, y in zip(a, b))
        total += len(a)
        ranges[mode] = {key: sorted([max(x[key] for x in a), max(x[key] for x in b)])
                        for key in ('new_peak_bytes', 'new_live_at_return_bytes', 'peak_excess_over_return_bytes',
                                    'tracer_metadata_after_snapshot_bytes')}
    return {'semantic_and_ownership_exact_match': True, 'raw_measurements_exact_match': original == replica,
            'byte_identical_windows': equal, 'compared_windows': total, 'per_mode_run_max_ranges_bytes': ranges,
            'result_sha256': digest(path('result')), 'replication_sha256': digest(path('replication')),
            'semantic_sha256': fingerprint(semantic_result(original)), 'preserved_artifacts': len(read('selection')['preserved'])}


def main(mode, case_index=None, arm=None):
    if mode == 'register':
        if sys.version_info[:2] != (3, 10) or platform.python_implementation() != 'CPython':
            raise ValueError('CPython 3.10 required')
        old = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_separation_registration_v1.json')).read_text())
        names = ['scripts/eval/event_allocation.py', 'src/sara_engine/evaluation/event_allocation.py',
                 'src/sara_engine/evaluation/event_allocation_execution.py', 'tests/test_event_allocation.py',
                 str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime(),
             'scope': 'Selection precedes implementation; code/tests frozen before 144 fresh diagnostic workers and one registered replication. No new learner experiment.',
             'files': {**old['files'], **{name: digest(project_path(name)) for name in names}}})
        verify()
        print('Allocation code registered before finite execution')
        return
    verify(worker=mode == 'worker')
    if mode == 'worker':
        if type(case_index) is not int or not 0 <= case_index < 48 or arm not in ('paired', 'neuron', 'scalar'):
            raise ValueError('Worker outside fixed roster')
        events, golden = cases(*inputs())[case_index]
        print(json.dumps(run_case(events, golden, arm), sort_keys=True, separators=(',', ':')))
        return
    if mode == 'replay' and path('replay').exists():
        receipt = read('replay')
        if receipt != replication_receipt(read('result'), read('replication')):
            raise ValueError('Immutable replication receipt differs')
        print('Previously consumed replication receipt verified; no new workers launched')
        return
    save('execution' if mode == 'execute' else 'replay_execution', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'registration_sha256': digest(path('registration')), 'selection_sha256': digest(path('selection'))})
    for suffix in ('registration', 'selection'):
        if read('execution')[suffix + '_sha256'] != digest(path(suffix)):
            raise ValueError('Consumed allocation contract changed')
    result = evaluate(*inputs())
    result.update(runtime=runtime(), preserved_artifacts=len(read('selection')['preserved']))
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        save('replication', result)
        save('replay', replication_receipt(read('result'), result))
    print(json.dumps({key: result[key] for key in ('case_count', 'worker_count', 'all_allocation_gates_passed', 'decision', 'preserved_artifacts')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay', 'worker'))
    parser.add_argument('--case', type=int, dest='case_index')
    parser.add_argument('--arm', choices=('paired', 'neuron', 'scalar'))
    args = parser.parse_args()
    main(args.mode, args.case_index, args.arm)
