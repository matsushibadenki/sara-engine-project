#!/usr/bin/env python3
"""Run only the registered saved-report serialization preflight."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.reactivation_codec import PROTOCOL_SHA256, loads, preflight
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory

BUDGETS = {'input_bytes_per_file': 2097152, 'input_bytes_total': 1048576,
           'max_depth': 32, 'max_nodes': 100000, 'artifact_bytes': 131072, 'all_artifacts_bytes': 1048576}
INPUTS = ['sparse_route_reactivation_result_v1.json', 'sparse_route_reactivation_failure_v1.json']


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'reactivation_codec_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return loads(path(suffix).read_bytes())


def save(suffix, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    used = sum(f.stat().st_size for f in path('selection').parent.glob('reactivation_codec_*_v1.json'))
    if len(raw) > BUDGETS['artifact_bytes'] or used+len(raw) > BUDGETS['all_artifacts_bytes']:
        raise ValueError('Static artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def verify():
    if digest(path('selection')) != PROTOCOL_SHA256:
        raise ValueError('Frozen codec selection changed')
    selection = read('selection')
    if selection['budgets'] != BUDGETS or selection['inputs'] != INPUTS:
        raise ValueError('Finite input roster or budgets changed')
    for pins in (selection['preserved'], read('registration')['sources']):
        for name, expected in pins.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned code/evidence changed: '+name)
    if read('registration')['python'] != platform.python_version():
        raise ValueError('Use registered Python patch release')
    if Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_replay_v1.json')).exists():
        raise ValueError('Failed learner replay was silently promoted')


def evaluate():
    inputs, total = [], 0
    for name in INPUTS:
        file = Path(processed_data_path('benchmark_fixtures', name))
        with file.open('rb') as stream:
            raw = stream.read(BUDGETS['input_bytes_per_file']+1)
        total += len(raw)
        if len(raw) > BUDGETS['input_bytes_per_file'] or total > BUDGETS['input_bytes_total']:
            raise ValueError('Fixed saved-input byte budget exceeded')
        inputs.append((loads(raw), raw))
    result = preflight(inputs[0][0], inputs[1][0], inputs[0][1])
    result.update(input_bytes_read=total, preserved_hashes=len(read('selection')['preserved']))
    return result


def main(mode):
    if mode == 'register':
        names = ['src/sara_engine/evaluation/reactivation_codec.py', 'scripts/eval/reactivation_codec.py',
                 'tests/test_reactivation_codec.py', 'src/sara_engine/utils/project_paths.py',
                 str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                             'python': platform.python_version(),
                             'sources': {name: digest(project_path(name)) for name in names}})
        verify()
        print('Saved-report codec implementation registered; no learner imported')
        return
    verify()
    if mode == 'replay' and not path('result').exists():
        raise ValueError('Retained static preflight result required')
    expected = {'exact_match': True, 'result_sha256': digest(path('result')), 'learner_executions': 0} if path('result').exists() else None
    if mode == 'replay' and path('replay').exists():
        if read('replay') != expected:
            raise ValueError('Immutable static replay receipt differs')
        print('Consumed static replay receipt verified; no learner started')
        return
    save('execution' if mode == 'execute' else 'replay_execution', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'registration_sha256': digest(path('registration'))})
    if read('execution')['registration_sha256'] != digest(path('registration')):
        raise ValueError('Consumed preflight binding differs')
    result = evaluate()
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Pure saved-report preflight replay differs')
        save('replay', expected)
    print(json.dumps({k: result[k] for k in ('codec_preflight_passed', 'input_bytes_read', 'learner_executions', 'full_replay_verified')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
