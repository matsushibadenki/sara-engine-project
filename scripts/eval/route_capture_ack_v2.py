#!/usr/bin/env python3
"""Capture-derived acknowledgement bridge verification; hand records only."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_capture_ack_preflight import analyze, canonical
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory


SELECTION_SHA = 'a0f31f68f833b91a4eecbf3ee376c233e983cc6e345f631b4d930eee0d5ca527'


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'route_capture_ack_' + suffix + '_v2.json'))


def digest(file): return sha256(Path(file).read_bytes()).hexdigest()


def save(suffix, obj):
    raw = canonical(obj) + b'\n'
    total = sum(p.stat().st_size for p in path('selection').parent.glob('route_capture_ack_*_v1.json'))
    if len(raw) > 262144 or total + len(raw) > 2097152: raise ValueError('Artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream: stream.write(raw)


def verify():
    if digest(path('selection')) != SELECTION_SHA: raise ValueError('Selection changed')
    spec = json.loads(path('selection').read_bytes())
    for name, pin in spec['preserved'].items():
        if digest(project_path(name)) != pin: raise ValueError('Historical evidence changed: ' + name)
    for name, pin in spec['reports'].items():
        if digest(processed_data_path('benchmark_fixtures', name)) != pin: raise ValueError('Frozen input changed')
    if Path(processed_data_path('benchmark_fixtures', 'sparse_route_reactivation_replay_v1.json')).exists():
        raise ValueError('Old replay failure reclassified')
    if path('registration').exists():
        reg = json.loads(path('registration').read_bytes())
        if reg['python'] != platform.python_version() or reg['selection_sha256'] != SELECTION_SHA:
            raise ValueError('Registered runtime/protocol changed')
        for name, pin in reg['sources'].items():
            if digest(project_path(name)) != pin: raise ValueError('Registered source changed')
    return spec


def compute(spec):
    reports = {name: Path(processed_data_path('benchmark_fixtures', name)).read_bytes() for name in spec['reports']}
    result = analyze(spec, reports)
    result['preserved_hashes'] = len(spec['preserved'])
    result['registration_sha256'] = digest(path('registration'))
    return result


def main(mode):
    spec = verify()
    if mode == 'register':
        sources = ['src/sara_engine/evaluation/route_capture_ack.py',
                   'src/sara_engine/evaluation/route_capture_ack_preflight.py',
                   'src/sara_engine/evaluation/route_owner_host.py',
                   'src/sara_engine/evaluation/route_outcome_admission.py',
                   'src/sara_engine/evaluation/route_capture_layout.py',
                   'src/sara_engine/evaluation/route_capture_layout_preflight.py',
                   'src/sara_engine/evaluation/route_capture_candidate.py',
                   'scripts/eval/route_capture_ack_v2.py', 'tests/test_route_capture_ack_v2.py',
                   'src/sara_engine/utils/project_paths.py']
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                             'selection_sha256': SELECTION_SHA, 'python': platform.python_version(),
                             'sources': {name: digest(project_path(name)) for name in sources}})
        return
    if not path('registration').exists(): raise ValueError('Registration required')
    if mode == 'verify':
        if path('failure').exists() or path('replay_failure').exists(): raise ValueError('Failed phase retained')
        if canonical(compute(spec)) != canonical(json.loads(path('result').read_bytes())):
            raise ValueError('Saved source audit differs')
        if json.loads(path('replay').read_bytes()) != {'exact_match': True, 'result_sha256': digest(path('result'))}:
            raise ValueError('Replay receipt differs')
        print('Capture acknowledgement bridge verified; actual model and producer remain unattached')
        return
    if mode == 'replay' and (not path('result').exists() or path('failure').exists()):
        raise ValueError('Successful primary required')
    save('execution' if mode == 'execute' else 'replay_execution',
         {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'registration_sha256': digest(path('registration'))})
    try:
        result = compute(spec)
        verify()
        if mode == 'execute': save('result', result)
        else:
            if canonical(result) != canonical(json.loads(path('result').read_bytes())):
                raise ValueError('Exact source audit replay differs')
            save('replay', {'exact_match': True, 'result_sha256': digest(path('result'))})
    except Exception as error:
        save('failure' if mode == 'execute' else 'replay_failure',
             {'error': type(error).__name__ + ': ' + str(error), 'retry_authorized': False})
        raise
    print(json.dumps({'hand_pairs': result['hand_pairs'], 'faults': len(result['faults_rejected']), 'preserved_hashes': result['preserved_hashes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay', 'verify'))
    main(parser.parse_args().mode)
