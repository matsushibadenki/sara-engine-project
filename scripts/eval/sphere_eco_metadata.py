#!/usr/bin/env python3
"""Preregister, meter and replay the finite SPHERE/ECO publisher metadata audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.evaluation.sphere_eco_metadata import validate_requests, evaluate
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    extension = '.jsonl' if suffix == 'requests' else '.json'
    return Path(processed_data_path('benchmark_fixtures', 'sphere_eco_metadata_' + suffix + '_v1' + extension))


def read(suffix):
    return json.loads(path(suffix).read_text())


def requests():
    return [json.loads(line) for line in path('requests').read_text().splitlines()] if path('requests').exists() else []


def save(suffix, value):
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    for name, digest in read('registration')['files'].items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Registered file changed: ' + name)
    for name, digest in read('selection')['preserved'].items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Earlier evidence changed: ' + name)


def main(args):
    mode = args.mode
    if mode == 'register':
        previous = json.loads(Path(processed_data_path('benchmark_fixtures', 'source_metadata_feasibility_registration_v1.json')).read_text())
        names = ['scripts/eval/sphere_eco_metadata.py', 'src/sara_engine/evaluation/sphere_eco_metadata.py',
                 'tests/test_sphere_eco_metadata.py', str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Selection, semantic rejection rule, implementation and tests registered before any new external metadata search/read. Earlier public metadata is known. Reserve each new search/page before navigation; cached finds are not new page navigations. No underlying data acquisition.',
             'files': {**previous['files'], **{name: file_sha256(project_path(name)) for name in names}}})
        verify()
        print('Finite metadata audit code registered before new external inspection')
        return
    verify()
    if mode == 'reserve':
        if path('execution').exists():
            raise ValueError('Consumed audit request ledger is sealed')
        ledger = requests()
        entry = {'id': len(ledger) + 1, 'source': args.source, 'kind': args.kind, 'target': args.target,
                 'reserved_at_utc': datetime.now(timezone.utc).isoformat()}
        validate_requests(read('selection'), ledger + [entry])
        destination = Path(ensure_parent_directory(str(path('requests'))))
        with destination.open('a') as stream:
            stream.write(json.dumps(entry, sort_keys=True) + '\n')
        print(json.dumps(entry))
        return
    if mode == 'execute':
        save('execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'sha256': {s: file_sha256(path(s)) for s in ('registration', 'requests', 'evidence', 'inputs')}})
    execution = read('execution')
    for suffix, digest in execution['sha256'].items():
        if file_sha256(path(suffix)) != digest:
            raise ValueError('Consumed audit input changed: ' + suffix)
    result = evaluate(read('selection'), requests(), read('evidence'), read('inputs'))
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Fresh-process metadata replay differs')
        receipt = {'exact_match': True, 'result_sha256': file_sha256(path('result')),
                   'previous_artifacts_preserved': len(read('selection')['preserved'])}
        if path('replay').exists():
            if receipt != read('replay'):
                raise ValueError('Immutable replay receipt differs')
        else:
            save('replay', receipt)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'reserve', 'execute', 'replay'))
    parser.add_argument('--source', choices=('SPHERE', 'ECO'))
    parser.add_argument('--kind', choices=('search', 'page'))
    parser.add_argument('--target')
    main(parser.parse_args())
