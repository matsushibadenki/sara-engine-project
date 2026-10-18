#!/usr/bin/env python3
"""Register and reproduce a metadata-only source acquisition-feasibility audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.evaluation.source_metadata_feasibility import evaluate
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'source_metadata_feasibility_' + suffix + '_v1.json'))


def read(suffix):
    return json.loads(path(suffix).read_text())


def save(suffix, value):
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    registration = read('registration')
    for name, digest in registration['files'].items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Pinned file changed: ' + name)
    for name, digest in read('selection')['preserved'].items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Previous evidence changed: ' + name)
    return registration


def main(mode):
    if mode == 'register':
        names = ['scripts/eval/source_metadata_feasibility.py',
                 'src/sara_engine/evaluation/source_metadata_feasibility.py',
                 'tests/test_source_metadata_feasibility.py']
        names += [str(path(s).relative_to(project_path())) for s in ('selection', 'evidence', 'inputs')]
        prior = json.loads(Path(processed_data_path('benchmark_fixtures', 'casas_roster_screen_registration_v1.json')).read_text())
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Source names, limits and conservative method registered before external searches in selection. Metadata now known; scalar implementation and scenarios pinned after metadata capture, before result calculation. This is not a blind source performance experiment.',
             'files': {**prior['code'], **{name: file_sha256(project_path(name)) for name in names}}})
        verify()
        print('Metadata-only calculation registered; candidate selection predates external searches')
        return
    registration = verify()
    if mode == 'execute':
        save('execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                           'registration_sha256': file_sha256(path('registration'))})
    elif read('execution')['registration_sha256'] != file_sha256(path('registration')):
        raise ValueError('Execution provenance changed')
    result = evaluate(read('selection'), read('evidence'), read('inputs'))
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Fresh-process metadata calculation differs')
        receipt = {'exact_match': True, 'result_sha256': file_sha256(path('result')),
                   'previous_artifacts_preserved': len(read('selection')['preserved'])}
        if path('replay').exists():
            if receipt != read('replay'):
                raise ValueError('Immutable replay receipt differs')
        else:
            save('replay', receipt)
    print(json.dumps({'decision': result['decision'], 'sources': result['sources'], 'work': result['work']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
