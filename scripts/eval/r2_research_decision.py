#!/usr/bin/env python3
"""Register and replay a bounded audit of known aggregate R2 receipts only."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.r2_research_decision import ROSTER, evaluate
from sara_engine.utils.project_paths import project_path, processed_data_path, ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'r2_research_decision_' + suffix + '_v1.json'))


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_text())


def save(suffix, value):
    with Path(ensure_parent_directory(str(path(suffix)))).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    for collection in (read('selection')['preserved'], read('registration')['files']):
        for name, expected in collection.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned evidence/code changed: ' + name)


def load_receipts(selection):
    if tuple(selection['roster']) != ROSTER:
        raise ValueError('Fixed receipt roster changed')
    receipts = {}
    total = 0
    for name in ROSTER:
        # Paths cannot be supplied by a caller or a receipt.
        source = Path(processed_data_path('benchmark_fixtures', name + '.json'))
        size = source.stat().st_size
        total += size
        if size > selection['budgets']['per_receipt_bytes'] or total > selection['budgets']['aggregate_receipt_bytes']:
            raise ValueError('Receipt read budget exceeded')
        raw = source.read_bytes()
        if len(raw) != size or sha256(raw).hexdigest() != selection['preserved'][str(source.relative_to(project_path()))]:
            raise ValueError('Receipt changed during read')
        receipts[name] = json.loads(raw)
    return receipts, total


def main(mode):
    if mode == 'register':
        selection = read('selection')
        if tuple(selection['roster']) != ROSTER:
            raise ValueError('Fixed receipt roster changed')
        previous = json.loads(Path(processed_data_path('benchmark_fixtures', 'sphere_eco_metadata_registration_v1.json')).read_text())
        names = ['scripts/eval/r2_research_decision.py', 'src/sara_engine/evaluation/r2_research_decision.py',
                 'tests/test_r2_research_decision.py', str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Existing results are known. Freeze roster, extraction, interpretation and work-accounting requirements before the new aggregate audit; no confirmatory new learner experiment.',
             'files': {**previous['files'], **{name: digest(project_path(name)) for name in names}}})
        verify()
        print('Read-only R2 audit registered; no source search or learner execution')
        return
    verify()
    if mode == 'execute':
        save('execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'registration_sha256': digest(path('registration')), 'selection_sha256': digest(path('selection'))})
    execution = read('execution')
    for key in ('registration', 'selection'):
        if digest(path(key)) != execution[key + '_sha256']:
            raise ValueError('Consumed audit input changed: ' + key)
    selection = read('selection')
    receipts, total = load_receipts(selection)
    result = evaluate(selection, receipts)
    result['receipt_bytes_read'] = total
    result['preserved_artifacts'] = len(selection['preserved'])
    verify()
    if mode == 'execute':
        save('result', result)
    else:
        if result != read('result'):
            raise ValueError('Read-only audit replay differs')
        replay = {'exact_match': True, 'result_sha256': digest(path('result')),
                  'preserved_artifacts': len(selection['preserved']), 'new_predictions_scored': 0}
        if path('replay').exists():
            if read('replay') != replay:
                raise ValueError('Immutable replay receipt differs')
        else:
            save('replay', replay)
    print(json.dumps({k: result[k] for k in ('decision', 'receipt_count', 'receipt_bytes_read', 'preserved_artifacts', 'audit_learning_writes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
