#!/usr/bin/env python3
"""Freeze and check one design-only credit contract; no model execution."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.local_credit_contract import ARMS, LIMITS, validate, witness
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory
from scripts.eval.r2_mechanism_decision import verify as verify_prior


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures', 'local_credit_contract_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_bytes())


def save(suffix, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if len(raw) > LIMITS['artifact_bytes']:
        raise ValueError('Static contract artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def verify():
    verify_prior()
    selection = read('selection')
    if selection['limits'] != LIMITS or selection['arms'] != list(ARMS):
        raise ValueError('Registered structural design changed')
    for pins in (selection['prior_receipts'], read('registration')['files']):
        for name, expected in pins.items():
            if digest(project_path(name)) != expected:
                raise ValueError('Pinned contract/evidence changed: '+name)


def evaluate():
    result = validate(witness())
    # Copy only already-retained aggregate prerequisites; never re-score.
    audit = json.loads(Path(processed_data_path('benchmark_fixtures', 'r2_mechanism_decision_result_v1.json')).read_bytes())
    result.update(schema='sara-local-credit-contract-result-v1',
                  prior_resource_prerequisites=audit['resource_prerequisites'],
                  independent_task=audit['independent_task'],
                  decision='hold_execution_and_resource_work_pending_separate_qualified_protocol',
                  new_predictions_scored=0, new_acquisitions=0, new_measurements=0,
                  applied_learning_writes=0, test_opened=False)
    return result


def main(mode):
    if mode == 'register':
        names = ['scripts/eval/local_credit_contract.py',
                 'src/sara_engine/evaluation/local_credit_contract.py',
                 'tests/test_local_credit_contract.py', str(path('selection').relative_to(project_path()))]
        save('registration', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                             'files': {n: digest(project_path(n)) for n in names}})
        verify()
        print('Static local-credit contract registered; no experiments authorized')
        return
    verify()
    if mode == 'execute':
        save('execution', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                           'registration_sha256': digest(path('registration'))})
        result = evaluate()
        verify()
        save('result', result)
    else:
        if read('execution')['registration_sha256'] != digest(path('registration')):
            raise ValueError('Execution binding differs')
        result = evaluate()
        if result != read('result'):
            raise ValueError('Static contract replay differs')
        receipt = {'exact_match': True, 'result_sha256': digest(path('result')),
                   'new_measurements': 0, 'applied_learning_writes': 0}
        if path('replay').exists():
            if read('replay') != receipt:
                raise ValueError('Immutable replay receipt differs')
        else:
            save('replay', receipt)
    print(json.dumps({'decision': result['decision'], 'actual_applied_writes': result['actual_applied_writes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'execute', 'replay'))
    main(parser.parse_args().mode)
