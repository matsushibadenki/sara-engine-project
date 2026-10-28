#!/usr/bin/env python3
"""Register and verify static addresses/metadata; contains no learner dispatch."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_fixed_allocation_contract import analyze,canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='84b926ef42392565c836c180eaadba15694935cc3451effeb87b48ea12caa896'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_fixed_allocation_contract_'+suffix+'_v1.json'))

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()

def read(suffix):return json.loads(path(suffix).read_text())


def save(suffix,data):
    raw=canonical(data)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_fixed_allocation_contract_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>2097152:raise ValueError('Static artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Frozen contract changed')
    contract=read('selection')
    for name,expected in contract['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Historical file changed: '+name)
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old failed replay reclassified')
    if path('registration').exists():
        registration=read('registration')
        if registration['python']!=platform.python_version():raise ValueError('Registered Python changed')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Registered source changed: '+name)
    return contract


def compute(contract):
    result=analyze(contract)
    result.update(registration_sha256=digest(path('registration')),preserved_hashes=len(contract['preserved']))
    return result


def main(mode):
    contract=verify()
    if mode=='register':
        names=['scripts/eval/route_fixed_allocation_contract.py','src/sara_engine/evaluation/route_fixed_allocation_contract.py',
               'tests/test_route_fixed_allocation_contract.py','src/sara_engine/utils/project_paths.py']
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'selection_sha256':SELECTION_SHA,'sources':{n:digest(project_path(n)) for n in names}})
        return
    if not path('registration').exists():raise ValueError('Static source registration required')
    if mode=='verify':
        if canonical(compute(contract))!=canonical(read('result')):raise ValueError('Static proof differs')
        if read('replay')!={'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0}:raise ValueError('Static replay binding differs')
        print('Static contract and pure replay verified; no learner dispatch exists');return
    if mode=='replay' and not path('result').exists():raise ValueError('Static primary required')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),'static_only':True})
    result=compute(contract);verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Static replay mismatch')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0})
    print(json.dumps(result['status']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
