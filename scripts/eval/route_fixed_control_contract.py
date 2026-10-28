#!/usr/bin/env python3
"""Static-only matched controls; no learner execution path."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_fixed_control_contract import analyze
from sara_engine.evaluation.route_fixed_allocation_contract import canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='17b6bcf2f081ba50f15ea109fadb44fc5602d1e350d39be43dd8d5636e23acae'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_fixed_control_contract_'+suffix+'_v1.json'))

def digest(file):return sha256(Path(file).read_bytes()).hexdigest()

def read(suffix):return json.loads(path(suffix).read_text())


def save(suffix,data):
    raw=canonical(data)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_fixed_control_contract_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>2097152:raise ValueError('Static control artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Control protocol changed')
    spec=read('selection')
    for name,expected in spec['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Preserved history changed: '+name)
    parent=Path(processed_data_path('benchmark_fixtures',spec['parent']['selection']))
    if digest(parent)!=spec['parent']['sha256']:raise ValueError('Parent contract binding differs')
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old failed replay reclassified')
    if path('registration').exists():
        registration=read('registration')
        if registration['python']!=platform.python_version():raise ValueError('Registered Python changed')
        if registration['selection_sha256']!=SELECTION_SHA:raise ValueError('Registration binding differs')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Frozen source changed: '+name)
    return spec


def compute(spec):
    parent=json.loads(Path(processed_data_path('benchmark_fixtures',spec['parent']['selection'])).read_text())
    saved=json.loads(Path(processed_data_path('benchmark_fixtures','route_fixed_allocation_contract_result_v1.json')).read_text())
    result=analyze(spec,parent,saved)
    result.update(registration_sha256=digest(path('registration')),preserved_hashes=len(spec['preserved']))
    return result


def main(mode):
    spec=verify()
    if mode=='register':
        sources=['scripts/eval/route_fixed_control_contract.py','src/sara_engine/evaluation/route_fixed_control_contract.py',
                 'tests/test_route_fixed_control_contract.py','src/sara_engine/evaluation/route_fixed_allocation_contract.py',
                 'src/sara_engine/utils/project_paths.py']
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'selection_sha256':SELECTION_SHA,'sources':{name:digest(project_path(name)) for name in sources}})
        return
    if not path('registration').exists():raise ValueError('Source registration required')
    if mode=='verify':
        if canonical(compute(spec))!=canonical(read('result')):raise ValueError('Static control proof differs')
        if read('replay')!={'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0}:raise ValueError('Replay proof differs')
        print('Static control contract verified; no learning or metric scoring');return
    if mode=='replay' and not path('result').exists():raise ValueError('Primary static proof required')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),'static_only':True})
    result=compute(spec);verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Static replay mismatch')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0})
    print(json.dumps(result['status']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
