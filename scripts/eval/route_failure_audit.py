#!/usr/bin/env python3
"""Register and execute only a finite audit of seven saved reports."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.reactivation_codec import loads
from sara_engine.evaluation.route_numeric_v2 import canonical
from sara_engine.evaluation.route_failure_audit import INPUTS, PROTOCOL_SHA256, evaluate
from sara_engine.utils.project_paths import processed_data_path, project_path, ensure_parent_directory

BUDGETS = {'input_files':7,'per_input_bytes':2097152,'all_input_bytes':2097152,
           'result_bytes':262144,'all_artifacts_bytes':1048576}


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures','route_failure_audit_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return loads(path(suffix).read_bytes())


def save(suffix,value):
    canonical(value)
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    used=sum(f.stat().st_size for f in path('selection').parent.glob('route_failure_audit_*_v1.json'))
    if len(raw)>BUDGETS['result_bytes'] or used+len(raw)>BUDGETS['all_artifacts_bytes']:
        raise ValueError('Finite audit artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def verify():
    if digest(path('selection'))!=PROTOCOL_SHA256:
        raise ValueError('Frozen finite audit selection changed')
    selection=read('selection')
    if selection['budgets']!=BUDGETS or tuple(selection['inputs'])!=INPUTS:
        raise ValueError('Fixed saved-input roster/budgets changed')
    for pins in (selection['preserved'],read('registration')['sources']):
        for name,expected in pins.items():
            if digest(project_path(name))!=expected:
                raise ValueError('Pinned evidence/code changed: '+name)
    if read('registration')['python']!=platform.python_version():
        raise ValueError('Use registered Python patch release')
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():
        raise ValueError('Old failed replay was silently promoted')


def load_inputs(selection):
    if tuple(selection['inputs'])!=INPUTS or selection['budgets']!=BUDGETS:
        raise ValueError('Finite parsed input scope changed')
    reports,hashes,total={},{},0
    for name in INPUTS:
        file=Path(processed_data_path('benchmark_fixtures',name))
        with file.open('rb') as stream:raw=stream.read(BUDGETS['per_input_bytes']+1)
        total+=len(raw)
        if len(raw)>BUDGETS['per_input_bytes'] or total>BUDGETS['all_input_bytes']:
            raise ValueError('Saved report byte budget exceeded')
        hashes[name]=sha256(raw).hexdigest()
        if hashes[name]!=selection['preserved']['data/processed/benchmark_fixtures/'+name]:
            raise ValueError('Parsed report hash differs')
        reports[name]=loads(raw)
    return reports,hashes,total


def compute():
    selection=read('selection')
    reports,hashes,total=load_inputs(selection)
    result=evaluate(selection,reports,hashes)
    result.update(input_bytes_read=total,preserved_hashes=len(selection['preserved']))
    return result


def main(mode):
    if mode=='register':
        names=['scripts/eval/route_failure_audit.py','src/sara_engine/evaluation/route_failure_audit.py',
               'tests/test_route_failure_audit.py','src/sara_engine/utils/project_paths.py',
               str(path('selection').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
                            'python':platform.python_version(),
                            'sources':{name:digest(project_path(name)) for name in names}})
        verify()
        print('Finite saved-report failure audit registered; no learner imported')
        return
    verify()
    if mode=='replay' and not path('result').exists():raise ValueError('Saved static audit result required')
    receipt={'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0} if path('result').exists() else None
    if mode=='replay' and path('replay').exists():
        if canonical(read('replay'))!=canonical(receipt):raise ValueError('Immutable audit replay receipt differs')
        print('Consumed audit replay verified; no learner started')
        return
    save('execution' if mode=='execute' else 'replay_execution',{
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration'))})
    if read('execution')['registration_sha256']!=digest(path('registration')):raise ValueError('Consumed audit binding differs')
    result=compute()
    verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Pure failure-audit replay differs')
        save('replay',receipt)
    print(json.dumps({'summary':result['summary'],'input_bytes_read':result['input_bytes_read'],'learner_executions':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
