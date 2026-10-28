#!/usr/bin/env python3
"""Exclusive pure audit phases over fixed saved factorial reports."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_factorial_audit import analyze,canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='fe67f5e06ed23237ce5cbb8db4dde0dd2087d959094a5d4486a2898079089952'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_factorial_audit_'+suffix+'_v1.json'))

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()

def read(suffix):return json.loads(path(suffix).read_text())


def save(suffix,data):
    raw=canonical(data)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_factorial_audit_*_v1.json'))
    if len(raw)>4194304 or used+len(raw)>8388608:raise ValueError('Audit artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as f:f.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Protocol changed')
    selection=read('selection')
    if read('scope')['selection_sha256']!=SELECTION_SHA:raise ValueError('Scope binding differs')
    for name,expected in selection['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Historical evidence changed: '+name)
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old replay failure must remain failed')
    if path('registration').exists():
        registration=read('registration')
        if registration['python']!=platform.python_version():raise ValueError('Registered Python changed')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Registered source changed: '+name)
    return selection


def compute(selection):
    inputs=[];total=0
    for name in selection['inputs']:
        p=Path(processed_data_path('benchmark_fixtures',name))
        with p.open('rb') as f:raw=f.read(8388609)
        total+=len(raw)
        if len(raw)>8388608 or total>16777216:raise ValueError('Input budget exceeded')
        inputs.append(raw)
    primary,replication,proof=inputs
    if primary!=replication:raise ValueError('Saved numeric replay mismatch')
    proof=json.loads(proof)
    if not proof['exact_replay_equal'] or proof['primary_sha256']!=sha256(primary).hexdigest() or proof['replication_sha256']!=sha256(replication).hexdigest():raise ValueError('Saved proof mismatch')
    result=analyze(json.loads(primary))
    result.update(input_bytes_read=total,preserved_hashes=len(selection['preserved']),
                  registration_sha256=digest(path('registration')))
    return result


def main(mode):
    selection=verify()
    if mode=='register':
        sources=['scripts/eval/route_factorial_audit.py','src/sara_engine/evaluation/route_factorial_audit.py','tests/test_route_factorial_audit.py','src/sara_engine/utils/project_paths.py',str(path('scope').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'sources':{s:digest(project_path(s)) for s in sources}})
        return
    if not path('registration').exists():raise ValueError('Registration required')
    if mode=='verify':
        result=compute(selection)
        if canonical(result)!=canonical(read('result')):raise ValueError('Saved audit differs')
        replay=read('replay')
        if replay['result_sha256']!=digest(path('result')) or not replay['exact_match']:raise ValueError('Replay receipt differs')
        print('Pure verification passed; no learner imported');return
    if mode=='replay' and not path('result').exists():raise ValueError('Primary audit required')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration'))})
    result=compute(selection);verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Pure replay mismatch')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0})
    print(json.dumps(result['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
