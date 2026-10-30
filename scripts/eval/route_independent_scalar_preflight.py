#!/usr/bin/env python3
"""Versioned independent arithmetic on the fixed eleven literal records."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_independent_scalar_preflight import analyze,canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='eeecf9fb7bb6ebbf9f34f7013faabe9e43b179f16eb2078c0f477832c2514b52'
KERNEL='src/sara_engine/evaluation/route_independent_scalar.py'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_independent_scalar_preflight_'+suffix+'_v1.json'))

def digest(file):return sha256(Path(file).read_bytes()).hexdigest()


def loads(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate fixed-input JSON key')
            result[key]=value
        return result
    def bad(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def read(suffix):return loads(path(suffix).read_bytes())


def save(suffix,value):
    raw=canonical(value)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_independent_scalar_preflight_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>2097152:raise ValueError('Independent arithmetic artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Frozen independent protocol changed')
    spec=read('selection')
    for name,expected in spec['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Historical evidence changed: '+name)
    if digest(processed_data_path('benchmark_fixtures',spec['input']['file']))!=spec['input']['sha256']:raise ValueError('Literal-source binding differs')
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old failed replay reclassified')
    if path('registration').exists():
        registration=read('registration')
        if registration['selection_sha256']!=SELECTION_SHA or registration['python']!=platform.python_version():raise ValueError('Registered protocol/runtime differs')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Frozen source changed: '+name)
    return spec


def compute(spec):
    with Path(processed_data_path('benchmark_fixtures',spec['input']['file'])).open('rb') as stream:raw=stream.read(262145)
    if len(raw)>262144:raise ValueError('Literal source byte budget exceeded')
    if sha256(raw).hexdigest()!=spec['input']['sha256']:raise ValueError('Literal input changed')
    rows=loads(raw)['local_rows']
    source=Path(project_path(KERNEL)).read_text()
    result=analyze(spec,rows,source)
    result.update(input_bytes_read=len(raw),input_sha256=sha256(raw).hexdigest(),
                  registration_sha256=digest(path('registration')),preserved_hashes=len(spec['preserved']))
    return result


def main(mode):
    spec=verify()
    if mode=='register':
        sources=[KERNEL,'scripts/eval/route_independent_scalar_preflight.py',
                 'src/sara_engine/evaluation/route_independent_scalar_preflight.py',
                 'tests/test_route_independent_scalar_preflight.py','src/sara_engine/utils/project_paths.py']
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'selection_sha256':SELECTION_SHA,'sources':{s:digest(project_path(s)) for s in sources}})
        return
    if not path('registration').exists():raise ValueError('Source registration required')
    if mode=='verify':
        result=compute(spec)
        if not result['summary']['all_exact'] or canonical(result)!=canonical(read('result')):raise ValueError('Independent arithmetic saved proof differs')
        if read('replay')!={'exact_match':True,'result_sha256':digest(path('result')),'new_learner_executions':0}:raise ValueError('Independent replay binding differs')
        print('Independent literal arithmetic verified; no learner or task scoring');return
    if mode=='replay':
        if not path('result').exists():raise ValueError('Primary result required')
        if path('failure').exists() or read('result')['summary']['all_exact'] is not True:raise ValueError('Failed primary cannot start replication')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),'stateless_literal_arithmetic_only':True})
    try:
        result=compute(spec);verify()
    except Exception as exc:
        save('failure' if mode=='execute' else 'replay_failure',{'phase':mode,'error':type(exc).__name__+': '+str(exc),'new_learner_executions':0,'retry_authorized':False})
        raise
    if mode=='execute':
        save('result',result)
        if not result['summary']['all_exact']:
            save('failure',{'phase':mode,'reason':'Literal golden mismatch; retained report','retry_authorized':False})
            raise ValueError('Golden mismatch; no retuning or retry')
    else:
        if not result['summary']['all_exact'] or canonical(result)!=canonical(read('result')):
            save('replay_failure',{'reason':'Exact independent replay differs','retry_authorized':False})
            raise ValueError('Exact replay differs')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'new_learner_executions':0})
    print(json.dumps(result['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
