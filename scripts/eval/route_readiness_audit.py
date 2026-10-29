#!/usr/bin/env python3
"""Bounded saved-proof audit only; imports no fixture or numeric calculator."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_readiness_audit import INPUTS,analyze,canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='08130c4ee4b9d2f749780822429ce4021425c97bd90cac34cbd45491b7a27785'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_readiness_audit_'+suffix+'_v1.json'))

def digest(file):return sha256(Path(file).read_bytes()).hexdigest()


def loads(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate saved-report field')
            result[key]=value
        return result
    def bad_constant(value):raise ValueError('Nonfinite saved-report constant: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad_constant)


def read(suffix):return loads(path(suffix).read_bytes())


def save(suffix,data):
    raw=canonical(data)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_readiness_audit_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>2097152:raise ValueError('Readiness artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Frozen readiness protocol changed')
    spec=read('selection')
    if tuple(spec['inputs'])!=INPUTS:raise ValueError('Input roster changed')
    for name,expected in spec['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Preserved evidence changed: '+name)
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old failed replay reclassified')
    if path('registration').exists():
        registration=read('registration')
        if registration['selection_sha256']!=SELECTION_SHA or registration['python']!=platform.python_version():raise ValueError('Registered protocol/runtime differs')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Registered source changed: '+name)
    return spec


def compute(spec):
    reports={};hashes={};total=0
    for name in INPUTS:
        p=Path(processed_data_path('benchmark_fixtures',name))
        with p.open('rb') as stream:raw=stream.read(8388609)
        total+=len(raw)
        if len(raw)>8388608 or total>8388608:raise ValueError('Finite saved-input budget exceeded')
        hashes[name]=sha256(raw).hexdigest()
        if hashes[name]!=spec['preserved']['data/processed/benchmark_fixtures/'+name]:raise ValueError('Saved input binding differs')
        reports[name]=loads(raw)
    result=analyze(spec,reports,hashes)
    result.update(input_bytes_read=total,source_input_sha256=hashes,preserved_hashes=len(spec['preserved']),
                  registration_sha256=digest(path('registration')))
    return result


def main(mode):
    spec=verify()
    if mode=='register':
        sources=['scripts/eval/route_readiness_audit.py','src/sara_engine/evaluation/route_readiness_audit.py',
                 'tests/test_route_readiness_audit.py','src/sara_engine/utils/project_paths.py']
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'selection_sha256':SELECTION_SHA,'sources':{s:digest(project_path(s)) for s in sources}})
        return
    if not path('registration').exists():raise ValueError('Readiness source registration required')
    if mode=='verify':
        if canonical(compute(spec))!=canonical(read('result')):raise ValueError('Saved readiness result differs')
        if read('replay')!={'exact_match':True,'result_sha256':digest(path('result')),'new_learner_executions':0}:raise ValueError('Readiness replay binding differs')
        print('Saved-proof readiness verified; no learning/scoring/acquisition/measurement');return
    if mode=='replay' and not path('result').exists():raise ValueError('Primary audit required')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),'saved_reports_only':True})
    result=compute(spec);verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Readiness replay differs')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'new_learner_executions':0})
    print(json.dumps({'readiness':result['readiness'],'selected_next_check':result['selected_next_check']['id'],'status':result['status']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
