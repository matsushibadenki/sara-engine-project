#!/usr/bin/env python3
"""Fresh registered v2 allocation diagnostic with bounded retained reports."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation import event_allocation_v2 as audit
from sara_engine.evaluation import allocation_report as codec
from sara_engine.utils.project_paths import project_path,processed_data_path,ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures','event_allocation_v2_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_bytes())


def runtime():
    return {'python':platform.python_version(),'implementation':platform.python_implementation(),
            'machine':platform.machine(),'byteorder':sys.byteorder}


def write(suffix,raw):
    selection=read('selection')
    report=suffix in ('result','replication')
    existing=list(path('selection').parent.glob('event_allocation_v2_*_v1.json'))
    aux=sum(f.stat().st_size for f in existing if f not in (path('result'),path('replication')))
    if (len(raw) > (codec.LIMIT if report else selection['budgets']['auxiliary_bytes'])
            or (not report and aux+len(raw)>selection['budgets']['auxiliary_bytes'])
            or sum(f.stat().st_size for f in existing)+len(raw)>selection['budgets']['total_artifact_bytes']):
        raise ValueError('Registered artifact byte budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:
        stream.write(raw)


def save(suffix,value):
    write(suffix,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def verify(worker=False):
    registration=read('registration')
    if registration['runtime']!=runtime() or sys.version_info[:2]!=(3,10):
        raise ValueError('Use the registered CPython runtime')
    if not worker:
        for collection in (read('selection')['preserved'],registration['files']):
            for name,expected in collection.items():
                if digest(project_path(name))!=expected:
                    raise ValueError('Pinned evidence/code changed: '+name)
    for suffix in ('result','replication','replay','replay_execution'):
        if Path(processed_data_path('benchmark_fixtures','event_allocation_'+suffix+'_v1.json')).exists():
            raise ValueError('Consumed v1 allocation run changed')


def require_phase(phase):
    if phase not in ('execute','replay') or path('failure').exists():
        raise ValueError('Invalid or failed diagnostic phase')
    marker=read('execution' if phase=='execute' else 'replay_execution')
    for suffix in ('registration','selection'):
        if marker[suffix+'_sha256']!=digest(path(suffix)):
            raise ValueError('Consumed phase contract changed')
    if path('result' if phase=='execute' else 'replication').exists():
        raise ValueError('Completed diagnostic phase cannot restart')
    if phase=='replay':
        codec.saved_replication_gate(path('result'),marker['result_sha256'],digest(path('registration')),
                                     digest(path('execution')),lambda value:None)


def marker(phase):
    result={'started_at_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,
            'registration_sha256':digest(path('registration')),'selection_sha256':digest(path('selection'))}
    if phase=='replay':result['result_sha256']=digest(path('result'))
    save('execution' if phase=='execute' else 'replay_execution',result)


def run_phase(phase):
    marker(phase)
    require_phase(phase)
    workers,witnesses=audit.collect(read('selection'),phase)
    value=audit.payload(workers,witnesses,digest(path('registration')),
                        digest(path('execution' if phase=='execute' else 'replay_execution')),runtime())
    raw=codec.encode(value)
    if codec.decode(raw)!=value:
        raise ValueError('Real report round-trip differs')
    verify()
    write('result' if phase=='execute' else 'replication',raw)
    return value


def receipt(original,replica):
    return {**audit.summarize(original,replica),'result_sha256':digest(path('result')),
            'replication_sha256':digest(path('replication')),'result_bytes':path('result').stat().st_size,
            'replication_bytes':path('replication').stat().st_size,
            'prior_artifacts_preserved':len(read('selection')['preserved'])}


def main(mode,case_index=None,arm=None,phase=None):
    if mode=='register':
        if sys.version_info[:2]!=(3,10) or platform.python_implementation()!='CPython':
            raise ValueError('CPython 3.10 required')
        old=json.loads(Path(processed_data_path('benchmark_fixtures','allocation_report_registration_v1.json')).read_text())
        names=['scripts/eval/event_allocation_v2.py','src/sara_engine/evaluation/event_allocation_v2.py',
               'tests/test_event_allocation_v2.py',str(path('selection').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime(),
             'scope':'Fresh v2 selected before implementation; immutable code/tests before finite primary and one gated independent replication. Failed v1 stays consumed.',
             'files':{**old['files'],**{name:digest(project_path(name)) for name in names}}})
        verify()
        print('Fresh allocation v2 code registered')
        return
    verify(worker=mode=='worker')
    if mode=='worker':
        require_phase(phase)
        if type(case_index)is not int or not 0<=case_index<48 or arm not in codec.MODES:
            raise ValueError('Worker outside fixed v2 roster')
        events,golden=audit.inputs(read('selection'))[case_index]
        print(json.dumps(audit.frozen.run_case(events,golden,arm),sort_keys=True,separators=(',',':')))
        return
    if path('failure').exists():raise ValueError('Failed v2 is consumed; no retry allowed')
    if mode=='execute':
        value=run_phase('execute')
        save('summary',{**audit.summarize(value),'result_sha256':digest(path('result')),
                        'result_bytes':path('result').stat().st_size,'prior_artifacts_preserved':len(read('selection')['preserved'])})
    elif path('replay').exists():
        expected=receipt(codec.decode(path('result').read_bytes()),codec.decode(path('replication').read_bytes()))
        if read('replay')!=expected:raise ValueError('Immutable replication receipt differs')
        print('Consumed v2 replication verified; no new worker launched')
    else:
        def replicate(original):
            replica=run_phase('replay')
            save('replay',receipt(original,replica))
        # Gate runs BEFORE marker creation, calibration or any worker launch.
        codec.saved_replication_gate(path('result'),digest(path('result')),digest(path('registration')),
                                     digest(path('execution')),replicate)
    print('Allocation v2 phase completed: '+mode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','worker'))
    parser.add_argument('--case',type=int,dest='case_index')
    parser.add_argument('--arm',choices=codec.MODES)
    parser.add_argument('--phase',choices=('execute','replay'))
    args=parser.parse_args()
    already_consumed=path('execution' if args.mode=='execute' else 'replay_execution').exists()
    try:
        main(args.mode,args.case_index,args.arm,args.phase)
    except Exception as error:
        if args.mode in ('execute','replay') and not already_consumed and path('execution' if args.mode=='execute' else 'replay_execution').exists() and not path('failure').exists():
            save('failure',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'phase':args.mode,
                 'status':'failed_v2_consumed','error_type':type(error).__name__,'error':str(error)[:2000],
                 'worker_stderr':str(getattr(error,'stderr','') or '')[-4000:],
                 'result_retained':path('result').exists(),'replication_retained':path('replication').exists(),
                 'actual_learning_writes':0,'predictions_scored':0,'no_retry':True})
        raise
