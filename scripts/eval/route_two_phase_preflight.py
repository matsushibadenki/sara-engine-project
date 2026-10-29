#!/usr/bin/env python3
"""Hand-fixture two-phase witness sealing; never runs a learner."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from sara_engine.evaluation.route_two_phase_preflight import analyze
from sara_engine.evaluation.route_fixed_allocation_contract import canonical
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

SELECTION_SHA='5c8a562798daf7b2c31a18dae5593cedeec8b4b2dadd6c7ce390225f540b64bf'


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_two_phase_preflight_'+suffix+'_v1.json'))

def digest(file):return sha256(Path(file).read_bytes()).hexdigest()

def read(suffix):return json.loads(path(suffix).read_text())


def save(suffix,data):
    raw=canonical(data)+b'\n'
    used=sum(p.stat().st_size for p in path('selection').parent.glob('route_two_phase_preflight_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>2097152:raise ValueError('Metadata preflight budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=SELECTION_SHA:raise ValueError('Frozen interface protocol changed')
    spec=read('selection')
    for name,expected in spec['preserved'].items():
        if digest(project_path(name))!=expected:raise ValueError('Historical file changed: '+name)
    if digest(processed_data_path('benchmark_fixtures',spec['parent']['file']))!=spec['parent']['sha256']:raise ValueError('Parent control contract binding differs')
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old failed replay reclassified')
    if path('registration').exists():
        registration=read('registration')
        if registration['python']!=platform.python_version() or registration['selection_sha256']!=SELECTION_SHA:raise ValueError('Registered runtime/protocol differs')
        for name,expected in registration['sources'].items():
            if digest(project_path(name))!=expected:raise ValueError('Registered source changed: '+name)
    return spec


def compute(spec):
    result=analyze(spec)
    result.update(registration_sha256=digest(path('registration')),preserved_hashes=len(spec['preserved']))
    return result


def main(mode):
    spec=verify()
    if mode=='register':
        names=['scripts/eval/route_two_phase_preflight.py','src/sara_engine/evaluation/route_two_phase_preflight.py',
               'tests/test_route_two_phase_preflight.py','src/sara_engine/evaluation/route_error_binding_preflight.py',
               'src/sara_engine/evaluation/route_numeric_delta_preflight.py',
               'src/sara_engine/evaluation/route_delta_ledger_preflight.py',
               'src/sara_engine/evaluation/route_interface_preflight.py',
               'src/sara_engine/evaluation/route_fixed_allocation_contract.py',
               'src/sara_engine/evaluation/route_fixed_control_contract.py','src/sara_engine/utils/project_paths.py']
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'selection_sha256':SELECTION_SHA,'sources':{n:digest(project_path(n)) for n in names}})
        return
    if not path('registration').exists():raise ValueError('Interface source registration required')
    if mode=='verify':
        if canonical(compute(spec))!=canonical(read('result')):raise ValueError('Metadata proof differs')
        if read('replay')!={'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0}:raise ValueError('Replay binding differs')
        print('Two-phase hand-witness seal verified; no learner or scoring');return
    if mode=='replay' and not path('result').exists():raise ValueError('Primary preflight required')
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),'metadata_only':True})
    result=compute(spec);verify()
    if mode=='execute':save('result',result)
    else:
        if canonical(result)!=canonical(read('result')):raise ValueError('Metadata replay differs')
        save('replay',{'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0})
    print(json.dumps(result['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay','verify'))
    main(parser.parse_args().mode)
