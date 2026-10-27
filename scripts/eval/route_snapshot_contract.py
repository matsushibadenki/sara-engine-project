#!/usr/bin/env python3
"""Static contract/capacity check only; never collects an actual model state."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.reactivation_codec import loads
from sara_engine.evaluation.route_snapshot_contract import (
    CATALOG,LIMITS,PROTOCOL_SHA256,capture,compare_pair,diagnose,encode,encode_collection,
    symbolic_state,
)
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

INPUTS=('route_numeric_v2_primary_v1.json','route_failure_audit_result_v1.json')


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures','route_snapshot_contract_'+suffix+'_v1.json'))


def digest(file):return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):return loads(path(suffix).read_bytes())


def save(suffix,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    used=sum(f.stat().st_size for f in path('selection').parent.glob('route_snapshot_contract_*_v1.json'))
    if len(raw)>262144 or used+len(raw)>1048576:raise ValueError('Registered static artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    if digest(path('selection'))!=PROTOCOL_SHA256:raise ValueError('Frozen snapshot contract changed')
    selection=read('selection')
    if selection['limits']!=LIMITS or selection['saved_inputs']!=list(INPUTS) or encode(selection['catalog'])!=encode(CATALOG):raise ValueError('Fixed scope/capacity/catalog changed')
    for pins in (selection['preserved'],read('registration')['sources']):
        for name,expected in pins.items():
            if digest(project_path(name))!=expected:raise ValueError('Pinned evidence/code changed: '+name)
    if read('registration')['python']!=platform.python_version():raise ValueError('Use registered Python patch release')
    if Path(processed_data_path('benchmark_fixtures','sparse_route_reactivation_replay_v1.json')).exists():raise ValueError('Old replay failure was reclassified')


def planned_roster(saved):
    result=[]
    for case in saved['payload']['cases']:
        identity=case['task']+':'+str(case['seed'])+':'+case['condition']
        for name in ('error_triggered','permanent'):
            arm=case['arms'][name]
            events=[(180,'initial')]+[(t,'start') for t in arm['exploration_starts']]+[(c['tick'],'commit') for c in arm['commits']]
            for tick,kind in sorted(events):
                for side in ('pre','post'):result.append({'case_id':identity,'arm':name,'tick':tick,'stage':side+'_'+kind})
            result.append({'case_id':identity,'arm':name,'tick':1024,'stage':'final'})
    return result


def evaluate():
    selection=read('selection');saved=[];total=0
    for name in INPUTS:
        file=Path(processed_data_path('benchmark_fixtures',name))
        with file.open('rb') as stream:raw=stream.read(2097153)
        total+=len(raw)
        if len(raw)>2097152 or total>2097152:raise ValueError('Fixed saved-input byte ceiling exceeded')
        if sha256(raw).hexdigest()!=selection['preserved']['data/processed/benchmark_fixtures/'+name]:raise ValueError('Saved planning input changed')
        saved.append(loads(raw))
    roster=planned_roster(saved[0])
    if encode(roster)!=encode(selection['boundary_roster']) or len(roster)!=160:raise ValueError('Finite prospective boundary plan differs')
    if saved[1]['summary']['exact_retention_failures']!=3 or saved[1]['component_passed'] is not False:raise ValueError('Prior negative audit differs')
    # Symbolic fixtures exercise the API, never stand in for missing old states.
    state=symbolic_state();before=deepcopy(state)
    pre=capture(state,roster[0],'symbolic_contract_fixture')
    state['pool']=tuple(pre['proposed_pool'])
    post=capture(state,roster[1],'symbolic_contract_fixture')
    pair=compare_pair(pre,post)
    expected=deepcopy(before);expected['pool']=state['pool']
    if state!=expected:raise ValueError('Collector mutated input learner fields')
    header={'schema':'sara-route-evidence-collection-v1','contract_sha256':PROTOCOL_SHA256,
            'catalog':CATALOG,'roster_sha256':sha256(encode(roster[:2])).hexdigest(),
            'provenance':'symbolic_contract_fixture','executor_binding':None}
    fixture=encode_collection(header,[pre,post],roster[:2])
    bound=160*8192+65536+160+1
    if bound>=2097152:raise ValueError('Registered conditional capacity bound failed')
    return {'schema':'sara-route-snapshot-contract-result-v1',
        'contract_preflight_passed':pair,'planned_snapshots':160,
        'planned_candidate_snapshots':sum(p['arm']=='error_triggered' for p in roster),
        'planned_permanent_snapshots':sum(p['arm']=='permanent' for p in roster),
        'registered_conservative_report_bound_bytes':bound,
        'tight_bound_given_framing_cap_bytes':160*8192+65536+159,
        'report_byte_ceiling':2097152,'symbolic_collection_bytes':len(fixture),
        'symbolic_fixture_sha256':sha256(fixture).hexdigest(),
        'evaluator_ids_from_old_reports':None,'symbolic_unknown_diagnostic':diagnose(post),
        'actual_snapshots_collected':0,'learner_executions':0,'new_predictions_scored':0,
        'new_measurements':0,'new_acquisitions':0,'old_failed_replay_reclassified':False,
        'old_discarded_trace_equality':None,'prior_component_passed':False,
        'test_opened':False,'conditions_relaxed':False,'future_collection_authorized':False,
        'new_experiment_authorized':False,'promotion_authorized':False,
        'input_bytes_read':total,'preserved_hashes':len(selection['preserved']),
        'decision':'Ready for separately authorized prospective hook integration only; no missing historical state reconstructed and no new trial.'}


def main(mode):
    if mode=='register':
        names=['scripts/eval/route_snapshot_contract.py','src/sara_engine/evaluation/route_snapshot_contract.py',
               'tests/test_route_snapshot_contract.py','src/sara_engine/utils/project_paths.py',str(path('selection').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                             'sources':{n:digest(project_path(n)) for n in names}})
        verify();print('Prospective snapshot contract registered; no collection authorized');return
    verify()
    if mode=='replay' and not path('result').exists():raise ValueError('Saved static result required')
    receipt={'exact_match':True,'result_sha256':digest(path('result')),'learner_executions':0} if path('result').exists() else None
    if mode=='replay' and path('replay').exists():
        if encode(read('replay'))!=encode(receipt):raise ValueError('Immutable static replay differs')
        print('Consumed static snapshot-contract replay verified; no model started');return
    save('execution' if mode=='execute' else 'replay_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration'))})
    if read('execution')['registration_sha256']!=digest(path('registration')):raise ValueError('Consumed static binding differs')
    result=evaluate();verify()
    if mode=='execute':save('result',result)
    else:
        if encode(result)!=encode(read('result')):raise ValueError('Pure contract replay differs')
        save('replay',receipt)
    print(json.dumps({k:result[k] for k in ('contract_preflight_passed','planned_snapshots','actual_snapshots_collected','learner_executions')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
