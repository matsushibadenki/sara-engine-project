#!/usr/bin/env python3
"""New fixed-condition baseline and real external capture, never old restart."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.reactivation_codec import ReportCodec,loads
from sara_engine.evaluation.route_numeric_v2 import canonical,payload
from sara_engine.evaluation.route_snapshot_contract import CATALOG,encode,encode_collection
from sara_engine.evaluation.route_capture_v3 import PROTOCOL_SHA256,Observer,analyze,evaluator_truth,record,validate_record
from sara_engine.utils.project_paths import processed_data_path,project_path,ensure_parent_directory

BUDGETS={'phases':2,'cases_per_phase':20,'arms_per_case':5,'snapshots':160,
         'numeric_report_bytes':2097152,'collection_bytes':2097152,'auxiliary_bytes':1048576,'all_artifacts_bytes':7340032}


def path(suffix):return Path(processed_data_path('benchmark_fixtures','route_capture_v3_'+suffix+'_v1.json'))


def prior(name):return Path(processed_data_path('benchmark_fixtures',name))


def digest(file):return sha256(Path(file).read_bytes()).hexdigest()


def read_file(file):
    with Path(file).open('rb') as stream:raw=stream.read(2097153)
    return loads(raw)


def read(suffix):return read_file(path(suffix))


def save_bytes(suffix,raw):
    report=suffix in ('baseline','capture');collection=suffix=='collection'
    cap=2097152 if report or collection else 1048576
    files=list(path('selection').parent.glob('route_capture_v3_*_v1.json'))
    used=sum(f.stat().st_size for f in files)
    aux=sum(f.stat().st_size for f in files if f not in (path('baseline'),path('capture'),path('collection')))
    if len(raw)>cap or used+len(raw)>7340032 or (not report and not collection and aux+len(raw)>1048576):
        raise ValueError('Registered capture/report/auxiliary budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def save(suffix,value):
    canonical(value)
    save_bytes(suffix,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def runtime():
    return {'implementation':platform.python_implementation(),'python':platform.python_version(),
            'version':sys.version,'machine':platform.machine(),'executable':sys.executable}


def verify():
    if digest(path('selection'))!=PROTOCOL_SHA256 or read('selection')['budgets']!=BUDGETS:raise ValueError('Frozen collection protocol/budgets changed')
    selection,registration=read('selection'),read('registration')
    for pins in (selection['preserved'],registration['sources']):
        for name,expected in pins.items():
            if digest(project_path(name))!=expected:raise ValueError('Pinned evidence/source changed: '+name)
    old=read_file(prior('sparse_route_reactivation_selection_v1.json'))
    spec={k:v for k,v in old.items() if k not in {'schema','registered_at_utc','authorization','preserved','stop'}}
    contract=read_file(prior('route_snapshot_contract_selection_v1.json'))
    if canonical(spec)!=canonical(selection['frozen_scientific_spec']) or encode(selection['boundary_roster'])!=encode(contract['boundary_roster']):raise ValueError('Scientific conditions or160 boundaries changed')
    old_runtime=read_file(prior('route_numeric_v2_registration_v1.json'))['runtime']
    if canonical(runtime())!=canonical(registration['runtime']) or canonical(runtime())!=canonical(old_runtime):raise ValueError('Exact registered CPython runtime required')
    if prior('sparse_route_reactivation_replay_v1.json').exists():raise ValueError('Old failed attempt reclassified')
    failed=read_file(prior('sparse_route_reactivation_failure_v1.json'))
    if failed['full_replay_verified'] is not False or failed['numeric_trace_equality'] is not None:raise ValueError('Old replay failure changed')
    if read_file(prior('route_snapshot_contract_result_v1.json'))['contract_preflight_passed'] is not True:raise ValueError('Snapshot preflight required')


def marker(phase):
    value=read(phase+'_execution')
    if set(value)!={'pid','started_at_utc','registration_sha256','selection_sha256'} or value['registration_sha256']!=digest(path('registration')) or value['selection_sha256']!=PROTOCOL_SHA256:raise ValueError('Phase binding changed')
    if type(value['pid']) is not int or value['pid']<=0 or datetime.fromisoformat(value['started_at_utc']).utcoffset() is None:raise ValueError('Actual process/time identity required')
    return value


def reference():return read_file(prior('sparse_route_reactivation_result_v1.json'))


def context(phase):
    marker(phase)
    return {'phase':phase,'registration_sha256':digest(path('registration')),
            'execution_sha256':digest(path(phase+'_execution')),'runtime':runtime(),
            'preserved_count':len(read('selection')['preserved'])}


def validated(phase):return validate_record(read(phase),reference(),context(phase))


def run_study(observer=None):
    from sara_engine.evaluation.sparse_route_reactivation import run_case,summarize
    spec=read('selection')['frozen_scientific_spec'];cases=[]
    with observer.install() if observer is not None else nullcontext():
        for task in spec['tasks']:
            for seed in spec['seeds']:
                for condition in spec['conditions']:
                    if observer is not None:observer.set_case(task+':'+str(seed)+':'+condition)
                    case=run_case(seed,task,condition);cases.append(case)
                    print(json.dumps({'task':task,'seed':seed,'condition':condition,
                                      'failed_gates':[k for k,v in case['component_gates'].items() if not v]}),flush=True)
    if len(cases)!=20 or any(len(c['arms'])!=5 for c in cases):raise ValueError('Fixed study roster changed')
    return payload(cases,summarize(cases),reference(),digest(path('registration')),len(read('selection')['preserved']))


def collection_header():
    roster=read('selection')['boundary_roster']
    return {'schema':'sara-route-evidence-collection-v1',
            'contract_sha256':read('selection')['snapshot_contract_sha256'],'catalog':CATALOG,
            'roster_sha256':sha256(encode(roster)).hexdigest(),'provenance':'future_external_state',
            'executor_binding':{'registration_sha256':digest(path('registration')),
                                'execution_sha256':digest(path('capture_execution'))}}


def validate_collection():
    collection=read('collection');roster=read('selection')['boundary_roster']
    if encode(collection['header'])!=encode(collection_header()):raise ValueError('Actual collection provenance differs')
    expected=encode_collection(collection['header'],collection['snapshots'],roster)
    if sha256(expected).hexdigest()!=digest(path('collection')):raise ValueError('Collection representation differs')
    return collection


def proof():
    base,captured=validated('baseline'),validated('capture')
    if marker('baseline')['pid']==marker('capture')['pid']:raise ValueError('Separate capture process required')
    if not ReportCodec(reference()).equivalent(base['payload'],captured['payload']):raise ValueError('Instrumentation changed numerical payload')
    collection=validate_collection();saved=read('analysis')
    expected=analyze(collection,captured['payload'],saved['diagnostics']['evaluator_truth'])
    if canonical(expected)!=canonical(saved['diagnostics']):raise ValueError('Saved evidence diagnosis differs')
    order=saved['actual_capture_order'];checks=saved['read_only_capture_checks']
    roster=read('selection')['boundary_roster']
    if len(order)!=160 or len(checks)!=160 or {encode(p) for p in order}!={encode(p) for p in roster}:raise ValueError('Actual capture order/coverage incomplete')
    case_index={c['task']+':'+str(c['seed'])+':'+c['condition']:c for c in captured['payload']['cases']}
    for point,check in zip(order,checks):
        if encode(point)!=encode(check['point']) or check['before_sha256']!=check['after_sha256']:raise ValueError('Capture changed full state or lost provenance')
        if point['stage']=='final' and check['before_sha256']!=case_index[point['case_id']]['arms'][point['arm']]['state_sha256']:raise ValueError('Final snapshot not linked to numerical state digest')
    return {'schema':'sara-route-capture-proof-v3','numerical_noninterference_passed':True,
        'historical_values_exact_match':True,'actual_snapshots':160,'complete_state_read_only_checks':160,
        'baseline_sha256':digest(path('baseline')),'capture_sha256':digest(path('capture')),
        'collection_sha256':digest(path('collection')),'analysis_sha256':digest(path('analysis')),
        'selection_sha256':PROTOCOL_SHA256,'registration_sha256':digest(path('registration')),
        'independent_processes':True,'summary':expected['summary'],
        'component_passed':captured['payload']['component_passed'],'old_failed_replay_reclassified':False,
        'old_discarded_trace_equality':None,'conditions_relaxed':False,'test_opened':False,
        'resource_measurements':0,'new_real_data_reads':0,'causal_attribution':None,
        'counterfactual_accuracy':None,'promotion_authorized':False}


def main(mode):
    if mode=='register':
        names=['scripts/eval/route_capture_v3.py','src/sara_engine/evaluation/route_capture_v3.py',
               'tests/test_route_capture_v3.py',str(path('selection').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime(),
                             'sources':{n:digest(project_path(n)) for n in names}})
        verify();print('Separate fixed-condition real collection v3 registered');return
    verify()
    if mode=='verify':
        expected=proof()
        if canonical(expected)!=canonical(read('result')):raise ValueError('Immutable capture proof differs')
        print(json.dumps(expected,sort_keys=True));return
    phase='baseline' if mode=='baseline' else 'capture'
    if path('baseline_failure').exists() or path('capture_failure').exists():raise ValueError('Consumed failed v3 phase cannot restart')
    if phase=='capture':
        validated('baseline')
        if marker('baseline')['pid']==os.getpid():raise ValueError('Fresh capture process required')
    save(phase+'_execution',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'registration_sha256':digest(path('registration')),'selection_sha256':PROTOCOL_SHA256})
    try:
        observer=Observer(read('selection')['boundary_roster']) if phase=='capture' else None
        value=run_study(observer)
        result=record(value,reference(),context(phase));save(phase,result)
        if observer is not None:
            snapshots=observer.finish()
            raw=encode_collection(collection_header(),snapshots,read('selection')['boundary_roster'])
            save_bytes('collection',raw)
            # Truth is produced after all learner work and hook installation ends.
            truth=evaluator_truth(value['cases'])
            diagnostics=analyze(loads(raw),result['payload'],truth)
            save('analysis',{'diagnostics':diagnostics,'actual_capture_order':observer.actual_order,
                             'read_only_capture_checks':observer.read_only_checks})
        verify();validated(phase)
        if phase=='capture':save('result',proof())
    except Exception as exc:
        save(phase+'_failure',{'status':'failed_consumed_v3_phase','phase':phase,
             'error_type':type(exc).__name__,'error':str(exc),'execution_sha256':digest(path(phase+'_execution')),
             'numeric_report_retained':path(phase).exists(),'collection_retained':path('collection').exists(),
             'retry_authorized':False})
        raise
    print(json.dumps({'phase':phase,'historical_values_exact_match':result['historical_values_exact_match'],
                      'component_passed':result['payload']['component_passed']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','baseline','collect','verify'))
    main(parser.parse_args().mode)
