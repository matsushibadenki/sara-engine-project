#!/usr/bin/env python3
"""Freeze, audit and replay 26 retained aggregate receipts; no experiments."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.r2_mechanism_decision import ROSTER,evaluate
from sara_engine.utils.project_paths import project_path,processed_data_path,ensure_parent_directory


def path(suffix):
    return Path(processed_data_path('benchmark_fixtures','r2_mechanism_decision_'+suffix+'_v1.json'))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def read(suffix):
    return json.loads(path(suffix).read_bytes())


def save(suffix,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    limits=read('selection')['budgets']
    if len(raw)>limits['result_bytes'] or sum(f.stat().st_size for f in path('selection').parent.glob('r2_mechanism_decision_*_v1.json'))+len(raw)>limits['total_artifact_bytes']:
        raise ValueError('Registered audit artifact budget exceeded')
    with Path(ensure_parent_directory(str(path(suffix)))).open('xb') as stream:stream.write(raw)


def verify():
    for collection in (read('selection')['preserved'],read('registration')['files']):
        for name,expected in collection.items():
            if digest(project_path(name))!=expected:
                raise ValueError('Pinned evidence/code changed: '+name)
    for suffix in ('result','replication','replay','replay_execution'):
        if Path(processed_data_path('benchmark_fixtures','event_allocation_'+suffix+'_v1.json')).exists():
            raise ValueError('Consumed failed v1 was resumed')


def load_receipts(selection):
    if tuple(selection['roster'])!=ROSTER:raise ValueError('Fixed aggregate receipt roster changed')
    limits=selection['budgets']
    if limits!={'receipt_files':26,'per_receipt_bytes':614400,'aggregate_receipt_bytes':2097152,'result_bytes':262144,'total_artifact_bytes':1048576}:
        raise ValueError('Fixed audit budget changed')
    receipts,total={},0
    for name in ROSTER:
        file=Path(processed_data_path('benchmark_fixtures',name+'.json'))
        with file.open('rb') as stream:raw=stream.read(limits['per_receipt_bytes']+1)
        total+=len(raw)
        if len(raw)>limits['per_receipt_bytes'] or total>limits['aggregate_receipt_bytes']:
            raise ValueError('Receipt parsing budget exceeded')
        receipts[name]=json.loads(raw)
    return receipts,total


def main(mode):
    if mode=='register':
        old=json.loads(Path(processed_data_path('benchmark_fixtures','event_allocation_v2_registration_v1.json')).read_text())
        names=['scripts/eval/r2_mechanism_decision.py','src/sara_engine/evaluation/r2_mechanism_decision.py',
               'tests/test_r2_mechanism_decision.py',str(path('selection').relative_to(project_path()))]
        save('registration',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'Known aggregate decision rules/candidate fixed before implementation; pure code/tests frozen before retained finite audit. No source acquisition, scoring, model execution or resource probes.',
             'files':{**old['files'],**{name:digest(project_path(name)) for name in names}}})
        verify()
        print('Finite mechanism/task decision audit registered')
        return
    verify()
    receipt={'exact_match':True,'result_sha256':digest(path('result')),
             'new_measurements':0,'diagnostic_worker_launches':0} if path('result').exists() else None
    if mode=='replay' and path('replay').exists():
        if read('replay')!=receipt:raise ValueError('Immutable decision-audit replay receipt differs')
        print('Consumed decision-audit replay verified; no experiments launched')
        return
    if mode=='replay' and receipt is None:raise ValueError('Saved audit result required before replay')
    save('execution' if mode=='execute' else 'replay_execution',{
         'started_at_utc':datetime.now(timezone.utc).isoformat(),'registration_sha256':digest(path('registration')),
         'selection_sha256':digest(path('selection'))})
    for suffix in ('registration','selection'):
        if read('execution')[suffix+'_sha256']!=digest(path(suffix)):
            raise ValueError('Consumed decision audit input changed')
    receipts,total=load_receipts(read('selection'))
    result=evaluate(read('selection'),receipts)
    result.update(receipt_bytes_read=total,prior_artifacts_preserved=len(read('selection')['preserved']))
    verify()
    if mode=='execute':save('result',result)
    else:
        if result!=read('result'):raise ValueError('Read-only decision audit replay differs')
        save('replay',receipt)
    print(json.dumps({k:result[k] for k in ('decision','receipt_count','receipt_bytes_read','prior_artifacts_preserved','new_measurements')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
