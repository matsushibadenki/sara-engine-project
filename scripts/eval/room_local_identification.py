#!/usr/bin/env python3
"""Post-support training-only identification; no held-out predictive evidence."""
import argparse
from collections import defaultdict
from heapq import heappush, heappop
import json
from pathlib import Path
from sara_engine.evaluation.room_temporal import extract, keys, CODE
from sara_engine.evaluation.motion_temporal import LocalReadout, digest, save
from sara_engine.utils.project_paths import raw_data_path, processed_data_path, project_path

ARMS=('current','current_fine','age','histogram','sequence','scalar_age')
PREFIX='room_local_identification'

def learn(rows,arm):
    model=LocalReadout(); queue=[]; delivered=0
    def flush(time):
        nonlocal delivered
        while queue and queue[0][0]<=time:
            due,_,receipt,y=heappop(queue)
            model.pending=receipt
            model.deliver(due,lambda:y,True)
            delivered+=1
    for i,(cutoff,due,history,y) in enumerate(rows):
        flush(cutoff)
        model.predict(keys(history,cutoff,arm),cutoff,due)
        receipt=model.pending;model.pending=None
        heappush(queue,(due,i,receipt,y))
    if queue:flush(max(item[0] for item in queue))
    return model,delivered

def identify(rows):
    reports={};models={}
    for arm in ARMS:
        model,n=learn(rows,arm);models[arm]=model
        encoded=json.dumps(model.cells,sort_keys=True)
        restored=LocalReadout();restored.cells=json.loads(encoded)
        if restored.cells!=model.cells:raise ValueError('Checkpoint differs')
        reports[arm]={'delivered_outcomes':n,'changed_writes':model.writes,'routes':len(model.cells),'checkpoint_sha256':__import__('hashlib').sha256(encoded.encode()).hexdigest()}
    groups=defaultdict(dict)
    for t,due,h,y in rows:
        k=keys(h,t,'age');model=models['age']
        p=model.predict(k,t,due);model.deliver(due,lambda:y,False)
        groups[k[0]][k[1]]=p
    witnesses=[]
    for current,branches in groups.items():
        if min(branches.values())<.5<max(branches.values()):
            witnesses.append({'current_address':current,'low_probability':min(branches.values()),'high_probability':max(branches.values())})
    return {'arms':reports,'learned_same_current_opposite_posterior_groups':len(witnesses),'witnesses':witnesses,'scalar_equivalent':models['age'].cells==models['scalar_age'].cells,'development_scores_computed':False,'test_opened':False,'scope':'Post-support descriptive training-only identification; no generalization, temporal necessity or SNN advantage claim'}

def main(mode):
    path=lambda suffix:processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')
    if mode=='register':
        code=(*CODE,'scripts/eval/room_local_identification.py')
        save(path('registration'),{'reason':'Frozen source gate failed: development has no positive occupancy and only 3/5 train contrast groups. Diagnostic added after this known result; not an unseen preregistered quality experiment. No split/feature/horizon/threshold change.','scope':'Only 5200 training examples; six matched count readouts; 10400 changed writes each; time-ordered receipt queue; due outcomes flushed before prediction at same timestamp; no development scoring, no production artifact','code':{p:digest(project_path(p)) for p in code},'source_sha256':digest(raw_data_path('r2_room','source.zip')),'evidence':{p.name:digest(p) for p in Path(processed_data_path('benchmark_fixtures')).glob('room_temporal*_v1.json')}})
        print('Training-only diagnostic registered');return
    reg=json.loads(Path(path('registration')).read_text())
    for p,h in reg['code'].items():
        if digest(project_path(p))!=h:raise ValueError('Code changed')
    for p,h in reg['evidence'].items():
        if digest(processed_data_path('benchmark_fixtures',p))!=h:raise ValueError('Evidence changed')
    if digest(raw_data_path('r2_room','source.zip'))!=reg['source_sha256']:raise ValueError('Source changed')
    if mode=='execute':save(path('execution'),{'started':True})
    rows,_=extract(raw_data_path('r2_room','source.zip'))
    report=identify(rows['train'])
    if mode=='execute':save(path('result'),report)
    elif report!=json.loads(Path(path('result')).read_text()):raise ValueError('Replay differs')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
