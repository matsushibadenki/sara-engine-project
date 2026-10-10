"""PIR-only prospective room-occupancy source and local-readout screen."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from sara_engine.evaluation.motion_temporal import LocalReadout, digest, save
from sara_engine.utils.project_paths import processed_data_path, raw_data_path, project_path

PREFIX = 'room_temporal'
CODE = ('src/sara_engine/evaluation/room_temporal.py','src/sara_engine/evaluation/motion_temporal.py','scripts/eval/room_temporal_preflight.py','src/sara_engine/utils/project_paths.py')
ARMS = ('current','current_fine','age','histogram','sequence','age_shuffled','scalar_age')

def keys(history, time, arm):
    base = (history[-1], time.hour, time.minute//15)
    if arm == 'current':
        detail = base
    elif arm == 'current_fine':
        detail = (history[-1],time.hour,time.minute)
    elif arm == 'histogram':
        detail = (base, tuple(sorted(Counter(history[:-1]).items())))
    elif arm == 'sequence':
        detail = (base,history[:-1])
    else:
        ages = []
        for sensor in range(2):
            last = next((i for i,v in enumerate(reversed(history[:-1]),1) if v[sensor]),16)
            ages.append(0 if last==1 else 1 if last<=4 else 2 if last<=8 else 3)
        detail = (base,tuple(ages))
    return repr(('base',base)),repr(('detail',detail))

def extract(archive):
    with ZipFile(archive) as z:
        rows=csv.DictReader(StringIO(z.read('Occupancy_Estimation.csv').decode()))
        opened=[]; sealed=0
        for row in rows:
            t=datetime.strptime(row['Date']+' '+row['Time'],'%Y/%m/%d %H:%M:%S')
            if t>=datetime(2018,1,1):
                sealed+=1
                continue
            v=tuple(int(row[k]) for k in ('S6_PIR','S7_PIR'))
            n=int(row['Room_Occupancy_Count'])
            if any(x not in (0,1) for x in v) or n not in (0,1,2,3):
                raise ValueError('Invalid observed sensor/annotation')
            opened.append((t,v,int(n>0)))
    if any(a[0]>=b[0] for a,b in zip(opened,opened[1:])):
        raise ValueError('Non-increasing timestamp')
    examples={'train':[],'development':[]}
    for i in range(15,len(opened)-2):
        window=opened[i-15:i+3]
        split='train' if window[-1][0]<datetime(2017,12,25) else 'development'
        if (window[0][0]<datetime(2017,12,25))!=(split=='train'):
            continue
        if any(not 0<(b[0]-a[0]).total_seconds()<=65 for a,b in zip(window,window[1:])):
            continue
        delay=(window[-1][0]-window[-3][0]).total_seconds()
        if not 60<=delay<=130:
            continue
        examples[split].append((window[-3][0],window[-1][0],tuple(v for _,v,_ in window[:-2]),window[-1][2]))
    return examples,{'sealed_timestamp_only_rows':sealed,'numeric_opened_rows':len(opened)}

def contrast_audit(examples):
    groups=defaultdict(lambda:defaultdict(Counter))
    for t,_,h,y in examples['train']:
        current,detail=keys(h,t,'age');groups[current][detail][y]+=1
    pairs=[]
    for current,branches in groups.items():
        neg=[h for h,c in branches.items() if c[0]>=4]
        pos=[h for h,c in branches.items() if c[1]>=4]
        if any(a!=b for a in neg for b in pos):
            pairs.append(current)
    support={s:{'count':len(rows),'labels':dict(Counter(r[-1] for r in rows)),'days':len({r[0].date() for r in rows})} for s,rows in examples.items()}
    passed=len(pairs)>=5 and all(min(v['labels'].get(0,0),v['labels'].get(1,0))>=100 and v['days']>=2 for v in support.values())
    return {'support':support,'contrast_groups':len(pairs),'passed':passed}

def shuffled_histories(rows):
    groups=defaultdict(list)
    for i,(t,_,h,_) in enumerate(rows):groups[keys(h,t,'current')[0]].append(i)
    result={}
    for indices in groups.values():
        for a,b in zip(indices,indices[1:]+indices[:1]):result[a]=rows[b][2]
    return result

def compare(examples):
    arms={}; hashes={}; learned={}
    for arm in ARMS:
        model=LocalReadout(); predictions=[]
        for split in ('train','development'):
            rows=examples[split]; shuffled=shuffled_histories(rows) if arm=='age_shuffled' else {}
            for i,(t,due,h,y) in enumerate(rows):
                address=keys(shuffled.get(i,h),t,arm)
                p=model.predict(address,t,due)
                if split=='development':predictions.append(p)
                model.deliver(due,lambda:y,split=='train')
        rows=examples['development']
        brier=sum((p-r[-1])**2 for p,r in zip(predictions,rows))/len(rows)
        ba=sum(sum((p>=.5)==bool(r[-1]) for p,r in zip(predictions,rows) if r[-1]==c)/sum(r[-1]==c for r in rows) for c in (0,1))/2
        arms[arm]={'brier':brier,'balanced_accuracy':ba,'changed_writes':model.writes,'development_writes':0,'routes':len(model.cells)}
        hashes[arm]=sha256(json.dumps(predictions).encode()).hexdigest()
        if arm=='age':
            # Audit naturally observed branches learned through delayed receipts.
            branches=defaultdict(list)
            for t,due,h,_ in examples['train']:
                k=keys(h,t,arm)
                p=model.predict(k,t,due);model.deliver(due,lambda:0,False)
                branches[k[0]].append((k[1],p))
            learned={'opposite_posterior_groups':sum(any(p<.5 for _,p in v) and any(p>.5 for _,p in v) for v in branches.values())}
    best=min(arms[a]['brier'] for a in ('current','current_fine'))
    quality=best-arms['age']['brier']>=.01 and arms['age']['balanced_accuracy']-max(arms[a]['balanced_accuracy'] for a in ('current','current_fine'))>=.02 and arms['age_shuffled']['brier']-arms['age']['brier']>=.005
    return {'arms':arms,'trace_hashes':hashes,'learned_identification':learned,'scalar_equivalent':hashes['age']==hashes['scalar_age'],'quality_passed':quality}

def main(mode):
    location=lambda suffix:processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')
    archive=raw_data_path('r2_room','source.zip')
    if mode=='register':
        protocol={'source':'https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation','attribution':'Singh & Chaudhari (2018), DOI 10.24432/C5P605, CC BY 4.0','source_sha256':'a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd','task':'Independently manually annotated occupancy >0 at two subsequent observations, 60-130s later','input':'Only both raw digital PIR sensors and clock; no temperature/light/sound/CO2/derived CO2 slope or occupancy labels','history':'16 observations, every adjacent interval >0 and <=65s; purge cross-split windows','train':'2017-12-22 through 2017-12-24','development':'2017-12-25 through 2017-12-26','sealed':'2018-01-01 onward; timestamp fields only','metadata_only_exposure':'CSV header, date counts and time intervals; 7 dates observed despite official description of 4 collection days; no numeric sensor/label read','support_gate':'Each split >=100 per class and >=2 calendar days; >=5 exact current PIR + quarter-hour clock groups with distinct age histories and >=4 observations per opposing future label','arms':ARMS,'learning':'Two changed integer counters per due receipt; prior strength8; capacity100000; frozen development','age':'Each sensor last active before current: last observation, 2-4, 5-8, or 9-16/absent; temporal detail retains shared current+clock','destruction':'Within each split and exact current+clock group, rotate whole history to next chronological example, without labels. Offline diagnostic, never deployable predictor. Singleton groups unchanged.','quality':'Brier improvement >=.01 and balanced accuracy >=.02 over best two current controls; Brier destruction loss >=.005','scope':'PIR-only low-bandwidth interface feasibility, few controlled days, correlated examples. No full-sensor superiority, SNN advantage, energy measurement, production adoption or test opening.'}
        preserved={str(p.relative_to(project_path())):digest(p) for directory in ('data/processed/benchmark_fixtures','models/appliances') for p in Path(project_path(directory)).glob('*.json')}
        save(location('protocol'),protocol)
        save(location('registration'),{'protocol_sha256':digest(location('protocol')),'code':{p:digest(project_path(p)) for p in CODE},'preserved':preserved})
        print('Room task registered before numeric inspection');return
    registration=json.loads(Path(location('registration')).read_text());protocol=json.loads(Path(location('protocol')).read_text())
    if digest(location('protocol'))!=registration['protocol_sha256'] or digest(archive)!=protocol['source_sha256']:raise ValueError('Protocol/source changed')
    for p,h in {**registration['code'],**registration['preserved']}.items():
        if digest(project_path(p))!=h:raise ValueError('Pinned evidence changed: '+p)
    if mode=='execute':save(location('execution'),{'started':True})
    examples,extraction=extract(archive);report={'extraction':extraction,'audit':contrast_audit(examples),'test_opened':False}
    if report['audit']['passed']:report['comparison']=compare(examples)
    if mode=='execute':save(location('result'),report)
    elif report!=json.loads(Path(location('result')).read_text()):raise ValueError('Replay differs')
    print(json.dumps(report,indent=2))
