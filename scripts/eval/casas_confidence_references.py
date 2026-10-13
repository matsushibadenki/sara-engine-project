#!/usr/bin/env python3
"""Restore the complete frozen sufficient-reference set without candidate tuning."""
import argparse,gzip,json
from pathlib import Path
from sara_engine.evaluation.casas_confidence import ConfidenceCounts,phase
from sara_engine.evaluation.casas_longitudinal import metrics
from sara_engine.evaluation.casas_180 import extract
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import project_path,processed_data_path,raw_data_path,model_path,ensure_parent_directory

PREFIX='casas_confidence_references'
ARMS=('current_fine','histogram')
CODE=('scripts/eval/casas_confidence_references.py','src/sara_engine/evaluation/casas_confidence.py',
      'src/sara_engine/evaluation/casas_longitudinal.py','src/sara_engine/evaluation/casas_180.py',
      'src/sara_engine/evaluation/casas_180_source.py','src/sara_engine/evaluation/sparse_count_replay.py',
      'src/sara_engine/evaluation/delayed_sparse_counts.py','src/sara_engine/utils/project_paths.py')


def path(suffix):return processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')

def save(destination,value,compressed=False):
    data=(json.dumps(value,sort_keys=True,indent=None if compressed else 2)+'\n').encode()
    if compressed:data=gzip.compress(data,mtime=0)
    with Path(ensure_parent_directory(destination)).open('xb') as stream:stream.write(data)


def main(mode):
    oldpath=processed_data_path('benchmark_fixtures','casas_confidence_result_v1.json')
    if mode=='register':
        preserved={str(p.relative_to(project_path())):file_sha256(p)
                   for directory in ('data/processed/benchmark_fixtures','models')
                   for p in Path(project_path(directory)).rglob('*')
                   if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        save(path('registration'),{'reason':'Original eight-arm diagnostic retained as negative; current_fine and histogram were omitted from its reference set. Supplement both before inspecting their scores, retain every candidate prediction and strengthen comparison to all original sufficient references. No threshold/feature/split/gain change, no independent-validation claim.','arms':ARMS,'code':{p:file_sha256(project_path(p)) for p in CODE},'preserved':preserved})
        print('Complete sufficient-reference supplement registered');return
    reg=json.loads(Path(path('registration')).read_text())
    for p,h in {**reg['code'],**reg['preserved']}.items():
        if file_sha256(project_path(p))!=h:raise ValueError('Pinned evidence changed')
    if mode=='execute':save(path('execution'),{'started':True})
    protocol=json.loads(Path(processed_data_path('benchmark_fixtures','casas_confidence_protocol_v1.json')).read_text())
    receipt=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_acquisition_v1.json')).read_text())
    examples,_=extract(raw_data_path('r2_casas_180','hh103.deflate'),receipt,protocol['splits'])
    rows={};model_hashes={}
    for arm in ARMS:
        model=ConfidenceCounts(arm);phase(model,examples['train'],True);cp=model.checkpoint()
        clone=ConfidenceCounts(arm);clone.cells=json.loads(json.dumps(cp['cells']));clone.changed_writes=cp['changed_writes']
        prediction=phase(model,examples['development'],False)
        if prediction!=phase(clone,examples['development'],False) or model.checkpoint()!=cp:raise ValueError('Reference replay differs')
        rows[arm]={**metrics(examples['development'],prediction),'changed_writes':model.changed_writes,'confirmation_writes':0,'trace_sha256':__import__('hashlib').sha256(json.dumps(prediction).encode()).hexdigest()}
        if model.changed_writes!=27972:raise ValueError('Writes not matched')
        state={'arm':arm,'minimum_detail_support':8,'counts':cp};destination=model_path('casas_confidence',arm+'_v1.json.gz')
        if mode=='execute':save(destination,state,True)
        elif json.loads(gzip.decompress(Path(destination).read_bytes()))!=state:raise ValueError('Saved reference differs')
        model_hashes[arm]=file_sha256(destination)
    old=json.loads(Path(oldpath).read_text())['comparison'];allrefs={**old['arms'],**rows}
    sufficient=('current','current_fine','histogram','transition','joint')
    best=min(allrefs[a]['daily_brier'] for a in sufficient);ba=max(allrefs[a]['balanced_accuracy'] for a in sufficient)
    candidate=old['arms']['confidence']
    gates={**old['gates'],'brier_gain':best-candidate['daily_brier']>=.01,'balanced_gain':candidate['balanced_accuracy']-ba>=.02}
    report={'references':rows,'model_sha256':model_hashes,'complete_reference_gates':gates,'passed':all(gates.values()),'candidate_scores_unchanged':True,'original_development_opened':False,'test_opened':False}
    if mode=='execute':save(path('result'),report)
    elif json.loads(json.dumps(report))!=json.loads(Path(path('result')).read_text()):raise ValueError('Supplement replay differs')
    display=json.loads(json.dumps(report))
    for row in display['references'].values():row.pop('days',None)
    print(json.dumps(display,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','execute','replay'));main(parser.parse_args().mode)
