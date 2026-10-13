#!/usr/bin/env python3
"""Preregister and execute one fixed training-only confidence hypothesis."""
import argparse,gzip,json
from pathlib import Path
from sara_engine.evaluation.casas_180 import extract,qualification
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.evaluation.casas_confidence import compare,ARMS
from sara_engine.utils.project_paths import project_path,processed_data_path,raw_data_path,model_path,ensure_parent_directory

PREFIX='casas_confidence'
CODE=('scripts/eval/casas_confidence_diagnostic.py','src/sara_engine/evaluation/casas_confidence.py',
      'src/sara_engine/evaluation/casas_180.py','src/sara_engine/evaluation/casas_180_source.py',
      'src/sara_engine/evaluation/casas_longitudinal.py','src/sara_engine/evaluation/sparse_count_replay.py',
      'src/sara_engine/evaluation/delayed_sparse_counts.py','src/sara_engine/utils/project_paths.py')


def path(suffix):return processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')

def save(destination,value,compressed=False):
    data=(json.dumps(value,sort_keys=True,indent=None if compressed else 2,allow_nan=False)+'\n').encode()
    if compressed:data=gzip.compress(data,mtime=0)
    with Path(ensure_parent_directory(destination)).open('xb') as stream:stream.write(data)


def main(mode):
    oldmeta=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_metadata_v1.json')).read_text())
    dates=sorted(set(oldmeta['recorded_date_counts'])-set(oldmeta['clock_regression_days']))
    inner={**oldmeta,'development_start':dates[120],'test_start':oldmeta['development_start']}
    if mode=='register':
        preserved={str(p.relative_to(project_path())):file_sha256(p)
                   for directory in ('data/processed/benchmark_fixtures','models')
                   for p in Path(project_path(directory)).rglob('*')
                   if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        save(path('protocol'),{'scope':'Training-only diagnostic on already consumed HH103 training data; original negative scores known. Register before new candidate scores. No unseen-data or adoption claim.',
                              'splits':inner,'arms':ARMS,'rule':'Fixed detail support8 equals existing prior strength8; below8 predict exact current-control probability; >=8 use joint posterior. No threshold grid/search.',
                              'updates':'All8 arms receive3 changed counter writes per due outcome; same shared current inputs, capacity100000/pending32/extraction16, frozen internal confirmation.',
                              'qualification':'All original class/day/history contrast and recurrence gates retained for internal folds; failure stops before scoring.',
                              'quality':'Same .01 daily Brier/.02 balanced gain over best current/transition/joint; .005 constant-gap and .01 shuffled losses; exact scalar equivalence.',
                              'seals':'Original development and test discarded by date before any other field; no retuning if gate fails; independent generalization remains future work.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),'code':{p:file_sha256(project_path(p)) for p in CODE},'preserved':preserved})
        print('Fixed support8 training-only hypothesis registered');return
    reg=json.loads(Path(path('registration')).read_text())
    for p,h in {**reg['code'],**reg['preserved']}.items():
        if file_sha256(project_path(p))!=h:raise ValueError('Pinned evidence changed: '+p)
    if file_sha256(path('protocol'))!=reg['protocol_sha256']:raise ValueError('Protocol changed')
    if mode=='execute':save(path('execution'),{'started':True})
    receipt=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_acquisition_v1.json')).read_text())
    examples,audit=extract(raw_data_path('r2_casas_180','hh103.deflate'),receipt,inner)
    report={'audit':audit,'qualification':qualification(examples,inner),'original_development_opened':False,'test_opened':False}
    if report['qualification']['passed']:
        result,states=compare(examples);report['comparison']=result;report['model_sha256']={}
        for arm,state in states.items():
            destination=model_path('casas_confidence',arm+'_v1.json.gz')
            if mode=='execute':save(destination,state,True)
            elif json.loads(gzip.decompress(Path(destination).read_bytes()))!=state:raise ValueError('Checkpoint changed')
            report['model_sha256'][arm]=file_sha256(destination)
    if mode=='execute':save(path('result'),report)
    elif json.loads(json.dumps(report))!=json.loads(Path(path('result')).read_text()):raise ValueError('Replay differs')
    display=json.loads(json.dumps(report))
    for arm in display.get('comparison',{}).get('arms',{}).values():arm.pop('days',None)
    print(json.dumps(display,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','execute','replay'));main(parser.parse_args().mode)
