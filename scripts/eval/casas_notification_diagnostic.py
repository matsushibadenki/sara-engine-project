#!/usr/bin/env python3
"""Preregistered score diagnostic after known HH102 contrast-support failure."""
import argparse
import json
from pathlib import Path
from sara_engine.evaluation.casas_notification import extract
from sara_engine.evaluation.casas_longitudinal import compare
from sara_engine.evaluation.motion_temporal import save,digest
from sara_engine.utils.project_paths import project_path,processed_data_path,raw_data_path,model_path

PREFIX='casas_notification_diagnostic'
CODE=('src/sara_engine/evaluation/casas_notification.py',
      'src/sara_engine/evaluation/casas_longitudinal.py',
      'src/sara_engine/evaluation/delayed_sparse_counts.py',
      'src/sara_engine/evaluation/motion_temporal.py',
      'src/sara_engine/utils/project_paths.py','scripts/eval/casas_notification_diagnostic.py')
SOURCE_HASH='7d9e976308e1cb5bc91cb39937848bfe073fc7b44c8a33cf166de203014b0091'


def main(mode):
    path=lambda suffix:processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')
    support_path=processed_data_path('benchmark_fixtures','casas_notification_result_v1.json')
    archive=raw_data_path('r2_casas_longitudinal','source.zip')
    if mode=='register':
        source_report=json.loads(Path(support_path).read_text())
        if source_report['support']['passed']:
            raise ValueError('Diagnostic is reserved for known failed contrast support')
        preserved={str(p.relative_to(project_path())):digest(p)
                   for directory in ('data/processed/benchmark_fixtures','models')
                   for p in Path(project_path(directory)).rglob('*.json')}
        save(path('registration'),{
            'reason':'Both classes exist on all 30/15 days, but the frozen rich-current contrast requirement failed (0/10). This known failure is never relaxed or relabeled as a pass. Diagnostic scores were not inspected before this registration.',
            'known_support':source_report['support'],
            'scope':'First-score descriptive comparison on already inspected support data; not fresh unseen source-support evidence, not proof of temporal necessity, not eligible for R2 promotion irrespective of metric gates.',
            'conditions':'Unchanged original target, current inputs, chronological splits, eight arms, local prior8 and metric thresholds. 6064 outcomes/arm, exactly18192 changed writes/arm; zero development updates. Due-ordered receipt queue, bounded pending/state, exact checkpoint replay. No test opening.',
            'code':{p:digest(project_path(p)) for p in CODE},'preserved':preserved,'source_sha256':SOURCE_HASH})
        print('Known-failed-support comparison registered before first learner scores')
        return
    reg=json.loads(Path(path('registration')).read_text())
    if digest(archive)!=reg['source_sha256']:raise ValueError('Source changed')
    for p,h in {**reg['code'],**reg['preserved']}.items():
        if digest(project_path(p))!=h:raise ValueError('Pinned evidence changed: '+p)
    if mode=='execute':save(path('execution'),{'started':True})
    examples,audit=extract(archive)
    comparison,checkpoints=compare(examples)
    report={'known_source_support':reg['known_support'],'comparison':comparison,
            'eligible_for_promotion':False,'test_opened':False,'model_sha256':{},
            'scope':'Post-support first-score diagnostic; original 0/10 contrast failure retained'}
    for arm,state in checkpoints.items():
        target=model_path('casas_hh102_diagnostic',arm+'_v1.json')
        if mode=='execute':save(target,state)
        elif json.loads(Path(target).read_text())!=state:raise ValueError('Published state differs')
        report['model_sha256'][arm]=digest(target)
    if mode=='execute':save(path('result'),report)
    elif json.loads(json.dumps(report))!=json.loads(Path(path('result')).read_text()):
        raise ValueError('Fresh-process replay differs')
    print(json.dumps({**report,'comparison':{**comparison,'arms':{a:{k:v for k,v in row.items() if k!='days'} for a,row in comparison['arms'].items()}}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
