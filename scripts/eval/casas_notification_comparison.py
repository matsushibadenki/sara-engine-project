#!/usr/bin/env python3
"""One-shot preregistration, execution and exact replay of fresh HH102 task."""
import argparse
import json
from pathlib import Path
from sara_engine.evaluation.casas_notification import extract
from sara_engine.evaluation.casas_longitudinal import ARMS, support, compare
from sara_engine.evaluation.motion_temporal import save, digest
from sara_engine.utils.project_paths import project_path, processed_data_path, raw_data_path, model_path

PREFIX = 'casas_notification'
SOURCE_HASH = '7d9e976308e1cb5bc91cb39937848bfe073fc7b44c8a33cf166de203014b0091'
CODE = ('src/sara_engine/evaluation/casas_notification.py',
        'src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/evaluation/motion_temporal.py',
        'src/sara_engine/utils/project_paths.py',
        'scripts/eval/casas_notification_comparison.py')


def main(mode):
    path = lambda suffix:processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')
    archive = raw_data_path('r2_casas_longitudinal','source.zip')
    if mode == 'register':
        protocol = {
            'source':'https://zenodo.org/records/15708568',
            'archive_sha256':SOURCE_HASH,'member':'labeled/hh102.csv',
            'attribution':'Cook et al. (2013), CASAS: A smart home in a box, DOI 10.1109/MC.2012.328; public research source, raw data not redistributed.',
            'independence':'Fresh distinct household; no HH102 input/outcome inspection before registration. Same acquisition archive, different source identity from HH101. No cross-household generalization claim.',
            'motivation':'HH101 Sleep task failed frozen support: sparse observable nighttime events. Retain that failure. New future physically recorded notification target uses no activity annotation or imputed occupancy.',
            'metadata_only_exposure':'62 dates, 2011-06-15 through 2011-08-15, no timestamp inversions; no sensor/message/annotation values inspected.',
            'task':'Kitchen-prefix sensor ON at the first binary notification 300-600 seconds after cutoff; negative is any other observed binary notification, never missing data or no-event absence.',
            'train':'2011-06-15 through 2011-07-14','development':'2011-07-15 through 2011-07-29','sealed':'2011-07-30 onward: date fields only; no sensor/message/annotation parsing',
            'input':'Raw current sensor ID and binary message, weekday and quarter-hour clock, eight binary events with adjacent gaps 0..300s; current notification state, not complete household physical state. Annotations and continuous sensors never inputs.',
            'eligibility':'At most one cutoff per60s; purge cross-split history/outcomes and interrupted histories. First event >=300s later must arrive within600s. Conditional on follow-up observation, not absence prediction.',
            'support':'Training >=500/class on >=20 days/class; development >=200/class on >=10 days/class; >=10 exact current+clock groups with distinct joint histories and >=4 observations per opposite future label.',
            'arms':ARMS,
            'routes':'Three changed integer counters per outcome: global, shared current+clock, detail; prior8; <=100000 local addresses; <=32 pending prediction receipts; <=16 extraction receipts.',
            'detail':'Current quarter-hour and5-minute controls; unordered full history histogram; lag1/lag4 sensor-message transitions; joint adds lag1/lag4 gaps binned <1,<10,<60,>=60 seconds; constant-gap sets both bins2; outcome half-rotation training only; identical scalar joint.',
            'causality':'Chronological due queue flushes due labels before next cutoff; supplier only read at exact due; same real writes every arm; frozen development, no test opening.',
            'quality':'Joint daily Brier improves >=.01 and balanced accuracy >=.02 versus best current/current_fine/histogram/transition; constant gap loss >=.005 and outcome-shuffle loss >=.01 daily Brier; exact scalar equivalence.',
            'scope':'CPU scalar sparse local readout feasibility; no matrices/GPU/global gradient; no SNN/energy/health/production claim. No test opening whether gates pass or fail.'}
        preserved = {str(p.relative_to(project_path())):digest(p)
                     for directory in ('data/processed/benchmark_fixtures','models')
                     for p in Path(project_path(directory)).rglob('*.json')}
        save(path('protocol'),protocol)
        save(path('registration'),{'protocol_sha256':digest(path('protocol')),
                                   'code':{p:digest(project_path(p)) for p in CODE},'preserved':preserved})
        print('Fresh HH102 task registered before sensor/message inspection')
        return
    reg=json.loads(Path(path('registration')).read_text())
    if digest(path('protocol'))!=reg['protocol_sha256'] or digest(archive)!=SOURCE_HASH:
        raise ValueError('Source or protocol changed')
    amendment=Path(path('parser_amendment'))
    code=json.loads(amendment.read_text())['code'] if amendment.exists() else reg['code']
    for p,h in {**code,**reg['preserved']}.items():
        if digest(project_path(p))!=h:raise ValueError('Pinned evidence changed: '+p)
    if mode=='execute':save(path('repaired_execution' if amendment.exists() else 'execution'),{'started':True})
    examples,audit=extract(archive)
    report={'extraction':audit,'support':support(examples),'test_opened':False}
    if report['support']['passed']:
        comparison,checkpoints=compare(examples)
        report['comparison']=comparison
        report['model_sha256']={}
        for arm,checkpoint in checkpoints.items():
            target=model_path('casas_hh102',arm+'_v1.json')
            if mode=='execute':save(target,checkpoint)
            elif json.loads(Path(target).read_text())!=checkpoint:raise ValueError('Saved model replay differs')
            report['model_sha256'][arm]=digest(target)
    if mode=='execute':save(path('result'),report)
    elif json.loads(json.dumps(report))!=json.loads(Path(path('result')).read_text()):
        raise ValueError('Independent replay differs')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
