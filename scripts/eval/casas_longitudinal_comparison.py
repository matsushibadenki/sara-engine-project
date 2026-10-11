#!/usr/bin/env python3
"""Register once, execute once, and replay frozen HH101 development evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from sara_engine.evaluation.casas_longitudinal import ARMS, extract, support, compare
from sara_engine.evaluation.motion_temporal import digest, save
from sara_engine.utils.project_paths import project_path, raw_data_path, processed_data_path, model_path

PREFIX = 'casas_longitudinal'
CODE = ('src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/evaluation/motion_temporal.py',
        'src/sara_engine/utils/project_paths.py',
        'scripts/eval/casas_longitudinal_comparison.py')
SOURCE_HASH = '7d9e976308e1cb5bc91cb39937848bfe073fc7b44c8a33cf166de203014b0091'


def main(mode):
    path = lambda suffix: processed_data_path('benchmark_fixtures', PREFIX+'_'+suffix+'_v2.json')
    archive = raw_data_path('r2_casas_longitudinal', 'source.zip')
    if mode == 'register':
        protocol = {
            'source': 'https://zenodo.org/records/15708568',
            'source_url': 'https://zenodo.org/records/15708568/files/labeled_data.zip?download=1',
            'source_sha256': SOURCE_HASH, 'member': 'labeled/hh101.csv',
            'attribution': 'Diane Cook and CASAS; Cook et al. (2013), CASAS: A smart home in a box, DOI 10.1109/MC.2012.328. Public research source; raw records not redistributed.',
            'independence': 'New WSU home, distinct from consumed UCI sources; externally annotated behavior, not direct physical occupancy measurement.',
            'selection': 'Aruba missing from current official bundle; HH101 is smallest numbered HH home, selected by file identity and date metadata, not label prevalence.',
            'prior_exposure': 'First data row was mistaken for header and displayed: 2012-07-20 10:38:54 OutsideDoor ON Step_Out begin. V1 stopped on a .22547-second timestamp regression on July21 before support/learner scoring; July20 and already opened July21 both excluded from this fresh v2 registration. Remaining inspection limited to file names, date counts and timestamp-only audit; no July22-onward message/annotation inspection before v2.',
            'train': '2012-07-22 <= cutoff and outcome < 2012-08-20',
            'development': '2012-08-20 <= cutoff and outcome < 2012-09-04',
            'sealed': '2012-09-04 onward, date/time fields only, no annotation or message access',
            'task': 'Sleep/Sleeping active at the first binary sensor event >=300s after cutoff and <=600s after cutoff; other actively annotated behavior is negative; missing annotation never means negative.',
            'annotation': 'Forward begin/end interval tracking, ends exclusive, max interval age 12h; active Sleep/Sleeping wins overlaps. Label eligibility is evaluator-only. This is conditional on future annotation coverage, not all-time occupancy detection.',
            'input': 'Exact current sensor identifier and binary ON/OFF/OPEN/CLOSE/CLOSED message, weekday and quarter-hour clock. Event notification state, not whole-home physical state. Continuous messages and annotations are never predictive inputs.',
            'history': 'Last 8 binary events; gaps 0..300s; no cross-split history; cutoff at most once per 60s; unresolved/late/unannotated/cross-split outcomes excluded and counted.',
            'support': 'Training >=500 examples/class on >=20 days/class; development >=200/class on >=10 days/class; >=10 current+clock groups with distinct joint histories and >=4 observations per opposing future label.',
            'arms': ARMS,
            'routes': '3 distinct changed integer count writes/outcome: global, shared current+clock, arm detail. Hierarchical prior strength8; 100000 total routes; 32 pending receipts.',
            'details': '5-minute current clock; unordered full history histogram; lag1/lag4 transition; joint adds gap bins <1,<10,<60,>=60 at lag1/lag4; constant gap sets both bins2; exact scalar_joint; training-only half-length outcome rotation.',
            'causality': 'Due-ordered receipt queue; no label supplier call before due; flush all due outcomes before next decision; frozen development zero writes.',
            'quality': 'Joint daily Brier >=.01 better and balanced accuracy >=.02 better than best current/current_fine/histogram/transition; constant-gap daily Brier loss >=.005; shuffled-outcome loss >=.01; scalar equivalence exact.',
            'resources': 'CPU-only sparse dictionaries, no GPU/matrices/global backpropagation. Pending/state ceilings fail closed. No physical energy claim.',
            'stop': 'Support failure stops before any arm training/scoring; thresholds, split and labels never retuned. No sealed test opening or production promotion in either outcome.'}
        preserved = {}
        for directory in ('data/processed/benchmark_fixtures', 'models'):
            for p in Path(project_path(directory)).rglob('*.json'):
                preserved[str(p.relative_to(project_path()))] = digest(p)
        save(path('protocol'), protocol)
        save(path('registration'), {'protocol_sha256':digest(path('protocol')),
                                  'code':{p:digest(project_path(p)) for p in CODE},
                                  'preserved':preserved})
        print('Registered before remaining sensor/annotation inspection')
        return
    reg = json.loads(Path(path('registration')).read_text())
    if digest(path('protocol')) != reg['protocol_sha256'] or digest(archive) != SOURCE_HASH:
        raise ValueError('Frozen source/protocol differs')
    code = reg['code']
    for p,h in {**code, **reg['preserved']}.items():
        if digest(project_path(p)) != h:
            raise ValueError('Pinned code or old evidence differs: '+p)
    if mode == 'execute':
        save(path('execution'), {'started':True})
    examples, audit = extract(archive)
    report = {'extraction':audit, 'support':support(examples), 'test_opened':False}
    if report['support']['passed']:
        comparison, checkpoints = compare(examples)
        report['comparison'] = comparison
        report['model_sha256'] = {}
        for arm,state in checkpoints.items():
            destination = model_path('casas_hh101', arm+'_v2.json')
            if mode == 'execute':
                save(destination, state)
            elif json.loads(Path(destination).read_text()) != state:
                raise ValueError('Published count checkpoint differs')
            report['model_sha256'][arm] = digest(destination)
    if mode == 'execute':
        save(path('result'), report)
    elif json.loads(json.dumps(report)) != json.loads(Path(path('result')).read_text()):
        raise ValueError('Fresh-process replay differs')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register','execute','replay'))
    main(parser.parse_args().mode)
