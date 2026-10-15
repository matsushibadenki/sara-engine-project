#!/usr/bin/env python3
"""Register, metadata-screen, execute and replay a fresh-household component test."""
import argparse
import gzip
import json
from pathlib import Path
from sara_engine.evaluation.casas_180 import extract, qualification
from sara_engine.evaluation.casas_180_source import file_sha256, metadata, member_lines
from sara_engine.evaluation.casas_evidence import ARMS, REFERENCES, POLICY, compare
from sara_engine.utils.project_paths import project_path, processed_data_path, raw_data_path, model_path, ensure_parent_directory

PREFIX = 'casas_evidence'
CODE = ('scripts/eval/casas_evidence_diagnostic.py', 'src/sara_engine/evaluation/casas_evidence.py',
        'src/sara_engine/evaluation/casas_180.py', 'src/sara_engine/evaluation/casas_180_source.py',
        'src/sara_engine/evaluation/casas_longitudinal.py', 'src/sara_engine/evaluation/sparse_count_replay.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py', 'src/sara_engine/utils/project_paths.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures', PREFIX+'_'+suffix+'_v1.json')


def save(destination, value, compressed=False):
    data = (json.dumps(value, sort_keys=True, indent=None if compressed else 2, allow_nan=False)+'\n').encode()
    if compressed:
        data = gzip.compress(data, mtime=0)
    with Path(ensure_parent_directory(destination)).open('xb') as stream:
        stream.write(data)


def read(suffix):
    return json.loads(Path(path(suffix)).read_text())


def verify(reg):
    for name, digest in {**reg['code'], **reg['preserved']}.items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Pinned evidence changed: '+name)
    if file_sha256(path('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')


def main(mode):
    if mode == 'register':
        preserved = {str(p.relative_to(project_path())): file_sha256(p)
                     for directory in ('data/processed/benchmark_fixtures', 'models')
                     for p in Path(project_path(directory)).rglob('*')
                     if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        save(path('protocol'), {
            'identity': 'HH104, smallest unused household; ZIP name/size/CRC/offset only inspected before registration',
            'source': 'Official CASAS data.zip at https://zenodo.org/records/15708568/files/data.zip?download=1',
            'selection': 'HH104 only. Require >=180 chronological recorded dates using timestamp/schema metadata only; no identity substitution after failure.',
            'split': 'Use unchanged outer metadata split. Within its training dates, first floor(2*N/3) dates fit; remaining training dates internal confirmation. Original development/test date-only sealed.',
            'arms': ARMS, 'sufficient_references': REFERENCES, 'policy': POLICY,
            'rule': 'Detail counts D and disjoint complement C=current-D; n=sum(D), m=sum(C). Activate joint iff n,m>0 and abs(Dpos/n-Cpos/m)>sqrt(log(80)/(2*n))+sqrt(log(80)/(2*m)). Otherwise exact global/current/current posterior. Prior8. One fixed rule; no threshold search. Hoeffding separation proxy only, not calibrated time-series significance.',
            'controls': 'Support8; ungated constant-gap; same evidence policy with constant gap bins2; half-rotation training outcomes; scalar duplicate exact trace. All five sufficient references from start.',
            'updates': 'Three changed local counter writes per due feedback, including inactive detail, for every arm. Capacity100000, pending32, extraction16. Frozen confirmation. Diagnostic branch counters are not learned.',
            'task': 'Unchanged casas_180 extraction: binary events, eight-event histories, <=300s gaps, >=60s cutoffs; first observed binary event 300..600s later Kitchen ON positive; no absent outcome negatives; split purging.',
            'qualification': 'Unchanged qualification(): >=500 per class/20 days fit, >=200 per class/10 days confirmation, >=10 rich current/history contrasts with >=4 per opposing label, >=.9 recurrence and >=10 current groups support8. Stop before learner scores on any failure.',
            'quality': 'All required: .01 daily Brier gain and .02 balanced accuracy gain over best of five sufficient references; constant-gap evidence loss >=.005; shuffled loss >=.01; exact scalar trace. Actual writes and checkpoint replay exact.',
            'stop': 'Training-only component. Failure terminal, no retuning, no independent development/test opening, no runtime or factor promotion. A pass requires separate preregistration for independent validation.'})
        save(path('registration'), {'protocol_sha256': file_sha256(path('protocol')),
             'code': {p: file_sha256(project_path(p)) for p in CODE}, 'preserved': preserved})
        print('Fresh HH104 evidence hypothesis registered before source values')
        return
    reg = read('registration')
    verify(reg)
    source = raw_data_path('r2_casas_evidence', 'hh104.deflate')
    receipt = read('acquisition')
    if mode == 'metadata':
        audit = {}
        meta = metadata(member_lines(source, receipt, audit))
        meta['selection'] = 'HH104 fixed by preregistration before timestamp-only screen'
        save(path('metadata'), meta)
        save(path('integrity'), audit)
        print(json.dumps({k:v for k,v in meta.items() if k != 'recorded_date_counts'}, indent=2))
        return
    meta = read('metadata')
    receipt = {**receipt, **read('integrity')}
    dates = sorted(set(meta['recorded_date_counts'])-set(meta['clock_regression_days']))
    inner = {**meta, 'development_start': dates[2*meta['train_recorded_days']//3],
             'test_start': meta['development_start']}
    if mode == 'execute':
        save(path('execution'), {'started': True, 'splits': inner,
             'inputs': {suffix:file_sha256(path(suffix)) for suffix in ('acquisition','metadata','integrity')}})
    else:
        for suffix, digest in read('execution')['inputs'].items():
            if file_sha256(path(suffix)) != digest:
                raise ValueError('Pinned input receipt changed')
    examples, audit = extract(source, receipt, inner)
    report = {'audit': audit, 'qualification': qualification(examples, inner),
              'original_development_opened': False, 'test_opened': False}
    if report['qualification']['passed']:
        result, states = compare(examples)
        report['comparison'] = result
        report['model_sha256'] = {}
        for arm, state in states.items():
            destination = model_path('casas_evidence', arm+'_v1.json.gz')
            if mode == 'execute':
                save(destination, state, True)
            elif json.loads(gzip.decompress(Path(destination).read_bytes())) != state:
                raise ValueError('Checkpoint changed')
            report['model_sha256'][arm] = file_sha256(destination)
    if mode == 'execute':
        save(path('result'), report)
    elif json.loads(json.dumps(report)) != read('result'):
        raise ValueError('Fresh-process replay differs')
    display = json.loads(json.dumps(report))
    for arm in display.get('comparison', {}).get('arms', {}).values():
        arm.pop('days', None)
    print(json.dumps(display, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'metadata', 'execute', 'replay'))
    main(parser.parse_args().mode)
