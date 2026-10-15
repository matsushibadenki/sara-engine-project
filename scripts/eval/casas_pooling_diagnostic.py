#!/usr/bin/env python3
"""Preregister one bounded pooling feasibility test within consumed fit dates."""
import argparse
import gzip
import json
from pathlib import Path
from sara_engine.evaluation.casas_180 import extract, qualification
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.evaluation.casas_pooling import ARMS, POLICY, REFERENCES, compare
from sara_engine.utils.project_paths import project_path, processed_data_path, raw_data_path, model_path
from scripts.eval.casas_evidence_diagnostic import save, CODE as SOURCE_CODE

CODE = SOURCE_CODE + ('scripts/eval/casas_pooling_diagnostic.py',
                      'src/sara_engine/evaluation/casas_pooling.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures', 'casas_pooling_'+suffix+'_v1.json')


def read_path(value):
    return json.loads(Path(value).read_text())


def source_path(suffix):
    return processed_data_path('benchmark_fixtures', 'casas_evidence_'+suffix+'_v1.json')


def verify(reg):
    if file_sha256(path('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')
    for name, digest in {**reg['code'], **reg['preserved']}.items():
        if file_sha256(project_path(name)) != digest:
            raise ValueError('Pinned evidence changed: '+name)


def main(mode):
    if mode == 'register':
        preserved = {str(p.relative_to(project_path())): file_sha256(p)
                     for directory in ('data/processed/benchmark_fixtures', 'models')
                     for p in Path(project_path(directory)).rglob('*')
                     if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        old = read_path(source_path('execution'))['splits']
        dates = sorted(d for d in old['recorded_date_counts']
                       if d < old['development_start'] and d not in old['clock_regression_days'])
        inner = {**old, 'development_start': dates[2*len(dates)//3],
                 'test_start': old['development_start']}
        save(path('protocol'), {
            'scope': 'Training-only feasibility on already consumed HH104 fit dates. Prior negative scores known. No fresh identity, unseen confirmation or independent generalization claim.',
            'splits': inner, 'arms': ARMS, 'sufficient_references': REFERENCES, 'policy': POLICY,
            'pool': 'Keep exact raw current event plus weekday/hour/15-minute clock context. Detail=(current, previous sensor ID equals current sensor ID, latest gap bin<2). Four branches at most per current. Gap bins0/1 mean <10s. No outcome-dependent clustering, fitting or threshold sweep.',
            'prediction': 'Same disjoint evidence rule log(80) as previous experiment; compare detail counts against current-detail. If inactive use exact global/current/current posterior with prior8. No changed evidence threshold. Ungated pool is descriptive control.',
            'controls': 'All five sufficient references; ungated pool; unchanged joint evidence; order/same-sensor-only pool; timing-only pool; constant-gap evidence (fast=False, matching old fixed bin2); half-rotation feedback evidence; scalar duplicate.',
            'updates': 'All13 arms exactly three changed local count writes per due feedback, including inactive detail. Capacity100000, pending32/extraction16. Frozen internal confirmation. Candidate state <=1+5*observed current contexts and <=4 detail branches/context.',
            'source_gate': 'Unchanged casas_180 qualification and task extraction. Both classes >=500/20 days fit and >=200/10 days internal confirmation; >=10 original rich-current opposing-history groups with >=4 opposing outcomes; recurrence>=.9 and >=10 contexts support8. Stop before model scoring if any source gate fails.',
            'identification': 'After fit, count contexts with two distinct pooled branches passing unchanged disjoint evidence in opposite directions relative to their shared current mean, each with >=4 outcomes of its direction. Require >=10 such contexts.',
            'feasibility': 'Require >=1% evidence-active internal-confirmation predictions on >=10 dates, opposing contexts>=10, and exact route/write bounds. All gates reported; failure does not suppress registered training-only comparison scores but blocks further independent validation.',
            'quality': 'Retain >=.01 daily Brier and >=.02 balanced gain over best five sufficient references, constant-gap evidence loss>=.005, shuffled loss>=.01, exact scalar equivalence. All feasibility and quality gates required; no performance promotion from feasibility alone.',
            'seals': 'All dates from previous HH104 confirmation start onward discarded by date before predictive fields. Older household negatives and models hash-pinned. No new source acquisition, retuning, prior-confirmation opening, independent development/test opening or factor/runtime promotion.'})
        save(path('registration'), {'protocol_sha256': file_sha256(path('protocol')),
             'code': {name:file_sha256(project_path(name)) for name in CODE}, 'preserved': preserved})
        print('Fixed four-branch pooling registered before new pooling execution')
        return
    reg = read_path(path('registration'))
    verify(reg)
    protocol = read_path(path('protocol'))
    if mode == 'execute':
        save(path('execution'), {'started': True, 'protocol_sha256': reg['protocol_sha256']})
    receipt = {**read_path(source_path('acquisition')), **read_path(source_path('integrity'))}
    examples, audit = extract(raw_data_path('r2_casas_evidence','hh104.deflate'), receipt, protocol['splits'])
    report = {'audit': audit, 'qualification': qualification(examples, protocol['splits']),
              'previous_confirmation_opened': False, 'original_development_opened': False, 'test_opened': False}
    if report['qualification']['passed']:
        report['comparison'], states = compare(examples)
        report['model_sha256'] = {}
        for arm, state in states.items():
            destination = model_path('casas_pooling', arm+'_v1.json.gz')
            if mode == 'execute':
                save(destination, state, True)
            elif json.loads(gzip.decompress(Path(destination).read_bytes())) != state:
                raise ValueError('Saved state changed')
            report['model_sha256'][arm] = file_sha256(destination)
    if mode == 'execute':
        save(path('result'), report)
    elif json.loads(json.dumps(report)) != read_path(path('result')):
        raise ValueError('Fresh-process replay differs')
    display = json.loads(json.dumps(report))
    for arm in display.get('comparison',{}).get('arms',{}).values():
        arm.pop('days',None)
    print(json.dumps(display,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
