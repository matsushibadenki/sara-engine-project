#!/usr/bin/env python3
"""Timestamp-only selection, preregistration, gated execution and exact replay."""
import argparse
import gzip
import json
from pathlib import Path
import re
from sara_engine.evaluation.casas_180_source import file_sha256, member_lines, metadata
from sara_engine.evaluation.casas_180 import extract, qualification
from sara_engine.evaluation.casas_longitudinal import ARMS, compare
from sara_engine.utils.project_paths import (project_path, raw_data_path, processed_data_path,
                                            model_path, ensure_parent_directory)

PREFIX = 'casas_180'
CODE = ('scripts/eval/casas_180_comparison.py',
        'src/sara_engine/evaluation/casas_180.py',
        'src/sara_engine/evaluation/casas_180_source.py',
        'src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/utils/project_paths.py')


def output(suffix):
    return processed_data_path('benchmark_fixtures', PREFIX+'_'+suffix+'_v1.json')


def save(path, value, compressed=False):
    data = (json.dumps(value, sort_keys=True, indent=None if compressed else 2,
                       separators=(',', ':') if compressed else None, allow_nan=False)+'\n').encode()
    if compressed:
        data = gzip.compress(data, mtime=0)
    with Path(ensure_parent_directory(path)).open('xb') as stream:
        stream.write(data)


def hashes(names):
    return {p:file_sha256(project_path(p)) for p in names}


def old_evidence():
    return {str(p.relative_to(project_path())):file_sha256(p)
            for directory in ('data/processed/benchmark_fixtures', 'models')
            for p in Path(project_path(directory)).rglob('*.json')}


def check(reg):
    for p,h in {**reg['code'], **reg['preserved']}.items():
        if file_sha256(project_path(p)) != h:
            raise ValueError('Pinned implementation or retained evidence changed: '+p)


def acquisition_metadata():
    source = raw_data_path('r2_casas_180', 'hh103.deflate')
    entry = json.loads(Path(raw_data_path('r2_casas_180', 'member_range.json')).read_text())
    headers = Path(raw_data_path('r2_casas_180', 'member.headers')).read_text()
    ranges = re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)', headers, re.I)
    if not ranges or tuple(map(int, ranges[-1])) != (entry['start'], entry['end'], entry['total_size']):
        raise ValueError('Server did not honor exact member range')
    entry['compressed_sha256'] = file_sha256(source)
    entry['source_url'] = 'https://zenodo.org/records/15708568/files/data.zip?download=1'
    entry['central_directory_tail_sha256'] = file_sha256(raw_data_path('r2_casas_180','tail.bin'))
    if not Path(output('selection')).exists():
        save(output('selection'), {
        'member':'data/hh103.csv', 'unused_identity':'HH103, excludes consumed HH101/HH102',
        'selection':'Smallest unused HH number; only timestamp/schema metadata inspected',
        'rules':'At least180 distinct recorded dates after excluding every date involved in a clock regression; first min(180,floor(2N/3)) valid dates training, next min(60,floor(N/6)) development, remaining dates sealed.',
        'code':hashes(('src/sara_engine/evaluation/casas_180_source.py',))})
    audit = {}
    meta = metadata(member_lines(source, entry, audit))
    entry.update(audit)
    save(output('acquisition'), entry)
    save(output('metadata'), meta)
    print(json.dumps({k:v for k,v in meta.items() if k!='recorded_date_counts'}, indent=2))


def main(mode):
    if mode == 'metadata':
        acquisition_metadata()
        return
    meta = json.loads(Path(output('metadata')).read_text())
    receipt = json.loads(Path(output('acquisition')).read_text())
    source = raw_data_path('r2_casas_180','hh103.deflate')
    if mode == 'register':
        protocol = {
            'source':'https://zenodo.org/records/15708568', 'identity':'HH103, fresh unused household',
            'attribution':'Cook et al. (2013), CASAS: A smart home in a box, DOI 10.1109/MC.2012.328; official partial member acquisition, no raw redistribution.',
            'acquisition_sha256':file_sha256(output('acquisition')), 'metadata_sha256':file_sha256(output('metadata')),
            'exposure':'Date, time, column count and byte integrity only before registration; no sensor/message/annotation values inspected.',
            'splits':{k:meta[k] for k in ('train_start','development_start','test_start','train_recorded_days','development_recorded_days','sealed_recorded_days')},
            'excluded_dates':meta['clock_regression_days'],
            'source_gate':'At least180 valid recorded dates; >=90% train rows use recurring raw current states; >=10 current/weekday/quarter-hour contexts with >=8 observations.',
            'unchanged_support':'Train >=500/class on >=20 days/class; development >=200/class on >=10 days/class; >=10 exact current+clock groups with different joint histories having >=4 observations per opposing future result.',
            'task':'First binary notification at >=300s and <=600s after cutoff; Kitchen-prefix sensor ON positive, any other recorded binary notification negative; silence/missing records never negative.',
            'inputs':'Exact current sensor ID and binary message, weekday and quarter-hour clock; last8 binary events with adjacent gaps0..300s; at most1 cutoff/60s. Notification state, not all-room physical state. No activity annotation/continuous sensor features.',
            'purging':'Cross-split history/outcome windows and interrupted history discarded. Metadata clock-invalid dates excluded; sealed date examined before all remaining fields.',
            'arms':ARMS,
            'updates':'3 changed integer counters per actual delayed training outcome; shared global/current/detail hierarchy, prior8,100000 routes,32 prediction receipts,16 extraction receipts. Due-ordered outcomes before next cutoff; zero development writes.',
            'controls':'Current,5-minute current,histogram,lag1/lag4 transition,joint lag1/lag4 gap bins,constant gaps2,half-rotated training outcomes,exact scalar joint; same existing address functions unchanged.',
            'unchanged_quality':'Joint daily Brier >=.01 better and balanced accuracy >=.02 better than best sufficient current/current_fine/histogram/transition; constant-gap Brier loss >=.005 and shuffled-outcome >=.01; exact scalar equivalence.',
            'stop':'Any source/support failure: no fitting/scoring, no diagnostic bypass, no threshold/split/target relaxation. Quality failure retained. Test remains sealed regardless of result.',
            'scope':'CPU sparse local readout comparison; no matrix/GPU/global gradient, energy measurement, cross-home transfer, SNN-specific advantage or production promotion.'}
        preserved = old_evidence()
        save(output('protocol'), protocol)
        save(output('registration'), {'protocol_sha256':file_sha256(output('protocol')),
                                      'code':hashes(CODE), 'preserved':preserved})
        print('Long-source conditions registered before sensor/message inspection')
        return
    reg = json.loads(Path(output('registration')).read_text())
    check(reg)
    if file_sha256(output('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')
    if mode == 'execute':
        save(output('execution'), {'started':True})
    examples, audit = extract(source, receipt, meta)
    report = {'extraction':audit, 'qualification':qualification(examples, meta), 'test_opened':False}
    if report['qualification']['passed']:
        comparison, checkpoints = compare(examples)
        report['comparison'] = comparison
        report['model_sha256'] = {}
        for arm,state in checkpoints.items():
            target = model_path('casas_hh103_180',arm+'_v1.json.gz')
            if mode == 'execute':
                save(target,state,compressed=True)
            elif json.loads(gzip.decompress(Path(target).read_bytes())) != state:
                raise ValueError('Published checkpoint replay differs')
            report['model_sha256'][arm] = file_sha256(target)
    if mode == 'execute':
        save(output('result'), report)
    elif json.loads(json.dumps(report)) != json.loads(Path(output('result')).read_text()):
        raise ValueError('Source or learner fresh-process replay differs')
    comparison = report.get('comparison')
    if comparison:
        display = {**comparison,'arms':{a:{k:v for k,v in row.items() if k!='days'} for a,row in comparison['arms'].items()}}
        print(json.dumps({**report,'comparison':display},indent=2))
    else:
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('metadata','register','execute','replay'))
    main(parser.parse_args().mode)
