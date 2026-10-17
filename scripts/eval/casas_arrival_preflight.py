#!/usr/bin/env python3
"""Register, metadata-screen and replay one fresh training-input arrival audit."""
import argparse
import json
from pathlib import Path
from sara_engine.evaluation.casas_arrival_preflight import measure, Limits, ARRIVAL_TARGET
from sara_engine.evaluation.casas_180_source import metadata, member_lines, file_sha256
from sara_engine.utils.project_paths import project_path, processed_data_path, raw_data_path, ensure_parent_directory

CODE = ('scripts/eval/casas_arrival_preflight.py','src/sara_engine/evaluation/casas_arrival_preflight.py',
        'src/sara_engine/evaluation/casas_180_source.py','src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/casas_pooling.py','src/sara_engine/evaluation/casas_evidence.py',
        'src/sara_engine/evaluation/casas_vocabulary_budget.py','src/sara_engine/evaluation/casas_observation_budget.py',
        'src/sara_engine/evaluation/casas_count_audit.py','src/sara_engine/evaluation/sparse_count_replay.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py','src/sara_engine/utils/project_paths.py')
REFERENCE = processed_data_path('benchmark_fixtures','casas_vocabulary_budget_inventory_v1.json')


def path(suffix):
    return processed_data_path('benchmark_fixtures','casas_arrival_preflight_'+suffix+'_v1.json')


def read(suffix):
    return json.loads(Path(path(suffix)).read_text())


def save(destination,value):
    with Path(ensure_parent_directory(destination)).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def main(mode):
    if mode == 'register':
        preserved = {str(p.relative_to(project_path())):file_sha256(p)
                     for directory in ('data/processed/benchmark_fixtures','models')
                     for p in Path(project_path(directory)).rglob('*')
                     if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        save(path('protocol'),{
            'selection_sha256':file_sha256(path('selection')),
            'scope':'Unused HH105 fixed by prior selection. Training inputs only: arrival, recurrence, event vocabulary and address growth. No outcome class computation, learner or quality scores. New training identity is consumed by measurement, not independent generalization evidence.',
            'source':'Official CASAS raw data.zip member data/hh105.csv; bounded range acquisition, SHA/size/CRC verification, timestamp-only >=180 chronological recorded-date screen. No identity substitution or split rescue.',
            'window':'Unchanged casas_180_source metadata outer splitter: first min(180,2*N//3) chronological recorded dates only; outer independent development/test discarded by date before time/sensor/message fields. No internal confirmation is opened separately.',
            'limits':vars(Limits()),'arrival_target':ARRIVAL_TARGET,
            'target_basis':'70,212 qualified observed outcomes equals previous common k5 additional-cohort planning target. Frozen before HH105 values; not a new task-quality or predictive gate.',
            'qualifying_arrival':'Same eight-binary-event history, adjacent gaps<=300s, >=60s cutoff spacing, first observed binary event at least300s later and at most600s. Missing/late/unresolved outcomes are never negative labels. Histories use original gap bins; pending bound16. Qualification uses timing and original binary schema only; no Kitchen label test.',
            'unknown_events':'Reference is HH104 retained20 inventory. All binary sensor/message pairs, including previously unseen pairs, are measured without remapping or exclusion. This is a string-label comparison across households, not physical-sensor identity. Original nonbinary rows remain outside the fixed task and are counted separately.',
            'addresses':'Use original pool_evidence and joint_evidence routes, exact raw current event/weekday/hour/15-minute bin. Maintain bounded audit address sets at due arrivals; precheck both then atomically commit. Three would-be readout count changes per qualified arrival for each arm, actual learner writes0. Address-set insertions and audit statistics are not learned count writes.',
            'budget_stop':'Whole-measurement stop on qualified/write, opened-row, vocabulary, route or pending limit; never skip an unknown event and resume. Remaining training rows and all independent periods are date-only while byte integrity is drained to EOF. Route limit stops before either arm commits the rejected arrival; report truncation and unresolved work.',
            'measurement':'Causal per-due-date growth of opened/binary rows, qualified arrivals, all stream event types, reference-novel event types, retained current/two-lag vocabulary, exact current recurrence and per-arm route counts. Report measured yield and recorded/calendar-day arrival rates, delays, late/unresolved outcomes, final event inventory and conditional retained-vocabulary envelopes.',
            'preflight_gates':'Require long source, complete training window within budgets, >=70212 qualified outcomes, original raw repeat fraction>=.9 and >=10 rich current contexts support8, both route ceilings and matched hypothetical write ceiling. Keep all class/day/opposing-outcome and quality gates unchanged and explicitly unassessed; never infer independent usefulness from this preflight.',
            'verification':'Synthetic extraction/address parity with original parser, delayed timing, novelty retention, atomic capacity stop, date-only seals after stop and at independent split, causal prefix growth, actual learner writes0, full source checksums and exact fresh-process summary/growth/inventory replay. Pin all older JSON/gzip evidence and code.',
            'stop':'No learner, label scoring, unknown-event rejection, threshold/current change, test opening or runtime/factor promotion. Failed arrival/recurrence cannot extend the window or switch households post hoc. Future comparative learning needs separate preregistration and original source/quality gates.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),
             'code':{name:file_sha256(project_path(name)) for name in CODE},'preserved':preserved})
        print('HH105 training-input measurement registered before timestamp/value inspection')
        return
    reg = read('registration')
    if file_sha256(path('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')
    def verify():
        for name,digest in {**reg['code'],**reg['preserved']}.items():
            if file_sha256(project_path(name)) != digest:
                raise ValueError('Pinned evidence changed: '+name)
    verify()
    receipt = read('acquisition')
    source = raw_data_path('r2_casas_arrival','hh105.deflate')
    if mode == 'metadata':
        integrity = {}
        meta = metadata(member_lines(source,receipt,integrity))
        meta['selection'] = 'HH105 fixed before acquisition; training-only input preflight'
        save(path('metadata'),meta)
        save(path('integrity'),integrity)
        print(json.dumps({k:v for k,v in meta.items() if k != 'recorded_date_counts'},indent=2))
        return
    meta = read('metadata')
    receipt = {**receipt,**read('integrity')}
    inputs = {suffix:file_sha256(path(suffix)) for suffix in ('acquisition','metadata','integrity')}
    inputs['reference'] = file_sha256(REFERENCE)
    if mode == 'execute':
        save(path('execution'),{'started':True,'protocol_sha256':reg['protocol_sha256'],'inputs':inputs})
    elif inputs != read('execution')['inputs']:
        raise ValueError('Input receipts changed')
    reference = json.loads(Path(REFERENCE).read_text())['shared_known_events']
    report = measure(member_lines(source,receipt),meta,reference)
    for name in ('inventory','growth'):
        value = report.pop(name)
        if mode == 'execute':
            save(path(name),value)
        elif json.loads(json.dumps(value)) != read(name):
            raise ValueError(name+' replay differs')
        report[name+'_sha256'] = file_sha256(path(name))
    verify()
    if mode == 'execute':
        save(path('result'),report)
    elif report != read('result'):
        raise ValueError('Fresh-process measurement differs')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','metadata','execute','replay'))
    main(parser.parse_args().mode)
