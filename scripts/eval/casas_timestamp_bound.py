#!/usr/bin/env python3
"""Preregister and replay timestamp-only upper bounds on consumed training windows."""
import argparse
import json
from pathlib import Path
from sara_engine.evaluation.casas_timestamp_bound import bound_lines, validate_counterexamples, BoundLimits, TARGET
from sara_engine.evaluation.casas_180_source import file_sha256, member_lines
from sara_engine.utils.project_paths import project_path, processed_data_path, raw_data_path, ensure_parent_directory

SOURCES = {
    'HH103': {'raw':['r2_casas_180','hh103.deflate'],'metadata':'casas_180_metadata_v1.json','metadata_fields':[],
              'acquisition':'casas_180_acquisition_v1.json','integrity':None,
              'baseline':'casas_180_result_v1.json','baseline_fields':['qualification','splits','train','count']},
    'HH104': {'raw':['r2_casas_evidence','hh104.deflate'],'metadata':'casas_pooling_protocol_v1.json','metadata_fields':['splits'],
              'acquisition':'casas_evidence_acquisition_v1.json','integrity':'casas_evidence_integrity_v1.json',
              'baseline':'casas_pooling_result_v1.json','baseline_fields':['qualification','splits','train','count']},
    'HH105': {'raw':['r2_casas_arrival','hh105.deflate'],'metadata':'casas_arrival_preflight_metadata_v1.json','metadata_fields':[],
              'acquisition':'casas_arrival_preflight_acquisition_v1.json','integrity':'casas_arrival_preflight_integrity_v1.json',
              'baseline':'casas_arrival_preflight_result_v1.json','baseline_fields':['qualified_outcomes']}}
CODE = ('scripts/eval/casas_timestamp_bound.py','src/sara_engine/evaluation/casas_timestamp_bound.py',
        'src/sara_engine/evaluation/casas_180_source.py','src/sara_engine/evaluation/casas_arrival_preflight.py',
        'src/sara_engine/evaluation/casas_longitudinal.py','src/sara_engine/evaluation/casas_pooling.py',
        'src/sara_engine/evaluation/casas_evidence.py','src/sara_engine/evaluation/casas_vocabulary_budget.py',
        'src/sara_engine/evaluation/casas_observation_budget.py','src/sara_engine/evaluation/casas_count_audit.py',
        'src/sara_engine/evaluation/sparse_count_replay.py','src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/utils/project_paths.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures','casas_timestamp_bound_'+suffix+'_v1.json')


def read_path(value):
    return json.loads(Path(value).read_text())


def fixture(name):
    return processed_data_path('benchmark_fixtures',name)


def select(value, fields):
    for field in fields:
        value = value[field]
    return value


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
            'scope':'Timestamp-only upper-bound method validation on existing consumed training windows only. No new household, acquisition, input-value inspection, scoring or learner. Earlier measured counts and negative results known.',
            'sources':SOURCES,'target':TARGET,'limits':vars(BoundLimits()),
            'source_limits':'Each existing member expanded size<=536870912 bytes, exact original compressed/expanded SHA/CRC checks. Sequential bounded decompression and date-only drain of excluded/sealed rows. Timestamp-row or queue-cap stop is inconclusive, never an exclusion.',
            'fields':'Read ISO date and training time only. Choose CSV/whitespace delimiter at the date boundary, never from event payload. Before time parsing, discard all dates from the existing training end onward and metadata clock exclusions. Sensor/message fields are never interpreted; any binary subsequence of raw timestamps must be covered.',
            'history_necessity':'Any valid eight-binary-event history with adjacent gaps<=300s implies the raw timestamp streak ending at its cutoff contains >=8 rows without a raw gap>300s. Mark all such raw times as potential cutoffs; no original cutoff-clock phase is simulated.',
            'minute_injection':'Original >=60s cutoff spacing implies at most one actual qualified cutoff per half-open clock-minute interval. Count intervals containing a potential cutoff with some future raw time300..600s later. This counting partition is not a change to the original 15-minute current context or actual extraction.',
            'interval_union':'Inside one minute all potential-cutoff result intervals [t+300,t+600] overlap, so their union is [first+300,last+600]. Store only first/last. At the first later raw time>=first+300, certify bucket iff time<=last+600, otherwise expire. The minute is closed before maturity. Pending first times stay within300s of present: at most6 minute buckets, guarded by16.',
            'soundness':'Every actual qualified cutoff maps injectively to a certified bucket: its history is raw-feasible and its binary result is a raw result in the union. Extra buckets are permitted. Complete training EOF excludes unresolved buckets; truncated scans return no usable upper bound. Raw rows treated as possible binary events define a superset, not a task filter change.',
            'naive_counterexample':'Eight rows at0s, then60s,119s,700s. Original all-binary greedy count0; making60s nonbinary yields count1. Do not use all-raw greedy extraction as an upper bound.',
            'validation':'Before real timestamp parsing, exhaust all8192 binary masks on three frozen synthetic traces in validate_counterexamples(), comparing with unchanged original preflight extractor. Validate duplicate bursts, inclusive300/600s, history300/301s, clock regression, midnight, event-payload poisoning, date-only seals and inconclusive budget stops in unit tests.',
            'real_checks':'For each source use only its frozen training end and stored actual training count. Require complete-stream upper>=existing actual count; otherwise stop with error. Compare exact summary/daily replay in a fresh process; preserve all prior JSON/gzip evidence and code hashes.',
            'decision':'Exclude input inspection only if a complete timestamp upper bound<70212. Otherwise cannot_exclude, not eligible/pass for source/class/quality or independent validation. Also report min(upper,263295,789885//3) as budget-capped upper, not altered arrival criteria.',
            'work':'Report total streamed rows including date-only sealed rows, training timestamps parsed, raw potential cutoff rows, candidate/eligible/expired/unresolved minute buckets, queue pushes/pops/peak and retrospective cutoff-date totals. Full checksum/decompression work remains; no energy or zero-I/O claim. No labels or readout writes.',
            'design_reference':'The user circuit-optimization memo motivates avoiding needless later work while accounting for screening/maintenance overhead. No quantization, structural plasticity, Transformer or runtime feature is introduced.',
            'stop':'No condition relaxation, threshold/current modification, unknown-event rejection, test opening, old-negative reclassification or comparative learner. No new identity acquisition during this method validation.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),
             'code':{name:file_sha256(project_path(name)) for name in CODE},'preserved':preserved})
        print('Timestamp-only upper-bound method registered before training timestamp audit')
        return
    reg = read_path(path('registration'))
    if file_sha256(path('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')
    def verify():
        for name,digest in {**reg['code'],**reg['preserved']}.items():
            if file_sha256(project_path(name)) != digest:
                raise ValueError('Pinned evidence changed: '+name)
    verify()
    if mode == 'execute':
        save(path('execution'),{'started':True,'protocol_sha256':reg['protocol_sha256']})
    validation = validate_counterexamples()
    if mode == 'execute':
        save(path('validation'),validation)
    elif validation != read_path(path('validation')):
        raise ValueError('Synthetic replay differs')
    report = {'sources':{},'validation_sha256':file_sha256(path('validation')),
              'new_identity_inputs_opened':False,'sensor_message_fields_inspected':False,
              'actual_learning_writes':0,'original_development_opened':False,'test_opened':False}
    daily = {}
    for identity,spec in read_path(path('protocol'))['sources'].items():
        metadata = select(read_path(fixture(spec['metadata'])),spec['metadata_fields'])
        receipt = read_path(fixture(spec['acquisition']))
        if spec['integrity']:
            receipt = {**receipt,**read_path(fixture(spec['integrity']))}
        if receipt['size'] > 536870912:
            raise ValueError('Existing source member exceeds size ceiling')
        source = raw_data_path(*spec['raw'])
        result = bound_lines(member_lines(source,receipt),metadata)
        actual = select(read_path(fixture(spec['baseline'])),spec['baseline_fields'])
        if result['timestamp_upper_bound'] is None or actual > result['timestamp_upper_bound']:
            raise ValueError('Timestamp screen incomplete or undercounts existing arrivals')
        result.update({'existing_training_arrivals':actual,'bound_minus_existing':result['timestamp_upper_bound']-actual,
                       'dominates_existing_arrivals':True,'training_start':metadata['train_start'],
                       'training_end_exclusive':metadata['development_start']})
        daily[identity] = result.pop('daily')
        report['sources'][identity] = result
    verify()
    if mode == 'execute':
        save(path('daily'),daily)
        report['daily_sha256'] = file_sha256(path('daily'))
        save(path('result'),report)
    else:
        if daily != read_path(path('daily')):
            raise ValueError('Daily timestamp audit replay differs')
        report['daily_sha256'] = file_sha256(path('daily'))
        if report != read_path(path('result')):
            raise ValueError('Fresh-process timestamp screen differs')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
