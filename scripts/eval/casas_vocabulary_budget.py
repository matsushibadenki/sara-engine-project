#!/usr/bin/env python3
"""Preregister known-combination and unknown-event state ceilings from fit counts."""
import argparse
import json
from pathlib import Path
from sara_engine.evaluation.casas_vocabulary_budget import analyze, ARMS
from sara_engine.evaluation.casas_count_audit import load_snapshot
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import project_path, processed_data_path, model_path, ensure_parent_directory

CODE = ('scripts/eval/casas_vocabulary_budget.py','src/sara_engine/evaluation/casas_vocabulary_budget.py',
        'src/sara_engine/evaluation/casas_count_audit.py','src/sara_engine/evaluation/casas_observation_budget.py',
        'src/sara_engine/evaluation/casas_evidence.py','src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/sparse_count_replay.py','src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/evaluation/casas_180_source.py','src/sara_engine/utils/project_paths.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures','casas_vocabulary_budget_'+suffix+'_v1.json')


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
            'scope':'Count-address-only state ceiling from consumed HH104 fitted pool/joint checkpoints. Earlier negatives and forecasts known. No raw source, score evaluation, learner, acquisition or new rejection policy.',
            'inputs':{arm:str(Path(model_path('casas_pooling',arm+'_v1.json.gz')).relative_to(project_path())) for arm in ARMS},
            'known_vocabulary':'Union of raw (sensor ID,message) events in both current tables and the two stored joint history positions. Pool booleans cannot reveal history identities; do not mistake a current-only set for the full retained event set. This is the vocabulary retained in these keys, not a claim to know all source events.',
            'fixed_current':'Same raw sensor/message, weekday7, hour24, 15-minute bins4: 672 contexts per current event. No clock coarsening, pooling across current addresses or evidence threshold changes. Three changed writes per observed outcome remains original accounting.',
            'envelopes':'For V shared known event types, current C=672*V. Pool has <=4*C detail routes, total <=1+5*C. Joint has <=16*C*V**2 details for two lag events and two four-bin gaps, total <=1+C+16*C*V**2. Include global and current routes. These are conservative syntactic upper bounds, not necessarily jointly reachable histories.',
            'worlds':'Exact address recurrence uses existing S. All new clock/history combinations over the fixed known vocabulary use the finite envelope. U genuinely new event types use the same envelope at V+U, regardless of identity. Unrestricted unknown events have no finite all-horizon bound; at the fixed +263295-outcome acquisition ceiling generic S+2*B still bounds count routes.',
            'report':'Known vocabulary cardinalities, history-only events, deterministic event inventory and hash; existing and finite-vocabulary route bounds, generic finite-acquisition bounds and min of applicable upper bounds. Certify capacity only if an upper bound<=100000; exceeding a bound is not proof of actual overflow.',
            'frontier':'Report largest total event cardinality whose full envelope<=100000, remaining hypothetical unknown-type allowance only if known envelope already certified, and one-new-type endpoint. Integer inversion is a cardinality certificate, not an event rejection/admission policy or threshold search.',
            'validation':'Unchanged count/current/global/write conservation, cross-input current hashes, known history union, toy route enumeration against the original address functions, monotone capacity frontiers, no learning/mutation, exact process/inventory replay and old artifact hashes.',
            'limits':'No unknown events are discarded or remapped. Fixed vocabulary is an assumption, not an observed future guarantee. No raw-row yield, acquisition duration, byte volume, physical memory/energy, predictive quality or statistical coverage inferred. Previous fixed-rate forecasts must not be combined with new-combination envelopes into a proven learner result.',
            'stop':'Descriptive capacity analysis only. Unlimited unseen-event feasibility remains unresolved. No acquisition, outcome reading, independent period opening, new input policy or factor/runtime promotion even if a finite-vocabulary bound fits.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),
             'code':{name:file_sha256(project_path(name)) for name in CODE},'preserved':preserved})
        print('Vocabulary state ceilings registered before count-address inspection')
        return
    reg = read('registration')
    if file_sha256(path('protocol')) != reg['protocol_sha256']:
        raise ValueError('Protocol changed')
    def verify():
        for name,digest in {**reg['code'],**reg['preserved']}.items():
            if file_sha256(project_path(name)) != digest:
                raise ValueError('Pinned evidence changed: '+name)
    verify()
    if mode == 'execute':
        save(path('execution'),{'started':True,'protocol_sha256':reg['protocol_sha256']})
    report = analyze({arm:load_snapshot(project_path(name),arm) for arm,name in read('protocol')['inputs'].items()})
    inventory = report.pop('inventory')
    if mode == 'execute':
        save(path('inventory'),inventory)
    elif json.loads(json.dumps(inventory)) != read('inventory'):
        raise ValueError('Vocabulary inventory differs')
    report['inventory_sha256'] = file_sha256(path('inventory'))
    verify()
    if mode == 'execute':
        save(path('result'),report)
    elif report != read('result'):
        raise ValueError('Fresh-process state bounds differ')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
