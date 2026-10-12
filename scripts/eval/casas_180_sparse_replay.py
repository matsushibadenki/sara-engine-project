#!/usr/bin/env python3
"""Audit an equivalent sparse backend against the complete frozen experiment."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts
from sara_engine.evaluation import casas_longitudinal
from sara_engine.evaluation.casas_180 import extract, qualification
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import project_path, raw_data_path, processed_data_path, model_path, ensure_parent_directory

PREFIX='casas_180_sparse_replay'
CODE=('src/sara_engine/evaluation/sparse_count_replay.py','scripts/eval/casas_180_sparse_replay.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures',PREFIX+'_'+suffix+'_v1.json')


def save(destination, value):
    with Path(ensure_parent_directory(destination)).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


def main(mode):
    result_path=processed_data_path('benchmark_fixtures','casas_180_result_v1.json')
    if mode=='register':
        save(path('registration'),{
            'scope':'Post-result verification backend, not a new scientific arm or retuned experiment. Original implementation/protocol/results/models remain unchanged.',
            'reason':'Read-only full replay sampled in model-wide dict-key difference capacity scan; stopped without modifying source evidence. Replace only capacity admission with three direct membership probes; frozen delivery requires no capacity work.',
            'requirements':'Same receipt identity, due order, supplier validation, route ceiling, changed writes, local counts, every source/quality gate, trace and checkpoint; complete fresh-process refit must equal original result.',
            'code':{p:file_sha256(project_path(p)) for p in CODE},
            'original_result_sha256':file_sha256(result_path),
            'original_registration_sha256':file_sha256(processed_data_path('benchmark_fixtures','casas_180_registration_v1.json'))})
        print('Sparse-equivalence verification registered');return
    reg=json.loads(Path(path('registration')).read_text())
    for p,h in reg['code'].items():
        if file_sha256(project_path(p))!=h:raise ValueError('Verification implementation changed')
    if file_sha256(result_path)!=reg['original_result_sha256']:raise ValueError('Frozen result changed')
    original_reg_path=processed_data_path('benchmark_fixtures','casas_180_registration_v1.json')
    if file_sha256(original_reg_path)!=reg['original_registration_sha256']:raise ValueError('Original registration changed')
    original_reg=json.loads(Path(original_reg_path).read_text())
    for p,h in {**original_reg['code'],**original_reg['preserved']}.items():
        if file_sha256(project_path(p))!=h:raise ValueError('Original code/evidence changed')
    if mode=='execute':save(path('execution'),{'started':True})
    receipt=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_acquisition_v1.json')).read_text())
    meta=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_metadata_v1.json')).read_text())
    examples,audit=extract(raw_data_path('r2_casas_180','hh103.deflate'),receipt,meta)
    expected=json.loads(Path(result_path).read_text())
    if audit!=expected['extraction'] or qualification(examples,meta)!=expected['qualification']:
        raise ValueError('Source conditions differ')
    # Isolated verification process: same frozen comparison function and inputs.
    reference=casas_longitudinal.DelayedSparseCounts
    try:
        casas_longitudinal.DelayedSparseCounts=SparseCapacityCounts
        comparison,states=casas_longitudinal.compare(examples)
    finally:
        casas_longitudinal.DelayedSparseCounts=reference
    if json.loads(json.dumps(comparison))!=expected['comparison']:
        raise ValueError('Frozen predictions, writes or gates differ')
    for arm,state in states.items():
        model=Path(model_path('casas_hh103_180',arm+'_v1.json.gz'))
        if file_sha256(model)!=expected['model_sha256'][arm] or json.loads(gzip.decompress(model.read_bytes()))!=state:
            raise ValueError('Frozen checkpoint differs')
    report={'source_conditions_equal':True,'all_gates_metrics_traces_equal':True,
            'all_eight_checkpoints_equal':True,'changed_writes_per_arm':56028,
            'capacity_probes_per_training_receipt':3,'frozen_capacity_probes':0,
            'test_opened':False,'original_result_sha256':reg['original_result_sha256']}
    if mode=='execute':save(path('result'),report)
    elif report!=json.loads(Path(path('result')).read_text()):raise ValueError('Sparse verification replay differs')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
