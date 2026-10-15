#!/usr/bin/env python3
"""Preregister and execute a read-only audit of two existing training checkpoints."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from sara_engine.evaluation.casas_count_audit import audit, load_snapshot
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import project_path, processed_data_path, model_path, ensure_parent_directory

ARMS = ('pool_evidence','joint_evidence')
CODE = ('scripts/eval/casas_count_support_audit.py','src/sara_engine/evaluation/casas_count_audit.py',
        'src/sara_engine/evaluation/casas_evidence.py','src/sara_engine/evaluation/casas_longitudinal.py',
        'src/sara_engine/evaluation/sparse_count_replay.py','src/sara_engine/evaluation/delayed_sparse_counts.py',
        'src/sara_engine/evaluation/casas_180_source.py','src/sara_engine/utils/project_paths.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures','casas_count_audit_'+suffix+'_v1.json')


def save(destination,value):
    with Path(ensure_parent_directory(destination)).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def read(suffix):
    return json.loads(Path(path(suffix)).read_text())


def main(mode):
    if mode == 'register':
        preserved = {str(p.relative_to(project_path())):file_sha256(p)
                     for directory in ('data/processed/benchmark_fixtures','models')
                     for p in Path(project_path(directory)).rglob('*')
                     if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
        save(path('protocol'),{
            'scope':'Read-only audit of consumed HH104 80-day fitted count states only. Prior negative results known. No raw source, confirmation predictions, new fitting or source acquisition.',
            'inputs':{arm:str(Path(model_path('casas_pooling',arm+'_v1.json.gz')).relative_to(project_path())) for arm in ARMS},
            'current':'Preserve exact stored raw event/weekday/hour/15-minute current address. Validate that detail counts partition each current, currents partition global and existing changed writes equal three times fit observations.',
            'evidence':'Unchanged strict delta > sqrt(log(80)/(2*n))+sqrt(log(80)/(2*m)), disjoint complement=current-detail. No threshold or current-context change. Cross-check every positive-complement classification against original evidence().',
            'partition':['no_complement: m=0, no disjoint observed alternative',
                         'support_impossible: n,m>0 and radius>=1, strict evidence impossible even at unit rate separation',
                         'margin_impossible: radius<1 but maximum feasible separation given n,m,total current positives<=radius',
                         'observed_below_bound: radius<1, feasible maximum>radius but observed separation<=radius',
                         'evidence_pass: observed separation>radius'],
            'margin_maximum':'Maximize abs(x/n-(P-x)/m) at integer endpoints x=max(0,P-m) and x=min(n,P). Preserve sample sizes and class margins. Do not misclassify rare-outcome constraints as weak observed association.',
            'report':'Deterministic gzip branch trace plus JSON summaries. All branch classifications and fixed summaries: branches, unique current contexts and fit observation mass per category, opposing evidence contexts with >=4 directional outcomes. Context categories overlap; branch/observation categories partition. Complements overlap; no independent-sample or p-value claim.',
            'validation':'Hand calculations, exhaustive small fixed-margin allocations, equality with existing evidence, partition/global/write conservation, identical current-count hashes across inputs, invalid checkpoint rejection, exact fresh-process replay and unchanged checkpoint bytes. Audit learning writes must be zero for both inputs.',
            'interpretation':'observed_below_bound is insufficient observed separation under this fixed rule, not proof of small population effect, absence of time information or adequate statistical power. Cumulative endpoints do not diagnose time drift or online activation coverage.',
            'stop':'Descriptive audit only; no winner selection, threshold search, new model, runtime/factor promotion or confirmation/development/test opening. Preserve all negative artifacts and seals.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),
             'code':{name:file_sha256(project_path(name)) for name in CODE},'preserved':preserved})
        print('Fixed training-count audit registered before new decomposition')
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
    report = {'scope':'Existing fitted counts only; zero new learning', 'arms':{},
              'raw_source_opened':False,'confirmation_opened':False,
              'original_development_opened':False,'test_opened':False}
    branches = {}
    for arm,name in read('protocol')['inputs'].items():
        source = project_path(name)
        report['arms'][arm] = audit(load_snapshot(source,arm))
        report['arms'][arm]['checkpoint_sha256'] = file_sha256(source)
        branches[arm] = report['arms'][arm].pop('branches')
    if len({(item['current_counts_sha256'],item['existing_changed_writes']) for item in report['arms'].values()}) != 1:
        raise ValueError('Audit inputs do not share the same fitted current counts')
    verify()
    trace = gzip.compress(json.dumps(branches,sort_keys=True,separators=(',',':')).encode(),mtime=0)
    trace_path = path('branches')+'.gz'
    report['branch_trace_sha256'] = sha256(trace).hexdigest()
    if mode == 'execute':
        with Path(ensure_parent_directory(trace_path)).open('xb') as stream:
            stream.write(trace)
        save(path('result'),report)
    elif report != read('result'):
        raise ValueError('Fresh-process decomposition differs')
    if Path(trace_path).read_bytes() != trace:
        raise ValueError('Branch trace replay differs')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
