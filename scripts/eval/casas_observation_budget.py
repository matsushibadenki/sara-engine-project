#!/usr/bin/env python3
"""Preregister conditional observation/state forecasts from existing fit counts."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from sara_engine.evaluation.casas_observation_budget import forecast, ADDITIONAL_OBSERVATIONS, ADDITIONAL_WRITES, STATE_CAPACITY
from sara_engine.evaluation.casas_count_audit import load_snapshot
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import project_path, processed_data_path, model_path, ensure_parent_directory

CODE = ('scripts/eval/casas_observation_budget.py','src/sara_engine/evaluation/casas_observation_budget.py',
        'src/sara_engine/evaluation/casas_count_audit.py','src/sara_engine/evaluation/casas_evidence.py',
        'src/sara_engine/evaluation/casas_longitudinal.py','src/sara_engine/evaluation/sparse_count_replay.py',
        'src/sara_engine/evaluation/delayed_sparse_counts.py','src/sara_engine/evaluation/casas_180_source.py',
        'src/sara_engine/utils/project_paths.py')


def path(suffix):
    return processed_data_path('benchmark_fixtures','casas_observation_budget_'+suffix+'_v1.json')


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
            'scope':'Conditional planning from already consumed HH104 80-day training count tables. Prior negatives known. No fitting, raw source, confirmation scores or data acquisition.',
            'inputs':{arm:str(Path(model_path('casas_pooling',arm+'_v1.json.gz')).relative_to(project_path())) for arm in ('pool_evidence','joint_evidence')},
            'fixed':'Exact current raw sensor/message, weekday, hour and 15-minute bins. Same detail routes and strict log(80) disjoint evidence rule; no threshold, current aggregation or model selection.',
            'acquisition_budget':{'additional_qualified_observed_outcomes':ADDITIONAL_OBSERVATIONS,'additional_changed_writes_per_arm':ADDITIONAL_WRITES,'count_route_capacity':STATE_CAPACITY},
            'budget_basis':'Fixed 16-fold cohort ceiling for the known 17,553-example fit: at most 15 additional cohorts, three real writes per new outcome per arm if ever executed. This is an explicit planning ceiling, not a measured data availability or energy budget.',
            'observed_rates':'Uniform integer replication k of the entire contingency table; both class rates and all history/current frequencies remain exactly fixed. Find minimal k by strict-bound integer inversion and verify k and k-1 with unchanged evidence(). No selective branch acquisition. Zero contrast or absent complement remains impossible under this assumption.',
            'oracle':'Separate optimistic per-branch fixed-margin reassignment: preserve k*n,k*m,k*P; maximize separation at allocation endpoints. These counterfactual branch assignments need not coexist; their eligible branch mass is only a branchwise upper bound, not simultaneous coverage, global feasibility or a new predictive result.',
            'targets':'Observed-rate forecast only: >=10 current contexts with distinct evidence branches in opposite directions and >=4 directional outcomes each; >=1% fitted observation mass in passing branches. These are count-planning targets, not replacements for original independent quality, confirmation coverage or active-day gates.',
            'state_assumptions':'Closed world: exact same addresses recur, no added routes. Open world: at most one new current and one new detail per additional outcome, so conservative routes=S+2*B; report full-budget and required-budget bounds plus state-safe whole-cohort ceiling. This is an upper bound, not measured growth; open-world events violate the closed-world frequency assumption.',
            'inversion':'Binary integer inversion of original radius at k*n,k*m, never a threshold sweep. Numerical ceiling 2**128; exceeding it is an explicit error. Required additional outcomes=(k-1)*N; writes=3*(k-1)*N. Both audited models must retain equal current-count hashes and historical writes.',
            'report':'Fixed budget endpoints and minimum full-cohort scale, per-branch deterministic gzip trace, route bounds, no finite scale reasons, and exact fresh-process replay. Actual acquisition and learning writes remain zero.',
            'limitations':'No guarantee of recurrence, stationarity, independent evidence, raw-row yield, duration, byte volume, latency, RAM bytes, physical energy or future usefulness. Fixed observed-zero contrast is not proof of population zero. Do not extrapolate calendar days from counts.',
            'stop':'No acquisition or runtime/factor promotion even if a conditional forecast passes. Preserve all negative results and sealed periods. Any future learner or data acquisition needs separate preregistration and original quality gates.'})
        save(path('registration'),{'protocol_sha256':file_sha256(path('protocol')),
             'code':{name:file_sha256(project_path(name)) for name in CODE},'preserved':preserved})
        print('Conditional observation budget registered before forecast')
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
    report = {'arms':{},'raw_source_opened':False,'confirmation_opened':False,
              'original_development_opened':False,'test_opened':False,
              'actual_learning_writes':0,'actual_acquired_observations':0}
    branches = {}
    for arm,name in read('protocol')['inputs'].items():
        report['arms'][arm] = forecast(load_snapshot(project_path(name),arm))
        branches[arm] = {scenario:result.pop('branches') for scenario,result in report['arms'][arm]['scenarios'].items()}
        report['arms'][arm]['checkpoint_sha256'] = file_sha256(project_path(name))
    if len({(a['current_counts_sha256'],a['existing_changed_writes']) for a in report['arms'].values()}) != 1:
        raise ValueError('Current counts or historical write matching changed')
    verify()
    trace = gzip.compress(json.dumps(branches,sort_keys=True,separators=(',',':')).encode(),mtime=0)
    report['branch_trace_sha256'] = sha256(trace).hexdigest()
    trace_path = path('branches')+'.gz'
    if mode == 'execute':
        with Path(ensure_parent_directory(trace_path)).open('xb') as stream:
            stream.write(trace)
        save(path('result'),report)
    elif report != read('result'):
        raise ValueError('Fresh-process forecast differs')
    if Path(trace_path).read_bytes() != trace:
        raise ValueError('Branch forecast trace differs')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
