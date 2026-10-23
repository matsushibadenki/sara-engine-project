"""Fresh v2 orchestration around immutable allocation measurement components."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sara_engine.evaluation import event_allocation as frozen
from sara_engine.evaluation import allocation_report as codec
from sara_engine.evaluation.allocation_report_preflight import roster
from sara_engine.utils.project_paths import project_path, processed_data_path


def inputs(selection):
    raw = Path(project_path(selection['input']['path'])).read_bytes()
    if sha256(raw).hexdigest() != selection['input']['sha256']:
        raise ValueError('Fixed synthetic source changed')
    previous = json.loads(Path(processed_data_path('benchmark_fixtures', 'event_work_separation_result_v1.json')).read_text())
    return frozen.cases(selection, json.loads(raw), previous)


def check_report_limits(selection):
    expected = dict(report_bytes=codec.LIMIT, workers=codec.WORKERS, calls_per_worker=codec.CALLS,
                    sites=codec.SITES, site_json_bytes=codec.SITE_BYTES, site_incidences=codec.INCIDENCES,
                    unsigned_integer_max=codec.U64)
    if selection['report_limits'] != expected:
        raise ValueError('Registered report domain changed')
    if codec.byte_proof(roster())['proved_upper_bound_bytes'] >= selection['budgets']['result_bytes']:
        raise ValueError('Preflight byte bound does not fit')


def collect(selection, phase):
    """Exactly 144 fresh serial workers per independently consumed v2 phase."""
    if phase not in ('execute', 'replay'):
        raise ValueError('Unknown diagnostic phase')
    check_report_limits(selection)
    cases = inputs(selection)
    calibration = frozen.witnesses()
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': project_path('src') + os.pathsep + project_path()}
    started, workers, registered_sites, incidences = time.monotonic(), [], set(), 0
    for case_index, (_, golden) in enumerate(cases):
        for mode in ('paired', 'neuron', 'scalar'):
            remaining = selection['budgets']['execution_timeout_seconds'] - (time.monotonic()-started)
            if remaining <= 0:
                raise ValueError('Diagnostic total-time budget exceeded')
            completed = subprocess.run([sys.executable, project_path('scripts/eval/event_allocation_v2.py'),
                'worker', '--case', str(case_index), '--arm', mode, '--phase', phase],
                cwd=project_path(), env=env, capture_output=True, text=True, check=True,
                timeout=min(remaining, selection['budgets']['worker_timeout_seconds']))
            if time.monotonic()-started > selection['budgets']['execution_timeout_seconds']:
                raise ValueError('Diagnostic total-time budget exceeded')
            if len(completed.stdout.encode()) > selection['budgets']['worker_stdout_bytes']:
                raise ValueError('Worker output byte budget exceeded')
            row = json.loads(completed.stdout)
            if (row['mode'] != mode or row['input_sha256'] != golden['input_sha256']
                    or row['receipt_sha256'] != golden['semantic_trace_sha256']
                    or any(row[k] != golden[k] for k in ('family','member','scenario','horizon'))):
                raise ValueError('Worker identity/receipt differs from fixed case')
            for record in row['records']:
                for name, site in record['allocation']['post_query_live_sites'].items():
                    if len(codec.canonical(name))-1 > codec.SITE_BYTES:
                        raise ValueError('Site name capacity exceeded')
                    registered_sites.add((site['owner'], name))
                    incidences += 1
            if len(registered_sites) > codec.SITES or incidences > codec.INCIDENCES:
                raise ValueError('Report site/incidence capacity exceeded')
            workers.append(row)
        if len(workers) % 24 == 0:
            print('Allocation v2 ' + phase + ' workers completed: ' + str(len(workers)) + '/144', flush=True)
    if len(workers) != selection['budgets']['workers_per_execution'] or len(workers) != 144:
        raise ValueError('Fixed worker count changed')
    return workers, calibration


def payload(workers, witnesses, registration, execution, runtime):
    return {'metadata': {'synthetic': False, 'registration_sha256': registration, 'execution_sha256': execution,
                         'prior_failure_sha256': sha256(Path(processed_data_path('benchmark_fixtures', 'event_allocation_failure_v1.json')).read_bytes()).hexdigest(),
                         'runtime': [runtime[k] for k in ('python','implementation','machine','byteorder')]},
            'roster': roster(), 'workers': workers, 'witnesses': witnesses}


def semantic(value):
    return frozen.semantic_result({'workers': value['workers'], 'witnesses': value['witnesses']})


def summarize(original, replica=None):
    """Descriptive per-window maxima only; no sums of peaks or resource ranking."""
    if replica is not None and semantic(original) != semantic(replica):
        raise ValueError('Independent replication changed semantic/ownership state')
    modes = {}
    for mode in ('paired','neuron','scalar'):
        workers = [w for w in original['workers'] if w['mode'] == mode]
        rows = [r for w in workers for r in w['records']]
        maxima = {k: max(r['allocation'][k] for r in rows) for k in codec.SCALARS}
        stages = {stage: ({k: max(r['allocation'][k] for r in rows if r['stage'] == stage)
                           for k in codec.SCALARS} if any(r['stage']==stage for r in rows) else None)
                  for stage in codec.STAGES}
        own = {k: max(r['ownership'][k] for r in rows) for k in codec.GRAPH}
        modes[mode] = {'workers':len(workers), 'recorded_calls':len(rows), 'max_new_traced_bytes':maxima,
                      'max_by_stage_bytes':stages, 'max_boundary_graph_bytes':own,
                      'windows_with_peak_excess':sum(r['allocation'][codec.SCALARS[2]] > 0 for r in rows),
                      'shared_input_calls':sum(w['shared_input_calls'] for w in workers),
                      'service_receipts':{arm:{key:sum(w['service_receipts'][arm][key] for w in workers)
                                        for key in codec.COUNTS} for arm in workers[0]['service_receipts']}}
    comparison = None
    if replica is not None:
        a = [r['allocation'] for w in original['workers'] for r in w['records']]
        b = [r['allocation'] for w in replica['workers'] for r in w['records']]
        comparison = {'semantic_and_ownership_exact_match':True,
                      'byte_identical_windows':sum(x==y for x,y in zip(a,b)), 'compared_windows':len(a),
                      'raw_witnesses_equal':original['witnesses']==replica['witnesses'],
                      'per_mode_run_max_ranges_bytes':{mode:{k:sorted([
                          max(r['allocation'][k] for w in value['workers'] if w['mode']==mode for r in w['records'])
                          for value in (original,replica)]) for k in codec.SCALARS} for mode in modes}}
    return {'schema':'sara-event-allocation-v2-summary-v1','case_count':48,'workers_per_phase':144,
            'modes':modes,'replication':comparison,'actual_learning_writes':0,'predictions_scored':0,
            'cpu_seconds':None,'energy_joules':None,'rss_bytes':None,'total_transient_peak_bytes':None,
            'snn_resource_advantage_established':False,'independent_usefulness_established':False,
            'runtime_promotion_authorized':False,'test_opening_authorized':False,
            'scope':'Per-call newly traced blocks under fixed GC/probe regime; graph ownership is a separate boundary measure. Neither is total process memory. Source-site snapshots are post-query live blocks, not freed allocation sites.',
            'decision':'retained_allocation_diagnostics_no_snn_advantage_established'}
