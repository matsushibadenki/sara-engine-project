"""Finite saved-report timing/retention audit; no learner or generator imports."""
from sara_engine.evaluation.reactivation_codec import ReportCodec
from sara_engine.evaluation.route_numeric_v2 import canonical, validate_record

PROTOCOL_SHA256 = 'c3a740c6049bd2b9e3f2a7facf63d2904e3c71b92f54249727dc9dcd43dcfeed'
INPUTS = ('sparse_route_reactivation_selection_v1.json', 'sparse_route_reactivation_result_v1.json',
          'sparse_route_reactivation_failure_v1.json', 'route_numeric_v2_selection_v1.json',
          'route_numeric_v2_primary_v1.json', 'route_numeric_v2_replication_v1.json', 'route_numeric_v2_result_v1.json')
ARMS = ('round_robin', 'permanent', 'periodic', 'error_triggered', 'shuffled_error')
UNKNOWN = ('missing_dependency_ids', 'route_score_margins', 'recent_route_support',
           'functional_substitution_proven', 'causal_effect_of_shorter_exploration', 'within_block_prediction_order')


def integer(value, low=0, high=1024):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Expected bounded integer metadata')
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError('Expected exact boolean')
    return value


def pool(value, size=None):
    if type(value) is not list or any(type(i) is not int or not 0 <= i < 45 for i in value):
        raise ValueError('Expected bounded route ID list')
    if len(set(value)) != len(value) or value != sorted(value) or (size is not None and len(value) != size):
        raise ValueError('Invalid route ID set or pool size')
    return set(value)


def lifecycle(arm, name):
    """Reconstruct sets only, never latent meaning or unrecorded scores."""
    starts, commits = arm['exploration_starts'], arm['commits']
    if type(starts) is not list or len(starts) > 64 or type(commits) is not list:
        raise ValueError('Lifecycle metadata exceeds fixed bound')
    for start in starts: integer(start, 180, 1023)
    if starts != sorted(set(starts)) or any(b < a+180 for a, b in zip(starts, starts[1:])):
        raise ValueError('Exploration windows overlap or repeat')
    if name in ('error_triggered', 'shuffled_error') and any(b < a+244 for a, b in zip(starts, starts[1:])):
        raise ValueError('Recorded exploration violates cooldown')
    expected = [start+180 for start in starts if start+180 < 1024]
    if [c['tick'] for c in commits] != expected:
        raise ValueError('Commit does not follow 180 observed receipts before final begin')
    unfinished = any(start+180 >= 1024 for start in starts)
    if boolean(arm['unfinished_exploration']) != unfinished:
        raise ValueError('Unfinished exploration flag disagrees')
    final = pool(arm['final_pool'], 45 if name == 'round_robin' else 9)
    if name == 'round_robin':
        if arm['initial_pool'] is not None or starts or commits or final != set(range(45)):
            raise ValueError('Full-bank control changed')
        initial = set(range(45))
    else:
        initial = pool(arm['initial_pool'], 9)
    if name == 'permanent' and (starts or commits):
        raise ValueError('Permanent control reactivated')
    if name == 'periodic' and starts != [436, 692, 948]:
        raise ValueError('Fixed periodic schedule changed')
    current = set(initial)
    for commit in commits:
        if set(commit) != {'tick', 'added', 'removed'}:
            raise ValueError('Unexpected commit fields')
        integer(commit['tick'], 0, 1023)
        added, removed = pool(commit['added']), pool(commit['removed'])
        if not removed <= current or added & current or added & removed:
            raise ValueError('Commit route membership is inconsistent')
        current = (current-removed)|added
        if len(current) != 9:
            raise ValueError('Commit changes nine-route ceiling')
    if current != final:
        raise ValueError('Final pool cannot be reconstructed')
    return {'starts': list(starts), 'commit_ticks': expected, 'unfinished': unfinished,
            'start_count': len(starts), 'commit_count': len(commits),
            'added_from_initial': sorted(final-initial), 'removed_from_initial': sorted(initial-final),
            'final_pool_reconciled': True}


def blocks(arm):
    values = arm['blocks']
    if len(values) != 16 or [b['end'] for b in values] != list(range(64, 1025, 64)):
        raise ValueError('Fixed 64-tick block layout changed')
    for block in values:
        exact = block['exact']
        if type(exact) is not float or not 0 <= exact <= 1 or not (exact*64).is_integer():
            raise ValueError('Saved block accuracy cannot represent an integer correct count')
    post = [b for b in values if b['end'] > 512]
    recovered = next((b['end']-512 for a, b in zip(post, post[1:])
                      if a['exact'] >= 0.90 and b['exact'] >= 0.90), None)
    saved = arm['recovery_ticks']
    if saved is not None: integer(saved, 1, 512)
    if saved != recovered:
        raise ValueError('Original recovery criterion does not reconstruct')
    return values, recovered


def boundary_bounds(block, commit):
    """Sharp subset-correct bounds, not guessed within-block predictions."""
    end, start = block['end'], block['end']-64
    old = max(0, min(end, commit)-start)
    new = 64-old
    correct = int(block['exact']*64)
    low, high = max(0, correct-old), min(correct, new)
    return {'start': start, 'end_exclusive': end, 'stored_total_correct': correct,
            'pre_commit_predictions': old, 'post_commit_predictions': new,
            'post_commit_correct_bounds': [low, high],
            'post_commit_accuracy_bounds': [low/new, high/new] if new else None}


def timing(arm, life):
    values, recovered = blocks(arm)
    eligible = [(start, end) for start, end in zip(life['starts'], [s+180 for s in life['starts']])
                if 512 <= end < 1024]
    if not eligible:
        return {'first_post_change_commit': None, 'reported_recovery_ticks': recovered}
    start, commit = eligible[0]
    first_full_start = ((commit+63)//64)*64
    end = first_full_start+128
    first_post_start = next((s for s in life['starts'] if s >= 512), None)
    result = {'first_start_at_or_after_change': first_post_start,
              'trigger_delay_ticks': first_post_start-512 if first_post_start is not None else None,
              'committed_window_start': start, 'first_post_change_commit': commit,
              'exploration_receipts': commit-start, 'commit_delay_ticks': commit-512,
              'window_pre_change_receipts': max(0, 512-start),
              'window_post_change_receipts': commit-max(start, 512),
              'deadline_remaining_after_commit': 768-commit,
              'first_wholly_post_commit_block_start': first_full_start,
              'earliest_wholly_post_commit_pair_end_delay': end-512,
              'two_wholly_post_commit_blocks_fit_deadline': end <= 768,
              'reported_recovery_ticks': recovered,
              'reported_pair_contains_pre_commit_predictions': recovered+512-128 < commit if recovered is not None else None,
              'deadline_last_pair': [boundary_bounds(b, commit) for b in values if b['end'] in (704, 768)],
              'original_gate_impossible': None, 'causal_effect_of_shorter_exploration': None}
    return result


def verify_saved_inputs(selection, reports, hashes):
    if tuple(reports) != INPUTS or tuple(selection['inputs']) != INPUTS:
        raise ValueError('Finite saved-input roster/order changed')
    for name in INPUTS:
        key = 'data/processed/benchmark_fixtures/'+name
        if hashes[name] != selection['preserved'][key]:
            raise ValueError('Saved input bytes changed')
    original, saved, failure, v2selection, primary, replica, proof = (reports[n] for n in INPUTS)
    spec = {k: v for k, v in original.items() if k not in {'schema','registered_at_utc','authorization','preserved','stop'}}
    if canonical(spec) != canonical(v2selection['frozen_scientific_spec']):
        raise ValueError('Original scientific conditions changed')
    if original['change_at'] != 512 or original['steps'] != 1024 or original['exploration_steps'] != 180 or original['gates']['block_steps'] != 64 or original['gates']['recovery_block_exact_min'] != 0.90 or original['gates']['recovery_consecutive_blocks'] != 2 or original['gates']['recovery_ticks_max'] != 256:
        raise ValueError('Fixed timing/evidence rule differs')
    if failure['full_replay_verified'] is not False or failure['numeric_trace_equality'] is not None or saved['component_passed'] is not False:
        raise ValueError('Old failure or negative result was reclassified')
    for key in ('numerical_reproduction_passed','primary_replication_exact_match','historical_values_exact_match','independent_processes'):
        if proof[key] is not True: raise ValueError('Saved numerical reproduction prerequisite failed')
    if proof['component_passed'] is not False or proof['old_failed_replay_reclassified'] is not False or proof['old_discarded_trace_equality'] is not None:
        raise ValueError('Numerical success erased old negative or unknown')
    for phase, report in (('primary', primary), ('replication', replica)):
        expected_sha = hashes['route_numeric_v2_'+phase+'_v1.json']
        if proof[phase+'_sha256'] != expected_sha:
            raise ValueError('Saved numerical proof/report binding differs')
        context = {'phase': phase, 'registration_sha256': proof['registration_sha256'],
                   'execution_sha256': selection['preserved']['data/processed/benchmark_fixtures/route_numeric_v2_'+phase+'_execution_v1.json'],
                   'runtime': primary['runtime'], 'preserved_count': len(v2selection['preserved']),
                   'historical_report_sha256': hashes[INPUTS[1]]}
        validate_record(report, saved, context)
        if report['test_opened'] is not False or report['conditions_relaxed'] is not False:
            raise ValueError('Saved seals or conditions changed')
    if not ReportCodec(saved).equivalent(primary['payload'], replica['payload']):
        raise ValueError('Reproduced numerical payloads differ')
    return original, primary['payload']


def evaluate(selection, reports, hashes):
    original, payload = verify_saved_inputs(selection, reports, hashes)
    identities = [(t, s, c) for t in original['tasks'] for s in original['seeds'] for c in original['conditions']]
    if [(c['task'],c['seed'],c['condition']) for c in payload['cases']] != identities:
        raise ValueError('Fixed 20-case roster changed')
    lifecycle_rows, detail, stationary = [], [], []
    for index, case in enumerate(payload['cases']):
        identity = {k:case[k] for k in ('task','seed','condition')}
        if set(case['arms']) != set(ARMS): raise ValueError('Missing sufficient comparator')
        lives = {}
        for name in ARMS:
            arm = case['arms'][name]
            lives[name] = lifecycle(arm, name)
            blocks(arm)
            lifecycle_rows.append({**identity,'arm':name,**lives[name],
                                  'source_pointer':'/payload/cases/'+str(index)+'/arms/'+name})
        candidate = case['arms']['error_triggered']
        if case['condition'] == 'stationary':
            stationary.append({**identity,'false_start_count':len(lives['error_triggered']['starts']),
                               'original_gates':case['component_gates']})
            continue
        timeline = timing(candidate, lives['error_triggered'])
        recovered = timeline['reported_recovery_ticks']
        reconstructed_gate = recovered is not None and recovered <= 256
        if boolean(case['component_gates']['recovery_within_256']) != reconstructed_gate:
            raise ValueError('Original negative recovery gate changed')
        retained = boolean(candidate['final_dependencies_retained'])
        if boolean(case['component_gates']['new_shared_dependencies_retained']) != retained:
            raise ValueError('Original exact-retention gate differs')
        high = candidate['tail_exact'] >= 0.90 and candidate['frozen']['exact_accuracy'] >= 0.90
        classification = 'retained' if retained else ('exact_retention_failure_with_high_reported_accuracy' if high else 'retention_failure_with_lower_reported_accuracy')
        detail.append({**identity,'candidate_timing':timeline,'periodic_timing':timing(case['arms']['periodic'],lives['periodic']),
                       'retention_class':classification,'reported_exact_retention':retained,
                       'reported_tail_exact':candidate['tail_exact'],'reported_frozen_exact':candidate['frozen']['exact_accuracy'],
                       'unknowns':{k:None for k in UNKNOWN},'source_pointer':'/payload/cases/'+str(index)})
    if len(detail) != 10 or len(stationary) != 10 or len(lifecycle_rows) != 100:
        raise ValueError('Fixed audit population differs')
    return {'schema':'sara-route-failure-audit-result-v1','saved_numerical_equivalence_verified':True,'source_report':INPUTS[4],
            'source_input_sha256':dict(hashes),
            'lifecycle_rows':lifecycle_rows,'changed_case_detail':detail,'stationary_controls':stationary,
            'summary':{'changed_cases':10,'pool_lifecycles_reconciled':100,
                'recovery_failures':sum(d['candidate_timing']['reported_recovery_ticks'] is None or d['candidate_timing']['reported_recovery_ticks']>256 for d in detail),
                'deadline_lacks_two_wholly_post_commit_blocks':sum(not d['candidate_timing']['two_wholly_post_commit_blocks_fit_deadline'] for d in detail),
                'exact_retention_failures':sum(not d['reported_exact_retention'] for d in detail),
                'high_accuracy_exact_retention_failures':sum(d['retention_class']=='exact_retention_failure_with_high_reported_accuracy' for d in detail),
                'stationary_false_starts':sum(d['false_start_count'] for d in stationary)},
            'interpretation':'Delayed pool commits leave insufficient room for two wholly post-commit blocks under the unchanged gate. Old/mixed blocks could still pass; this is not proof of impossibility or causal benefit from shortening exploration. Exact-route failure and finite predictive accuracy are distinct; missing route identities/ranks/support are unrecorded.',
            'old_failed_replay_reclassified':False,'old_discarded_trace_equality':None,
            'component_passed':False,'learner_executions':0,'new_predictions_scored':0,'new_measurements':0,
            'new_acquisitions':0,'test_opened':False,'conditions_relaxed':False,
            'new_experiment_authorized':False,'promotion_authorized':False}
