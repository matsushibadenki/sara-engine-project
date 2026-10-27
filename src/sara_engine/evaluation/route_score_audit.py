"""Finite exact-arithmetic score audit of captured evidence only."""
from fractions import Fraction
from hashlib import sha256

from sara_engine.evaluation.route_snapshot_contract import (
    CATALOG,costs,diagnose,encode,encode_collection,pool,window_support,
)

PROTOCOL_SHA256='95e5facb0a7f81e470d954cebc254dcae4163e013430ee77cc6c961c3608af29'
INPUTS=('route_capture_v3_collection_v1.json','route_capture_v3_analysis_v1.json','route_capture_v3_result_v1.json')
COEFFICIENT=Fraction.from_float(0.005)


def f(value):
    if type(value) is not float:raise ValueError('Original binary64 value required')
    return Fraction.from_float(value)


def ratio(row):return f(row[1])/f(row[2]) if row[2] else Fraction(1,4)


def ordering(rows,dependency):
    values=costs(rows);output=CATALOG[dependency]['output']
    group=sorted((i for i,r in enumerate(CATALOG) if r['output']==output),key=lambda i:(values[i],i))
    return group.index(dependency)+1,group[2]


def terms(rows,dependency,competitor):
    """Rational accounting, with all original floating arithmetic left intact."""
    if CATALOG[dependency]['output']!=CATALOG[competitor]['output']:raise ValueError('Same-output comparator required')
    error=ratio(rows[dependency])-ratio(rows[competitor])
    width=COEFFICIENT*(len(CATALOG[dependency]['inputs'])-len(CATALOG[competitor]['inputs']))
    values=costs(rows);margin=values[dependency]-values[competitor]
    rounding=f(margin)-error-width
    if error+width+rounding!=f(margin):raise ValueError('Exact cumulative decomposition failed')
    return {'cumulative_error_gap':str(error),'width_gap':str(width),
        'rounding_residual':str(rounding),'actual_float_margin':margin,
        'actual_float_margin_exact':str(f(margin)),
        'dependency_score':values[dependency],'cutoff_score':values[competitor],
        'widths':[len(CATALOG[dependency]['inputs']),len(CATALOG[competitor]['inputs'])],
        'exact_identity':True}


def window_terms(start,end,dependency,competitor):
    evidence=window_support(start,end)
    old_d,old_c=start['rows'][dependency],start['rows'][competitor]
    end_d,end_c=end['rows'][dependency],end['rows'][competitor]
    ed,ec=f(end_d[1]),f(end_c[1]);wd,wc=f(end_d[2]),f(end_c[2])
    od,oc=f(old_d[1]),f(old_c[1]);owd,owc=f(old_d[2]),f(old_c[2])
    if not 0<owd<wd or not 0<owc<wc:raise ValueError('Positive old/recent weight components required')
    old_component=od/wd-oc/wc
    new_component=(ed-od)/wd-(ec-oc)/wc
    cumulative=ed/wd-ec/wc
    recent=(ed-od)/(wd-owd)-(ec-oc)/(wc-owc)
    mix_difference=cumulative-recent
    base=terms(end['rows'],dependency,competitor)
    width=Fraction(base['width_gap']);rounding=Fraction(base['rounding_residual'])
    if old_component+new_component!=cumulative or recent+mix_difference+width+rounding!=f(base['actual_float_margin']):
        raise ValueError('Exact old/window mixture identity failed')
    recent_costs=costs(evidence['route_window_rows'])
    recent_float_margin=recent_costs[dependency]-recent_costs[competitor]
    recent_rounding=f(recent_float_margin)-recent-width
    if recent+width+recent_rounding!=f(recent_float_margin):raise ValueError('Window float residual identity failed')
    recent_rank,recent_cutoff=ordering(evidence['route_window_rows'],dependency)
    return {'pre_window_contribution_to_cumulative_error_gap':str(old_component),
        'window_contribution_to_cumulative_error_gap':str(new_component),
        'window_only_error_gap_same_pair':str(recent),
        'cumulative_minus_window_only_error_gap':str(mix_difference),
        'old_weight_shares':[str(owd/wd),str(owc/wc)],
        'window_float_margin_same_pair':recent_float_margin,
        'window_float_rounding_residual':str(recent_rounding),
        'window_rank':recent_rank,'window_own_cutoff_route':recent_cutoff,
        'window_own_cutoff_float_margin':recent_costs[dependency]-recent_costs[recent_cutoff],
        'window_same_pair_observations':[evidence['route_window_rows'][i][0] for i in (dependency,competitor)],
        'exact_mixture_identities':True,'counterfactual_accuracy':None}


def verify_inputs(selection,reports,hashes):
    if tuple(reports)!=INPUTS or tuple(selection['inputs'])!=INPUTS:raise ValueError('Finite input roster differs')
    for name in INPUTS:
        if hashes[name]!=selection['preserved']['data/processed/benchmark_fixtures/'+name]:raise ValueError('Pinned evidence bytes differ')
    collection,analysis,proof=(reports[n] for n in INPUTS)
    if proof['collection_sha256']!=hashes[INPUTS[0]] or proof['analysis_sha256']!=hashes[INPUTS[1]]:raise ValueError('Collection/analysis proof binding differs')
    for name in ('baseline','capture'):
        if proof[name+'_sha256']!=selection['preserved']['data/processed/benchmark_fixtures/route_capture_v3_'+name+'_v1.json']:raise ValueError('Numerical reference proof differs')
    if proof['registration_sha256']!=selection['preserved']['data/processed/benchmark_fixtures/route_capture_v3_registration_v1.json'] or proof['selection_sha256']!=selection['preserved']['data/processed/benchmark_fixtures/route_capture_v3_selection_v1.json']:raise ValueError('Actual capture provenance differs')
    for key in ('numerical_noninterference_passed','historical_values_exact_match','independent_processes'):
        if proof[key] is not True:raise ValueError('Actual numerical collection prerequisite failed')
    if type(proof['actual_snapshots']) is not int or proof['actual_snapshots']!=160 or proof['complete_state_read_only_checks']!=160:raise ValueError('Actual160-boundary evidence required')
    for key in ('component_passed','old_failed_replay_reclassified','conditions_relaxed','test_opened','promotion_authorized'):
        if proof[key] is not False:raise ValueError('Old negative/seal/condition boundary changed')
    if proof['old_discarded_trace_equality'] is not None:raise ValueError('Discarded old replay remains unknown')
    header=collection['header']
    expected_binding={'registration_sha256':proof['registration_sha256'],
                      'execution_sha256':selection['preserved']['data/processed/benchmark_fixtures/route_capture_v3_capture_execution_v1.json']}
    if header['provenance']!='future_external_state' or encode(header['executor_binding'])!=encode(expected_binding):raise ValueError('Actual snapshot binding differs')
    raw=encode_collection(header,collection['snapshots'],selection['boundary_roster'])
    if sha256(raw).hexdigest()!=hashes[INPUTS[0]]:raise ValueError('Canonical complete collection differs')
    checks=analysis['read_only_capture_checks'];order=analysis['actual_capture_order']
    if len(checks)!=160 or len(order)!=160 or {encode(p) for p in order}!={encode(p) for p in selection['boundary_roster']}:raise ValueError('Actual capture order missing/repeated')
    for p,check in zip(order,checks):
        if encode(p)!=encode(check['point']) or check['before_sha256']!=check['after_sha256']:raise ValueError('Capture read-only evidence differs')
    return collection,analysis['diagnostics']


def evaluate(selection,reports,hashes):
    collection,diagnostics=verify_inputs(selection,reports,hashes)
    snapshots=collection['snapshots'];truth=diagnostics['evaluator_truth']
    index={(s['point']['case_id'],s['point']['arm'],s['point']['tick'],s['point']['stage']):s for s in snapshots}
    boundary_rows=[];row_index={}
    for snapshot_index,snapshot in enumerate(snapshots):
        point=snapshot['point'];phase='initial' if point['tick']<512 else 'final'
        ids=pool(truth[point['case_id']][phase],3)
        if sorted(CATALOG[i]['output'] for i in ids)!=[0,1,2]:raise ValueError('Evaluator truth must cover all outputs')
        for dependency in ids:
            rank,competitor=ordering(snapshot['rows'],dependency)
            row={'point':point,'snapshot_index':snapshot_index,'truth_phase':phase,
                'dependency_route':dependency,'same_output_cumulative_cutoff_route':competitor,
                'rank':rank,'rank_excluded':rank>3,'actually_retained':dependency in snapshot['pool'],
                **terms(snapshot['rows'],dependency,competitor)}
            row_index[(encode(point),dependency)]=len(boundary_rows)
            boundary_rows.append(row)
    window_rows=[]
    for post in snapshots:
        point=post['point']
        if point['stage']!='post_commit':continue
        identity,arm,tick=point['case_id'],point['arm'],point['tick']
        start=index[(identity,arm,tick-180,'pre_start')];end=index[(identity,arm,tick,'pre_commit')]
        evidence=window_support(start,end)
        for dependency in truth[identity]['final' if tick>=512 else 'initial']:
            _,competitor=ordering(end['rows'],dependency)
            pointer=row_index[(encode(point),dependency)]
            row={'case_id':identity,'arm':arm,'start':tick-180,'commit':tick,
                 'dependency_route':dependency,'fixed_cumulative_cutoff_route':competitor,
                 'boundary_row_index':pointer,**window_terms(start,end,dependency,competitor)}
            if (dependency in evidence['window_ranked_pool'])!=(row['window_rank']<=3):raise ValueError('Saved window ranking differs')
            window_rows.append(row)
    focus=[]
    saved_missing={(d['case_id'],d['route_id']):d for d in diagnostics['missing_at_commit']}
    for i,row in enumerate(window_rows):
        base=boundary_rows[row['boundary_row_index']]
        if base['rank_excluded']:
            key=(row['case_id'],row['dependency_route'])
            if key not in saved_missing:raise ValueError('New exclusion contradicts saved diagnosis')
            saved=saved_missing[key]
            if base['rank']!=saved['rank'] or base['actual_float_margin']!=saved['cost_margin_to_third'] or row['window_rank']<=3:
                raise ValueError('Captured exclusion/rank/margin differs')
            focus.append({'case_id':row['case_id'],'dependency_route':row['dependency_route'],
                          'boundary_row_index':row['boundary_row_index'],'window_row_index':i})
    if len(boundary_rows)!=480 or len(window_rows)!=30 or len(focus)!=len(saved_missing):raise ValueError('Fixed finite comparison population differs')
    def sign_count(field,value):
        return sum((Fraction(boundary_rows[r['boundary_row_index']][field])>0)-(Fraction(boundary_rows[r['boundary_row_index']][field])<0)==value for r in focus)
    return {'schema':'sara-route-score-audit-result-v1','boundary_comparisons':boundary_rows,
        'window_comparisons':window_rows,'missing_dependency_focus':focus,
        'summary':{'captured_boundaries':160,'boundary_dependency_comparisons':480,
                  'window_dependency_comparisons':30,'missing_commit_dependencies':len(focus),
                  'missing_with_width_term_favoring_dependency':sign_count('width_gap',-1),
                  'missing_with_width_term_opposing_dependency':sign_count('width_gap',1),
                  'missing_with_zero_width_difference':sign_count('width_gap',0),
                  'missing_with_positive_cumulative_error_gap':sign_count('cumulative_error_gap',1)},
        'exact_all_identities_verified':True,'coefficient_binary64_exact':str(COEFFICIENT),
        'interpretation':'Cumulative and recent errors are not independent addends. Exact mixture/difference identities and explicit rounding preserve original float ranks. Comparators remain fixed for each temporal comparison; window-own cutoffs are reported separately. No new predictor, rule change or causal/counterfactual performance is evaluated.',
        'causal_attribution':None,'counterfactual_accuracy':None,'functional_substitution_proven':None,
        'learner_executions':0,'actual_new_snapshots':0,'new_predictions_scored':0,'new_measurements':0,
        'new_acquisitions':0,'conditions_relaxed':False,'test_opened':False,'component_passed':False,
        'old_failed_replay_reclassified':False,'old_discarded_trace_equality':None,
        'new_experiment_authorized':False,'promotion_authorized':False}
