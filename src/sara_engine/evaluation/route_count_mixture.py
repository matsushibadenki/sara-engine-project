"""Finite scalar count/lookup algebra on captured boundaries only."""
from fractions import Fraction
from hashlib import sha256

from sara_engine.evaluation.route_score_audit import INPUTS,verify_inputs
from sara_engine.evaluation.route_snapshot_contract import CATALOG,encode,window_support

PROTOCOL_SHA256='77b5c86f5e6c5fc18e87b6c10dce4d26238fb03f6df5a286b2cf867eccac47ce'
LOOKUP_COLUMNS=['n0','n1','positive_support_lookup_exact','positive_support_lookup_float','float_rounding_residual']
KEY_COLUMNS=['route_window_index','key_index','start_recipe','end_recipe','recent_counts',
    'old_empirical_rate','recent_empirical_rate','old_weight','recent_weight','prior_weight',
    'old_positive_contribution','recent_positive_contribution','prior_positive_contribution',
    'conditional_formula_if_allocated','endpoint_lookup_change','end_minus_recent_empirical',
    'old_deviation_from_recent','prior_deviation_from_recent','recent_label_state','old_opposite_recent_label_count']
LAYOUT=[(i,k) for i,r in enumerate(CATALOG) for k in range(2**len(r['inputs']))]


def counts(value):
    if type(value) not in (tuple,list) or len(value)!=2 or any(type(n) is not int or not 0<=n<=1024 for n in value) or sum(value)>1024:
        raise ValueError('Original bounded integer class counts required')
    return tuple(value)


def lookup(value):
    """A conditional table lookup, never an emitted or scored prediction.

    Zero support does not encode key allocation, so actual lookup is unknown.
    """
    n0,n1=counts(value);n=n0+n1
    if not n:return [n0,n1,None,None,None]
    exact=Fraction(n1+1,n+2)
    empirical=Fraction(n1,n);sample_weight=Fraction(n,n+2);prior_weight=Fraction(2,n+2)
    if sample_weight+prior_weight!=1 or sample_weight*empirical+prior_weight*Fraction(1,2)!=exact:
        raise ValueError('Original smoothing identity failed')
    original_float=(n1+1)/(n+2)
    residual=Fraction.from_float(original_float)-exact
    if exact+residual!=Fraction.from_float(original_float):raise ValueError('Lookup rounding identity failed')
    return [n0,n1,str(exact),original_float,str(residual)]


def mixture(old,recent):
    a0,a1=counts(old);b0,b1=counts(recent)
    end=counts((a0+b0,a1+b1));a=a0+a1;b=b0+b1;n=a+b;den=n+2
    old_rate=Fraction(a1,a) if a else None;recent_rate=Fraction(b1,b) if b else None
    aw,bw,pw=Fraction(a,den),Fraction(b,den),Fraction(2,den)
    ac,bc,pc=Fraction(a1,den),Fraction(b1,den),Fraction(1,den)
    conditional=Fraction(end[1]+1,den)
    if aw+bw+pw!=1 or ac+bc+pc!=conditional:raise ValueError('Exact old/recent/prior identity failed')
    change=conditional-Fraction(a1+1,a+2) if a and n else None
    gap=conditional-recent_rate if b else None
    old_deviation=aw*(old_rate-recent_rate) if a and b else (Fraction(0) if b else None)
    prior_deviation=pw*(Fraction(1,2)-recent_rate) if b else None
    if b and old_deviation+prior_deviation!=gap:raise ValueError('Exact recent empirical deviation identity failed')
    label='absent' if not b else ('only0' if b1==0 else ('only1' if b0==0 else 'mixed'))
    opposite=a1 if label=='only0' else (a0 if label=='only1' else None)
    def text(v):return str(v) if v is not None else None
    return [text(old_rate),text(recent_rate),str(aw),str(bw),str(pw),str(ac),str(bc),str(pc),
            str(conditional),text(change),text(gap),text(old_deviation),text(prior_deviation),label,opposite]


def cell(row,key):return counts(row[3][2*key:2*key+2])


def evaluate(selection,reports,hashes):
    collection,diagnostics=verify_inputs(selection,reports,hashes)
    snapshots=collection['snapshots'];recipes=[];recipe_ids={};references=[];snapshot_summary=[];ledger=sha256()
    def intern(value):
        key=counts(value)
        if key not in recipe_ids:
            recipe_ids[key]=len(recipes);recipes.append(lookup(key))
        return recipe_ids[key]
    for snapshot_index,snapshot in enumerate(snapshots):
        refs=[];supports=[]
        for route,key in LAYOUT:
            values=cell(snapshot['rows'][route],key);rid=intern(values);refs.append(rid)
            supports.append(sum(values))
            ledger.update(encode([snapshot_index,route,key,rid,recipes[rid]])+b'\n')
        references.append(refs)
        positive=[n for n in supports if n]
        snapshot_summary.append({'snapshot_index':snapshot_index,'key_cells':150,
            'positive_support_keys':len(positive),'zero_support_presence_unknown_keys':150-len(positive),
            'minimum_positive_support':min(positive) if positive else None,'maximum_support':max(supports),
            'counts_reconcile':all(sum(sum(cell(row,k)) for k in range(2**len(CATALOG[i]['inputs'])))==row[0] for i,row in enumerate(snapshot['rows']))})
    index={(s['point']['case_id'],s['point']['arm'],s['point']['tick'],s['point']['stage']):i for i,s in enumerate(snapshots)}
    truth=diagnostics['evaluator_truth'];route_rows=[];key_rows=[];focus=[]
    saved_missing={(d['case_id'],d['route_id']) for d in diagnostics['missing_at_commit']}
    windows=0
    for end_index,post in enumerate(snapshots):
        point=post['point']
        if point['stage']!='post_commit':continue
        identity,arm,tick=point['case_id'],point['arm'],point['tick']
        start_index=index[(identity,arm,tick-180,'pre_start')]
        pre_index=index[(identity,arm,tick,'pre_commit')]
        start,end=snapshots[start_index],snapshots[pre_index]
        window=window_support(start,end);windows+=1
        for route,(old,current,delta) in enumerate(zip(start['rows'],end['rows'],window['route_window_rows'])):
            route_row={'case_id':identity,'arm':arm,'start':tick-180,'commit':tick,'route_id':route,
                'old_observations':old[0],'recent_observations':delta[0],'end_observations':current[0],
                'old_route_count_share':str(Fraction(old[0],current[0])),
                'old_route_error_weight_share':str(Fraction.from_float(old[2])/Fraction.from_float(current[2])),
                'true_dependency':route in truth[identity]['final'],
                'retained_at_commit':route in post['pool']}
            route_index=len(route_rows);route_rows.append(route_row)
            for key in range(2**len(CATALOG[route]['inputs'])):
                before=cell(old,key);after=cell(current,key)
                recent=counts((after[0]-before[0],after[1]-before[1]))
                if recent!=cell(delta,key):raise ValueError('Window count subtraction differs')
                parts=mixture(before,recent)
                old_recipe,end_recipe=intern(before),intern(after)
                if sum(after)>0 and recipes[end_recipe][2]!=parts[8]:raise ValueError('Endpoint lookup disagrees with mixture')
                values=[route_index,key,old_recipe,end_recipe,list(recent),*parts]
                if len(values)!=len(KEY_COLUMNS):raise ValueError('Declared key-column layout changed')
                row_index=len(key_rows);key_rows.append(values)
                if (identity,route) in saved_missing:focus.append({'case_id':identity,'route_id':route,'key_index':key,
                    'route_window_index':route_index,'window_key_row_index':row_index})
    if len(references)!=160 or sum(map(len,references))!=24000 or windows!=10 or len(route_rows)!=450 or len(key_rows)!=1500 or len(focus)!=8:
        raise ValueError('Fixed count-audit population differs')
    focus_dicts=[dict(zip(KEY_COLUMNS,key_rows[f['window_key_row_index']])) for f in focus]
    return {'schema':'sara-route-count-mixture-result-v1',
        'lookup_dictionary_columns':LOOKUP_COLUMNS,'lookup_dictionary':recipes,
        'cell_layout':[list(v) for v in LAYOUT],'snapshot_cell_references':references,
        'snapshot_summaries':snapshot_summary,'cell_ledger_sha256':ledger.hexdigest(),
        'route_window_summaries':route_rows,'window_key_columns':KEY_COLUMNS,'window_key_rows':key_rows,
        'missing_dependency_focus_keys':focus,
        'summary':{'boundaries':160,'key_cells_audited':24000,'unique_count_recipes':len(recipes),
            'zero_support_presence_unknown_cells':sum(s['zero_support_presence_unknown_keys'] for s in snapshot_summary),
            'exploration_windows':10,'route_windows':450,'window_key_mixtures':1500,'focus_key_mixtures':8,
            'focus_keys_with_unanimous_recent_labels':sum(d['recent_label_state'] in ('only0','only1') for d in focus_dicts),
            'focus_keys_with_old_opposite_labels':sum((d['old_opposite_recent_label_count'] or 0)>0 for d in focus_dicts)},
        'exact_lookup_and_mixture_identities_verified':True,
        'interpretation':'Conditional lookup algebra at captured endpoints, not queried predictions. Zero support leaves key presence/actual lookup unknown. Window empirical rates are observed count summaries, not a recent-only/reset learner. Route count shares, key smoothing weights and weighted-error shares are distinct. Within-window order, pre-update probabilities and key-level losses are unrecorded.',
        'within_window_probability_history':None,'per_key_prequential_loss_attribution':None,
        'causal_attribution':None,'counterfactual_accuracy':None,'functional_substitution_proven':None,
        'learner_executions':0,'model_lookup_queries':0,'new_predictions_scored':0,'actual_new_snapshots':0,
        'new_measurements':0,'new_acquisitions':0,'component_passed':False,'conditions_relaxed':False,'test_opened':False,
        'old_failed_replay_reclassified':False,'old_discarded_trace_equality':None,'new_experiment_authorized':False,'promotion_authorized':False}
