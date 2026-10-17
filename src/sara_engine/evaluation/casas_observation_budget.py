"""Conditional observation budgets from frozen counts, without new learning."""
from collections import defaultdict
import math
from sara_engine.evaluation.casas_count_audit import audit
from sara_engine.evaluation.casas_evidence import evidence

ADDITIONAL_OBSERVATIONS = 263295
ADDITIONAL_WRITES = 789885
STATE_CAPACITY = 100000
SCENARIOS = ('observed_rates', 'fixed_margin_oracle')


def minimum_scale(n, m, contrast):
    """Invert the unchanged strict bound using integer uniform cohort multipliers."""
    if type(n) is not int or type(m) is not int or n <= 0 or m < 0:
        raise ValueError('Positive detail and nonnegative complement sizes required')
    if not math.isfinite(contrast) or not 0 <= contrast <= 1:
        raise ValueError('Contrast must be finite in [0,1]')
    if not m or not contrast:
        return None
    def passes(k):
        return contrast > math.sqrt(math.log(80)/(2*k*n))+math.sqrt(math.log(80)/(2*k*m))
    upper = 1
    while not passes(upper):
        upper *= 2
        if upper > 2**128:
            raise ValueError('Forecast exceeds bounded numerical search')
    lower = 1
    while lower < upper:
        middle = (lower+upper)//2
        if passes(middle):
            upper = middle
        else:
            lower = middle+1
    return lower


def scaled_counts(detail, current, k, oracle):
    """Counterfactual arithmetic only; never store a fitted model or deliver feedback."""
    if type(k) is not int or k < 1:
        raise ValueError('Positive integer scale required')
    total = [v*k for v in current]
    branch = [v*k for v in detail]
    if oracle:
        n, m, positives = sum(branch), sum(total)-sum(branch), total[1]
        if m:
            endpoints = (max(0,positives-m), min(n,positives))
            x = max(endpoints, key=lambda p:abs(p/n-(positives-p)/m))
            branch = [n-x,x]
    return branch,total


def required_cohort(records, observations):
    """Uniform scaling of the whole cohort; no branch-targeted acquisition plan."""
    contexts = defaultdict(lambda: {1:[], -1:[]})
    eligible = []
    for item in records:
        k = item['minimum_scale']
        if k is None:
            continue
        eligible.append((k,item['detail_observations']))
        if item['direction'] and item['direction_outcomes']:
            directional = max(k,(4+item['direction_outcomes']-1)//item['direction_outcomes'])
            contexts[item['current_key']][item['direction']].append(directional)
    opposing = sorted(max(min(v[1]),min(v[-1])) for v in contexts.values() if v[1] and v[-1])
    context_scale = opposing[9] if len(opposing) >= 10 else None
    covered, coverage_scale = 0,None
    for k,count in sorted(eligible):
        covered += count
        if 100*covered >= observations:
            coverage_scale = k
            break
    needed = max(context_scale,coverage_scale) if context_scale is not None and coverage_scale is not None else None
    return needed, opposing, eligible


def forecast(state):
    validated = audit(state)
    cells = state['counts']['cells']
    observations = validated['fit_observations']
    routes = len(cells)
    if observations <= 0 or state['counts']['capacity'] != STATE_CAPACITY or routes > STATE_CAPACITY:
        raise ValueError('Unsupported cohort or state ceiling')
    cap = min(1+ADDITIONAL_OBSERVATIONS//observations, 1+ADDITIONAL_WRITES//(3*observations))
    # At most one new current and one new detail address per additional outcome.
    robust_scale = 1+(STATE_CAPACITY-routes)//(2*observations)
    report = {'arm':state['arm'],'fit_observations':observations,
              'existing_changed_writes':validated['existing_changed_writes'],
              'current_counts_sha256':validated['current_counts_sha256'],
              'existing_routes':routes,'actual_new_observations':0,'actual_learning_writes':0,
              'cohort_scale_cap':cap,'state_safe_whole_cohort_scale':robust_scale,
              'open_world_additional_observation_ceiling':(STATE_CAPACITY-routes)//2,
              'cap_extra_observations':(cap-1)*observations,
              'cap_extra_writes':3*(cap-1)*observations,
              'open_world_routes_at_cap':routes+2*(cap-1)*observations,
              'scenarios':{}}
    for scenario in SCENARIOS:
        records = []
        oracle = scenario == 'fixed_margin_oracle'
        for row in validated['branches']:
            detail, current = cells[row['detail_key']], cells[row['current_key']]
            contrast = row['margin_maximum_separation'] if oracle else row['observed_separation']
            k = minimum_scale(row['detail_observations'],row['complement_observations'],contrast or 0)
            branch,total = scaled_counts(detail,current,1,oracle)
            m = sum(total)-sum(branch)
            direction = (1 if branch[1]/sum(branch) > (total[1]-branch[1])/m else -1) if contrast and m else 0
            if k is not None:
                a,b = scaled_counts(detail,current,k,oracle)
                if not evidence(a,b):
                    raise ValueError('Inverted budget fails unchanged evidence rule')
                if k > 1:
                    a,b = scaled_counts(detail,current,k-1,oracle)
                    if evidence(a,b):
                        raise ValueError('Inverted budget is not minimal')
            records.append({'current_key':row['current_key'],'detail_key':row['detail_key'],
                            'detail_observations':sum(detail),'minimum_scale':k,
                            'nonfinite_reason':('no_complement' if not m else 'zero_contrast') if k is None else None,
                            'direction':direction,'direction_outcomes':branch[1 if direction > 0 else 0] if direction else 0})
        if oracle:
            # Per-branch endpoint assignments need not coexist in one table.
            needed, opposing = None, []
            eligible = [(r['minimum_scale'],r['detail_observations']) for r in records if r['minimum_scale'] is not None]
        else:
            needed, opposing, eligible = required_cohort(records,observations)
        mass = sum(count for k,count in eligible if k <= cap)
        robust_mass = sum(count for k,count in eligible if k <= min(cap,robust_scale))
        required_extra = (needed-1)*observations if needed is not None else None
        report['scenarios'][scenario] = {
            'branches_without_finite_scale':sum(r['minimum_scale'] is None for r in records),
            'no_complement_branches':sum(r['nonfinite_reason'] == 'no_complement' for r in records),
            'zero_contrast_branches':sum(r['nonfinite_reason'] == 'zero_contrast' for r in records),
            'branches_within_cap':sum(k <= cap for k,_ in eligible),
            'fit_mass_within_cap':mass,'fit_mass_fraction_within_cap':mass/observations,
            'opposing_contexts_within_cap':None if oracle else sum(k <= cap for k in opposing),
            'fit_mass_within_state_safe_cap':robust_mass,
            'opposing_contexts_within_state_safe_cap':None if oracle else sum(k <= min(cap,robust_scale) for k in opposing),
            'minimum_global_scale_for_targets':needed,'required_extra_observations':required_extra,
            'required_extra_writes':3*required_extra if required_extra is not None else None,
            'closed_world_routes_required':routes,
            'open_world_routes_at_required_scale':routes+2*required_extra if required_extra is not None else None,
            'branchwise_oracle_only':oracle,
            'closed_world_budget_feasible':None if oracle else needed is not None and needed <= cap,
            'conservative_state_budget_feasible':None if oracle else needed is not None and needed <= min(cap,robust_scale),
            'branches':records}
    return report
