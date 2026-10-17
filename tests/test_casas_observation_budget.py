"""Strict integer budgets, coherent cohort targets and no new model updates."""
from copy import deepcopy
import math
import pytest
from sara_engine.evaluation.casas_observation_budget import minimum_scale, scaled_counts, forecast, required_cohort
from sara_engine.evaluation.casas_evidence import evidence


def state(size=8, contexts=10):
    cells = {'global':[contexts*size,contexts*size]}
    for hour in range(contexts):
        current = repr(('current',(('Bedroom','ON'),2,hour,0)))
        cells[current] = [size,size]
        cells[repr(('pool',current,False,False))] = [size,0]
        cells[repr(('pool',current,False,True))] = [0,size]
    return {'arm':'pool_evidence','policy':'same_sensor_fast10_four_branches_v1',
            'counts':{'schema':1,'capacity':100000,'changed_writes':6*contexts*size,'cells':cells}}


def test_strict_inversion_matches_integer_scan_and_existing_rule():
    for n in range(1,9):
        for m in range(1,9):
            for x,y in ((0,m),(n,0),(n//2,m//2)):
                detail, current = [n-x,x],[n+m-x-y,x+y]
                delta = abs(x/n-y/m)
                k = minimum_scale(n,m,delta)
                if not delta:
                    assert k is None
                    continue
                expected = next(i for i in range(1,10000) if evidence(*scaled_counts(detail,current,i,False)))
                assert k == expected
    assert minimum_scale(8,8,1) == 2
    assert minimum_scale(9,9,1) == 1
    assert minimum_scale(8,0,1) is None
    assert minimum_scale(100,100,0) is None
    for args in ((0,1,.5),(1,-1,.5),(1,1,float('nan')),(1,1,2)):
        with pytest.raises(ValueError):minimum_scale(*args)


def test_full_cohort_prediction_is_read_only_with_matching_three_write_forecast():
    source = state()
    before = deepcopy(source)
    result = forecast(source)
    assert source == before
    actual = result['scenarios']['observed_rates']
    assert actual['minimum_global_scale_for_targets'] == 2
    assert actual['required_extra_observations'] == 160
    assert actual['required_extra_writes'] == 480
    assert actual['closed_world_budget_feasible'] and actual['conservative_state_budget_feasible']
    assert result['actual_learning_writes'] == result['actual_new_observations'] == 0
    assert result['existing_changed_writes'] == 480
    oracle = result['scenarios']['fixed_margin_oracle']
    assert oracle['branchwise_oracle_only']
    assert oracle['closed_world_budget_feasible'] is None
    assert oracle['opposing_contexts_within_cap'] is None


def test_observed_zero_remains_unforecastable_but_oracle_is_only_lower_bound():
    source = state(size=100)
    for key in source['counts']['cells']:
        if key.startswith("('pool'"):
            source['counts']['cells'][key] = [50,50]
    result = forecast(source)
    observed = result['scenarios']['observed_rates']
    assert observed['zero_contrast_branches'] == 20
    assert observed['minimum_global_scale_for_targets'] is None
    assert not observed['closed_world_budget_feasible']
    assert result['scenarios']['fixed_margin_oracle']['branches_within_cap'] == 20
    assert result['scenarios']['fixed_margin_oracle']['closed_world_budget_feasible'] is None


def test_target_requires_distinct_opposing_contexts_not_summed_branch_counts():
    records = [{'current_key':'same','minimum_scale':1,'detail_observations':10,
                'direction':direction,'direction_outcomes':10} for direction in (-1,1) for _ in range(10)]
    needed, contexts, _ = required_cohort(records,200)
    assert needed is None and len(contexts) == 1


def test_state_growth_bound_and_no_alternative_forecast():
    source = state(size=1,contexts=1)
    cells = source['counts']['cells']
    key = next(k for k in cells if k.startswith("('pool'"))
    cells[key] = [1,1]
    other = next(k for k in cells if k.startswith("('pool'") and k != key)
    del cells[other]
    result = forecast(source)
    assert result['scenarios']['observed_rates']['no_complement_branches'] == 1
    assert result['open_world_routes_at_cap'] == len(cells)+2*result['cap_extra_observations']
    assert result['cap_extra_writes'] == 3*result['cap_extra_observations']


def test_registered_forecast_preserves_inputs_and_seals():
    import gzip
    import json
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import project_path, processed_data_path
    def path(suffix):
        return processed_data_path('benchmark_fixtures','casas_observation_budget_'+suffix+'_v1.json')
    reg = json.loads(Path(path('registration')).read_text())
    result = json.loads(Path(path('result')).read_text())
    assert file_sha256(path('protocol')) == reg['protocol_sha256']
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    assert not any(result[k] for k in ('raw_source_opened','confirmation_opened',
                                       'original_development_opened','test_opened'))
    assert result['actual_learning_writes'] == result['actual_acquired_observations'] == 0
    assert file_sha256(path('branches')+'.gz') == result['branch_trace_sha256']
    trace = json.loads(gzip.decompress(Path(path('branches')+'.gz').read_bytes()))
    for arm,scale,extra,routes in (('pool_evidence',4,52659,114537),('joint_evidence',5,70212,155746)):
        report = result['arms'][arm]
        observed = report['scenarios']['observed_rates']
        assert report['cohort_scale_cap'] == 16
        assert report['cap_extra_writes'] == 789885
        assert report['state_safe_whole_cohort_scale'] == 3
        assert observed['minimum_global_scale_for_targets'] == scale
        assert observed['required_extra_observations'] == extra
        assert observed['required_extra_writes'] == 3*extra
        assert observed['open_world_routes_at_required_scale'] == routes > 100000
        assert observed['closed_world_budget_feasible']
        assert not observed['conservative_state_budget_feasible']
        assert report['scenarios']['fixed_margin_oracle']['closed_world_budget_feasible'] is None
        for scenario,records in trace[arm].items():
            summary = report['scenarios'][scenario]
            assert sum(r['minimum_scale'] is None for r in records) == summary['branches_without_finite_scale']
            assert sum(r['detail_observations'] for r in records if r['minimum_scale'] is not None and r['minimum_scale'] <= 16) == summary['fit_mass_within_cap']


def test_directional_outcome_support_is_retained_in_required_cohort():
    records = []
    for context in range(10):
        for direction in (-1,1):
            records.append({'current_key':str(context),'minimum_scale':1,'detail_observations':1,
                            'direction':direction,'direction_outcomes':1})
    needed,opposing,_ = required_cohort(records,20)
    assert needed == 4 and opposing == [4]*10
