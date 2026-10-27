"""Saved-count identities and lossless evidence coverage, never a new learner."""
import ast
from copy import deepcopy
from fractions import Fraction
import inspect
import sys

import pytest

from sara_engine.evaluation import route_count_mixture as audit
from sara_engine.evaluation.reactivation_codec import loads
from sara_engine.evaluation.route_snapshot_contract import encode,CATALOG
from scripts.eval import route_count_mixture as runner


@pytest.fixture(scope='module')
def inputs():return runner.load_inputs(runner.read('selection'))


@pytest.fixture(scope='module')
def result(inputs):return audit.evaluate(runner.read('selection'),inputs[0],inputs[1])


def test_complete_population_and_lossless_codec(result,inputs):
    assert len(result['snapshot_cell_references'])==160
    assert sum(map(len,result['snapshot_cell_references']))==24000
    assert len(result['window_key_rows'])==1500 and len(result['route_window_summaries'])==450
    assert len(result['missing_dependency_focus_keys'])==8
    assert loads(encode(result))==result
    assert len(encode(result))<2097152
    recipes=result['lookup_dictionary'];snapshots=inputs[0][audit.INPUTS[0]]['snapshots']
    for index,refs in enumerate(result['snapshot_cell_references']):
        for position,recipe_id in enumerate(refs):
            route,key=result['cell_layout'][position]
            assert recipes[recipe_id][:2]==list(audit.cell(snapshots[index]['rows'][route],key))


def test_every_positive_lookup_uses_original_single_prior_and_exact_rounding(result):
    for recipe in result['lookup_dictionary']:
        n0,n1,exact,value,residual=recipe;n=n0+n1
        if not n:
            assert exact is value is residual is None
        else:
            assert Fraction(exact)==Fraction(n1+1,n+2)
            assert value==(n1+1)/(n+2)
            assert Fraction(exact)+Fraction(residual)==Fraction.from_float(value)
            assert Fraction(n,n+2)*Fraction(n1,n)+Fraction(2,n+2)*Fraction(1,2)==Fraction(exact)


def test_all_window_mixtures_exact_with_unknown_zero_weight_rates(result):
    recipes=result['lookup_dictionary']
    for values in result['window_key_rows']:
        d=dict(zip(result['window_key_columns'],values))
        old=recipes[d['start_recipe']][:2];end=recipes[d['end_recipe']][:2];recent=d['recent_counts']
        assert [a+b for a,b in zip(old,recent)]==end
        aw,bw,pw=(Fraction(d[k]) for k in ('old_weight','recent_weight','prior_weight'))
        assert aw+bw+pw==1
        ac,bc,pc=(Fraction(d[k]) for k in ('old_positive_contribution','recent_positive_contribution','prior_positive_contribution'))
        assert ac+bc+pc==Fraction(d['conditional_formula_if_allocated'])
        if sum(old):assert Fraction(d['old_empirical_rate'])==Fraction(old[1],sum(old))
        else:assert d['old_empirical_rate'] is None and d['endpoint_lookup_change'] is None
        if sum(recent):
            assert Fraction(d['old_deviation_from_recent'])+Fraction(d['prior_deviation_from_recent'])==Fraction(d['end_minus_recent_empirical'])
        else:assert d['recent_empirical_rate'] is d['end_minus_recent_empirical'] is None


def test_zero_support_does_not_become_a_point_five_prediction():
    assert audit.lookup([0,0])==[0,0,None,None,None]
    result=audit.mixture([0,0],[0,0])
    assert result[8]=='1/2'  # Conditional formula only, not known key presence.
    assert result[0] is result[1] is result[9] is result[10] is None
    assert result[-2]=='absent'


def test_no_recent_observations_does_not_fabricate_recent_labels():
    result=audit.mixture([3,2],[0,0])
    assert result[1] is None and result[10] is result[11] is result[12] is None
    assert result[9]=='0' and result[-2]=='absent'


def test_old_and_recent_smoothed_probabilities_are_not_naively_added(result):
    recipes=result['lookup_dictionary'];found=False
    for values in result['window_key_rows']:
        d=dict(zip(result['window_key_columns'],values));old=recipes[d['start_recipe']][:2];recent=d['recent_counts']
        a,b=sum(old),sum(recent)
        if a and b:
            naive=Fraction(a,a+b)*Fraction(old[1]+1,a+2)+Fraction(b,a+b)*Fraction(recent[1]+1,b+2)
            if naive!=Fraction(d['conditional_formula_if_allocated']):found=True;break
    assert found


@pytest.mark.parametrize('bad',[[True,0],[-1,1],[1.0,0],[1024,1],[0],[None,1]])
def test_invalid_count_fields_rejected(bad):
    with pytest.raises(ValueError):audit.lookup(bad)
    with pytest.raises(ValueError):audit.mixture([0,0],bad)


def test_count_and_error_weight_shares_are_separate_not_equated(result):
    for row in result['route_window_summaries']:
        assert row['recent_observations']==36
        assert row['old_observations']+36==row['end_observations']
        assert Fraction(row['old_route_count_share'])==Fraction(row['old_observations'],row['end_observations'])
    assert any(Fraction(r['old_route_count_share'])!=Fraction(r['old_route_error_weight_share']) for r in result['route_window_summaries'])


def test_old_opposite_labels_are_counts_not_a_causal_accuracy_claim(result):
    for focus in result['missing_dependency_focus_keys']:
        d=dict(zip(result['window_key_columns'],result['window_key_rows'][focus['window_key_row_index']]))
        recent=d['recent_counts'];old=result['lookup_dictionary'][d['start_recipe']][:2]
        if d['recent_label_state']=='only0':assert d['old_opposite_recent_label_count']==old[1]
        if d['recent_label_state']=='only1':assert d['old_opposite_recent_label_count']==old[0]
    assert result['causal_attribution'] is result['counterfactual_accuracy'] is result['per_key_prequential_loss_attribution'] is None
    assert result['learner_executions']==result['model_lookup_queries']==result['new_predictions_scored']==0
    assert not result['component_passed'] and not result['test_opened'] and not result['conditions_relaxed']


@pytest.mark.parametrize('change',['hash','proof','negative','old_unknown','scope'])
def test_altered_evidence_or_conditions_fail_closed(inputs,change):
    reports,hashes,_=inputs;reports,hashes=deepcopy(reports),dict(hashes)
    selection=deepcopy(runner.read('selection'))
    if change=='hash':hashes[audit.INPUTS[0]]='0'*64
    elif change=='proof':reports[audit.INPUTS[2]]['numerical_noninterference_passed']=False
    elif change=='negative':reports[audit.INPUTS[2]]['component_passed']=True
    elif change=='old_unknown':reports[audit.INPUTS[2]]['old_discarded_trace_equality']=True
    else:selection['inputs'].append('unregistered.json')
    with pytest.raises(ValueError):audit.evaluate(selection,reports,hashes)


def test_budgets_fixed_and_inputs_not_mutated(inputs):
    selection=deepcopy(runner.read('selection'));selection['budgets']['result_bytes']+=1
    with pytest.raises(ValueError):runner.load_inputs(selection)
    before=encode(inputs[0]);audit.evaluate(runner.read('selection'),inputs[0],inputs[1])
    assert encode(inputs[0])==before


def test_no_learner_query_or_evaluator_regeneration_dependencies():
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    tree=ast.parse(inspect.getsource(audit))
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports=={'fractions','hashlib','sara_engine.evaluation.route_score_audit','sara_engine.evaluation.route_snapshot_contract'}
    assert not hasattr(audit,'Observer') and not hasattr(audit,'evaluator_truth')


def test_registered_and_consumed_pure_result():
    if runner.path('registration').exists():runner.verify()
    if runner.path('result').exists():
        assert encode(runner.read('result'))==encode(runner.compute())
        with pytest.raises(FileExistsError):runner.main('execute')
