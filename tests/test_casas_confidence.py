"""Support gating, exact fallback, local writes and old-period seals."""
from datetime import datetime,timedelta
from dataclasses import replace
from pathlib import Path
import json
import pytest
from sara_engine.evaluation.casas_confidence import ConfidenceCounts,phase
from sara_engine.evaluation.casas_longitudinal import Example
from sara_engine.evaluation.casas_180 import extract_lines
from sara_engine.evaluation.casas_180_source import file_sha256
from sara_engine.utils.project_paths import processed_data_path,project_path

T=datetime(2011,6,15,10)
H=(('Bedroom','ON'),)*8


def example(i=0):return Example(T+timedelta(seconds=i),T+timedelta(seconds=i+.5),H,(2,)*7,i%2)


def test_exact_current_fallback_and_support_transition():
    current=ConfidenceCounts('current');candidate=ConfidenceCounts('confidence')
    for i in range(9):
        e=example(i);a=current.predict_example(e);b=candidate.predict_example(e)
        assert a.probability==b.probability
        for m,r in ((current,a),(candidate,b)):m.deliver(r,r.due,lambda:e.label,training=True)
    assert candidate.fallbacks==8 and candidate.overrides==1
    assert candidate.changed_writes==current.changed_writes==27


def test_receipt_replacement_preserves_identity_binding_and_due_checks():
    m=ConfidenceCounts('confidence');e=example();r=m.predict_example(e)
    assert m.pending[r.identity] is r
    with pytest.raises(ValueError):m.deliver(r,e.cutoff,lambda:pytest.fail('Early outcome read'),training=True)
    m.deliver(r,r.due,lambda:1,training=True)
    assert m.changed_writes==3


def test_support8_uses_local_past_counts_for_distinct_histories():
    model=ConfidenceCounts('confidence')
    for i in range(16):
        e=example(i)
        if i%2:e=replace(e,history=(('KitchenA','ON'),)*7+H[-1:])
        r=model.predict_example(e);model.deliver(r,r.due,lambda:i%2,training=True)
    e=example(20);a=model.predict_example(e);model.deliver(a,a.due,lambda:0,training=False)
    e=replace(example(21),history=(('KitchenA','ON'),)*7+H[-1:])
    b=model.predict_example(e)
    assert a.probability<.5<b.probability and model.overrides==2
    assert model.changed_writes==48


def test_original_development_fields_are_sealed_in_internal_split():
    meta=json.loads(Path(processed_data_path('benchmark_fixtures','casas_confidence_protocol_v1.json')).read_text())['splits']
    rows,audit=extract_lines(['2011-12-13 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE',
                              '2012-02-11 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE'],meta)
    assert audit['sealed_date_only_rows']==2 and not rows['development']


def test_registration_and_previous_negatives_remain_intact():
    for name in ('casas_confidence_registration_v1.json','casas_confidence_references_registration_v1.json'):
        reg=json.loads(Path(processed_data_path('benchmark_fixtures',name)).read_text())
        for p,h in {**reg['code'],**reg['preserved']}.items():assert file_sha256(project_path(p))==h


def test_frozen_result_actual_writes_and_scalar_equivalence():
    r=json.loads(Path(processed_data_path('benchmark_fixtures','casas_confidence_result_v1.json')).read_text())
    c=r['comparison']
    assert {a['changed_writes'] for a in c['arms'].values()}=={27972}
    assert all(a['confirmation_writes']==0 for a in c['arms'].values())
    assert c['traces']['confidence']==c['traces']['scalar_confidence']
    assert c['arms']['confidence']['overrides']+c['arms']['confidence']['fallbacks']==9352
    assert not c['passed'] and not r['original_development_opened'] and not r['test_opened']

def test_complete_sufficient_reference_set_and_no_relaxed_gate():
    supplemental=json.loads(Path(processed_data_path('benchmark_fixtures','casas_confidence_references_result_v1.json')).read_text())
    original=json.loads(Path(processed_data_path('benchmark_fixtures','casas_confidence_result_v1.json')).read_text())['comparison']
    assert set(supplemental['references'])=={'current_fine','histogram'}
    assert {r['changed_writes'] for r in supplemental['references'].values()}=={27972}
    refs={**original['arms'],**supplemental['references']}
    names=('current','current_fine','histogram','transition','joint')
    candidate=refs['confidence'];gates=supplemental['complete_reference_gates']
    assert gates['brier_gain']==(min(refs[a]['daily_brier'] for a in names)-candidate['daily_brier']>=.01)
    assert gates['balanced_gain']==(candidate['balanced_accuracy']-max(refs[a]['balanced_accuracy'] for a in names)>=.02)
    assert not supplemental['passed'] and supplemental['candidate_scores_unchanged']
