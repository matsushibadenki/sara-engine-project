"""Disjoint evidence, exact fallback, delayed local updates and replay."""
from dataclasses import replace
from datetime import datetime, timedelta
import pytest
from sara_engine.evaluation.casas_evidence import EvidenceCounts, evidence, phase, ARMS, REFERENCES
from sara_engine.evaluation.casas_longitudinal import Example


def example(i=0, label=0):
    t = datetime(2011, 1, 1) + timedelta(seconds=i)
    return Example(t, t+timedelta(seconds=.5), (('Bedroom','ON'),)*8, (2,)*7, label)


def test_disjoint_evidence_requires_opposing_supported_rates():
    assert not evidence((0,0), (0,0))
    assert not evidence((100,0), (100,0))
    assert not evidence((50,50), (100,100))
    assert not evidence((1,0), (1,1))
    assert evidence((100,0), (100,100))
    assert evidence((0,100), (100,100))
    with pytest.raises(ValueError):
        evidence((2,0), (1,0))


def test_fallback_matches_current_and_actual_updates_match():
    a, b = EvidenceCounts('current'), EvidenceCounts('evidence')
    rows = [example(i, i%2) for i in range(20)]
    assert phase(a, rows, True) == phase(b, rows, True)
    assert a.changed_writes == b.changed_writes == 60
    assert b.fallbacks == 20


def test_opposing_histories_activate_using_only_past_feedback():
    model = EvidenceCounts('evidence')
    rows = [replace(example(i, i%2), history=(('Kitchen','ON'),)*7+(('Bedroom','ON'),))
            if i%2 else example(i) for i in range(400)]
    phase(model, rows, True)
    a = model.predict_example(example(401))
    model.deliver(a, a.due, lambda:0, training=False)
    b = model.predict_example(replace(example(402), history=rows[1].history))
    assert a.probability < .5 < b.probability
    assert model.overrides > 0 and model.changed_writes == 1200
    with pytest.raises(ValueError):
        model.deliver(b, b.cutoff, lambda:pytest.fail('Early outcome accessed'), training=True)
    model.deliver(b, b.due, lambda:1, training=False)
    state = model.snapshot()
    restored = EvidenceCounts.from_snapshot(state)
    assert phase(model, [example(403)], False) == phase(restored, [example(403)], False)
    assert model.snapshot() == state


def test_full_controls_and_unknown_policy():
    assert set(REFERENCES) == {'current','current_fine','histogram','transition','joint'}
    assert len(ARMS) == 11
    state = EvidenceCounts('evidence').snapshot()
    state['policy'] = 'retuned'
    with pytest.raises(ValueError):
        EvidenceCounts.from_snapshot(state)


def test_due_order_flushes_before_next_prediction_and_freezes_counts():
    rows = [replace(example(i), due=example(i).cutoff+timedelta(seconds=3))
            for i in range(10)]
    model = EvidenceCounts('evidence')
    phase(model, rows, True)
    assert model.changed_writes == 30 and model.peak_pending == 3
    before = model.snapshot()
    phase(model, [example(20), example(21)], False)
    assert model.snapshot() == before and not model.pending


def test_internal_confirmation_keeps_original_periods_date_only():
    from sara_engine.evaluation.casas_180 import extract_lines
    rows, audit = extract_lines(['2012-01-01 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE'],
        {'train_start':'2011-01-01', 'development_start':'2011-05-01',
         'test_start':'2011-07-01', 'clock_regression_days':[]})
    assert audit['sealed_date_only_rows'] == 1
    assert not rows['train'] and not rows['development']


def test_preregistration_preserves_prior_code_and_negative_evidence():
    import json
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import processed_data_path, project_path
    reg = json.loads(Path(processed_data_path('benchmark_fixtures',
                       'casas_evidence_registration_v1.json')).read_text())
    for name, digest in {**reg['code'], **reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    protocol = processed_data_path('benchmark_fixtures','casas_evidence_protocol_v1.json')
    assert file_sha256(protocol) == reg['protocol_sha256']


def test_registered_hh104_result_is_terminal_negative_with_all_seals():
    import json
    from pathlib import Path
    from sara_engine.evaluation.casas_180_source import file_sha256
    from sara_engine.utils.project_paths import processed_data_path, model_path
    result = json.loads(Path(processed_data_path('benchmark_fixtures',
                          'casas_evidence_result_v1.json')).read_text())
    assert result['qualification']['passed']
    assert result['qualification']['contrast_groups'] == 101
    assert result['audit']['sealed_date_only_rows'] == 3845589
    comparison = result['comparison']
    assert set(comparison['arms']) == set(ARMS)
    assert {a['changed_writes'] for a in comparison['arms'].values()} == {82062}
    assert all(a['confirmation_writes'] == 0 for a in comparison['arms'].values())
    assert comparison['arms']['evidence']['overrides'] == 0
    assert comparison['arms']['evidence']['fallbacks'] == 15552
    assert comparison['traces']['evidence'] == comparison['traces']['current']
    assert comparison['traces']['evidence'] == comparison['traces']['scalar_evidence']
    assert not comparison['passed']
    assert not result['original_development_opened'] and not result['test_opened']
    for arm, digest in result['model_sha256'].items():
        assert file_sha256(model_path('casas_evidence', arm+'_v1.json.gz')) == digest
