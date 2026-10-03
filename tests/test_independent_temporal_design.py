"""Identifiability is checked before learning, not inferred from fit scores."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pytest
from sara_engine.evaluation.independent_temporal_design import (
    independent_signature, synthetic_pairs, audit_design)


def test_pairs_control_histogram_and_suffix_without_reference_labels():
    for pair in synthetic_pairs():
        assert pair.mode_zero[:2] == (0, 1)
        assert pair.mode_one[:2] == (1, 0)
        assert pair.mode_zero[2:] == pair.mode_one[2:]
        assert Counter(pair.mode_zero) == Counter(pair.mode_one)
        assert sorted(set(pair.mode_zero), key=pair.mode_zero.index)[:2] == [0, 1]
        last_zero = {route: index for index, route in enumerate(pair.mode_zero)}
        last_one = {route: index for index, route in enumerate(pair.mode_one)}
        assert last_zero == last_one


def test_independent_controls_never_call_reference(monkeypatch):
    from sara_engine.evaluation import electricity_learner
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent control invoked reference')
    monkeypatch.setattr(electricity_learner, 'unit_activity', forbidden)
    pair = synthetic_pairs()[0]
    for control in ('leaky_analog', 'first_visit', 'histogram'):
        a = independent_signature(pair.mode_zero, control)
        b = independent_signature(pair.mode_one, control)
        assert (a == b) == (control == 'histogram')
        assert all(0 < strength <= 1 for _, strength in a)


def test_unsupported_lif_target_is_rejected_before_scoring():
    result = audit_design()
    assert result['control_integrity_passed']
    assert result['opposite_target_signature_collisions']['retained_lif'] == result['pairs']
    assert not result['lif_learning_comparison_ready']
    assert not result['candidate_scored']
    assert not result['real_data_read']


def test_reset_and_bounded_input():
    pair = synthetic_pairs()[0]
    first = independent_signature(pair.mode_zero, 'leaky_analog')
    independent_signature(pair.mode_one, 'leaky_analog')
    assert independent_signature(pair.mode_zero, 'leaky_analog') == first
    with pytest.raises(ValueError):
        independent_signature((0 for _ in range(1000000)), 'first_visit')
    with pytest.raises(ValueError):
        independent_signature((True,) * 95, 'histogram')


def test_frozen_design_does_not_authorize_learning():
    path = Path(__file__).resolve().parents[1] / 'data/processed/benchmark_fixtures/independent_temporal_design_v1.json'
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == '59b13be6f44a539d9d9661cd4457a58daf518e08ca0b2673881ab815fd7ec270'
    protocol = json.loads(raw)
    assert not protocol['candidate_scoring_authorized']
    assert not protocol['real_data_access_authorized']
