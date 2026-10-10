"""Pre-score isolation, representation and budget regression coverage."""
from copy import deepcopy
from pathlib import Path

import pytest

from sara_engine.evaluation.temporal_boundary_learning import (
    ARMS, audit_inputs, encode, load_protocol, paired_inputs,
)


def small_protocol():
    protocol = deepcopy(load_protocol())
    protocol['pairs_per_family'] = {'training': 2, 'evaluation': 2}
    return protocol


def test_frozen_protocol_and_reproducible_input_audit():
    protocol = small_protocol()
    report = audit_inputs(protocol)
    assert report == audit_inputs(protocol)
    assert report['passed'] and not report['learning_executed']
    assert all(row['pairs'] == 10 for row in report['families'].values())


def test_raw_histories_and_seeds_are_disjoint():
    protocol = small_protocol()
    histories = {}
    for split in ('training', 'evaluation'):
        histories[split] = {events for family in protocol['families']
                            for seed in protocol[split + '_seeds']
                            for _, pair in paired_inputs(protocol, split, seed, family)
                            for _, events in pair}
    assert not histories['training'] & histories['evaluation']
    protocol['evaluation_seeds'][0] = protocol['training_seeds'][0]
    with pytest.raises(ValueError, match='Seed overlap'):
        audit_inputs(protocol)


def test_source_route_renaming_does_not_change_features():
    events = ((2, 0), (5, 1), (5, 3), (2, 1))
    renamed = ((22, 0), (17, 1), (17, 3), (22, 1))
    for arm in ARMS:
        assert encode(events, arm)[0] == encode(renamed, arm)[0]


@pytest.mark.parametrize('events', [(), ((0, 1),), ((True, 0),), ((0, 0), (24, 1)),
                                    ((0, 0), (1, 0)), ((0, 0), (1, 256)),
                                    ((0, 0),) + ((1, 1),) * 95])
def test_invalid_and_oversize_inputs_fail_closed(events):
    with pytest.raises(ValueError):
        encode(events, 'joint')


def test_maximum_encoding_has_unique_bounded_channels():
    events = ((0, 0),) + tuple((index % 24, 255) for index in range(94))
    for arm in ARMS:
        features, costs = encode(events, arm)
        assert len(features) == len({channel for channel, _ in features})
        assert costs['active_channels'] <= 191
        assert costs['wire_bytes'] == 194
        assert costs['projection_visits'] <= 285
        assert max(channel for channel, _ in features) <= 26672


def test_wrong_information_expectation_and_pool_overlap_fail():
    protocol = small_protocol()
    protocol['expected_distinct']['middle_order'].append('histogram')
    with pytest.raises(ValueError, match='information boundary'):
        audit_inputs(protocol)
    protocol = small_protocol()
    protocol['route_pools']['evaluation'][0] = 0
    with pytest.raises(ValueError, match='Route pool overlap'):
        audit_inputs(protocol)


def test_budget_failure_is_not_reported_as_pass():
    protocol = small_protocol()
    protocol['budgets']['max_active_channels'] = 1
    with pytest.raises(ValueError, match='resource budget'):
        audit_inputs(protocol)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match='Unknown arm'):
        encode(((0, 0),), 'label')


@pytest.mark.parametrize('filename,message', [
    ('temporal_boundary_learning_v1.json', 'learning protocol changed'),
    ('temporal_capability_boundary_v1.json', 'baseline changed'),
])
def test_changed_frozen_artifact_rejected(monkeypatch, filename, message):
    original = Path.read_bytes

    def altered(path):
        raw = original(path)
        return raw + b' ' if path.name == filename else raw

    monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(ValueError, match=message):
        load_protocol()
