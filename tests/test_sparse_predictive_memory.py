"""Causal, capacity and composition controls for the isolated memory study."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.sparse_predictive_memory import (
    ARMS, PROTOCOL_SHA256, PredictiveMemory, evaluate, run_seed, stream,
)
from sara_engine.utils.project_paths import processed_data_path


def protocol():
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_predictive_memory_v1.json")).read_bytes()
    assert sha256(raw).hexdigest() == PROTOCOL_SHA256
    return json.loads(raw)


@pytest.mark.parametrize("arm", ARMS)
def test_causal_receipt_and_rejected_updates_are_atomic(arm):
    model = PredictiveMemory(arm)
    with pytest.raises(ValueError):
        model.observe((3, 4, 5), 1)
    assert model.begin((0, 1, 2), 0) == (None, None, None)
    before = deepcopy(model.__dict__)
    for action in (lambda: model.begin((0, 1, 2), 1),
                   lambda: model.observe((3, 4, 5), 0),
                   lambda: model.observe((3, 4, 5), 2),
                   lambda: model.observe((True, 4, 5), 1),
                   lambda: model.observe((3, 4), 1)):
        with pytest.raises(ValueError):
            action()
        assert model.__dict__ == before
    assert model.predict((0, 1, 2)) == (None, None, None)
    model.observe((3, 4, 5), 1)
    assert model.predict((0, 1, 2)) == (3, 4, 5)
    with pytest.raises(ValueError):
        model.observe((6, 7, 8), 1)
    with pytest.raises(ValueError):
        model.begin((0, 1, 2), 0)


@pytest.mark.parametrize("arm", ARMS)
def test_budget_and_fifo_under_sustained_novelty(arm):
    model = PredictiveMemory(arm, 24)
    for t in range(200):
        model.begin((t, t, t), t)
        model.observe((t + 1, t + 1, t + 1), t + 1)
        assert model.retained_scalars() <= 24
        assert all(len(table) <= model.capacity for table in model.tables)
    assert model.writes == 600 and model.evictions > 0
    assert model.predict((0, 0, 0)) == (None, None, None)
    assert model.predict((199, 199, 199)) == (200, 200, 200)


def test_composition_has_independent_scalar_reference_and_missing_factor_abstention():
    rows = stream(735, (5, 7, 11), 430)
    model = PredictiveMemory("provided_factors")
    reference = [{}, {}, {}]
    for t in range(44):
        assert model.begin(rows[t], t) == tuple(reference[i].get(rows[t][i]) for i in range(3))
        model.observe(rows[t + 1], t + 1)
        for i in range(3):
            reference[i][rows[t][i]] = rows[t + 1][i]
    for row in rows[44:]:
        assert model.predict(row) == tuple(reference[i].get(row[i]) for i in range(3))
    model.tables[1].clear()
    predicted = model.predict(rows[44])
    assert predicted[1] is None
    assert predicted[0] == rows[45][0] and predicted[2] == rows[45][2]
    with pytest.raises(ValueError):
        model.predict(iter(range(1000000)))


def test_frozen_continuation_is_disjoint_and_replays():
    config = protocol()
    rows = stream(config["seeds"][0], config["periods"], 430)
    assert len(set(rows[:385])) == 385
    assert not set(rows[:44]) & set(rows[44:385])
    model = PredictiveMemory("joint")
    for t in range(44):
        model.begin(rows[t], t)
        model.observe(rows[t + 1], t + 1)
    tables = deepcopy(model.tables)
    result = evaluate(model, rows[44:], set(rows[:44]))
    assert result["novel_contexts"] == 341
    assert result["novel_exact_accuracy"] == 0
    assert evaluate(model, rows[44:], set(rows[:44])) == result
    assert model.tables == tables and model.writes == 132


def test_registered_component_gates_and_claim_boundaries():
    config = protocol()
    assert config["promotion_authorized"] is False
    result = run_seed(config["seeds"][0], config)
    assert all(result["gates"].values())
    assert result["arms"]["provided_factors"]["exact_accuracy"] == 1
    assert result["controls"]["reset_middle_factor"]["component_accuracy"] == pytest.approx(2 / 3)
    assert result["controls"]["copy_current_state_exact_accuracy"] == 0
