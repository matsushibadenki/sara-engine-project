"""Development-only checks for causal bounded routing and evaluator integrity."""
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.sparse_factor_routing import (
    ARMS, PROTOCOL_SHA256, ROUTES, LocalRouteMemory, Task, evaluate,
    reference_predict, run_case, synthesize, train,
)
from sara_engine.utils.project_paths import processed_data_path


def protocol():
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_factor_routing_v1.json")).read_bytes()
    assert sha256(raw).hexdigest() == PROTOCOL_SHA256
    return json.loads(raw)


def test_protocol_and_candidate_bounds():
    config = protocol()
    assert len(ROUTES) == config["candidate_routes"] == 45
    assert len(set(ROUTES)) == 45
    assert max(len(key) for key, _ in ROUTES) == config["max_key_width"] == 2
    assert config["arms"] == list(ARMS)
    assert not config["promotion_authorized"]
    assert not set(config["seeds"]) & {seed + 100000 + i for seed in config["seeds"] for i in range(4)}


@pytest.mark.parametrize("arm", ARMS)
def test_causal_receipts_and_atomic_invalid_inputs(arm):
    task = Task(7001, "interacting")
    model = LocalRouteMemory(arm, task.provided_routes() if arm == "provided_routes" else None)
    with pytest.raises(ValueError):
        model.observe((1, 0, 1), 1)
    prediction, _ = model.begin((0, 1, 0, 1, 0), 0)
    assert prediction == (None, None, None)
    before = deepcopy(model.__dict__)
    for action in (lambda: model.begin((0, 1, 0, 1, 0), 1),
                   lambda: model.observe((1, 0, 1), 0),
                   lambda: model.observe((1, 0, 1), 2),
                   lambda: model.observe((True, 0, 1), 1),
                   lambda: model.observe((1, 0), 1),
                   lambda: model.predict(iter(range(1000000))),
                   lambda: model.predict((2, 0, 0, 0, 0))):
        with pytest.raises(ValueError):
            action()
        assert model.__dict__ == before
    model.observe((1, 0, 1), 1)
    assert model.updates == len(model.routes) <= 45
    with pytest.raises(ValueError):
        model.observe((1, 0, 1), 1)
    with pytest.raises(ValueError):
        model.begin((0, 1, 0, 1, 0), 0)


def test_learned_arms_reject_oracle_hints_and_share_training_evidence():
    config = protocol()
    task = Task(7002, "interacting")
    rows = task.samples("train", 64)
    learners = []
    for arm in ARMS[1:4]:
        with pytest.raises(ValueError):
            LocalRouteMemory(arm, task.provided_routes())
        model = LocalRouteMemory(arm)
        metrics = train(model, rows, config)
        assert metrics["outcome_updates"] == 64 * 45
        learners.append(model)
    for model in learners[1:]:
        assert model.tables == learners[0].tables
        assert model.errors == learners[0].errors
        assert model.observations == learners[0].observations


@pytest.mark.parametrize("arm", ARMS)
def test_exhaustive_binary_keys_bound_every_table(arm):
    task = Task(7006, "independent")
    model = LocalRouteMemory(arm, task.provided_routes() if arm == "provided_routes" else None)
    events = list(product((0, 1), repeat=5))
    for t in range(128):
        current = events[t % len(events)]
        model.begin(current, t)
        model.observe(current[:3], t + 1)
        assert all(len(table) <= 2 ** len(route[0]) for route, table in zip(model.routes, model.tables))
    assert all(len(table) == 2 ** len(route[0]) for route, table in zip(model.routes, model.tables))
    assert model.updates == 128 * len(model.routes)


def test_generator_split_and_paired_interventions():
    task = Task(7003, "interacting")
    training = task.samples("train", 512)
    for row in training:
        assert tuple(row.events[task.inputs.index(i)] for i in range(3)) != (1, 1, 1)
    novel = task.samples("novel_combination", 64)
    assert not {r.events for r in training} & {r.events for r in novel}
    assert len({r.events for r in novel}) == 4
    for row in task.samples("nuisance_shift", 64):
        raw = [row.events[task.inputs.index(i)] for i in range(5)]
        assert raw[3:] == [1 - raw[0], 1 - raw[1]]
    pairs = task.samples("intervention_pairs", 64)
    for a, b in zip(pairs[::2], pairs[1::2]):
        changed = [i for i in range(5) if a.events[i] != b.events[i]]
        assert len(changed) == 1 and task.inputs[changed[0]] < 3
        assert a.outcome != b.outcome
    with pytest.raises(ValueError):
        task.samples("intervention_pairs", 3)


def test_input_inhibition_is_separable_from_output_coverage():
    models = [LocalRouteMemory(arm) for arm in ARMS[1:4]]
    preferred = (((0,), 0), ((0, 1), 1), ((0, 2), 2))
    independent = (((1,), 1), ((2,), 2))
    for model in models:
        model.observations = [16] * 45
        model.errors = [1.0] * 45
        for route in preferred:
            model.errors[ROUTES.index(route)] = 0.0
        for route in independent:
            model.errors[ROUTES.index(route)] = 0.01
    output_selected = [ROUTES[i] for i in models[1].select()[0]]
    full_selected = [ROUTES[i] for i in models[2].select()[0]]
    assert output_selected == list(preferred)
    assert full_selected == [preferred[0], *independent]
    assert len({slot for _, slot in output_selected}) == len({slot for _, slot in full_selected}) == 3


def test_missing_conflicting_and_permuted_packets():
    assert synthesize(((0, 0.9), (2, 0.1))) == (0.9, None, 0.1)
    assert synthesize(((0, 0.9), (0, 0.1), (1, 0.2))) == (None, 0.2, None)
    packets = ((0, 0.91), (0, 0.83), (0, 0.77))
    assert synthesize(packets) == synthesize(tuple(reversed(packets)))
    with pytest.raises(ValueError):
        synthesize(packets + ((1, 0.5),))
    with pytest.raises(ValueError):
        synthesize(((0, float("nan")),))


def test_local_residual_and_scalar_reference():
    model = LocalRouteMemory("learned_full_inhibition")
    current, target = (0, 1, 0, 1, 0), (1, 0, 1)
    index = ROUTES.index(((0,), 0))
    model.begin(current, 0)
    model.observe(target, 1)
    assert model.errors[index] == 0.25
    assert model.tables[index][(0,)] == [0, 1]
    model.begin(current, 1)
    model.observe(target, 2)
    assert model.errors[index] == pytest.approx(0.25 + ((2 / 3 - 1) ** 2 - 0.25) / 32)
    selected = (index, ROUTES.index(((1,), 1)), ROUTES.index(((2,), 2)))
    assert model.predict(current, selected)[0] == (0.75, 0.25, 0.75)
    for events in product((0, 1), repeat=5):
        assert model.predict(events, selected)[0] == reference_predict(model, events, selected)


def test_frozen_routing_and_reset_preserve_other_predictions():
    config = protocol()
    task = Task(7004, "interacting")
    model = LocalRouteMemory("provided_routes", task.provided_routes())
    metrics = train(model, task.samples("train", 512), config)
    assert metrics["peak_logical_slots"] <= config["max_logical_scalar_slots"]
    assert all(len(table) <= 4 for table in model.tables)
    rows = task.samples("novel_combination", 64)
    before = deepcopy(model.__dict__)
    result = evaluate(model, rows)
    assert result["exact_accuracy"] == 1
    assert evaluate(model, rows) == result and model.__dict__ == before
    model.tables[1].clear()
    reset = evaluate(model, rows, (0, 1, 2))
    assert reset["exact_accuracy"] == 0
    assert reset["component_accuracy"] == pytest.approx(2 / 3)


def test_development_case_preserves_negative_scientific_gates():
    config = protocol()
    config["training_steps"] = 64
    config["evaluation_steps_per_split"] = 16
    result = run_case(7005, "interacting", config)
    assert all(result["integrity"].values())
    assert result["specialization_passed"] == all(result["specialization_gates"].values())
    assert result["controls"]["untrained"]["component_coverage"] == 0
