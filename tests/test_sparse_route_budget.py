"""Development-seed checks of actual update equality and shared-route handling."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.sparse_factor_routing import ROUTES, Task, evaluate
from sara_engine.evaluation.sparse_route_budget import (
    ARMS, PROTOCOL_SHA256, UPDATES, WARMUP, BudgetRouteMemory, run_case, summarize, train_budget,
)
from sara_engine.utils.project_paths import processed_data_path


def protocol():
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_route_budget_v1.json")).read_bytes()
    assert sha256(raw).hexdigest() == PROTOCOL_SHA256
    return json.loads(raw)


def test_protocol_constants_and_fresh_seeds():
    config = protocol()
    assert config["arms"] == list(ARMS)
    assert config["warmup_steps"] == WARMUP == 180
    assert config["updates_per_tick"] == UPDATES == 9
    assert config["candidate_count"] == len(ROUTES) == 45
    assert config["training_steps"] * UPDATES == 4608
    assert not config["promotion_authorized"]
    old = json.loads(Path(processed_data_path("benchmark_fixtures", "sparse_factor_routing_v1.json")).read_bytes())
    assert not set(config["seeds"]) & set(old["seeds"])


@pytest.mark.parametrize("strategy", ARMS)
def test_exact_active_updates_and_frozen_inactive_routes(strategy):
    model = BudgetRouteMemory(strategy, 8101)
    rows = Task(8101, "interacting").samples("train", 192)
    for t, row in enumerate(rows):
        _, trace = model.begin(row.events, t)
        before = deepcopy((model.tables, model.errors, model.observations))
        active = trace["updated_routes"]
        assert len(set(active)) == 9
        assert all(sum(model.routes[i][1] == slot for i in active) == 3 for slot in range(3))
        model.observe(row.outcome, t + 1)
        for i in range(45):
            if i not in active:
                assert model.tables[i] == before[0][i]
                assert model.errors[i] == before[1][i]
                assert model.observations[i] == before[2][i]
            else:
                assert model.observations[i] == before[2][i] + 1
                assert sum(map(sum, model.tables[i].values())) == sum(map(sum, before[0][i].values())) + 1
                inputs, slot = model.routes[i]
                old_counts = before[0][i].get(tuple(row.events[j] for j in inputs))
                probability = 0.5 if old_counts is None else (old_counts[1] + 1) / (sum(old_counts) + 2)
                expected_error = before[1][i] + ((probability - row.outcome[slot]) ** 2 - before[1][i]) / 32
                assert model.errors[i] == pytest.approx(expected_error)
            assert len(model.tables[i]) <= 2 ** len(model.routes[i][0])
        if t == 179:
            assert model.observations == [36] * 45
            assert len(model.pool) == 45
        if t == 180:
            assert len(model.pool) == (45 if strategy == "round_robin" else 9)
    assert model.updates == 192 * 9


def test_identical_warmup_state_across_strategies():
    config = protocol()
    rows = Task(8102, "independent").samples("train", 180)
    models = [BudgetRouteMemory(arm, 8102) for arm in ARMS]
    logs = [train_budget(model, rows, config) for model in models]
    assert len({log["update_ledger_sha256"] for log in logs}) == 1
    assert all(model.tables == models[0].tables and model.errors == models[0].errors for model in models)


def test_rejected_updates_are_atomic_and_future_outcomes_hidden():
    model = BudgetRouteMemory("residual_prune", 8103)
    with pytest.raises(ValueError):
        model.observe((1, 0, 1), 1)
    assert model.begin((0, 1, 0, 1, 0), 0)[0] == (None, None, None)
    before = deepcopy(model.__dict__)
    for action in (lambda: model.begin((0, 1, 0, 1, 0), 1),
                   lambda: model.observe((1, 0, 1), 0),
                   lambda: model.observe((1, 0, 1), 2),
                   lambda: model.observe((True, 0, 1), 1),
                   lambda: model.begin(iter(range(1000000)), 1)):
        with pytest.raises(ValueError):
            action()
        assert model.__dict__ == before
    model.observe((1, 0, 1), 1)
    with pytest.raises(ValueError):
        model.observe((1, 0, 1), 1)


def test_residual_pruning_preserves_useful_shared_inputs_without_truth_hints():
    model = BudgetRouteMemory("residual_prune", 8104)
    model.steps = 180
    model.observations = [36] * 45
    model.errors = [1.0] * 45
    useful = (((0,), 0), ((1, 2), 1), ((2,), 2))
    for route in useful:
        model.errors[ROUTES.index(route)] = 0.0
    model.narrow()
    assert all(ROUTES.index(route) in model.pool for route in useful)
    assert [model.routes[i] for i in model.select()[0]] == list(useful)
    pool = model.pool
    model.errors = [1 - value for value in model.errors]
    model.narrow()
    assert model.pool == pool


def test_frozen_readout_ignores_retired_low_error_routes():
    model = BudgetRouteMemory("random_prune", 8105)
    model.steps = 180
    model.observations = [36] * 45
    model.narrow()
    for i in range(45):
        model.errors[i] = 1.0 if i in model.pool else 0.0
    assert set(model.select()[0]) <= set(model.pool)
    before = deepcopy(model.__dict__)
    evaluate(model, Task(8105, "independent").samples("novel_combination", 16))
    assert model.__dict__ == before


def test_development_case_ledger_replay_and_negative_gates():
    config = protocol()
    config["training_steps"] = 192
    config["evaluation_steps_per_split"] = 16
    result = run_case(8106, "interacting", config)
    assert all(result["integrity"].values())
    assert all(arm["training"]["outcome_route_updates"] == 1728 for arm in result["arms"].values())
    assert result["controls"]["shuffled_outcome_route_updates"] == 1728
    summary = summarize([result], config)
    assert summary["component_passed"] == (all(result["component_gates"].values()) and all(summary["aggregate_gates"].values()))
