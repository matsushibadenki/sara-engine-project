"""Registration and decision tests do not score Gymnasium CartPole."""
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.evaluation.cartpole_local_policy import CartPoleLocalPolicy
from sara_engine.evaluation.cartpole_registered_runner import (
    VARIANTS, _reserve_one_shot, _run_episode, development_decision, load_frozen_protocol,
    run_registered_development,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/cartpole_terminal_prereg_v1.json"


class StubEnvironment:
    def __init__(self, reward):
        self.reward = reward
        self.actions = []
        self.seed = None

    def reset(self, *, seed):
        self.actions = []
        self.seed = seed
        return (0.0, 0.0, 0.0, 0.0), {}

    def step(self, action):
        self.actions.append(action)
        return (0.0, 0.0, 0.0, 0.0), self.reward, len(self.actions) == 3, False, {}


def _synthetic_record(steps, *, updates=0, action_digest="same"):
    return {
        "steps": steps, "input_events": steps * 4,
        "internal_event_work": steps * 4, "peak_policy_state_bytes": 1000,
        "agent_cpu_ns": 1000, "terminal_updates": updates,
        "actions_sha256": action_digest,
    }


def _synthetic_runs(*, c_steps=120):
    lengths = {
        "A_scalar_local": 80, "B_compact_event": 80,
        "C_stateful_spiking": c_steps, "C_spike_bypass": 80,
        "C_state_reset": 90, "A_no_learning": 80,
    }
    runs = []
    for run_id in range(5):
        variants = {}
        for variant in VARIANTS:
            episodes = [_synthetic_record(lengths[variant]) for _ in range(32)]
            variants[variant] = {
                "training": [_synthetic_record(lengths[variant]) for _ in range(128)],
                "development": episodes,
                "final_weights_sha256": "same" if variant in VARIANTS[:2] else variant,
            }
        runs.append({"run_id": run_id, "variants": variants})
    return runs


def test_protocol_is_canonical_source_pinned_and_fail_closed():
    digest = sha256(PROTOCOL.read_bytes()).hexdigest()
    assert load_frozen_protocol(digest)["heldout_opening_authorized"] is False
    with pytest.raises(ValueError, match="digest mismatch"):
        load_frozen_protocol("0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        run_registered_development(expected_protocol_sha256="0" * 64)


def test_episode_runner_uses_only_terminal_feedback_and_no_eval_updates():
    first = CartPoleLocalPolicy("B_compact_event", 11)
    second = CartPoleLocalPolicy("B_compact_event", 99)
    high_reward_env = StubEnvironment(999.0)
    low_reward_env = StubEnvironment(-999.0)
    first_result = _run_episode(high_reward_env, first, seed=210000, learn=False)
    second_result = _run_episode(low_reward_env, second, seed=210000, learn=False)
    assert high_reward_env.seed == low_reward_env.seed == 210000
    assert first_result["steps"] == second_result["steps"] == 3
    assert first_result["terminal_feedback"] == second_result["terminal_feedback"] == 3 / 500
    assert first_result["actions_sha256"] == second_result["actions_sha256"]
    assert first_result["terminal_updates"] == second_result["terminal_updates"] == 0
    assert first.weight_snapshot() == second.weight_snapshot()


def test_action_randomness_resets_from_each_episode_seed():
    policy = CartPoleLocalPolicy("A_scalar_local", 1)
    other = CartPoleLocalPolicy("A_scalar_local", 2)
    first = _run_episode(StubEnvironment(0), policy, seed=210007, learn=False)
    _run_episode(StubEnvironment(0), policy, seed=210008, learn=False)
    repeated = _run_episode(StubEnvironment(0), policy, seed=210007, learn=False)
    independent = _run_episode(StubEnvironment(0), other, seed=210007, learn=False)
    assert first["actions_sha256"] == repeated["actions_sha256"] == independent["actions_sha256"]


def test_development_gate_is_a_conjunction_and_never_opens_heldout():
    positive = development_decision(_synthetic_runs())
    assert positive["passed"] is True
    assert positive["heldout_opening_authorized"] is False
    negative = development_decision(_synthetic_runs(c_steps=80))
    assert negative["passed"] is False
    assert negative["checks"]["C_minus_B_at_least_20"] is False
    assert negative["checks"]["C_minus_B_positive_on_four_runs"] is False


def test_one_shot_reservation_survives_a_failed_or_duplicate_attempt(tmp_path):
    output = tmp_path / "development.json"
    _reserve_one_shot(output, "a" * 64)
    assert output.exists() is False
    assert (tmp_path / "development.json.lock").read_text() == "a" * 64 + "\n"
    with pytest.raises(FileExistsError):
        _reserve_one_shot(output, "a" * 64)
