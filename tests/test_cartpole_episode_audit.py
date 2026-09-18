"""Synthetic-only runner tests never instantiate Gymnasium CartPole."""
from types import SimpleNamespace

import pytest

from sara_engine.evaluation.cartpole_episode_audit import audit_synthetic_episode
from sara_engine.evaluation.cartpole_local_policy import CartPoleLocalPolicy


class SyntheticEpisode:
    spec = SimpleNamespace(id="SyntheticLifecycle-v1")

    def __init__(self, *, length=3, reward=999.0):
        self.length = length
        self.reward = reward
        self.actions = []
        self.reset_seed = None

    def reset(self, *, seed):
        self.reset_seed = seed
        self.actions = []
        return (0.0, 0.0, 0.0, 0.0), {}

    def step(self, action):
        self.actions.append(action)
        return ((0.0, 0.0, 0.0, 0.0), self.reward,
                len(self.actions) == self.length, False, {})


def test_synthetic_audit_discards_step_reward_and_applies_one_terminal_update():
    env = SyntheticEpisode(reward=999.0)
    policy = CartPoleLocalPolicy("B_compact_event", 21)
    audit = audit_synthetic_episode(env, policy, seed=110005, learn=True)
    assert env.reset_seed == 110005
    assert len(env.actions) == 3
    assert audit.steps == 3
    assert audit.terminal_feedback == 3 / 500
    assert audit.input_events == 12
    assert audit.internal_event_work == 12 + audit.terminal_updates
    assert audit.spikes == 0
    assert audit.terminal_updates > 0
    assert audit.peak_policy_state_bytes > 0
    assert audit.agent_cpu_ns > 0


def test_evaluation_mode_does_not_update_and_reward_never_leaks():
    first = CartPoleLocalPolicy("A_scalar_local", 7)
    second = CartPoleLocalPolicy("A_scalar_local", 7)
    audit_a = audit_synthetic_episode(SyntheticEpisode(reward=999.0), first,
                                      seed=210000, learn=False)
    audit_b = audit_synthetic_episode(SyntheticEpisode(reward=-999.0), second,
                                      seed=210000, learn=False)
    assert audit_a.steps == audit_b.steps == 3
    assert audit_a.terminal_updates == audit_b.terminal_updates == 0
    assert first.weight_snapshot() == second.weight_snapshot() == (0.0,) * 40


def test_real_cartpole_identity_fails_closed():
    env = SyntheticEpisode()
    env.spec = SimpleNamespace(id="CartPole-v1")
    policy = CartPoleLocalPolicy("C_stateful_spiking", 3)
    with pytest.raises(ValueError, match="Only synthetic"):
        audit_synthetic_episode(env, policy, seed=1, learn=True)
    assert env.reset_seed is None
    assert policy.episode_open is False
