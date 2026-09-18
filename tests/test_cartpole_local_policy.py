"""Policy-contract tests use synthetic events, never CartPole outcomes."""
import pytest

from sara_engine.evaluation.cartpole_local_policy import ARMS, CartPoleLocalPolicy
from sara_engine.evaluation.event_unit_causal_isolation import UnitEvent


def events(time: int, bins=(2, 2, 2, 2)):
    return tuple(UnitEvent(channel * 5 + bucket, time, 0)
                 for channel, bucket in enumerate(bins))


def test_scalar_and_compact_have_identical_decisions_and_local_updates():
    scalar = CartPoleLocalPolicy("A_scalar_local", 17)
    compact = CartPoleLocalPolicy("B_compact_event", 17)
    for reward in (0.8, 0.2, 1.0):
        for policy in (scalar, compact):
            policy.begin_episode()
        for time in range(12):
            sample = events(time, (time % 5, 2, 3, 1))
            assert scalar.choose(sample) == compact.choose(sample)
        for policy in (scalar, compact):
            policy.finish_episode(reward, learn=True)
        assert scalar.weight_snapshot() == compact.weight_snapshot()
    assert scalar.terminal_updates == compact.terminal_updates
    assert scalar.event_work == compact.event_work


def test_development_feedback_never_mutates_weights():
    for arm in ARMS:
        policy = CartPoleLocalPolicy(arm, 3)
        before = policy.weight_snapshot()
        policy.begin_episode()
        policy.choose(events(0))
        policy.finish_episode(1.0, learn=False)
        assert policy.weight_snapshot() == before
        assert policy.terminal_updates == 0


def test_spike_arm_and_controls_have_bounded_work_and_distinct_activity():
    spiking = CartPoleLocalPolicy("C_stateful_spiking", 7)
    bypass = CartPoleLocalPolicy("C_stateful_spiking", 7, spike_bypass=True)
    reset = CartPoleLocalPolicy("C_stateful_spiking", 7, state_reset_each_step=True)
    for policy in (spiking, bypass, reset):
        policy.begin_episode()
        for time in range(6):
            assert policy.choose(events(time)) in (0, 1)
        policy.finish_episode(1.0, learn=True)
        assert len(policy.weight_snapshot()) == 40
        ceiling = 6 * (44 if policy.state_reset_each_step else
                       24 if not policy.spike_bypass else 4) + 40
        if not policy.state_reset_each_step and not policy.spike_bypass:
            ceiling += 20
        assert policy.event_work <= ceiling
    assert bypass.spikes == 0
    assert spiking.spikes < reset.spikes
    assert spiking.weight_snapshot() != bypass.weight_snapshot()


def test_episode_lifecycle_rejects_invalid_calls():
    policy = CartPoleLocalPolicy("A_scalar_local", 1)
    with pytest.raises(ValueError):
        policy.choose(events(0))
    policy.begin_episode()
    with pytest.raises(ValueError):
        policy.begin_episode()
    with pytest.raises(ValueError):
        policy.choose(events(1))
    with pytest.raises(ValueError):
        policy.choose(events(0, (2, 2, 2, 5)))
    with pytest.raises(ValueError):
        policy.finish_episode(0.5, learn=True)
    policy.choose(events(0))
    with pytest.raises(ValueError):
        policy.finish_episode(float("nan"), learn=True)
    policy.finish_episode(0.5, learn=True)
    with pytest.raises(ValueError):
        policy.finish_episode(0.5, learn=True)
