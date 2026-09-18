"""Synthetic-only audit of the CartPole policy episode lifecycle.

Real CartPole scoring is deliberately refused until a complete experiment
preregistration and its evaluator are frozen separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import process_time_ns
from typing import Any

from sara_engine.evaluation.cartpole_local_policy import CartPoleLocalPolicy
from sara_engine.evaluation.cartpole_sparse_reward import (
    MAX_EPISODE_STEPS, encode_observation, terminal_feedback,
)
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size


@dataclass(frozen=True)
class EpisodeAudit:
    steps: int
    terminal_feedback: float
    input_events: int
    internal_event_work: int
    spikes: int
    terminal_updates: int
    peak_policy_state_bytes: int
    agent_cpu_ns: int


def audit_synthetic_episode(env: Any, policy: CartPoleLocalPolicy, *, seed: int,
                            learn: bool) -> EpisodeAudit:
    """Exercise an injected synthetic environment without exposing real scores."""
    if (getattr(getattr(env, "spec", None), "id", None) != "SyntheticLifecycle-v1"
            or type(seed) is not int or type(learn) is not bool
            or not isinstance(policy, CartPoleLocalPolicy)):
        raise ValueError("Only synthetic episode auditing is authorized")
    if policy.episode_open:
        raise ValueError("Policy already has an open episode")

    observation, _ = env.reset(seed=seed)
    before_work = policy.event_work
    before_spikes = policy.spikes
    before_updates = policy.terminal_updates
    start = process_time_ns()
    policy.begin_episode()
    peak_state = _deep_size(policy)
    cpu_ns = process_time_ns() - start
    for step_index in range(MAX_EPISODE_STEPS):
        start = process_time_ns()
        events = encode_observation(observation, step_index)
        action = policy.choose(events)
        peak_state = max(peak_state, _deep_size(policy))
        cpu_ns += process_time_ns() - start
        observation, _discarded_default_reward, terminated, truncated, _ = env.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("Synthetic environment terminal flags must be boolean")
        if terminated or truncated:
            start = process_time_ns()
            feedback = terminal_feedback(steps=step_index + 1,
                                         terminated=terminated, truncated=truncated)
            policy.finish_episode(feedback, learn=learn)
            cpu_ns += process_time_ns() - start
            return EpisodeAudit(
                steps=step_index + 1,
                terminal_feedback=feedback,
                input_events=(step_index + 1) * 4,
                internal_event_work=policy.event_work - before_work,
                spikes=policy.spikes - before_spikes,
                terminal_updates=policy.terminal_updates - before_updates,
                peak_policy_state_bytes=peak_state,
                agent_cpu_ns=cpu_ns,
            )
    raise ValueError("Synthetic environment exceeded the 500-step CartPole limit")
