"""Bounded event interface for CartPole-v1 with terminal-only feedback."""
from __future__ import annotations

import math
import platform
from typing import Sequence

from sara_engine.evaluation.event_unit_causal_isolation import UnitEvent


ENVIRONMENT_ID = "CartPole-v1"
EXPECTED_GYMNASIUM_VERSION = "1.0.0"
EXPECTED_PYTHON_VERSION = "3.10.18"
EXPECTED_NUMPY_VERSION = "1.25.2"
MAX_EPISODE_STEPS = 500
OBSERVATION_CUTPOINTS = (
    (-1.0, -0.25, 0.25, 1.0),
    (-1.0, -0.25, 0.25, 1.0),
    (-0.10, -0.025, 0.025, 0.10),
    (-1.0, -0.25, 0.25, 1.0),
)


def encode_observation(observation: Sequence[float], step_index: int) -> tuple[UnitEvent, ...]:
    """Encode four observed scalars as four sparse routes at one logical step."""
    if len(observation) != 4 or type(step_index) is not int or not 0 <= step_index < MAX_EPISODE_STEPS:
        raise ValueError("CartPole observation or step index is invalid")
    events = []
    for channel, (raw, cuts) in enumerate(zip(observation, OBSERVATION_CUTPOINTS)):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("CartPole observation must be finite")
        bucket = sum(value > cut for cut in cuts)
        events.append(UnitEvent(channel * 5 + bucket, step_index, 0))
    return tuple(events)


def terminal_feedback(*, steps: int, terminated: bool, truncated: bool) -> float:
    """Emit one bounded outcome only after episode termination or truncation."""
    if (type(steps) is not int or not 1 <= steps <= MAX_EPISODE_STEPS
            or type(terminated) is not bool or type(truncated) is not bool
            or not (terminated or truncated)):
        raise ValueError("Terminal feedback requires a completed CartPole episode")
    return steps / MAX_EPISODE_STEPS


def inspect_environment() -> dict:
    """Read API identity and one deterministic transition without model scoring."""
    import gymnasium as gym
    import numpy as np

    if (gym.__version__ != EXPECTED_GYMNASIUM_VERSION
            or platform.python_version() != EXPECTED_PYTHON_VERSION
            or np.__version__ != EXPECTED_NUMPY_VERSION):
        raise ValueError("CartPole runtime differs from the registered task")
    env = gym.make(ENVIRONMENT_ID)
    try:
        if (env.spec.id != ENVIRONMENT_ID or env.spec.max_episode_steps != MAX_EPISODE_STEPS
                or env.action_space.n != 2 or env.observation_space.shape != (4,)):
            raise ValueError("CartPole environment contract changed")
        initial, _ = env.reset(seed=110000)
        first_events = encode_observation(initial, 0)
        observed, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            raise ValueError("CartPole ended on the API inspection step")
        next_events = encode_observation(observed, 1)
        return {
            "environment_id": env.spec.id,
            "gymnasium_version": gym.__version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "max_episode_steps": env.spec.max_episode_steps,
            "observation_channels": len(first_events),
            "events_per_observation": len(next_events),
            "action_count": env.action_space.n,
            "evaluator_invoked": False,
        }
    finally:
        env.close()
