"""Matched local learners for the terminal-feedback CartPole task.

This module deliberately has no environment runner. It cannot score a split.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from sara_engine.evaluation.event_unit_causal_isolation import UnitEvent
from sara_engine.neuro.neuron import Neuron


ARMS = ("A_scalar_local", "B_compact_event", "C_stateful_spiking")
ROUTES = 20
ACTIONS = 2
TRACE_DECAY = 0.99
LEARNING_RATE = 0.05
EXPLORATION = 0.10
REWARD_REFERENCE = 0.5
WEIGHT_BOUND = 2.0


class CartPoleLocalPolicy:
    """One terminal local-credit rule with three route-processing backends."""

    def __init__(self, arm: str, seed: int, *, spike_bypass: bool = False,
                 state_reset_each_step: bool = False):
        if arm not in ARMS or type(seed) is not int:
            raise ValueError("Invalid policy arm or seed")
        if (spike_bypass or state_reset_each_step) and arm != "C_stateful_spiking":
            raise ValueError("Spiking controls require the spiking arm")
        self.arm = arm
        self.spike_bypass = spike_bypass
        self.state_reset_each_step = state_reset_each_step
        self.rng = random.Random(seed)
        self.weights = {} if arm == "A_scalar_local" else [0.0] * (ROUTES * ACTIONS)
        self.trace = [0.0] * (ROUTES * ACTIONS)
        self.neurons: list[Neuron] = []
        self.episode_open = False
        self.next_time = 0
        self.event_work = 0
        self.spikes = 0
        self.terminal_updates = 0

    def begin_episode(self) -> None:
        if self.episode_open:
            raise ValueError("Previous episode is still open")
        self.trace = [0.0] * (ROUTES * ACTIONS)
        self.neurons = ([Neuron(route, num_branches=1) for route in range(ROUTES)]
                        if self.arm == "C_stateful_spiking" else [])
        self.next_time = 0
        self.episode_open = True

    def _weight(self, index: int) -> float:
        return self.weights.get(index, 0.0) if isinstance(self.weights, dict) else self.weights[index]

    def _features(self, events: Sequence[UnitEvent]) -> tuple[int, ...]:
        if (len(events) != 4 or any(type(e.route) is not int or not 0 <= e.route < ROUTES
                                   or e.time != self.next_time or e.branch != 0 for e in events)):
            raise ValueError("Expected four current-time route events")
        routes = tuple(e.route for e in events)
        if len(set(route // 5 for route in routes)) != 4:
            raise ValueError("Expected one route per observation channel")
        self.event_work += 4
        if self.arm != "C_stateful_spiking" or self.spike_bypass:
            return routes
        if self.state_reset_each_step:
            self.neurons = [Neuron(route, num_branches=1) for route in range(ROUTES)]
        for route in routes:
            self.neurons[route].add_input_to_branch(0, 1.6)
        active = tuple(route for route, neuron in enumerate(self.neurons) if neuron.step())
        self.event_work += ROUTES
        self.spikes += len(active)
        return active

    def choose(self, events: Sequence[UnitEvent]) -> int:
        if not self.episode_open or self.next_time >= 500:
            raise ValueError("No available episode decision")
        features = self._features(events)
        scores = [sum(self._weight(route * ACTIONS + action) for route in features)
                  for action in range(ACTIONS)]
        explore_draw = self.rng.random()
        action_draw = self.rng.random()
        if explore_draw < EXPLORATION:
            action = int(action_draw >= 0.5)
        elif scores[0] == scores[1]:
            action = int(action_draw >= 0.5)
        else:
            action = int(scores[1] > scores[0])
        for index, value in enumerate(self.trace):
            self.trace[index] = value * TRACE_DECAY
        for route in features:
            index = route * ACTIONS + action
            self.trace[index] = min(1.0, self.trace[index] + (1.0 - TRACE_DECAY))
        self.next_time += 1
        return action

    def finish_episode(self, reward: float, *, learn: bool) -> None:
        if not self.episode_open or self.next_time < 1 or type(learn) is not bool:
            raise ValueError("No completed episode to finish")
        reward = float(reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("Terminal reward must be bounded")
        if learn:
            delta = LEARNING_RATE * (reward - REWARD_REFERENCE)
            for index, eligibility in enumerate(self.trace):
                if eligibility > 0.0:
                    updated = self._weight(index) + delta * eligibility
                    updated = max(-WEIGHT_BOUND, min(WEIGHT_BOUND, updated))
                    if isinstance(self.weights, dict):
                        self.weights[index] = updated
                    else:
                        self.weights[index] = updated
                    self.terminal_updates += 1
                    self.event_work += 1
        self.episode_open = False
        self.neurons = []
        self.trace = [0.0] * (ROUTES * ACTIONS)

    def weight_snapshot(self) -> tuple[float, ...]:
        return tuple(self._weight(index) for index in range(ROUTES * ACTIONS))
