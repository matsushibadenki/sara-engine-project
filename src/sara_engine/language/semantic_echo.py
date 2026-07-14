"""Finite multi-timescale sparse echo dynamics for temporal language tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .semantic_events import LanguageEvent


EchoKey = Tuple[str, str, str, str]
EchoPrefix = Tuple[str, str, str]


@dataclass(frozen=True)
class EchoDecision:
    feature: str
    axis: str
    score: float
    kind: str


@dataclass(frozen=True)
class EchoTrace:
    time: int
    active_echoes: int
    comparisons: int
    updates: int
    decisions: Tuple[EchoDecision, ...]
    abstained: bool


class SparseSemanticEchoField:
    """A bounded local resonance field with explicit decay and no dense state."""

    TAU = {"fast": 2.0, "medium": 6.0, "slow": 18.0}

    def __init__(
        self,
        *,
        tiers: Tuple[str, ...] = ("fast", "medium", "slow"),
        max_echoes: int = 24,
        max_comparisons: int = 32,
        threshold: float = 0.35,
        enable_role_binding: bool = True,
    ) -> None:
        if not tiers or any(tier not in self.TAU for tier in tiers):
            raise ValueError("tiers must contain known echo tiers")
        if max_echoes < 1 or max_comparisons < 1:
            raise ValueError("echo and comparison limits must be positive")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be within (0, 1]")
        self.tiers = tuple(tiers)
        self.max_echoes = int(max_echoes)
        self.max_comparisons = int(max_comparisons)
        self.threshold = float(threshold)
        self.enable_role_binding = bool(enable_role_binding)
        self.time = 0
        self._echoes: Dict[EchoKey, Tuple[float, int]] = {}
        self._recent: List[LanguageEvent] = []

    def reset(self) -> None:
        self.time = 0
        self._echoes.clear()
        self._recent.clear()

    def step(self, event: LanguageEvent, *, gap: int = 1) -> EchoTrace:
        if gap < 1:
            raise ValueError("gap must be at least one")
        self.time += int(gap)
        self._decay(gap)
        comparisons = 0
        decisions: List[EchoDecision] = []
        key_prefix = self._prefix(event)
        previous_values = [value[0] for key, value in self._echoes.items() if key[:3] == key_prefix]
        score = 0.0
        if previous_values:
            score = min(1.0, max(previous_values) + 0.5)
            comparisons += 1
            if score >= self.threshold:
                decisions.append(EchoDecision(event.feature, event.axis, round(score, 6), "reactivation"))
        for prior in reversed(self._recent):
            if comparisons >= self.max_comparisons:
                break
            comparisons += 1
            prior_key = self._prefix(prior)
            prior_active = self._has_active(prior_key)
            if event.role and prior.role == event.role and prior.feature != event.feature and prior_active and self.enable_role_binding:
                decisions.append(EchoDecision(f"{prior.feature}->{event.feature}", "binding", 1.0, "role_binding"))
            elif prior.feature == event.feature and prior.axis == event.axis and prior_active:
                decisions.append(EchoDecision(event.feature, event.axis, 1.0, "local_match"))
        for tier in self.tiers:
            self._echoes[(*key_prefix, tier)] = (1.0, self.time)
        self._recent.append(event)
        self._recent = self._recent[-self.max_comparisons :]
        self._trim()
        if event.role == "claim" and any(
            prior.role == event.role
            and prior.feature != event.feature
            and self._has_active(self._prefix(prior))
            for prior in self._recent
        ):
            decisions.append(EchoDecision(event.feature, event.axis, 1.0, "contradiction"))
        unique = tuple(dict.fromkeys(decisions))
        abstained = not bool(unique) or any(decision.kind == "contradiction" for decision in unique)
        return EchoTrace(self.time, len(self._echoes), comparisons, 1, unique, abstained)

    def run(self, events: Iterable[Tuple[int, LanguageEvent]]) -> Tuple[EchoTrace, ...]:
        return tuple(self.step(event, gap=gap) for gap, event in events)

    def state_dict(self) -> Dict[str, object]:
        """Return a deterministic, bounded snapshot of transient echo state."""
        echoes = [
            {
                "axis": key[0],
                "feature": key[1],
                "role": key[2],
                "tier": key[3],
                "strength": round(value[0], 6),
                "last_time": value[1],
            }
            for key, value in sorted(self._echoes.items())
        ]
        recent = [event.__dict__ for event in self._recent]
        return {"schema": "sara-semantic-echo-state-v1", "time": self.time, "echoes": echoes, "recent": recent}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore a validated transient state without admitting unbounded records."""
        if state.get("schema") != "sara-semantic-echo-state-v1":
            raise ValueError("unsupported semantic echo state schema")
        time = state.get("time")
        echoes = state.get("echoes")
        recent = state.get("recent", [])
        if not isinstance(time, int) or time < 0:
            raise ValueError("echo state time must be a non-negative integer")
        if not isinstance(echoes, list) or len(echoes) > self.max_echoes:
            raise ValueError("echo state exceeds the configured echo limit")
        if not isinstance(recent, list) or len(recent) > self.max_comparisons:
            raise ValueError("echo state exceeds the configured comparison history limit")
        restored: Dict[EchoKey, Tuple[float, int]] = {}
        for item in echoes:
            if not isinstance(item, dict):
                raise ValueError("echo entries must be objects")
            key = (str(item.get("axis", "")), str(item.get("feature", "")), str(item.get("role", "")), str(item.get("tier", "")))
            strength = item.get("strength")
            last_time = item.get("last_time")
            if key[3] not in self.tiers or not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 1.0 or not isinstance(last_time, int) or last_time > time:
                raise ValueError("invalid echo entry")
            restored[key] = (float(strength), last_time)
        restored_recent: List[LanguageEvent] = []
        for item in recent:
            if not isinstance(item, dict):
                raise ValueError("recent entries must be objects")
            restored_recent.append(LanguageEvent(**{key: item[key] for key in ("time", "axis", "feature", "source", "evidence_type", "confidence", "role")}))
        self.time = time
        self._echoes = restored
        self._recent = restored_recent

    def serialized_state_bytes(self) -> int:
        payload = json.dumps(self.state_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return len(payload.encode("utf-8"))

    def _decay(self, gap: int) -> None:
        updated: Dict[EchoKey, Tuple[float, int]] = {}
        for key, (strength, last_time) in self._echoes.items():
            tier = key[3]
            value = strength * math.exp(-float(gap) / self.TAU[tier])
            if value >= 0.08:
                updated[key] = (value, last_time)
        self._echoes = updated

    def _trim(self) -> None:
        if len(self._echoes) <= self.max_echoes:
            return
        ranked = sorted(self._echoes.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        self._echoes = dict(ranked[: self.max_echoes])

    @staticmethod
    def _prefix(event: LanguageEvent) -> EchoPrefix:
        return event.axis, event.feature, event.role

    def _has_active(self, prefix: EchoPrefix) -> bool:
        return any(key[:3] == prefix for key in self._echoes)
