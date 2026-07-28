"""Bounded sparse repetition-dependent memory consolidation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Set


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class RepetitionConsolidationConfig:
    """Configuration for sparse local repetition-dependent consolidation."""

    capacity: int = 256
    max_events: int = 100_000
    max_sources_per_memory: int = 8
    max_memory_id_bytes: int = 256
    max_source_ref_bytes: int = 512
    learning_rate: float = 0.35
    consolidation_rate: float = 0.22
    verification_rate: float = 0.30
    contradiction_rate: float = 0.40
    spacing_target: int = 5
    massed_gain_floor: float = 0.25
    successful_recall_multiplier: float = 1.30
    base_half_life: float = 32.0
    stability_half_life_gain: float = 7.0
    prune_threshold: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            "capacity",
            "max_events",
            "max_sources_per_memory",
            "max_memory_id_bytes",
            "max_source_ref_bytes",
            "spacing_target",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "consolidation_rate",
            "verification_rate",
            "contradiction_rate",
            "massed_gain_floor",
            "prune_threshold",
        ):
            value = _require_finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        recall_multiplier = _require_finite(
            "successful_recall_multiplier",
            self.successful_recall_multiplier,
        )
        if recall_multiplier < 1.0:
            raise ValueError(
                "successful_recall_multiplier must be at least 1.0"
            )
        half_life = _require_finite("base_half_life", self.base_half_life)
        if half_life <= 0.0:
            raise ValueError("base_half_life must be greater than 0.0")
        half_life_gain = _require_finite(
            "stability_half_life_gain",
            self.stability_half_life_gain,
        )
        if half_life_gain < 0.0:
            raise ValueError(
                "stability_half_life_gain must be non-negative"
            )


@dataclass
class _MemoryTrace:
    memory_id: str
    retrieval_strength: float = 0.0
    stability: float = 0.0
    verification_strength: float = 0.0
    repetitions: int = 0
    support_events: int = 0
    contradiction_events: int = 0
    successful_recalls: int = 0
    last_timestep: int = 0
    source_last_seen: Dict[str, int] = field(default_factory=dict)
    verified_sources: Set[str] = field(default_factory=set)


class RepetitionDependentConsolidator:
    """Sparse local plasticity with spacing, forgetting, and source isolation."""

    def __init__(
        self,
        config: Optional[RepetitionConsolidationConfig] = None,
    ) -> None:
        self.config = config or RepetitionConsolidationConfig()
        self._traces: Dict[str, _MemoryTrace] = {}
        self._clock = 0
        self._event_count = 0
        self._eviction_count = 0
        self._rejected_event_count = 0

    @property
    def clock(self) -> int:
        return self._clock

    def advance(self, timestep: int) -> Dict[str, Any]:
        """Advance the deterministic logical clock without global state writes."""
        target = self._validate_timestep(timestep)
        if target < self._clock:
            raise ValueError("timestep must be monotonic")
        self._clock = target
        return {
            "clock": self._clock,
            "memory_units": len(self._traces),
            "state_budget_ok": self.state_budget_ok(),
            "global_memory_rewrite": False,
        }

    def observe(
        self,
        *,
        memory_id: str,
        timestep: int,
        source_ref: str = "",
        outcome: str = "support",
        recall_success: bool = False,
        verified: bool = False,
    ) -> Dict[str, Any]:
        """Apply one bounded local support or contradiction event."""
        normalized_memory_id = self._validate_text(
            "memory_id",
            memory_id,
            self.config.max_memory_id_bytes,
            allow_empty=False,
        )
        normalized_source_ref = self._validate_text(
            "source_ref",
            source_ref,
            self.config.max_source_ref_bytes,
            allow_empty=True,
        )
        event_timestep = self._validate_timestep(timestep)
        if event_timestep < self._clock:
            raise ValueError("timestep must be monotonic")
        if outcome not in {"support", "contradiction"}:
            raise ValueError("outcome must be support or contradiction")
        if outcome == "contradiction" and recall_success:
            raise ValueError(
                "contradiction cannot be marked as a successful recall"
            )
        if verified and not normalized_source_ref:
            raise ValueError("verified evidence requires source_ref")
        if self._event_count >= self.config.max_events:
            self._rejected_event_count += 1
            return {
                "mutation_allowed": False,
                "reason": "event_budget_exhausted",
                "clock": self._clock,
                "event_count": self._event_count,
                "state_budget_ok": self.state_budget_ok(),
                "bptt_used": False,
                "dense_matrix_used": False,
                "gpu_used": False,
            }

        self._clock = event_timestep
        trace = self._traces.get(normalized_memory_id)
        created = trace is None
        if trace is None:
            trace = _MemoryTrace(
                memory_id=normalized_memory_id,
                last_timestep=event_timestep,
            )
            self._traces[normalized_memory_id] = trace
        before = self._project_trace(trace, event_timestep)
        trace.retrieval_strength = before["retrieval_strength"]
        trace.stability = before["stability"]
        gap = (
            self.config.spacing_target
            if created
            else event_timestep - trace.last_timestep
        )
        spacing_multiplier = (
            1.0
            if created
            else self.config.massed_gain_floor
            + (1.0 - self.config.massed_gain_floor)
            * min(1.0, gap / self.config.spacing_target)
        )

        new_verified_source = False
        source_budget_exhausted = False
        source_digest = (
            hashlib.sha256(normalized_source_ref.encode("utf-8")).hexdigest()
            if normalized_source_ref
            else ""
        )
        if source_digest:
            known_source = source_digest in trace.source_last_seen
            if (
                not known_source
                and len(trace.source_last_seen)
                >= self.config.max_sources_per_memory
            ):
                source_budget_exhausted = True
            else:
                trace.source_last_seen[source_digest] = event_timestep
                new_verified_source = bool(
                    verified and source_digest not in trace.verified_sources
                )
                if new_verified_source:
                    trace.verified_sources.add(source_digest)

        if outcome == "support":
            recall_multiplier = (
                self.config.successful_recall_multiplier
                if recall_success
                else 1.0
            )
            retrieval_gain = min(
                1.0,
                self.config.learning_rate
                * spacing_multiplier
                * recall_multiplier,
            )
            stability_gain = min(
                1.0,
                self.config.consolidation_rate
                * spacing_multiplier
                * recall_multiplier,
            )
            trace.retrieval_strength += retrieval_gain * (
                1.0 - trace.retrieval_strength
            )
            trace.stability += stability_gain * (1.0 - trace.stability)
            if new_verified_source:
                trace.verification_strength += (
                    self.config.verification_rate
                    * (1.0 - trace.verification_strength)
                )
            trace.support_events += 1
            trace.repetitions += 1
            if recall_success:
                trace.successful_recalls += 1
        else:
            depression = self.config.contradiction_rate
            trace.retrieval_strength *= 1.0 - depression
            trace.stability *= 1.0 - depression
            trace.verification_strength *= 1.0 - depression
            trace.contradiction_events += 1

        trace.retrieval_strength = _clamp01(trace.retrieval_strength)
        trace.stability = _clamp01(trace.stability)
        trace.verification_strength = _clamp01(
            trace.verification_strength
        )
        trace.last_timestep = event_timestep
        self._event_count += 1
        evicted_memory_id = self._enforce_capacity()
        after = self.read(normalized_memory_id)
        return {
            "mutation_allowed": True,
            "memory_id": normalized_memory_id,
            "outcome": outcome,
            "created": created,
            "gap": gap,
            "spacing_multiplier": spacing_multiplier,
            "recall_success": bool(recall_success),
            "verified": bool(verified),
            "new_verified_source": new_verified_source,
            "source_budget_exhausted": source_budget_exhausted,
            "before": before,
            "after": after,
            "evicted_memory_id": evicted_memory_id,
            "clock": self._clock,
            "event_count": self._event_count,
            "state_budget_ok": self.state_budget_ok(),
            "bptt_used": False,
            "dense_matrix_used": False,
            "gpu_used": False,
        }

    def read(
        self,
        memory_id: str,
        *,
        timestep: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read a projected trace without mutating unrelated memory state."""
        target = self._clock if timestep is None else self._validate_timestep(
            timestep
        )
        if target < self._clock:
            raise ValueError("timestep must not precede the current clock")
        trace = self._traces.get(str(memory_id))
        if trace is None:
            return None
        return self._project_trace(trace, target)

    def state_budget_ok(self) -> bool:
        return (
            len(self._traces) <= self.config.capacity
            and self._event_count <= self.config.max_events
            and all(
                len(trace.source_last_seen)
                <= self.config.max_sources_per_memory
                for trace in self._traces.values()
            )
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return a deterministic bounded state view at the current clock."""
        return {
            "schema": "sara-repetition-consolidation-state-v1",
            "clock": self._clock,
            "event_count": self._event_count,
            "rejected_event_count": self._rejected_event_count,
            "eviction_count": self._eviction_count,
            "memory_units": len(self._traces),
            "capacity": self.config.capacity,
            "max_events": self.config.max_events,
            "max_sources_per_memory": (
                self.config.max_sources_per_memory
            ),
            "state_budget_ok": self.state_budget_ok(),
            "production_integrated": False,
            "entries": [
                self._project_trace(self._traces[memory_id], self._clock)
                for memory_id in sorted(self._traces)
            ],
        }

    def _project_trace(
        self,
        trace: _MemoryTrace,
        timestep: int,
    ) -> Dict[str, Any]:
        gap = max(0, int(timestep) - int(trace.last_timestep))
        half_life = self.config.base_half_life * (
            1.0
            + self.config.stability_half_life_gain * trace.stability
        )
        retrieval_decay = 0.5 ** (gap / half_life)
        stability_decay = 0.5 ** (gap / (half_life * 2.0))
        return {
            "memory_id": trace.memory_id,
            "retrieval_strength": _clamp01(
                trace.retrieval_strength * retrieval_decay
            ),
            "stability": _clamp01(trace.stability * stability_decay),
            "verification_strength": _clamp01(
                trace.verification_strength
            ),
            "repetitions": trace.repetitions,
            "support_events": trace.support_events,
            "contradiction_events": trace.contradiction_events,
            "successful_recalls": trace.successful_recalls,
            "last_timestep": trace.last_timestep,
            "projected_timestep": int(timestep),
            "elapsed": gap,
            "verified_source_count": len(trace.verified_sources),
            "observed_source_count": len(trace.source_last_seen),
        }

    def _enforce_capacity(self) -> Optional[str]:
        if len(self._traces) <= self.config.capacity:
            return None
        remove_id = min(
            self._traces,
            key=lambda memory_id: (
                self._retention_priority(
                    self._project_trace(
                        self._traces[memory_id],
                        self._clock,
                    )
                ),
                self._traces[memory_id].last_timestep,
                memory_id,
            ),
        )
        self._traces.pop(remove_id, None)
        self._eviction_count += 1
        return remove_id

    @staticmethod
    def _retention_priority(state: Mapping[str, Any]) -> float:
        return (
            0.50 * float(state["retrieval_strength"])
            + 0.30 * float(state["stability"])
            + 0.20 * float(state["verification_strength"])
        )

    @staticmethod
    def _validate_timestep(timestep: int) -> int:
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise ValueError("timestep must be a non-negative integer")
        if timestep < 0:
            raise ValueError("timestep must be a non-negative integer")
        return timestep

    @staticmethod
    def _validate_text(
        name: str,
        value: str,
        max_bytes: int,
        *,
        allow_empty: bool,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not allow_empty and not value:
            raise ValueError(f"{name} must be non-empty")
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError(f"{name} exceeds byte limit")
        return value
