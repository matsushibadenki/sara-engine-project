from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple


RESONANCE_CHANNELS = (
    "local_coincidence",
    "prediction_consistency",
    "verifier_confidence",
    "cross_modal_agreement",
    "reward_signal",
    "novelty_signal",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class ResonanceCreditResult:
    update_allowed: bool
    decision: str
    resonance_score: float
    active_channel_count: int
    signed_modulation: float
    updates: Dict[Tuple[int, int], float]
    event_cost: int
    state_budget_units: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_allowed": self.update_allowed,
            "decision": self.decision,
            "resonance_score": self.resonance_score,
            "active_channel_count": self.active_channel_count,
            "signed_modulation": self.signed_modulation,
            "updates": {
                f"{pre_id}:{post_id}": value
                for (pre_id, post_id), value in sorted(self.updates.items())
            },
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "trace": dict(self.trace),
        }


class SparseResonanceCreditAssigner:
    """Gates sparse local plasticity when independent evidence channels resonate."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        channel_threshold: float = 0.55,
        min_resonant_channels: int = 3,
        max_links: int = 256,
        weight_clip: float = 1.5,
        min_metabolic_headroom: float = 0.2,
    ) -> None:
        self.learning_rate = _clamp(learning_rate, 0.0, 1.0)
        self.channel_threshold = _clamp(channel_threshold, 0.0, 1.0)
        self.min_resonant_channels = max(1, int(min_resonant_channels))
        self.max_links = max(1, int(max_links))
        self.weight_clip = max(0.1, float(weight_clip))
        self.min_metabolic_headroom = _clamp(min_metabolic_headroom, 0.0, 1.0)
        self.weights: MutableMapping[Tuple[int, int], float] = {}
        self.update_count = 0
        self.freeze_count = 0

    def apply(
        self,
        eligibility: Mapping[Tuple[int, int], float],
        signals: Mapping[str, Any],
    ) -> ResonanceCreditResult:
        normalized = {
            channel: _clamp(float(signals.get(channel, 0.0) or 0.0), 0.0, 1.0)
            for channel in RESONANCE_CHANNELS
        }
        active_channels = [
            channel
            for channel, value in normalized.items()
            if value >= self.channel_threshold
        ]
        resonance_score = (
            sum(normalized[channel] for channel in active_channels)
            / float(max(1, len(active_channels)))
        )
        contradiction = _clamp(float(signals.get("contradiction", 0.0) or 0.0), 0.0, 1.0)
        metabolic_headroom = _clamp(
            float(signals.get("metabolic_headroom", 1.0) or 0.0),
            0.0,
            1.0,
        )
        abstained = bool(signals.get("abstained", False))
        source_backed = bool(signals.get("source_backed", False))

        decision = "reinforce"
        if abstained:
            decision = "freeze_abstention"
        elif not source_backed:
            decision = "freeze_unverified_source"
        elif contradiction >= self.channel_threshold:
            decision = "freeze_contradiction"
        elif metabolic_headroom < self.min_metabolic_headroom:
            decision = "freeze_metabolic_budget"
        elif len(active_channels) < self.min_resonant_channels:
            decision = "freeze_insufficient_resonance"

        update_allowed = decision == "reinforce"
        signed_modulation = 0.0
        updates: Dict[Tuple[int, int], float] = {}
        if update_allowed:
            reward_polarity = _clamp(
                float(signals.get("reward_polarity", 1.0) or 0.0),
                -1.0,
                1.0,
            )
            signed_modulation = (
                resonance_score
                * reward_polarity
                * metabolic_headroom
                * (1.0 - contradiction)
            )
            for key, trace_value in sorted(eligibility.items()):
                if key not in self.weights and len(self.weights) >= self.max_links:
                    continue
                delta = self.learning_rate * float(trace_value) * signed_modulation
                current = self.weights.get(key, 0.0)
                updated = _clamp(current + delta, -self.weight_clip, self.weight_clip)
                self.weights[key] = updated
                updates[key] = round(delta, 6)
            self.update_count += 1
        else:
            self.freeze_count += 1

        return ResonanceCreditResult(
            update_allowed=update_allowed,
            decision=decision,
            resonance_score=round(resonance_score, 6),
            active_channel_count=len(active_channels),
            signed_modulation=round(signed_modulation, 6),
            updates=updates,
            event_cost=len(normalized) + len(eligibility),
            state_budget_units=len(self.weights),
            trace={
                "channels": normalized,
                "active_channels": active_channels,
                "contradiction": contradiction,
                "metabolic_headroom": metabolic_headroom,
                "source_backed": source_backed,
                "abstained": abstained,
                "min_resonant_channels": self.min_resonant_channels,
                "update_count": self.update_count,
                "freeze_count": self.freeze_count,
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-sparse-resonance-credit-state-v1",
            "weights": {
                f"{pre_id}:{post_id}": value
                for (pre_id, post_id), value in sorted(self.weights.items())
            },
            "update_count": self.update_count,
            "freeze_count": self.freeze_count,
        }
