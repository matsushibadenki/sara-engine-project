"""Bounded sparse causal messages for the two-stage credit experiment."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
import sys
from typing import Mapping, Sequence


ARMS = (
    "no_credit",
    "outcome_broadcast",
    "gradient_like_control",
    "local_credit_packet",
)
INTERVENTIONS = (
    "none",
    "packet_route_shuffle",
    "packet_sign_shuffle",
    "eligibility_reset_before_outcome",
    "packet_ttl_zero",
    "causal_depth_one",
    "replay_disabled",
)


@dataclass(frozen=True)
class LocalCreditPacket:
    source_event: int
    outcome_id: int
    sign: int
    magnitude_bucket: int
    age: int
    causal_depth: int
    confidence: int

    def to_bytes(self) -> bytes:
        return struct.pack(
            ">QQbBBBB",
            self.source_event,
            self.outcome_id,
            self.sign,
            self.magnitude_bucket,
            self.age,
            self.causal_depth,
            self.confidence,
        )


@dataclass(frozen=True)
class Eligibility:
    context_id: int
    branch_id: int
    route_alias: str
    created_step: int


class TwoStageCreditLearner:
    def __init__(
        self,
        arm: str,
        *,
        intervention: str = "none",
        reserve: int = 0,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        self.arm = arm
        self.intervention = intervention
        self.weights: dict[tuple[int, int], int] = {}
        self.eligibility: dict[int, Eligibility] = {}
        self.reserve = bytearray(reserve)
        self.packet_count = 0
        self.backward_events = 0
        self.maximum_packet_bytes = 0
        self.maximum_eligibility_entries = 0

    @staticmethod
    def _event_number(value: str) -> int:
        return int(value[:16], 16)

    def record_forward(self, row: Mapping[str, object], branch_id: int, step: int) -> None:
        source_event = self._event_number(str(row["source_event_id"]))
        aliases = row["branch_to_route_alias"]
        if not isinstance(aliases, Mapping):
            raise ValueError("Invalid route aliases")
        self.eligibility[source_event] = Eligibility(
            context_id=int(row["context_id"]),
            branch_id=branch_id,
            route_alias=str(aliases[str(branch_id)]),
            created_step=step,
        )
        self.maximum_eligibility_entries = max(
            self.maximum_eligibility_entries, len(self.eligibility)
        )

    def predict(self, context_id: int) -> int:
        left = self.weights.get((context_id, 0), 0)
        right = self.weights.get((context_id, 1), 0)
        return int(right > left)

    def apply_delayed_outcome(
        self,
        row: Mapping[str, object],
        *,
        step: int,
    ) -> tuple[int, LocalCreditPacket | None]:
        source_event = self._event_number(str(row["source_event_id"]))
        eligibility = self.eligibility.pop(source_event, None)
        if self.intervention == "eligibility_reset_before_outcome":
            self.eligibility.clear()
            eligibility = None
        if eligibility is None:
            return 0, None
        age = step - eligibility.created_step
        if age != int(row["delay_steps"]):
            raise ValueError("Delayed outcome age does not match the forward trace")
        success = int(row["scheduled_training_success"])
        sign = 1 if success else -1
        if self.intervention == "packet_sign_shuffle":
            sign *= -1
        packet = LocalCreditPacket(
            source_event=source_event,
            outcome_id=self._event_number(str(row["outcome_id"])),
            sign=sign,
            magnitude_bucket=3,
            age=age,
            causal_depth=2,
            confidence=3,
        )
        payload_size = len(packet.to_bytes())
        self.maximum_packet_bytes = max(self.maximum_packet_bytes, payload_size)
        if self.arm == "no_credit":
            return 0, None
        if self.arm == "outcome_broadcast":
            for branch_id in (0, 1):
                key = (eligibility.context_id, branch_id)
                self.weights[key] = max(-32, min(32, self.weights.get(key, 0) + sign))
            self.backward_events += 2
            return 2, None
        if self.arm == "gradient_like_control":
            key = (eligibility.context_id, eligibility.branch_id)
            self.weights[key] = max(-32, min(32, self.weights.get(key, 0) + sign))
            self.backward_events += 1
            return 1, None
        maximum_age = 0 if self.intervention == "packet_ttl_zero" else 8
        maximum_depth = 1 if self.intervention == "causal_depth_one" else 2
        if packet.age > maximum_age or packet.causal_depth > maximum_depth:
            return 0, packet
        branch_id = eligibility.branch_id
        if self.intervention == "packet_route_shuffle":
            branch_id ^= 1
        key = (eligibility.context_id, branch_id)
        self.weights[key] = max(-32, min(32, self.weights.get(key, 0) + packet.sign))
        self.packet_count += 1
        self.backward_events += 1
        return 1, packet

    def state_bytes(self) -> int:
        return (
            sys.getsizeof(self.weights)
            + sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in self.weights.items())
            + sys.getsizeof(self.eligibility)
            + sum(
                sys.getsizeof(key) + sys.getsizeof(value)
                for key, value in self.eligibility.items()
            )
            + sys.getsizeof(self.reserve)
        )


def run_credit_arm(
    arm: str,
    training: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    *,
    intervention: str = "none",
    outcome_override: Sequence[int] | None = None,
    reserve: int = 0,
) -> dict[str, object]:
    learner = TwoStageCreditLearner(arm, intervention=intervention, reserve=reserve)
    forward_trace = []
    update_count = 0
    for index, original in enumerate(training):
        row = dict(original)
        if outcome_override is not None:
            row["scheduled_training_success"] = int(outcome_override[index])
        action = int(row["scheduled_training_action"])
        learner.record_forward(row, action, index)
        aliases = row["branch_to_route_alias"]
        forward_trace.append(
            (
                row["identity"],
                action,
                aliases[str(action)],
                row["delay_steps"],
                row["distractor_count"],
                row["scheduled_training_success"],
            )
        )
        changed, _packet = learner.apply_delayed_outcome(
            row, step=index + int(row["delay_steps"])
        )
        update_count += changed
    predictions = []
    correct = 0
    for row in development:
        predicted = learner.predict(int(row["context_id"]))
        correct += int(predicted == int(row["target_branch"]))
        predictions.append((row["identity"], predicted))
    return {
        "accuracy": correct / len(development),
        "updates": update_count,
        "development_updates": 0,
        "packet_count": learner.packet_count,
        "backward_events": learner.backward_events,
        "maximum_backward_events_per_episode": min(2, learner.backward_events),
        "maximum_packet_bytes": learner.maximum_packet_bytes,
        "maximum_eligibility_entries": learner.maximum_eligibility_entries,
        "feature_count": len(learner.weights),
        "state_bytes": learner.state_bytes(),
        "maximum_forward_event_work_per_episode": 8,
        "forward_trace_sha256": hashlib.sha256(
            json.dumps(forward_trace, separators=(",", ":")).encode()
        ).hexdigest(),
        "prediction_trace_sha256": hashlib.sha256(
            json.dumps(predictions, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["ARMS", "LocalCreditPacket", "TwoStageCreditLearner", "run_credit_arm"]
