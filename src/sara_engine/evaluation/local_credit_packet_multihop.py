"""Two trainable local circuits with one-edge, non-gradient credit messages."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import sys
from typing import Sequence

from sara_engine.evaluation.local_credit_packet import LocalCreditPacket


ARMS = (
    "no_A_credit",
    "global_outcome_broadcast",
    "direct_anchor_shortcut",
    "local_credit_packet",
    "oracle_control",
)
INTERVENTIONS = (
    "none",
    "B_local_map_reset",
    "B_local_target_shuffle",
    "B_to_A_route_shuffle",
    "B_to_A_sign_shuffle",
    "A_eligibility_reset",
    "packet_delivery_disabled",
)


@dataclass(frozen=True)
class AEligibility:
    source_event: int
    cue: int
    active_branch: int


@dataclass(frozen=True)
class BLocalOutcome:
    cue: int
    target: int


@dataclass(frozen=True)
class GlobalOutcomeToB:
    source_event: int
    outcome_id: int
    B_action: int
    success: int


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        return total + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set)):
        return total + sum(_deep_size(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return total + sum(
            _deep_size(getattr(value, field.name), seen) for field in fields(value)
        )
    return total


class CircuitA:
    def __init__(self, reserve: int = 0) -> None:
        self.weights: dict[tuple[int, int], int] = {}
        self.eligibility: dict[int, AEligibility] = {}
        self.reserve = bytearray(reserve)
        self.updates = 0
        self.maximum_eligibility_entries = 0

    def predict(self, cue: int) -> int:
        return int(self.weights.get((cue, 1), 0) > self.weights.get((cue, 0), 0))

    def record(self, source_event: int, cue: int, action: int) -> None:
        if source_event in self.eligibility:
            raise ValueError("Duplicate A source event")
        self.eligibility[source_event] = AEligibility(source_event, cue, action)
        self.maximum_eligibility_entries = max(
            self.maximum_eligibility_entries, len(self.eligibility)
        )
        if len(self.eligibility) > 1:
            raise ValueError("A eligibility budget exceeded")

    def consume(self, source_event: int) -> AEligibility:
        try:
            return self.eligibility.pop(source_event)
        except KeyError as exc:
            raise ValueError("Missing A eligibility") from exc

    def update(self, cue: int, branch: int, sign: int) -> None:
        key = (cue, branch)
        self.weights[key] = max(-64, min(64, self.weights.get(key, 0) + sign))
        self.updates += 1


class CircuitB:
    def __init__(self) -> None:
        self.local_votes: dict[int, int] = {}
        self.local_updates = 0

    def observe_local(self, outcome: BLocalOutcome) -> None:
        self.local_votes[outcome.cue] = max(
            -128,
            min(
                128,
                self.local_votes.get(outcome.cue, 0)
                + (1 if outcome.target else -1),
            ),
        )
        self.local_updates += 1

    def local_prediction(self, cue: int) -> int:
        return int(self.local_votes.get(cue, 0) > 0)

    def forward(self, A_action: int, B_cue: int) -> int:
        return A_action ^ self.local_prediction(B_cue)

    def make_packet(
        self,
        outcome: GlobalOutcomeToB,
        *,
        B_cue: int,
        A_action: int,
        intervention: str = "none",
    ) -> LocalCreditPacket:
        # Recover the desired downstream bit from B's own action and success.
        # Only B can remove its private local code before credit moves to A.
        desired_B_action = outcome.B_action ^ (1 - outcome.success)
        local_code = 0 if intervention == "B_local_map_reset" else self.local_prediction(B_cue)
        desired_A_action = desired_B_action ^ local_code
        sign = 1 if A_action == desired_A_action else -1
        if intervention == "B_to_A_sign_shuffle":
            sign *= -1
        return LocalCreditPacket(
            source_event=outcome.source_event,
            outcome_id=outcome.outcome_id,
            sign=sign,
            magnitude_bucket=3,
            age=2,
            causal_depth=2,
            confidence=3,
        )


def run_multihop_arm(
    arm: str,
    calibration: Sequence[dict],
    training: Sequence[dict],
    development: Sequence[dict],
    *,
    intervention: str = "none",
    shuffled_B_targets: Sequence[int] | None = None,
    shuffled_global_outcomes: Sequence[int] | None = None,
    reserve: int = 0,
) -> dict[str, object]:
    if arm not in ARMS or intervention not in INTERVENTIONS:
        raise ValueError("Unknown arm or intervention")
    if shuffled_B_targets is not None and len(shuffled_B_targets) != len(calibration) + len(training):
        raise ValueError("B target shuffle length mismatch")
    if shuffled_global_outcomes is not None and len(shuffled_global_outcomes) != len(training):
        raise ValueError("Global outcome shuffle length mismatch")
    A = CircuitA(reserve=reserve)
    B = CircuitB()
    for index, row in enumerate(calibration):
        target = (
            int(row["B_local_target"])
            if shuffled_B_targets is None
            else int(shuffled_B_targets[index])
        )
        B.observe_local(BLocalOutcome(int(row["B_cue"]), target))
    forward_trace = []
    packet_count = 0
    maximum_packet_bytes = 0
    maximum_backward_events = 0
    peak_state_bytes = _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes))
    for index, row in enumerate(training):
        A_cue = int(row["A_cue"])
        B_cue = int(row["B_cue"])
        A_action = int(row["forced_A_action"])
        B_action = int(row["forced_B_action"])
        source = int(str(row["source_event"])[:16], 16)
        outcome_id = int(str(row["outcome_id"])[:16], 16)
        A.record(source, A_cue, A_action)
        peak_state_bytes = max(
            peak_state_bytes,
            _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)),
        )
        forward_trace.append(
            (row["identity"], A_action, B_action, row["global_success"])
        )
        local_target = (
            int(row["B_local_target"])
            if shuffled_B_targets is None
            else int(shuffled_B_targets[len(calibration) + index])
        )
        B.observe_local(BLocalOutcome(B_cue, local_target))
        peak_state_bytes = max(
            peak_state_bytes,
            _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)),
        )
        global_success = (
            int(row["global_success"])
            if shuffled_global_outcomes is None
            else int(shuffled_global_outcomes[index])
        )
        outcome = GlobalOutcomeToB(source, outcome_id, B_action, global_success)
        if intervention == "A_eligibility_reset":
            A.eligibility.clear()
            receipt = None
        else:
            receipt = A.consume(source)
        emitted = 0
        if arm == "global_outcome_broadcast" and receipt is not None:
            sign = 1 if outcome.success else -1
            A.update(receipt.cue, 0, sign)
            A.update(receipt.cue, 1, sign)
            emitted = 2
        elif arm == "direct_anchor_shortcut" and receipt is not None:
            A.update(receipt.cue, receipt.active_branch, 1 if outcome.success else -1)
            emitted = 1
        elif arm == "oracle_control" and receipt is not None:
            desired = int(row["A_target"])
            A.update(receipt.cue, receipt.active_branch, 1 if receipt.active_branch == desired else -1)
            emitted = 1
        elif arm == "local_credit_packet" and receipt is not None and intervention != "packet_delivery_disabled":
            packet = B.make_packet(
                outcome,
                B_cue=B_cue,
                A_action=A_action,
                intervention=intervention,
            )
            if packet.source_event != receipt.source_event or packet.causal_depth != 2:
                raise ValueError("Invalid B-to-A packet")
            branch = receipt.active_branch
            if intervention == "B_to_A_route_shuffle":
                branch ^= 1
            A.update(receipt.cue, branch, packet.sign)
            packet_count += 1
            emitted = 2
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        maximum_backward_events = max(maximum_backward_events, emitted)
        peak_state_bytes = max(
            peak_state_bytes,
            _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)),
        )
    prediction_trace = []
    correct = 0
    A_correct = 0
    B_local_correct = 0
    for row in development:
        A_action = A.predict(int(row["A_cue"]))
        B_local = B.local_prediction(int(row["B_cue"]))
        B_action = A_action ^ B_local
        correct += int(B_action == int(row["global_target"]))
        A_correct += int(A_action == int(row["A_target"]))
        B_local_correct += int(B_local == int(row["B_local_target"]))
        prediction_trace.append((row["identity"], A_action, B_action))
    digest = lambda value: hashlib.sha256(  # noqa: E731
        json.dumps(value, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "accuracy": correct / len(development),
        "A_accuracy": A_correct / len(development),
        "B_local_accuracy": B_local_correct / len(development),
        "A_updates": A.updates,
        "B_local_updates": B.local_updates,
        "development_updates": 0,
        "packet_count": packet_count,
        "maximum_packet_bytes": maximum_packet_bytes,
        "maximum_backward_events_per_episode": maximum_backward_events,
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "A_feature_count": len(A.weights),
        "B_feature_count": len(B.local_votes),
        "state_bytes": _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)),
        "peak_state_bytes": peak_state_bytes,
        "maximum_forward_event_work_per_episode": 8,
        "forward_trace_sha256": digest(forward_trace),
        "prediction_trace_sha256": digest(prediction_trace),
    }


__all__ = ["CircuitA", "CircuitB", "GlobalOutcomeToB", "run_multihop_arm"]
