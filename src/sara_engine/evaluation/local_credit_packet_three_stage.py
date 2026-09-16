"""Bounded targeted-anchor replay for the three-stage credit experiment."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import struct
import sys
from typing import Mapping, Sequence

from sara_engine.evaluation.local_credit_packet import LocalCreditPacket


ARMS = (
    "no_credit",
    "outcome_broadcast",
    "direct_packet",
    "packet_targeted_replay",
    "gradient_like_control",
)
CONTROLS = (
    "none",
    "replay_disabled",
    "anchor_route_shuffle",
    "anchor_context_shuffle",
    "packet_sign_shuffle",
    "anchor_expired",
    "causal_depth_two",
)


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
    if isinstance(value, (tuple, list, set, frozenset)):
        return total + sum(_deep_size(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return total + sum(
            _deep_size(getattr(value, field.name), seen) for field in fields(value)
        )
    return total


@dataclass(frozen=True)
class EpisodicAnchor:
    source_event: str
    route_digest: str
    context_bucket: int
    created_step: int
    expiry_step: int

    def to_bytes(self) -> bytes:
        return (
            bytes.fromhex(self.source_event)
            + bytes.fromhex(self.route_digest)
            + struct.pack(">BII", self.context_bucket, self.created_step, self.expiry_step)
        )


@dataclass(frozen=True)
class DirectEligibility:
    source_event: str
    context_bucket: int
    branch_id: int
    created_step: int


def _route_branch(route_digest: str, namespace: str) -> int:
    for branch in (0, 1):
        for alias in ("route-alpha", "route-beta"):
            expected = hashlib.sha256(
                f"{namespace}|route|{branch}|{alias}".encode()
            ).hexdigest()[:24]
            if route_digest == expected:
                return branch
    raise ValueError("Unknown route digest")


class ThreeStageCreditLearner:
    def __init__(
        self,
        arm: str,
        namespace: str,
        *,
        intervention: str = "none",
        reserve: int = 0,
    ) -> None:
        if arm not in ARMS or intervention not in CONTROLS:
            raise ValueError("Unknown arm or intervention")
        self.arm = arm
        self.namespace = namespace
        self.intervention = intervention
        self.weights: dict[tuple[int, int], int] = {}
        self.direct: dict[str, DirectEligibility] = {}
        self.anchors: dict[str, EpisodicAnchor] = {}
        self.reserve = bytearray(reserve)
        self.peak_direct_entries = 0
        self.peak_anchor_entries = 0
        self.peak_state_bytes = 0
        self.maximum_anchor_bytes = 0
        self.anchor_lookups = 0
        self.maximum_anchor_lookups_per_outcome = 0
        self.maximum_packet_bytes = 0
        self.maximum_backward_events_per_episode = 0
        self.backward_events = 0
        self.updates = 0
        self.expired_direct_count = 0
        self.successful_replays = 0

    def record_forward(self, row: Mapping[str, object]) -> None:
        source = str(row["source_event"])
        if source in self.direct or source in self.anchors:
            raise ValueError("Duplicate source event")
        branch = int(row["forced_action"])
        context = int(row["context_id"])
        self.direct[source] = DirectEligibility(
            source_event=source,
            context_bucket=context,
            branch_id=branch,
            created_step=int(row["source_step"]),
        )
        anchor = EpisodicAnchor(
            source_event=source,
            route_digest=str(row["route_digest"]),
            context_bucket=context,
            created_step=int(row["anchor_step"]),
            expiry_step=int(row["anchor_expiry_step"]),
        )
        self.anchors[source] = anchor
        self.maximum_anchor_bytes = max(
            self.maximum_anchor_bytes, len(anchor.to_bytes())
        )
        self.peak_direct_entries = max(self.peak_direct_entries, len(self.direct))
        self.peak_anchor_entries = max(self.peak_anchor_entries, len(self.anchors))
        self.peak_state_bytes = max(self.peak_state_bytes, self.state_bytes())
        if self.peak_direct_entries > 8 or self.peak_anchor_entries > 8:
            raise ValueError("Outstanding trace budget exceeded")

    def predict(self, context_id: int) -> int:
        return int(
            self.weights.get((context_id, 1), 0)
            > self.weights.get((context_id, 0), 0)
        )

    def _update(self, context: int, branch: int, sign: int) -> None:
        key = (context, branch)
        self.weights[key] = max(-48, min(48, self.weights.get(key, 0) + sign))
        self.updates += 1

    def resolve_outcome(self, row: Mapping[str, object], *, success: int | None = None) -> None:
        source = str(row["source_event"])
        outcome_step = int(row["outcome_step"])
        direct = self.direct.pop(source, None)
        if direct is None:
            raise ValueError("Outcome has no direct source")
        direct_valid = outcome_step - direct.created_step <= 8
        self.expired_direct_count += int(not direct_valid)
        anchor = self.anchors.pop(source, None)
        if anchor is None:
            raise ValueError("Outcome has no anchor")
        observed_success = int(row["outcome_success"] if success is None else success)
        sign = 1 if observed_success else -1
        if self.intervention == "packet_sign_shuffle":
            sign *= -1
        emitted = 0
        if self.arm == "outcome_broadcast":
            self._update(direct.context_bucket, 0, sign)
            self._update(direct.context_bucket, 1, sign)
            emitted = 2
        elif self.arm == "direct_packet" and direct_valid:
            self._update(direct.context_bucket, direct.branch_id, sign)
            emitted = 1
        elif self.arm == "gradient_like_control":
            self._update(direct.context_bucket, direct.branch_id, sign)
            emitted = 1
        elif self.arm == "packet_targeted_replay" and self.intervention not in (
            "replay_disabled", "causal_depth_two"
        ):
            self.anchor_lookups += 1
            self.maximum_anchor_lookups_per_outcome = max(
                self.maximum_anchor_lookups_per_outcome, 1
            )
            if self.intervention != "anchor_expired" and (
                anchor.created_step <= outcome_step < anchor.expiry_step
            ):
                route = anchor.route_digest
                context = anchor.context_bucket
                branch = _route_branch(route, self.namespace)
                if self.intervention == "anchor_route_shuffle":
                    branch ^= 1
                if self.intervention == "anchor_context_shuffle":
                    context = (context + 1) % 8
                packet = LocalCreditPacket(
                    source_event=int(source[:16], 16),
                    outcome_id=int(str(row["outcome_id"])[:16], 16),
                    sign=sign,
                    magnitude_bucket=3,
                    age=min(255, outcome_step - anchor.created_step),
                    causal_depth=3,
                    confidence=3,
                )
                self.maximum_packet_bytes = max(
                    self.maximum_packet_bytes, len(packet.to_bytes())
                )
                self._update(context, branch, packet.sign)
                self.successful_replays += 1
                emitted = 1
        self.backward_events += emitted
        self.maximum_backward_events_per_episode = max(
            self.maximum_backward_events_per_episode, emitted
        )

    def state_bytes(self) -> int:
        return _deep_size((self.weights, self.direct, self.anchors, self.reserve))


def run_three_stage_arm(
    arm: str,
    training: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    *,
    namespace: str,
    intervention: str = "none",
    shuffled_outcomes: Sequence[int] | None = None,
    reserve: int = 0,
) -> dict[str, object]:
    learner = ThreeStageCreditLearner(
        arm, namespace, intervention=intervention, reserve=reserve
    )
    waves: dict[int, list[tuple[int, Mapping[str, object]]]] = {}
    for index, row in enumerate(training):
        waves.setdefault(int(row["wave"]), []).append((index, row))
    forward_trace = []
    for wave in sorted(waves):
        group = waves[wave]
        if len(group) != 8:
            raise ValueError("Incomplete training wave")
        for _index, row in group:
            learner.record_forward(row)
            forward_trace.append(
                (row["identity"], row["forced_action"], row["route_digest"],
                 row["source_step"], row["outcome_step"], row["outcome_success"])
            )
        for index, row in group:
            success = None if shuffled_outcomes is None else shuffled_outcomes[index]
            learner.resolve_outcome(row, success=success)
    predictions = []
    correct = 0
    for row in development:
        prediction = learner.predict(int(row["context_id"]))
        correct += int(prediction == int(row["target_branch"]))
        predictions.append((row["identity"], prediction))
    digest = lambda value: hashlib.sha256(  # noqa: E731
        json.dumps(value, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "accuracy": correct / len(development),
        "updates": learner.updates,
        "development_updates": 0,
        "backward_events": learner.backward_events,
        "maximum_backward_events_per_episode": learner.maximum_backward_events_per_episode,
        "maximum_anchor_lookups_per_outcome": learner.maximum_anchor_lookups_per_outcome,
        "anchor_lookups": learner.anchor_lookups,
        "successful_replays": learner.successful_replays,
        "expired_direct_count": learner.expired_direct_count,
        "peak_direct_entries": learner.peak_direct_entries,
        "peak_anchor_entries": learner.peak_anchor_entries,
        "peak_state_bytes": learner.peak_state_bytes,
        "maximum_anchor_bytes": learner.maximum_anchor_bytes,
        "maximum_packet_bytes": learner.maximum_packet_bytes,
        "maximum_forward_event_work_per_episode": 12,
        "feature_count": len(learner.weights),
        "state_bytes": learner.state_bytes(),
        "forward_trace_sha256": digest(forward_trace),
        "prediction_trace_sha256": digest(predictions),
    }


__all__ = ["EpisodicAnchor", "ThreeStageCreditLearner", "run_three_stage_arm"]
