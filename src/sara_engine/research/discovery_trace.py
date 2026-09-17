"""Replay decision transcripts for observable-action audits."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Iterable

from .discovery_replay import DiscoveryReplayWorld, ReplayRecord, ReplayView


MAX_TRACE_ROUNDS = 1024


@dataclass(frozen=True)
class ReplayDecision:
    before_view_sha256: str
    actions: tuple[str, ...]
    revealed_node_ids: tuple[str, ...]
    after_view_sha256: str


@dataclass(frozen=True)
class ReplayTranscript:
    tree_sha256: str
    max_batch: int
    max_reveals: int
    decisions: tuple[ReplayDecision, ...]
    final_view_sha256: str


def _view_sha256(view: ReplayView) -> str:
    payload = {
        "revealed": [record.__dict__ for record in view.revealed],
        "available_actions": view.available_actions,
        "rounds": view.rounds,
        "reveals": view.reveals,
        "stopped": view.stopped,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("ascii")
    return sha256(canonical).hexdigest()


def run_traced_replay_policy(
    world: DiscoveryReplayWorld,
    policy: Callable[[ReplayView], Iterable[str]],
    *,
    max_rounds: int = 32,
) -> ReplayTranscript:
    """Record exactly the exposed view and valid actions; no source tree enters policy."""
    if not isinstance(world, DiscoveryReplayWorld) or not callable(policy):
        raise ValueError("Invalid replay controller arguments")
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_TRACE_ROUNDS:
        raise ValueError("Replay round budget must be positive")
    if world.view.rounds or world.view.stopped:
        raise ValueError("Traced replay requires a fresh world")
    decisions = []
    for _ in range(max_rounds):
        before = world.view
        actions = tuple(policy(before))
        after = world.reveal(actions)
        prior_ids = {record.node_id for record in before.revealed}
        decisions.append(ReplayDecision(
            before_view_sha256=_view_sha256(before),
            actions=actions,
            revealed_node_ids=tuple(record.node_id for record in after.revealed
                                    if record.node_id not in prior_ids),
            after_view_sha256=_view_sha256(after),
        ))
        if after.stopped:
            break
    return ReplayTranscript(
        tree_sha256=world.tree_sha256,
        max_batch=world._max_batch,
        max_reveals=world._max_reveals,
        decisions=tuple(decisions),
        final_view_sha256=_view_sha256(world.view),
    )


def verify_replay_transcript(records: Iterable[ReplayRecord],
                             transcript: ReplayTranscript) -> ReplayView:
    """Rebuild every observable transition; this cannot detect out-of-band policy reads."""
    if not isinstance(transcript, ReplayTranscript):
        raise ValueError("Invalid replay transcript")
    if not 1 <= len(transcript.decisions) <= MAX_TRACE_ROUNDS:
        raise ValueError("Invalid replay transcript length")
    world = DiscoveryReplayWorld(records, max_batch=transcript.max_batch,
                                 max_reveals=transcript.max_reveals)
    if world.tree_sha256 != transcript.tree_sha256:
        raise ValueError("Replay transcript tree mismatch")
    for decision in transcript.decisions:
        if not isinstance(decision, ReplayDecision):
            raise ValueError("Invalid replay decision")
        before = world.view
        if _view_sha256(before) != decision.before_view_sha256:
            raise ValueError("Replay transcript prior view mismatch")
        if not isinstance(decision.actions, tuple):
            raise ValueError("Replay transcript actions must be a tuple")
        try:
            after = world.reveal(decision.actions)
        except (TypeError, ValueError) as exc:
            raise ValueError("Replay transcript contains an unavailable action") from exc
        prior_ids = {record.node_id for record in before.revealed}
        newly_revealed = tuple(record.node_id for record in after.revealed
                               if record.node_id not in prior_ids)
        if newly_revealed != decision.revealed_node_ids or _view_sha256(after) != decision.after_view_sha256:
            raise ValueError("Replay transcript reveal mismatch")
    if _view_sha256(world.view) != transcript.final_view_sha256:
        raise ValueError("Replay transcript final view mismatch")
    return world.view
