"""Post-run coverage and missing-evidence audit for discovery replay."""
from __future__ import annotations

from dataclasses import dataclass

from .discovery_replay import DiscoveryReplayWorld


@dataclass(frozen=True)
class ReplayCoverageAudit:
    recorded_nonroot: int
    revealed_nonroot: int
    coverage: float
    unrevealed_recorded: int
    budget_unreachable: int
    missing_recorded: int
    missing_revealed: int
    failed_recorded: int
    stopped: bool


def audit_replay(world: DiscoveryReplayWorld) -> ReplayCoverageAudit:
    """Inspect complete history after a run; never pass this to a policy."""
    if not isinstance(world, DiscoveryReplayWorld):
        raise ValueError("A discovery replay world is required")
    records = world._records
    root_id = records[0].node_id
    root_children = world._children.get(root_id, ())
    minimum_reveals = {root_id: 0}
    for position, node_id in enumerate(root_children, 1):
        minimum_reveals[node_id] = position
    for record in records[1:]:
        if record.node_id not in minimum_reveals:
            minimum_reveals[record.node_id] = minimum_reveals[record.parent_id] + 1
    nonroot = records[1:]
    revealed = world._revealed
    count = len(nonroot)
    revealed_count = sum(record.node_id in revealed for record in nonroot)
    return ReplayCoverageAudit(
        recorded_nonroot=count,
        revealed_nonroot=revealed_count,
        coverage=revealed_count / count if count else 1.0,
        unrevealed_recorded=count - revealed_count,
        budget_unreachable=sum(minimum_reveals[record.node_id] > world._max_reveals
                               for record in nonroot),
        missing_recorded=sum(record.status == "missing" for record in nonroot),
        missing_revealed=sum(record.status == "missing" and record.node_id in revealed
                             for record in nonroot),
        failed_recorded=sum(record.status == "failed" for record in nonroot),
        stopped=world.view.stopped,
    )
