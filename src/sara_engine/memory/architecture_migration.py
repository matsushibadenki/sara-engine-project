"""Replay verified Event Memory into a new architecture without mutating legacy state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from sara_engine.memory.event_state_cache import (
    CacheAdmissionResult,
    EventStateCandidate,
    EventStateEntry,
    VerifiedHierarchicalEventStateCache,
)


@dataclass(frozen=True)
class ArchitectureMigrationPolicy:
    source_architecture_version: str
    target_architecture_version: str
    max_replay_candidates: int = 32
    min_utility: float = 0.0
    schema: str = "sara-architecture-migration-policy-v1"


@dataclass(frozen=True)
class ArchitectureMigrationHold:
    entry_id: str
    reason: str
    architecture_version: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "entry_id": self.entry_id,
            "reason": self.reason,
            "architecture_version": self.architecture_version,
        }


@dataclass(frozen=True)
class ArchitectureMigrationPlan:
    policy: ArchitectureMigrationPolicy
    legacy_reference_entry_ids: Tuple[str, ...]
    replay_candidates: Tuple[EventStateCandidate, ...]
    held_entries: Tuple[ArchitectureMigrationHold, ...]
    schema: str = "sara-architecture-migration-plan-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "policy": {
                "schema": self.policy.schema,
                "source_architecture_version": self.policy.source_architecture_version,
                "target_architecture_version": self.policy.target_architecture_version,
                "max_replay_candidates": self.policy.max_replay_candidates,
                "min_utility": self.policy.min_utility,
            },
            "legacy_reference_entry_ids": list(self.legacy_reference_entry_ids),
            "replay_entry_ids": [item.entry_id for item in self.replay_candidates],
            "held_entries": [item.to_dict() for item in self.held_entries],
        }


@dataclass(frozen=True)
class ArchitectureMigrationResult:
    plan: ArchitectureMigrationPlan
    admissions: Tuple[CacheAdmissionResult, ...]
    schema: str = "sara-architecture-migration-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        admitted_count = sum(1 for item in self.admissions if item.accepted)
        return {
            "schema": self.schema,
            "plan": self.plan.to_dict(),
            "admitted_count": admitted_count,
            "blocked_count": len(self.admissions) - admitted_count,
            "admissions": [item.to_dict() for item in self.admissions],
            "legacy_cache_mutated": False,
        }


class ArchitectureMigrationCoordinator:
    """Builds a source-backed replay bridge between architecture versions."""

    def __init__(self, policy: ArchitectureMigrationPolicy) -> None:
        if not policy.source_architecture_version:
            raise ValueError("source_architecture_version is required")
        if not policy.target_architecture_version:
            raise ValueError("target_architecture_version is required")
        if policy.source_architecture_version == policy.target_architecture_version:
            raise ValueError("source and target architecture versions must differ")
        self.policy = policy

    def build_plan(
        self,
        legacy_cache: VerifiedHierarchicalEventStateCache,
    ) -> ArchitectureMigrationPlan:
        replay_candidates: List[EventStateCandidate] = []
        held_entries: List[ArchitectureMigrationHold] = []
        legacy_reference_entry_ids: List[str] = []
        entries = sorted(
            legacy_cache.entries.values(),
            key=lambda entry: (-float(entry.utility), entry.entry_id),
        )
        for entry in entries:
            legacy_reference_entry_ids.append(entry.entry_id)
            hold_reason = self._hold_reason(entry)
            if hold_reason:
                held_entries.append(
                    ArchitectureMigrationHold(
                        entry_id=entry.entry_id,
                        reason=hold_reason,
                        architecture_version=entry.architecture_version,
                    )
                )
                continue
            if len(replay_candidates) >= max(1, int(self.policy.max_replay_candidates)):
                held_entries.append(
                    ArchitectureMigrationHold(
                        entry_id=entry.entry_id,
                        reason="hold_replay_budget",
                        architecture_version=entry.architecture_version,
                    )
                )
                continue
            replay_candidates.append(self._candidate_from_entry(entry))
        return ArchitectureMigrationPlan(
            policy=self.policy,
            legacy_reference_entry_ids=tuple(legacy_reference_entry_ids),
            replay_candidates=tuple(replay_candidates),
            held_entries=tuple(held_entries),
        )

    def migrate(
        self,
        legacy_cache: VerifiedHierarchicalEventStateCache,
        target_cache: VerifiedHierarchicalEventStateCache,
    ) -> ArchitectureMigrationResult:
        plan = self.build_plan(legacy_cache)
        admissions = tuple(target_cache.admit(candidate) for candidate in plan.replay_candidates)
        return ArchitectureMigrationResult(plan=plan, admissions=admissions)

    def _hold_reason(self, entry: EventStateEntry) -> str:
        if entry.architecture_version != self.policy.source_architecture_version:
            return "hold_architecture_version_mismatch"
        if not entry.observed or not entry.verified:
            return "hold_unverified_legacy_entry"
        if not entry.own_latent_id:
            return "hold_missing_canonical_concept_key"
        if float(entry.utility) < float(self.policy.min_utility):
            return "hold_low_utility"
        return ""

    def _candidate_from_entry(self, entry: EventStateEntry) -> EventStateCandidate:
        return EventStateCandidate.from_verified_evidence(
            verifier_id="architecture-migration-coordinator",
            evidence={
                "source_entry": entry.to_dict(),
                "target_architecture_version": self.policy.target_architecture_version,
            },
            entry_id=(
                f"migration:{self.policy.target_architecture_version}:{entry.entry_id}"
            ),
            signature=tuple(entry.signature),
            source_ref=entry.source_ref,
            source_revision=entry.source_revision,
            time_segment=entry.time_segment,
            own_latent_id=entry.own_latent_id,
            causal_predecessors=tuple(entry.causal_predecessors),
            confidence=entry.confidence,
            uncertainty=entry.uncertainty,
            source_reliability=entry.source_reliability,
            resonance_score=entry.resonance_score,
            sequence_support_score=entry.sequence_support_score,
            sequence_support_count=entry.sequence_support_count,
            credit_score=entry.credit_score,
            credit_responsibility=entry.credit_responsibility,
            credit_confidence=entry.credit_confidence,
            credit_longevity=entry.credit_longevity,
            metabolic_headroom=1.0,
            observed=True,
            source_backed=True,
            verified=True,
            contradicted=False,
            abstained=False,
            event_cost=max(1, int(entry.event_cost)),
            expires_at=entry.expires_at,
            architecture_version=self.policy.target_architecture_version,
            migration_source_architecture_version=entry.architecture_version,
        )
