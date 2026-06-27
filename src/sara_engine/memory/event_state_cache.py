from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from sara_engine.dynamics import memory_self_state_alignment


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_ids(values: Iterable[int], limit: int) -> Tuple[int, ...]:
    return tuple(sorted({int(value) for value in values})[: max(1, int(limit))])


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set)) / float(len(union))


@dataclass(frozen=True)
class EventStateCandidate:
    entry_id: str
    signature: Tuple[int, ...]
    source_ref: str
    time_segment: int
    source_revision: str = ""
    own_latent_id: str = ""
    causal_predecessors: Tuple[str, ...] = ()
    confidence: float = 1.0
    uncertainty: float = 0.0
    source_reliability: float = 1.0
    resonance_score: float = 0.0
    sequence_support_score: float = 0.0
    sequence_support_count: int = 0
    metabolic_headroom: float = 1.0
    observed: bool = True
    source_backed: bool = True
    verified: bool = True
    contradicted: bool = False
    abstained: bool = False
    event_cost: int = 0
    expires_at: Optional[int] = None


@dataclass(frozen=True)
class EventStateEntry:
    entry_id: str
    signature: Tuple[int, ...]
    source_ref: str
    source_revision: str
    time_segment: int
    own_latent_id: str
    causal_predecessors: Tuple[str, ...]
    confidence: float
    uncertainty: float
    source_reliability: float
    resonance_score: float
    sequence_support_score: float
    sequence_support_count: int
    observed: bool
    verified: bool
    event_cost: int
    tier: str
    utility: float
    access_count: int = 0
    expires_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "signature": list(self.signature),
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "time_segment": self.time_segment,
            "own_latent_id": self.own_latent_id,
            "causal_predecessors": list(self.causal_predecessors),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "source_reliability": self.source_reliability,
            "resonance_score": self.resonance_score,
            "sequence_support_score": self.sequence_support_score,
            "sequence_support_count": self.sequence_support_count,
            "observed": self.observed,
            "verified": self.verified,
            "event_cost": self.event_cost,
            "tier": self.tier,
            "utility": self.utility,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class CacheAdmissionResult:
    accepted: bool
    decision: str
    entry_id: str
    tier: Optional[str]
    event_cost: int
    state_budget_units: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "decision": self.decision,
            "entry_id": self.entry_id,
            "tier": self.tier,
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class CacheRetrievalResult:
    abstained: bool
    decision: str
    matches: Tuple[Dict[str, Any], ...]
    event_cost: int
    scanned_entries: int
    reactivation_hints: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstained": self.abstained,
            "decision": self.decision,
            "matches": [dict(match) for match in self.matches],
            "event_cost": self.event_cost,
            "scanned_entries": self.scanned_entries,
            "reactivation_hints": [
                dict(hint) for hint in self.reactivation_hints
            ],
        }


@dataclass(frozen=True)
class CacheRefreshResult:
    updated: bool
    entry_id: str
    previous_tier: str | None
    new_tier: str | None
    previous_utility: float
    new_utility: float
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updated": bool(self.updated),
            "entry_id": self.entry_id,
            "previous_tier": self.previous_tier,
            "new_tier": self.new_tier,
            "previous_utility": float(self.previous_utility),
            "new_utility": float(self.new_utility),
            "trace": dict(self.trace),
        }


class VerifiedHierarchicalEventStateCache:
    """Stores and retrieves verified sparse event states under hard budgets."""

    TIER_ORDER = ("recent", "consolidated", "durable")

    def __init__(
        self,
        *,
        retention_profile: str = "logarithmic",
        max_entries: int = 12,
        max_signature_width: int = 64,
        max_causal_links: int = 8,
        top_k: int = 3,
        min_resonance: float = 0.65,
        durable_resonance: float = 0.85,
        min_metabolic_headroom: float = 0.2,
        retrieval_threshold: float = 0.35,
        merge_threshold: float = 0.75,
    ) -> None:
        if retention_profile not in {"fixed", "linear", "logarithmic"}:
            raise ValueError("retention_profile must be fixed, linear, or logarithmic")
        self.retention_profile = retention_profile
        self.max_entries = max(1, int(max_entries))
        self.max_signature_width = max(1, int(max_signature_width))
        self.max_causal_links = max(1, int(max_causal_links))
        self.top_k = max(1, int(top_k))
        self.min_resonance = _clamp01(min_resonance)
        self.durable_resonance = max(self.min_resonance, _clamp01(durable_resonance))
        self.min_metabolic_headroom = _clamp01(min_metabolic_headroom)
        self.retrieval_threshold = _clamp01(retrieval_threshold)
        self.merge_threshold = _clamp01(merge_threshold)
        self.entries: Dict[str, EventStateEntry] = {}
        self.admission_count = 0
        self.block_count = 0
        self.merge_count = 0
        self.eviction_count = 0
        self.expiry_count = 0
        self.retrieval_count = 0
        self.lifecycle_trace: List[Dict[str, Any]] = []

    def admit(self, candidate: EventStateCandidate) -> CacheAdmissionResult:
        normalized_signature = _bounded_ids(
            candidate.signature,
            self.max_signature_width,
        )
        decision = self._admission_decision(candidate, normalized_signature)
        if decision != "admit":
            self.block_count += 1
            result = CacheAdmissionResult(
                accepted=False,
                decision=decision,
                entry_id=str(candidate.entry_id),
                tier=None,
                event_cost=len(normalized_signature),
                state_budget_units=len(self.entries),
                trace=self._admission_trace(candidate),
            )
            self.lifecycle_trace.append({"operation": "admission", **result.to_dict()})
            return result

        utility = self._utility(candidate)
        tier = self._select_tier(candidate, utility)
        duplicate = self._find_duplicate(
            normalized_signature,
            candidate.own_latent_id,
            candidate.source_ref,
        )
        if duplicate is not None:
            merged = replace(
                duplicate,
                signature=_bounded_ids(
                    tuple(duplicate.signature) + tuple(normalized_signature),
                    self.max_signature_width,
                ),
                confidence=max(duplicate.confidence, _clamp01(candidate.confidence)),
                uncertainty=min(duplicate.uncertainty, _clamp01(candidate.uncertainty)),
                source_reliability=max(
                    duplicate.source_reliability,
                    _clamp01(candidate.source_reliability),
                ),
                resonance_score=max(
                    duplicate.resonance_score,
                    _clamp01(candidate.resonance_score),
                ),
                sequence_support_score=max(
                    duplicate.sequence_support_score,
                    _clamp01(candidate.sequence_support_score),
                ),
                sequence_support_count=max(
                    int(duplicate.sequence_support_count),
                    max(0, int(candidate.sequence_support_count)),
                ),
                utility=max(duplicate.utility, utility),
                access_count=duplicate.access_count + 1,
            )
            self.entries[duplicate.entry_id] = merged
            self.merge_count += 1
            result = CacheAdmissionResult(
                accepted=True,
                decision="merge_verified_duplicate",
                entry_id=duplicate.entry_id,
                tier=duplicate.tier,
                event_cost=len(normalized_signature) + len(duplicate.signature),
                state_budget_units=len(self.entries),
                trace=self._admission_trace(candidate),
            )
            self.lifecycle_trace.append({"operation": "merge", **result.to_dict()})
            return result

        entry = EventStateEntry(
            entry_id=str(candidate.entry_id),
            signature=normalized_signature,
            source_ref=str(candidate.source_ref),
            source_revision=str(candidate.source_revision),
            time_segment=int(candidate.time_segment),
            own_latent_id=str(candidate.own_latent_id),
            causal_predecessors=tuple(
                sorted({str(value) for value in candidate.causal_predecessors})
            )[: self.max_causal_links],
            confidence=_clamp01(candidate.confidence),
            uncertainty=_clamp01(candidate.uncertainty),
            source_reliability=_clamp01(candidate.source_reliability),
            resonance_score=_clamp01(candidate.resonance_score),
            sequence_support_score=_clamp01(candidate.sequence_support_score),
            sequence_support_count=max(0, int(candidate.sequence_support_count)),
            observed=True,
            verified=True,
            event_cost=max(0, int(candidate.event_cost)),
            tier=tier,
            utility=utility,
            expires_at=candidate.expires_at,
        )
        self.entries[entry.entry_id] = entry
        self.admission_count += 1
        evicted = self._enforce_budget(preferred_entry_id=entry.entry_id)
        accepted = entry.entry_id in self.entries
        result = CacheAdmissionResult(
            accepted=accepted,
            decision="admit_verified" if accepted else "evict_budget",
            entry_id=entry.entry_id,
            tier=entry.tier if accepted else None,
            event_cost=len(normalized_signature) + len(evicted),
            state_budget_units=len(self.entries),
            trace={**self._admission_trace(candidate), "evicted_entry_ids": evicted},
        )
        self.lifecycle_trace.append({"operation": "admission", **result.to_dict()})
        return result

    def retrieve(
        self,
        signature: Iterable[int],
        *,
        own_latent_id: str = "",
        causal_context: Iterable[str] = (),
        source_ref: str = "",
        self_state_ids: Iterable[int] = (),
        now_segment: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> CacheRetrievalResult:
        query = _bounded_ids(signature, self.max_signature_width)
        if now_segment is not None:
            self.expire(now_segment)
        causal_set = {str(value) for value in causal_context}
        self_state_tuple = tuple(int(value) for value in self_state_ids)
        scored: List[Tuple[float, EventStateEntry, Dict[str, float]]] = []
        event_cost = len(query)
        for entry in self.entries.values():
            overlap = _jaccard(query, entry.signature)
            latent_agreement = float(
                bool(own_latent_id) and own_latent_id == entry.own_latent_id
            )
            causal_agreement = float(
                bool(causal_set.intersection(entry.causal_predecessors))
            )
            source_agreement = float(bool(source_ref) and source_ref == entry.source_ref)
            sequence_support = _clamp01(entry.sequence_support_score)
            self_state_alignment = _clamp01(
                memory_self_state_alignment(
                    own_latent_id=entry.own_latent_id,
                    source_ref=entry.source_ref,
                    self_state_ids=self_state_tuple,
                )
            )
            temporal_relevance = (
                0.0
                if now_segment is None
                else 1.0
                / float(1 + abs(int(now_segment) - int(entry.time_segment)))
            )
            score = (
                0.50 * overlap
                + 0.15 * latent_agreement
                + 0.10 * causal_agreement
                + 0.10 * source_agreement
                + 0.04 * entry.source_reliability
                + 0.03 * entry.confidence
                + 0.03 * temporal_relevance
                + 0.05 * sequence_support
                + 0.05 * self_state_alignment
            )
            event_cost += len(entry.signature)
            scored.append(
                (
                    score,
                    entry,
                    {
                        "sparse_overlap": round(overlap, 6),
                        "latent_agreement": latent_agreement,
                        "causal_agreement": causal_agreement,
                        "source_agreement": source_agreement,
                        "sequence_support": round(sequence_support, 6),
                        "self_state_alignment": round(self_state_alignment, 6),
                        "temporal_relevance": round(temporal_relevance, 6),
                    },
                )
            )

        limit = max(1, int(top_k or self.top_k))
        matches: List[Dict[str, Any]] = []
        for score, entry, components in sorted(
            scored,
            key=lambda item: (-item[0], -item[1].utility, item[1].entry_id),
        ):
            if score < self.retrieval_threshold or len(matches) >= limit:
                continue
            matches.append(
                {
                    "entry_id": entry.entry_id,
                    "score": round(score, 6),
                    "tier": entry.tier,
                    "source_ref": entry.source_ref,
                    "time_segment": entry.time_segment,
                    "utility": entry.utility,
                    "components": components,
                }
            )
            self.entries[entry.entry_id] = replace(
                entry,
                access_count=entry.access_count + 1,
            )

        self.retrieval_count += 1
        abstained = not matches
        result = CacheRetrievalResult(
            abstained=abstained,
            decision="abstain_insufficient_evidence" if abstained else "retrieve_verified",
            matches=tuple(matches),
            event_cost=event_cost,
            scanned_entries=len(scored),
            reactivation_hints=tuple(
                {
                    "entry_id": match["entry_id"],
                    "route": "verified_event_state_reactivation",
                    "activation": match["score"],
                    "source_ref": match["source_ref"],
                    "mutates_durable_state": False,
                }
                for match in matches
            ),
        )
        self.lifecycle_trace.append({"operation": "retrieval", **result.to_dict()})
        return result

    def refresh_from_consolidation(
        self,
        replay_events: Iterable[Mapping[str, Any]],
    ) -> Tuple[CacheRefreshResult, ...]:
        results: List[CacheRefreshResult] = []
        for replay_event in replay_events:
            entry_id = str(replay_event.get("memory_id", replay_event.get("entry_id", "")) or "")
            if not entry_id:
                continue
            entry = self.entries.get(entry_id)
            if entry is None:
                results.append(
                    CacheRefreshResult(
                        updated=False,
                        entry_id=entry_id,
                        previous_tier=None,
                        new_tier=None,
                        previous_utility=0.0,
                        new_utility=0.0,
                        trace={"decision": "skip_missing_entry"},
                    )
                )
                continue
            updated_entry, trace = self._refreshed_entry(entry, replay_event)
            self.entries[entry_id] = updated_entry
            result = CacheRefreshResult(
                updated=updated_entry != entry,
                entry_id=entry_id,
                previous_tier=entry.tier,
                new_tier=updated_entry.tier,
                previous_utility=float(entry.utility),
                new_utility=float(updated_entry.utility),
                trace=trace,
            )
            self.lifecycle_trace.append({"operation": "consolidation_refresh", **result.to_dict()})
            results.append(result)
        self._enforce_budget(preferred_entry_id="")
        return tuple(results)

    def expire(self, now_segment: int) -> List[str]:
        expired = sorted(
            entry_id
            for entry_id, entry in self.entries.items()
            if entry.expires_at is not None and entry.expires_at <= int(now_segment)
        )
        for entry_id in expired:
            del self.entries[entry_id]
        self.expiry_count += len(expired)
        if expired:
            self.lifecycle_trace.append(
                {
                    "operation": "expiry",
                    "now_segment": int(now_segment),
                    "expired_entry_ids": expired,
                }
            )
        return expired

    def state_dict(self) -> Dict[str, Any]:
        tier_counts = {
            tier: sum(1 for entry in self.entries.values() if entry.tier == tier)
            for tier in self.TIER_ORDER
        }
        return {
            "schema": "sara-verified-event-state-cache-v1",
            "retention_profile": self.retention_profile,
            "max_entries": self.max_entries,
            "max_signature_width": self.max_signature_width,
            "max_causal_links": self.max_causal_links,
            "top_k": self.top_k,
            "min_resonance": self.min_resonance,
            "durable_resonance": self.durable_resonance,
            "min_metabolic_headroom": self.min_metabolic_headroom,
            "retrieval_threshold": self.retrieval_threshold,
            "merge_threshold": self.merge_threshold,
            "tier_counts": tier_counts,
            "entry_count": len(self.entries),
            "admission_count": self.admission_count,
            "block_count": self.block_count,
            "merge_count": self.merge_count,
            "eviction_count": self.eviction_count,
            "expiry_count": self.expiry_count,
            "retrieval_count": self.retrieval_count,
            "entries": [
                entry.to_dict()
                for entry in sorted(
                    self.entries.values(),
                    key=lambda item: (self.TIER_ORDER.index(item.tier), item.entry_id),
                )
            ],
            "lifecycle_trace_tail": self.lifecycle_trace[-32:],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "VerifiedHierarchicalEventStateCache":
        if not isinstance(state, Mapping):
            raise ValueError("cache state must be a mapping")
        if state.get("schema") != "sara-verified-event-state-cache-v1":
            raise ValueError("unsupported event-state cache schema")
        raw_entries = state.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("cache entries must be a list")
        cache = cls(
            retention_profile=str(state.get("retention_profile", "")),
            max_entries=int(state.get("max_entries", 0)),
            max_signature_width=int(state.get("max_signature_width", 0)),
            max_causal_links=int(state.get("max_causal_links", 8)),
            top_k=int(state.get("top_k", 3)),
            min_resonance=float(state.get("min_resonance", 0.65)),
            durable_resonance=float(state.get("durable_resonance", 0.85)),
            min_metabolic_headroom=float(
                state.get("min_metabolic_headroom", 0.2)
            ),
            retrieval_threshold=float(state.get("retrieval_threshold", 0.35)),
            merge_threshold=float(state.get("merge_threshold", 0.75)),
        )
        if len(raw_entries) > cache.max_entries:
            raise ValueError("cache state exceeds max_entries")
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError("cache entry must be a mapping")
            entry_id = str(raw.get("entry_id", ""))
            source_ref = str(raw.get("source_ref", ""))
            signature_raw = raw.get("signature")
            tier = str(raw.get("tier", ""))
            if not entry_id or not source_ref:
                raise ValueError("cache entry requires entry_id and source_ref")
            if not isinstance(signature_raw, list) or not signature_raw:
                raise ValueError("cache entry requires a non-empty signature list")
            if len(signature_raw) > cache.max_signature_width:
                raise ValueError("cache entry signature exceeds configured width")
            if tier not in cache.TIER_ORDER:
                raise ValueError("cache entry has an invalid tier")
            if entry_id in cache.entries:
                raise ValueError("cache state contains duplicate entry_id")
            causal_raw = raw.get("causal_predecessors", [])
            if not isinstance(causal_raw, list):
                raise ValueError("causal_predecessors must be a list")
            if len(causal_raw) > cache.max_causal_links:
                raise ValueError("cache entry exceeds causal-link budget")
            entry = EventStateEntry(
                entry_id=entry_id,
                signature=_bounded_ids(
                    (int(value) for value in signature_raw),
                    cache.max_signature_width,
                ),
                source_ref=source_ref,
                source_revision=str(raw.get("source_revision", "")),
                time_segment=int(raw.get("time_segment", 0)),
                own_latent_id=str(raw.get("own_latent_id", "")),
                causal_predecessors=tuple(str(value) for value in causal_raw),
                confidence=_clamp01(raw.get("confidence", 0.0)),
                uncertainty=_clamp01(raw.get("uncertainty", 1.0)),
                source_reliability=_clamp01(raw.get("source_reliability", 0.0)),
                resonance_score=_clamp01(raw.get("resonance_score", 0.0)),
                sequence_support_score=_clamp01(raw.get("sequence_support_score", 0.0)),
                sequence_support_count=max(0, int(raw.get("sequence_support_count", 0) or 0)),
                observed=bool(raw.get("observed", False)),
                verified=bool(raw.get("verified", False)),
                event_cost=max(0, int(raw.get("event_cost", 0))),
                tier=tier,
                utility=_clamp01(raw.get("utility", 0.0)),
                access_count=max(0, int(raw.get("access_count", 0))),
                expires_at=(
                    None
                    if raw.get("expires_at") is None
                    else int(raw.get("expires_at"))
                ),
            )
            if not entry.observed or not entry.verified:
                raise ValueError("durable cache state must be observed and verified")
            cache.entries[entry_id] = entry
        for name in (
            "admission_count",
            "block_count",
            "merge_count",
            "eviction_count",
            "expiry_count",
            "retrieval_count",
        ):
            setattr(cache, name, max(0, int(state.get(name, 0))))
        return cache

    def _admission_decision(
        self,
        candidate: EventStateCandidate,
        signature: Sequence[int],
    ) -> str:
        if not signature:
            return "block_empty_signature"
        if candidate.abstained:
            return "block_abstention"
        if candidate.contradicted:
            return "block_contradiction"
        if not candidate.observed:
            return "block_predicted_only"
        if not candidate.source_backed or not candidate.source_ref:
            return "block_unverified_source"
        if not candidate.verified:
            return "block_failed_verification"
        if _clamp01(candidate.metabolic_headroom) < self.min_metabolic_headroom:
            return "block_metabolic_budget"
        if _clamp01(candidate.resonance_score) < self.min_resonance:
            return "block_insufficient_resonance"
        return "admit"

    def _admission_trace(self, candidate: EventStateCandidate) -> Dict[str, Any]:
        return {
            "observed": bool(candidate.observed),
            "source_backed": bool(candidate.source_backed),
            "source_revision": str(candidate.source_revision),
            "verified": bool(candidate.verified),
            "contradicted": bool(candidate.contradicted),
            "abstained": bool(candidate.abstained),
            "resonance_score": _clamp01(candidate.resonance_score),
            "sequence_support_score": _clamp01(candidate.sequence_support_score),
            "sequence_support_count": max(0, int(candidate.sequence_support_count)),
            "metabolic_headroom": _clamp01(candidate.metabolic_headroom),
            "retention_profile": self.retention_profile,
        }

    def _utility(self, candidate: EventStateCandidate) -> float:
        return round(
            0.40 * _clamp01(candidate.resonance_score)
            + 0.25 * _clamp01(candidate.confidence)
            + 0.20 * _clamp01(candidate.source_reliability)
            + 0.10 * (1.0 - _clamp01(candidate.uncertainty))
            + 0.05 * _clamp01(candidate.sequence_support_score),
            6,
        )

    def _select_tier(self, candidate: EventStateCandidate, utility: float) -> str:
        if self.retention_profile == "fixed":
            return "recent"
        if self.retention_profile == "linear":
            age_band = max(0, int(candidate.time_segment)) % len(self.TIER_ORDER)
            return self.TIER_ORDER[age_band]
        if candidate.resonance_score >= self.durable_resonance and utility >= 0.82:
            return "durable"
        if candidate.resonance_score >= self.min_resonance + 0.1:
            return "consolidated"
        return "recent"

    def _tier_capacities(self) -> Dict[str, int]:
        if self.retention_profile == "fixed":
            return {"recent": min(4, self.max_entries), "consolidated": 0, "durable": 0}
        if self.retention_profile == "linear":
            base, remainder = divmod(self.max_entries, 3)
            return {
                tier: base + int(index < remainder)
                for index, tier in enumerate(self.TIER_ORDER)
            }
        bounded_total = min(self.max_entries, 8)
        recent = max(1, bounded_total // 2)
        consolidated = max(1, bounded_total // 4)
        durable = max(1, bounded_total - recent - consolidated)
        return {
            "recent": recent,
            "consolidated": consolidated,
            "durable": durable,
        }

    def _find_duplicate(
        self,
        signature: Sequence[int],
        own_latent_id: str,
        source_ref: str,
    ) -> Optional[EventStateEntry]:
        candidates = [
            entry
            for entry in self.entries.values()
            if (
                _jaccard(signature, entry.signature) >= self.merge_threshold
                and (
                    (bool(source_ref) and source_ref == entry.source_ref)
                    or (
                        bool(own_latent_id)
                        and own_latent_id == entry.own_latent_id
                    )
                )
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: (entry.utility, entry.entry_id))

    def _refreshed_entry(
        self,
        entry: EventStateEntry,
        replay_event: Mapping[str, Any],
    ) -> Tuple[EventStateEntry, Dict[str, Any]]:
        baseline_retention = _clamp01(replay_event.get("baseline_retention", 0.0))
        post_retention = _clamp01(replay_event.get("post_retention", baseline_retention))
        baseline_noise = _clamp01(replay_event.get("baseline_noise", 1.0))
        post_noise = _clamp01(replay_event.get("post_noise", baseline_noise))
        health_before = _clamp01(replay_event.get("health_before", entry.utility))
        health_after = _clamp01(replay_event.get("health_after", health_before))
        phase = str(replay_event.get("phase", "") or "")
        selected_branch = str(replay_event.get("selected_branch", "") or "")
        branch_count = max(1, int(replay_event.get("latent_branch_count", 1) or 1))

        consolidation_score = _clamp01(
            0.40 * post_retention
            + 0.25 * health_after
            + 0.20 * (1.0 - post_noise)
            + 0.10 * _clamp01(entry.sequence_support_score)
            + 0.05 * min(1.0, float(branch_count) / 3.0)
        )
        updated_utility = round(
            _clamp01(0.70 * entry.utility + 0.30 * consolidation_score),
            6,
        )
        if phase == "crystal" and (
            post_retention >= 0.76
            and health_after >= 0.74
            and post_noise <= 0.24
            and updated_utility >= 0.78
        ):
            new_tier = "durable"
            decision = "promote_durable_phase"
        elif phase == "glass" and (
            post_retention >= 0.66
            and health_after >= 0.66
            and post_noise <= 0.34
            and updated_utility >= 0.64
        ):
            new_tier = "consolidated" if entry.tier == "recent" else entry.tier
            decision = "promote_consolidated_phase" if entry.tier == "recent" else "retain_tier"
        elif phase == "liquid":
            new_tier = "recent"
            decision = "retain_liquid_recent" if entry.tier == "recent" else "defer_recent_phase"
        elif (
            post_retention >= 0.82
            and health_after >= 0.80
            and post_noise <= 0.20
            and updated_utility >= 0.82
        ):
            new_tier = "durable"
            decision = "promote_durable"
        elif (
            post_retention >= 0.70
            and health_after >= 0.70
            and post_noise <= 0.30
            and updated_utility >= 0.68
        ):
            new_tier = "consolidated" if entry.tier == "recent" else entry.tier
            decision = "promote_consolidated" if entry.tier == "recent" else "retain_tier"
        elif post_retention < 0.55 or health_after < 0.55:
            new_tier = "recent"
            decision = "defer_recent"
        else:
            new_tier = entry.tier
            decision = "retain_tier"

        refreshed = replace(
            entry,
            tier=new_tier,
            utility=updated_utility,
            access_count=entry.access_count + 1,
        )
        trace = {
            "decision": decision,
            "phase": phase or "unknown",
            "baseline_retention": baseline_retention,
            "post_retention": post_retention,
            "baseline_noise": baseline_noise,
            "post_noise": post_noise,
            "health_before": health_before,
            "health_after": health_after,
            "consolidation_score": consolidation_score,
            "selected_branch": selected_branch,
            "latent_branch_count": branch_count,
        }
        return refreshed, trace

    def _enforce_budget(self, *, preferred_entry_id: str) -> List[str]:
        capacities = self._tier_capacities()
        evicted: List[str] = []
        for tier in self.TIER_ORDER:
            tier_entries = [
                entry for entry in self.entries.values() if entry.tier == tier
            ]
            overflow = len(tier_entries) - capacities[tier]
            if overflow <= 0:
                continue
            if self.retention_profile == "fixed":
                ranked = sorted(
                    tier_entries,
                    key=lambda entry: (
                        entry.time_segment,
                        entry.entry_id == preferred_entry_id,
                    ),
                )
            else:
                ranked = sorted(
                    tier_entries,
                    key=lambda entry: (
                        entry.utility,
                        entry.access_count,
                        -entry.time_segment,
                        entry.entry_id == preferred_entry_id,
                    ),
                )
            for entry in ranked[:overflow]:
                self.entries.pop(entry.entry_id, None)
                evicted.append(entry.entry_id)
        while len(self.entries) > self.max_entries:
            entry = min(
                self.entries.values(),
                key=lambda item: (
                    item.utility,
                    item.access_count,
                    -item.time_segment,
                ),
            )
            self.entries.pop(entry.entry_id, None)
            evicted.append(entry.entry_id)
        self.eviction_count += len(evicted)
        return sorted(evicted)
