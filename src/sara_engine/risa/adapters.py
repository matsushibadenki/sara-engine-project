from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from sara_engine.memory.event_state_cache import EventStateCandidate

from .kernel import SARAAlignedRisaKernel
from .models import RisaObservation


def _normalize_token(value: str, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    pieces = text.replace("::", "_").replace(":", "_").replace("->", "_to_").split()
    return "_".join(piece for piece in pieces if piece) or fallback


def observation_from_verified_relation(
    relation: Any,
    *,
    timestamp: int | None = None,
) -> RisaObservation:
    relation_name = _normalize_token(relation.relation, "related")
    source_token = _normalize_token(relation.source_event_id, "unknown_source")
    target_token = _normalize_token(relation.target_event_id, "unknown_target")
    context_tags = [
        f"delay:{int(relation.delay_lower_ms)}-{int(relation.delay_upper_ms)}",
        f"confidence:{round(float(relation.confidence), 3)}",
    ]
    if relation.lineage.source_ref:
        context_tags.append(f"source:{relation.lineage.source_ref}")
    return RisaObservation(
        timestamp=int(
            relation.delay_upper_ms if timestamp is None else timestamp
        ),
        event_id=str(relation.record_id),
        actor=source_token,
        action=relation_name,
        observed_effects=[target_token],
        context_tags=context_tags,
        source_ref=str(relation.lineage.source_ref),
        verified=True,
        resonance_score=float(relation.confidence),
        credit_longevity=min(
            1.0,
            float(max(0, int(relation.evidence_count)))
            / float(max(1, int(relation.evidence_count) + int(relation.counterexample_count))),
        ),
        event_energy=min(1.0, float(max(0.0, relation.prediction_gain))),
    )


def observation_from_event_state_candidate(
    candidate: EventStateCandidate,
    *,
    event_id_prefix: str = "event_state",
) -> RisaObservation:
    actor = _normalize_token(candidate.source_ref or candidate.own_latent_id, "event_memory")
    action = "stabilize"
    effects = [
        _normalize_token(item, "cause")
        for item in candidate.causal_predecessors[:3]
    ] or [_normalize_token(candidate.entry_id, "event_state")]
    return RisaObservation(
        timestamp=int(candidate.time_segment),
        event_id=f"{event_id_prefix}:{candidate.entry_id}",
        actor=actor,
        action=action,
        observed_effects=effects,
        context_tags=[
            f"source_backed:{str(bool(candidate.source_backed)).lower()}",
            f"observed:{str(bool(candidate.observed)).lower()}",
            f"resonance:{round(float(candidate.resonance_score), 3)}",
        ],
        source_ref=str(candidate.source_ref),
        verified=bool(candidate.verified and candidate.source_backed and candidate.observed),
        resonance_score=float(candidate.resonance_score),
        credit_longevity=float(candidate.credit_longevity),
        event_energy=float(candidate.credit_score),
    )


def observation_from_bundle_admission(
    result: Any,
) -> RisaObservation:
    candidate = result.candidate
    actor = _normalize_token(candidate.source_ref, "bundle")
    return RisaObservation(
        timestamp=int(candidate.time_segment),
        event_id=f"bundle:{candidate.entry_id}",
        actor=actor,
        action="bind",
        observed_effects=[_normalize_token(candidate.entry_id, "bundle_event")],
        context_tags=[
            f"promotion:{result.promotion_decision}",
            f"modalities:{len(candidate.causal_predecessors)}",
        ],
        source_ref=str(candidate.source_ref),
        verified=bool(result.promotion_allowed),
        resonance_score=float(candidate.resonance_score),
        credit_longevity=float(candidate.credit_longevity),
        event_energy=float(candidate.credit_score),
    )


def observation_from_reactivation_hint(
    hint: Mapping[str, object],
    *,
    time_segment: int = 0,
) -> RisaObservation:
    entry_id = str(hint.get("entry_id", "") or "")
    source_ref = str(hint.get("source_ref", "") or "")
    route = str(hint.get("route", "") or "verified_event_state_reactivation")
    activation = float(hint.get("activation", 0.0) or 0.0)
    actor = _normalize_token(source_ref or entry_id, "memory")
    return RisaObservation(
        timestamp=int(time_segment),
        event_id=f"reactivation:{entry_id or actor}",
        actor=actor,
        action="reactivate",
        observed_effects=[_normalize_token(entry_id or route, "memory_trace")],
        context_tags=[f"route:{route}", f"activation:{round(activation, 3)}"],
        source_ref=source_ref,
        verified=True,
        resonance_score=min(1.0, activation),
        credit_longevity=min(1.0, activation),
        event_energy=min(1.0, activation),
    )


def ingest_verified_surface_into_risa(
    kernel: SARAAlignedRisaKernel,
    *,
    verified_relations: Sequence[Any] = (),
    event_state_candidates: Sequence[EventStateCandidate] = (),
    bundle_admissions: Sequence[Any] = (),
) -> List[RisaObservation]:
    observations: List[RisaObservation] = []
    observations.extend(observation_from_verified_relation(item) for item in verified_relations)
    observations.extend(
        observation_from_event_state_candidate(item)
        for item in event_state_candidates
        if item.verified
    )
    observations.extend(
        observation_from_bundle_admission(item)
        for item in bundle_admissions
        if item.promotion_allowed
    )
    kernel.ingest_observations(observations)
    return observations


def event_state_candidate_from_entry(entry: Any) -> EventStateCandidate:
    return EventStateCandidate(
        entry_id=str(getattr(entry, "entry_id", "")),
        signature=tuple(int(value) for value in getattr(entry, "signature", ()) or ()),
        source_ref=str(getattr(entry, "source_ref", "")),
        source_revision=str(getattr(entry, "source_revision", "")),
        time_segment=int(getattr(entry, "time_segment", 0) or 0),
        own_latent_id=str(getattr(entry, "own_latent_id", "")),
        causal_predecessors=tuple(
            str(value) for value in getattr(entry, "causal_predecessors", ()) or ()
        ),
        confidence=float(getattr(entry, "confidence", 0.0) or 0.0),
        uncertainty=float(getattr(entry, "uncertainty", 0.0) or 0.0),
        source_reliability=float(getattr(entry, "source_reliability", 0.0) or 0.0),
        resonance_score=float(getattr(entry, "resonance_score", 0.0) or 0.0),
        sequence_support_score=float(getattr(entry, "sequence_support_score", 0.0) or 0.0),
        sequence_support_count=int(getattr(entry, "sequence_support_count", 0) or 0),
        credit_score=float(getattr(entry, "credit_score", 0.0) or 0.0),
        credit_responsibility=float(getattr(entry, "credit_responsibility", 0.0) or 0.0),
        credit_confidence=float(getattr(entry, "credit_confidence", 0.0) or 0.0),
        credit_longevity=float(getattr(entry, "credit_longevity", 0.0) or 0.0),
        metabolic_headroom=1.0,
        observed=bool(getattr(entry, "observed", False)),
        source_backed=True,
        verified=bool(getattr(entry, "verified", False)),
        contradicted=False,
        abstained=False,
        event_cost=int(getattr(entry, "event_cost", 0) or 0),
        expires_at=getattr(entry, "expires_at", None),
        architecture_version=str(
            getattr(entry, "architecture_version", "sara-architecture-v1")
            or "sara-architecture-v1"
        ),
        migration_source_architecture_version=str(
            getattr(entry, "migration_source_architecture_version", "") or ""
        ),
    )


def ingest_event_memory_cycle_into_risa(
    kernel: SARAAlignedRisaKernel,
    *,
    ingest_result: Any,
    cache: Any | None = None,
    retrieval_result: Any | None = None,
    now_segment: int | None = None,
) -> List[RisaObservation]:
    observations: List[RisaObservation] = []
    verified_relations = tuple(getattr(ingest_result, "verified_relations", ()) or ())
    bundle_admissions = tuple(
        getattr(ingest_result, "multimodal_bundle_admissions", ()) or ()
    )
    observations.extend(
        ingest_verified_surface_into_risa(
            kernel,
            verified_relations=verified_relations,
            bundle_admissions=bundle_admissions,
        )
    )

    if cache is not None and retrieval_result is not None:
        matched_entry_ids = [
            str(match.get("entry_id", "") or "")
            for match in getattr(retrieval_result, "matches", ()) or ()
            if isinstance(match, Mapping)
        ]
        matched_candidates = [
            event_state_candidate_from_entry(cache.entries[entry_id])
            for entry_id in matched_entry_ids
            if getattr(cache, "entries", None) is not None and entry_id in cache.entries
        ]
        if matched_candidates:
            observations.extend(
                ingest_verified_surface_into_risa(
                    kernel,
                    event_state_candidates=matched_candidates,
                )
            )
        hint_time = int(now_segment if now_segment is not None else 0)
        hint_observations = [
            observation_from_reactivation_hint(hint, time_segment=hint_time)
            for hint in getattr(retrieval_result, "reactivation_hints", ()) or ()
            if isinstance(hint, Mapping)
        ]
        if hint_observations:
            kernel.ingest_observations(hint_observations)
            observations.extend(hint_observations)

    return observations


def extract_verified_event_state_candidates(
    entries: Iterable[Mapping[str, object]],
) -> List[EventStateCandidate]:
    candidates: List[EventStateCandidate] = []
    for raw in entries:
        if not raw:
            continue
        candidates.append(
            EventStateCandidate(
                entry_id=str(raw.get("entry_id", "")),
                signature=tuple(int(value) for value in raw.get("signature", ()) or ()),
                source_ref=str(raw.get("source_ref", "")),
                source_revision=str(raw.get("source_revision", "")),
                time_segment=int(raw.get("time_segment", 0) or 0),
                own_latent_id=str(raw.get("own_latent_id", "")),
                causal_predecessors=tuple(str(value) for value in raw.get("causal_predecessors", ()) or ()),
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                uncertainty=float(raw.get("uncertainty", 0.0) or 0.0),
                source_reliability=float(raw.get("source_reliability", 0.0) or 0.0),
                resonance_score=float(raw.get("resonance_score", 0.0) or 0.0),
                sequence_support_score=float(raw.get("sequence_support_score", 0.0) or 0.0),
                sequence_support_count=int(raw.get("sequence_support_count", 0) or 0),
                credit_score=float(raw.get("credit_score", 0.0) or 0.0),
                credit_responsibility=float(raw.get("credit_responsibility", 0.0) or 0.0),
                credit_confidence=float(raw.get("credit_confidence", 0.0) or 0.0),
                credit_longevity=float(raw.get("credit_longevity", 0.0) or 0.0),
                metabolic_headroom=float(raw.get("metabolic_headroom", 0.0) or 0.0),
                observed=bool(raw.get("observed", False)),
                source_backed=bool(raw.get("source_backed", False)),
                verified=bool(raw.get("verified", False)),
                contradicted=bool(raw.get("contradicted", False)),
                abstained=bool(raw.get("abstained", False)),
                event_cost=int(raw.get("event_cost", 0) or 0),
                architecture_version=str(
                    raw.get("architecture_version", "sara-architecture-v1")
                    or "sara-architecture-v1"
                ),
                migration_source_architecture_version=str(
                    raw.get("migration_source_architecture_version", "") or ""
                ),
            )
        )
    return candidates
