from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from sara_engine.dynamics import (
    PersistentSelfStateController,
    relation_self_state_alignment,
    stable_self_state_id,
)
from sara_engine.learning.adaptive_credit import summarize_event_memory_credit
from sara_engine.memory.multimodal_event_bundle_admission import (
    MultimodalBundleAdmissionResult,
    build_multimodal_event_state_candidate,
)
from sara_engine.multimodal.synesthetic_binding import SparseEventBundle

from .candidate_proposals import (
    CandidateEvent,
    CandidateRelation,
    ObservedEvent,
    ProposalLineage,
    VerifiedRelation,
    make_verified_relation,
)
from .change_detection import ChangePoint, ScalarChangeDetector
from .episode_segmentation import BoundedEpisode, EpisodeSegmenter
from .frequent_sequence import FrequentSequence, FrequentSequenceMiner
from .prediction_gain import PredictionGainEstimator
from .proposal_lineage import ProposalLineageLedgerEntry, build_lineage_ledger_entry
from .proposal_verifier import ProposalVerificationResult, ProposalVerifier
from .synchrony_detector import SynchronyDetector
from .temporal_eventizer import TemporalEventizer


@dataclass(frozen=True)
class EventMemoryIngestResult:
    change_points: Tuple[ChangePoint, ...]
    observed_events: Tuple[ObservedEvent, ...]
    accepted_candidate_events: Tuple[CandidateEvent, ...]
    rejected_candidate_events: Tuple[CandidateEvent, ...]
    episodes: Tuple[BoundedEpisode, ...]
    frequent_sequences: Tuple[FrequentSequence, ...]
    candidate_relations: Tuple[CandidateRelation, ...]
    verified_relations: Tuple[VerifiedRelation, ...]
    lineage_ledger: Tuple[ProposalLineageLedgerEntry, ...]
    candidate_event_verifications: Tuple[ProposalVerificationResult, ...]
    relation_verifications: Tuple[ProposalVerificationResult, ...]
    multimodal_bundle_admissions: Tuple[MultimodalBundleAdmissionResult, ...]
    traces: Dict[str, Any]
    schema: str = "sara-event-memory-ingest-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "change_points": [item.to_dict() for item in self.change_points],
            "observed_events": [item.to_dict() for item in self.observed_events],
            "accepted_candidate_events": [item.to_dict() for item in self.accepted_candidate_events],
            "rejected_candidate_events": [item.to_dict() for item in self.rejected_candidate_events],
            "episodes": [item.to_dict() for item in self.episodes],
            "frequent_sequences": [item.to_dict() for item in self.frequent_sequences],
            "candidate_relations": [item.to_dict() for item in self.candidate_relations],
            "verified_relations": [item.to_dict() for item in self.verified_relations],
            "lineage_ledger": [item.to_dict() for item in self.lineage_ledger],
            "candidate_event_verifications": [item.to_dict() for item in self.candidate_event_verifications],
            "relation_verifications": [item.to_dict() for item in self.relation_verifications],
            "multimodal_bundle_admissions": [
                item.to_dict() for item in self.multimodal_bundle_admissions
            ],
            "traces": dict(self.traces),
        }


class EventMemoryIngestPipeline:
    """Runs a bounded ingest loop from scalar changes to verified relation proposals."""

    def __init__(
        self,
        *,
        change_detector: ScalarChangeDetector | None = None,
        temporal_eventizer: TemporalEventizer | None = None,
        episode_segmenter: EpisodeSegmenter | None = None,
        sequence_miner: FrequentSequenceMiner | None = None,
        synchrony_detector: SynchronyDetector | None = None,
        prediction_gain_estimator: PredictionGainEstimator | None = None,
        verifier: ProposalVerifier | None = None,
        persistent_self_state: PersistentSelfStateController | None = None,
    ) -> None:
        self.change_detector = change_detector or ScalarChangeDetector()
        self.temporal_eventizer = temporal_eventizer or TemporalEventizer()
        self.episode_segmenter = episode_segmenter or EpisodeSegmenter()
        self.sequence_miner = sequence_miner or FrequentSequenceMiner()
        self.synchrony_detector = synchrony_detector or SynchronyDetector()
        self.prediction_gain_estimator = prediction_gain_estimator or PredictionGainEstimator()
        self.verifier = verifier or ProposalVerifier()
        self.persistent_self_state = persistent_self_state

    def ingest_streams(
        self,
        streams: Sequence[Mapping[str, Any]],
        *,
        source_ref: str,
        source_hash: str,
        candidate_events: Sequence[CandidateEvent] = (),
        reactivation_hints: Sequence[Mapping[str, Any]] = (),
        multimodal_bundles: Sequence[SparseEventBundle] = (),
    ) -> EventMemoryIngestResult:
        change_points: List[ChangePoint] = []
        observed_events: List[ObservedEvent] = []
        for stream in streams:
            changes = self.change_detector.detect(
                stream.get("samples", ()) or (),
                stream_id=str(stream.get("stream_id", "") or ""),
                modality=str(stream.get("modality", "") or ""),
            )
            change_points.extend(changes)
            observed_events.extend(
                self.temporal_eventizer.eventize(
                    changes,
                    source_ref=source_ref,
                    source_hash=source_hash,
                )
            )

        candidate_verifications = tuple(self.verifier.verify_event(item) for item in candidate_events)
        accepted_candidate_events = tuple(
            candidate
            for candidate, verification in zip(candidate_events, candidate_verifications)
            if verification.accepted
        )
        rejected_candidate_events = tuple(
            candidate
            for candidate, verification in zip(candidate_events, candidate_verifications)
            if not verification.accepted
        )

        episodes = tuple(
            self.episode_segmenter.segment(
                observed_events,
                candidate_events=accepted_candidate_events,
            )
        )
        frequent_sequences = tuple(
            self.sequence_miner.mine(
                episodes,
                observed_events,
                candidate_events=accepted_candidate_events,
            )
        )

        relation_surface = list(observed_events) + list(accepted_candidate_events)
        synchrony_relations = self.synchrony_detector.propose_relations(relation_surface)
        prediction_relations = self.prediction_gain_estimator.propose_relations(relation_surface)
        candidate_relations = tuple(sorted(
            list(synchrony_relations) + list(prediction_relations),
            key=lambda item: (item.record_id, item.relation),
        ))

        persistent_self_state_trace = self._persistent_self_state_trace(
            observed_events=observed_events,
            accepted_candidate_events=accepted_candidate_events,
            reactivation_hints=reactivation_hints,
            source_ref=source_ref,
            source_hash=source_hash,
        )
        self_state_ids = tuple(
            int(value)
            for value in persistent_self_state_trace.get("self_state_ids", ())
        )
        relation_verifications = tuple(
            self.verifier.verify_relation_with_self_state(
                item,
                self_state_alignment=relation_self_state_alignment(
                    item.source_event_id,
                    item.target_event_id,
                    self_state_ids,
                ),
            )
            for item in candidate_relations
        )
        verified_relations = tuple(
            make_verified_relation(self._flatten_record(verification.promoted_record))
            for verification in relation_verifications
            if verification.promoted_record
        )
        bundle_admission_results = tuple(
            build_multimodal_event_state_candidate(
                bundle,
                time_segment=int(bundle.time_chunk_id),
            )
            for bundle in multimodal_bundles
        )

        ledger = tuple(
            self._build_ledger_entries(
                observed_events=observed_events,
                accepted_candidate_events=accepted_candidate_events,
                rejected_candidate_events=rejected_candidate_events,
                candidate_relations=candidate_relations,
                verified_relations=verified_relations,
            )
        )
        traces = {
            "change_detection": {
                "stream_count": len(streams),
                "change_point_count": len(change_points),
                "threshold": self.change_detector.threshold,
                "refractory_ms": self.change_detector.refractory_ms,
            },
            "eventization": {
                "schema": "sara-temporal-eventization-trace-v1",
                "emitted_count": len(observed_events),
                "suppressed_count": max(0, len(change_points) - len(observed_events)),
                "merge_window_ms": self.temporal_eventizer.merge_window_ms,
                "stream_count": len(streams),
            },
            "episode_segmentation": self.episode_segmenter.last_trace.to_dict(),
            "frequent_sequence": self.sequence_miner.last_trace.to_dict(),
            "synchrony": self.synchrony_detector.last_trace.to_dict(),
            "prediction_gain": self.prediction_gain_estimator.last_trace.to_dict(),
            "verification": {
                "accepted_candidate_event_count": len(accepted_candidate_events),
                "rejected_candidate_event_count": len(rejected_candidate_events),
                "verified_relation_count": len(verified_relations),
                "rejected_relation_count": len(candidate_relations) - len(verified_relations),
            },
            "adaptive_credit": {
                "accepted_candidate_event_summaries": [
                    {
                        "record_id": event.record_id,
                        **self._candidate_event_credit_summary(event),
                    }
                    for event in accepted_candidate_events
                ],
                "verified_relation_summaries": [
                    {
                        "record_id": relation.record_id,
                        **self._relation_credit_summary(relation),
                    }
                    for relation in verified_relations
                ],
            },
            "multimodal_bundle_admission": {
                "bundle_count": len(multimodal_bundles),
                "promotion_allowed_count": len(
                    [item for item in bundle_admission_results if item.promotion_allowed]
                ),
                "promotion_blocked_count": len(
                    [item for item in bundle_admission_results if not item.promotion_allowed]
                ),
                "results": [item.to_dict() for item in bundle_admission_results],
            },
            "persistent_self_state": persistent_self_state_trace,
        }
        return EventMemoryIngestResult(
            change_points=tuple(sorted(change_points, key=lambda item: (item.time_ms, item.stream_id))),
            observed_events=tuple(sorted(observed_events, key=lambda item: (item.local_time_ms, item.record_id))),
            accepted_candidate_events=accepted_candidate_events,
            rejected_candidate_events=rejected_candidate_events,
            episodes=episodes,
            frequent_sequences=frequent_sequences,
            candidate_relations=candidate_relations,
            verified_relations=verified_relations,
            lineage_ledger=ledger,
            candidate_event_verifications=candidate_verifications,
            relation_verifications=relation_verifications,
            multimodal_bundle_admissions=bundle_admission_results,
            traces=traces,
        )

    def _build_ledger_entries(
        self,
        *,
        observed_events: Sequence[ObservedEvent],
        accepted_candidate_events: Sequence[CandidateEvent],
        rejected_candidate_events: Sequence[CandidateEvent],
        candidate_relations: Sequence[CandidateRelation],
        verified_relations: Sequence[VerifiedRelation],
    ) -> Iterable[ProposalLineageLedgerEntry]:
        for record in observed_events:
            yield self._ledger_from_record(
                record_id=record.record_id,
                record_type=record.record_type,
                lineage=record.lineage,
            )
        for record in list(accepted_candidate_events) + list(rejected_candidate_events):
            yield self._ledger_from_record(
                record_id=record.record_id,
                record_type=record.record_type,
                lineage=record.lineage,
            )
        for record in candidate_relations:
            yield self._ledger_from_record(
                record_id=record.record_id,
                record_type=record.record_type,
                lineage=record.lineage,
            )
        for record in verified_relations:
            yield self._ledger_from_record(
                record_id=record.record_id,
                record_type=record.record_type,
                lineage=record.lineage,
            )

    def _ledger_from_record(
        self,
        *,
        record_id: str,
        record_type: str,
        lineage: ProposalLineage,
    ) -> ProposalLineageLedgerEntry:
        return build_lineage_ledger_entry(
            {
                "record_id": record_id,
                "record_type": record_type,
                "source_ref": lineage.source_ref,
                "source_hash": lineage.source_hash,
                "extractor_name": lineage.extractor_name,
                "extractor_version": lineage.extractor_version,
                "parent_ids": lineage.parent_ids,
                "observed_anchor_ids": lineage.observed_anchor_ids,
                "proposal_model": lineage.proposal_model,
                "proposal_config_hash": lineage.proposal_config_hash,
            }
        )

    def _flatten_record(self, promoted_record: Mapping[str, Any] | None) -> Dict[str, Any]:
        if promoted_record is None:
            return {}
        payload = dict(promoted_record)
        lineage = payload.pop("lineage", {})
        if isinstance(lineage, Mapping):
            payload.setdefault("source_ref", str(lineage.get("source_ref", "") or ""))
            payload.setdefault("source_hash", str(lineage.get("source_hash", "") or ""))
            payload.setdefault("extractor_name", str(lineage.get("extractor_name", "") or ""))
            payload.setdefault("extractor_version", str(lineage.get("extractor_version", "") or ""))
            payload.setdefault("parent_ids", lineage.get("parent_ids", ()) or ())
            payload.setdefault("observed_anchor_ids", lineage.get("observed_anchor_ids", ()) or ())
            payload.setdefault("proposal_model", str(lineage.get("proposal_model", "") or ""))
            payload.setdefault("proposal_config_hash", str(lineage.get("proposal_config_hash", "") or ""))
        return payload

    def _persistent_self_state_trace(
        self,
        *,
        observed_events: Sequence[ObservedEvent],
        accepted_candidate_events: Sequence[CandidateEvent],
        reactivation_hints: Sequence[Mapping[str, Any]],
        source_ref: str,
        source_hash: str,
    ) -> Dict[str, Any]:
        controller = self.persistent_self_state
        if controller is None:
            core_ids = (
                stable_self_state_id(source_ref),
                stable_self_state_id(source_hash),
            )
            controller = PersistentSelfStateController(core_event_ids=core_ids)
        external_ids = [
            stable_self_state_id(event.record_id)
            for event in observed_events
        ]
        external_ids.extend(
            stable_self_state_id(event.record_id)
            for event in accepted_candidate_events
        )
        trace = controller.step(
            external_event_ids=tuple(external_ids),
            reactivation_hints=reactivation_hints,
        )
        trace["external_event_count"] = len(external_ids)
        trace["reactivation_hint_count"] = len(tuple(reactivation_hints))
        trace["controller_snapshot"] = controller.snapshot()
        return trace

    def _candidate_event_credit_summary(
        self,
        event: CandidateEvent,
    ) -> Dict[str, float]:
        evidence = max(1.0, float(max(0, int(event.evidence_count))))
        counterexamples = float(max(0, int(event.counterexample_count)))
        return summarize_event_memory_credit(
            (
                {
                    "responsibility": max(0.0, float(event.prediction_gain)) * min(1.0, evidence / 3.0),
                    "confidence": float(event.confidence),
                    "longevity": evidence / (evidence + counterexamples + 1.0),
                },
            )
        )

    def _relation_credit_summary(
        self,
        relation: VerifiedRelation,
    ) -> Dict[str, float]:
        evidence = max(1.0, float(max(0, int(relation.evidence_count))))
        counterexamples = float(max(0, int(relation.counterexample_count)))
        return summarize_event_memory_credit(
            (
                {
                    "responsibility": max(0.0, float(relation.prediction_gain)) * min(1.0, evidence / 4.0),
                    "confidence": float(relation.confidence),
                    "longevity": evidence / (evidence + counterexamples + 1.0),
                },
            )
        )
