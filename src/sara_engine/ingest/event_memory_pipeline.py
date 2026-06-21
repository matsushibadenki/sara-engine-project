from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    ) -> None:
        self.change_detector = change_detector or ScalarChangeDetector()
        self.temporal_eventizer = temporal_eventizer or TemporalEventizer()
        self.episode_segmenter = episode_segmenter or EpisodeSegmenter()
        self.sequence_miner = sequence_miner or FrequentSequenceMiner()
        self.synchrony_detector = synchrony_detector or SynchronyDetector()
        self.prediction_gain_estimator = prediction_gain_estimator or PredictionGainEstimator()
        self.verifier = verifier or ProposalVerifier()

    def ingest_streams(
        self,
        streams: Sequence[Mapping[str, Any]],
        *,
        source_ref: str,
        source_hash: str,
        candidate_events: Sequence[CandidateEvent] = (),
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

        relation_verifications = tuple(self.verifier.verify_relation(item) for item in candidate_relations)
        verified_relations = tuple(
            make_verified_relation(self._flatten_record(verification.promoted_record))
            for verification in relation_verifications
            if verification.promoted_record
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
