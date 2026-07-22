"""Typed ingestion records and verification helpers for Event Memory pipelines.

This package uses lazy exports to avoid importing unrelated heavy submodules
when callers only need a small subset such as candidate proposal types.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CandidateEvent": ("sara_engine.ingest.candidate_proposals", "CandidateEvent"),
    "CandidateRelation": ("sara_engine.ingest.candidate_proposals", "CandidateRelation"),
    "ConceptCrystalCandidate": ("sara_engine.ingest.candidate_proposals", "ConceptCrystalCandidate"),
    "ObservedEvent": ("sara_engine.ingest.candidate_proposals", "ObservedEvent"),
    "ProposalLineage": ("sara_engine.ingest.candidate_proposals", "ProposalLineage"),
    "VerifiedRelation": ("sara_engine.ingest.candidate_proposals", "VerifiedRelation"),
    "make_candidate_event": ("sara_engine.ingest.candidate_proposals", "make_candidate_event"),
    "make_candidate_relation": ("sara_engine.ingest.candidate_proposals", "make_candidate_relation"),
    "make_observed_event": ("sara_engine.ingest.candidate_proposals", "make_observed_event"),
    "make_proposal_lineage": ("sara_engine.ingest.candidate_proposals", "make_proposal_lineage"),
    "make_verified_relation": ("sara_engine.ingest.candidate_proposals", "make_verified_relation"),
    "ProposalLineageLedgerEntry": ("sara_engine.ingest.proposal_lineage", "ProposalLineageLedgerEntry"),
    "build_lineage_ledger_entry": ("sara_engine.ingest.proposal_lineage", "build_lineage_ledger_entry"),
    "ProposalVerificationResult": ("sara_engine.ingest.proposal_verifier", "ProposalVerificationResult"),
    "ProposalVerifier": ("sara_engine.ingest.proposal_verifier", "ProposalVerifier"),
    "EventMemoryIngestPipeline": ("sara_engine.ingest.event_memory_pipeline", "EventMemoryIngestPipeline"),
    "EventMemoryIngestResult": ("sara_engine.ingest.event_memory_pipeline", "EventMemoryIngestResult"),
    "ChangePoint": ("sara_engine.ingest.change_detection", "ChangePoint"),
    "ScalarChangeDetector": ("sara_engine.ingest.change_detection", "ScalarChangeDetector"),
    "EventizationTrace": ("sara_engine.ingest.temporal_eventizer", "EventizationTrace"),
    "TemporalEventizer": ("sara_engine.ingest.temporal_eventizer", "TemporalEventizer"),
    "BoundedEpisode": ("sara_engine.ingest.episode_segmentation", "BoundedEpisode"),
    "MultimodalEpisodeBridgeResult": (
        "sara_engine.ingest.episode_segmentation",
        "MultimodalEpisodeBridgeResult",
    ),
    "bridge_verified_bundle_to_episode": (
        "sara_engine.ingest.episode_segmentation",
        "bridge_verified_bundle_to_episode",
    ),
    "EpisodeSegmentationTrace": ("sara_engine.ingest.episode_segmentation", "EpisodeSegmentationTrace"),
    "EpisodeSegmenter": ("sara_engine.ingest.episode_segmentation", "EpisodeSegmenter"),
    "FrequentSequence": ("sara_engine.ingest.frequent_sequence", "FrequentSequence"),
    "FrequentSequenceTrace": ("sara_engine.ingest.frequent_sequence", "FrequentSequenceTrace"),
    "FrequentSequenceMiner": ("sara_engine.ingest.frequent_sequence", "FrequentSequenceMiner"),
    "PredictionGainEstimator": ("sara_engine.ingest.prediction_gain", "PredictionGainEstimator"),
    "PredictionGainTrace": ("sara_engine.ingest.prediction_gain", "PredictionGainTrace"),
    "SynchronyDetector": ("sara_engine.ingest.synchrony_detector", "SynchronyDetector"),
    "SynchronyTrace": ("sara_engine.ingest.synchrony_detector", "SynchronyTrace"),
    "RelationStabilityAssessor": ("sara_engine.ingest.relation_stability", "RelationStabilityAssessor"),
    "StableRelationSummary": ("sara_engine.ingest.relation_stability", "StableRelationSummary"),
    "SequenceRelationSupport": ("sara_engine.ingest.sequence_relation_support", "SequenceRelationSupport"),
    "summarize_sequence_relation_support": (
        "sara_engine.ingest.sequence_relation_support",
        "summarize_sequence_relation_support",
    ),
    "ConceptAuditResult": ("sara_engine.ingest.concept_crystallization_guard", "ConceptAuditResult"),
    "ConceptCrystallizationGuard": (
        "sara_engine.ingest.concept_crystallization_guard",
        "ConceptCrystallizationGuard",
    ),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sara_engine.ingest' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
