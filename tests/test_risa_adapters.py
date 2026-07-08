import importlib.util
import os
import sys
from types import SimpleNamespace


def _load_module(name: str, relative_path: str):
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            *relative_path.split("/"),
        )
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


candidate_module = _load_module("candidate_proposals_test", "sara_engine/ingest/candidate_proposals.py")
cache_module = _load_module("event_state_cache_test", "sara_engine/memory/event_state_cache.py")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sara_engine.risa import (  # type: ignore  # noqa: E402
    SARAAlignedRisaKernel,
    extract_verified_event_state_candidates,
    ingest_verified_surface_into_risa,
    observation_from_bundle_admission,
)

ProposalLineage = candidate_module.ProposalLineage
VerifiedRelation = candidate_module.VerifiedRelation
EventStateCandidate = cache_module.EventStateCandidate


def _verified_relation(record_id: str, source_event_id: str, target_event_id: str) -> VerifiedRelation:
    return VerifiedRelation(
        record_id=record_id,
        relation="predicts",
        source_event_id=source_event_id,
        target_event_id=target_event_id,
        delay_lower_ms=10,
        delay_upper_ms=40,
        confidence=0.9,
        lineage=ProposalLineage(
            source_ref="fixture://session-a",
            source_hash="hash-a",
            extractor_name="fixture",
            extractor_version="v1",
        ),
        evidence_count=3,
        counterexample_count=0,
        prediction_gain=0.4,
        verification="verified",
    )


def _bundle_result():
    candidate = EventStateCandidate(
        entry_id="bundle:fixture-hard",
        signature=(1, 2, 11, 12),
        source_ref="fixture://hard",
        source_revision="bundle-rev:fixture",
        time_segment=3,
        own_latent_id="bundle:fixture-hard",
        causal_predecessors=("language-hard", "vision-hard"),
        confidence=0.9,
        uncertainty=0.1,
        source_reliability=0.9,
        resonance_score=0.88,
        sequence_support_score=1.0,
        sequence_support_count=1,
        credit_score=0.81,
        credit_responsibility=0.8,
        credit_confidence=0.9,
        credit_longevity=1.0,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        event_cost=8,
    )
    return SimpleNamespace(
        candidate=candidate,
        promotion_allowed=True,
        promotion_decision="promote_verified_multimodal_bundle",
    )


def test_risa_adapter_ingests_verified_relation_surface_into_kernel() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    observations = ingest_verified_surface_into_risa(
        kernel,
        verified_relations=(
            _verified_relation("rel-1", "vision:cluster_a", "audio:cluster_shared"),
            _verified_relation("rel-2", "vision:cluster_b", "audio:cluster_shared"),
        ),
    )

    assert len(observations) == 2
    concept = kernel.state.graph.get_node("concept:shared_predicts_audio_cluster_shared")
    assert concept is not None
    assert kernel.state.concept_members["concept:shared_predicts_audio_cluster_shared"] == [
        "vision_cluster_a",
        "vision_cluster_b",
    ]


def test_risa_adapter_supports_bundle_admission_observation() -> None:
    result = _bundle_result()
    observation = observation_from_bundle_admission(result)

    assert result.promotion_allowed is True
    assert observation.action == "bind"
    assert observation.verified is True
    assert observation.source_ref.startswith("fixture://")


def test_risa_adapter_can_extract_event_state_candidates_from_cache_dicts() -> None:
    raw_entries = [
        EventStateCandidate(
            entry_id="entry-1",
            signature=(1, 2, 3),
            source_ref="fixture://entry",
            source_revision="rev-1",
            time_segment=7,
            own_latent_id="latent-1",
            causal_predecessors=("cause-a", "cause-b"),
            confidence=0.9,
            uncertainty=0.1,
            source_reliability=0.8,
            resonance_score=0.85,
            sequence_support_score=0.5,
            sequence_support_count=2,
            credit_score=0.7,
            credit_responsibility=0.6,
            credit_confidence=0.8,
            credit_longevity=0.9,
            metabolic_headroom=1.0,
            observed=True,
            source_backed=True,
            verified=True,
            event_cost=5,
        ).__dict__
    ]

    entries = extract_verified_event_state_candidates(raw_entries)

    assert len(entries) == 1
    assert entries[0].entry_id == "entry-1"
    assert entries[0].verified is True
    assert entries[0].causal_predecessors == ("cause-a", "cause-b")
