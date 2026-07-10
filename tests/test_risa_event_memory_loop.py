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


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sara_engine.risa import (  # type: ignore  # noqa: E402
    SARAAlignedRisaKernel,
    ingest_event_memory_cycle_into_risa,
)

candidate_module = _load_module("candidate_proposals_loop_test", "sara_engine/ingest/candidate_proposals.py")
cache_module = _load_module("event_state_cache_loop_test", "sara_engine/memory/event_state_cache.py")

ProposalLineage = candidate_module.ProposalLineage
VerifiedRelation = candidate_module.VerifiedRelation
EventStateCandidate = cache_module.EventStateCandidate
VerifiedHierarchicalEventStateCache = cache_module.VerifiedHierarchicalEventStateCache


def _verified_relation(record_id: str, source_event_id: str, target_event_id: str) -> VerifiedRelation:
    return VerifiedRelation(
        record_id=record_id,
        relation="predicts",
        source_event_id=source_event_id,
        target_event_id=target_event_id,
        delay_lower_ms=10,
        delay_upper_ms=20,
        confidence=0.92,
        lineage=ProposalLineage(
            source_ref="fixture://session-loop",
            source_hash="hash-loop",
            extractor_name="fixture",
            extractor_version="v1",
        ),
        evidence_count=3,
        counterexample_count=0,
        prediction_gain=0.45,
        verification="verified",
    )


def _candidate(entry_id: str, own_latent_id: str) -> EventStateCandidate:
    return EventStateCandidate(
        entry_id=entry_id,
        signature=(1, 3, 5, 7),
        source_ref="fixture://memory",
        source_revision="rev-memory",
        time_segment=12,
        own_latent_id=own_latent_id,
        causal_predecessors=("cause-a", "cause-b"),
        confidence=0.92,
        uncertainty=0.08,
        source_reliability=0.87,
        resonance_score=0.9,
        sequence_support_score=0.6,
        sequence_support_count=2,
        credit_score=0.8,
        credit_responsibility=0.78,
        credit_confidence=0.82,
        credit_longevity=0.93,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        event_cost=6,
    )


def test_risa_loop_ingests_ingest_result_and_cache_reactivation_surface() -> None:
    kernel = SARAAlignedRisaKernel(min_support=1, min_distinct_actors=1)
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1)
    candidate = _candidate("entry-1", "latent:memory")
    admission = cache.admit(candidate)
    assert admission.accepted is True
    retrieval = cache.retrieve(candidate.signature, own_latent_id="latent:memory", now_segment=12)
    assert retrieval.abstained is False

    ingest_result = SimpleNamespace(
        verified_relations=(
            _verified_relation("rel-1", "vision:cluster_a", "audio:cluster_shared"),
            _verified_relation("rel-2", "vision:cluster_b", "audio:cluster_shared"),
        ),
        multimodal_bundle_admissions=(),
    )

    observations = ingest_event_memory_cycle_into_risa(
        kernel,
        ingest_result=ingest_result,
        cache=cache,
        retrieval_result=retrieval,
        now_segment=12,
    )

    assert len(observations) >= 4
    assert kernel.state.graph.get_node("concept:shared_predicts_audio_cluster_shared") is not None
    assert kernel.state.graph.get_node("concept:shared_stabilize_cause-a") is not None
    reactivation_concept = kernel.state.graph.get_node("concept:shared_reactivate_entry-1")
    assert reactivation_concept is not None
    assert reactivation_concept.dormant is False


def test_risa_loop_can_run_without_cache_surface() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    ingest_result = SimpleNamespace(
        verified_relations=(
            _verified_relation("rel-1", "vision:cluster_a", "audio:cluster_shared"),
            _verified_relation("rel-2", "vision:cluster_b", "audio:cluster_shared"),
        ),
        multimodal_bundle_admissions=(),
    )

    observations = ingest_event_memory_cycle_into_risa(
        kernel,
        ingest_result=ingest_result,
    )

    assert len(observations) == 2
    assert kernel.state.graph.get_node("concept:shared_predicts_audio_cluster_shared") is not None
