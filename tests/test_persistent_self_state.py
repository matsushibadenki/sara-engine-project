from sara_engine.dynamics import (
    concept_self_state_alignment,
    PersistentSelfStateController,
    relation_self_state_alignment,
    SparseInternalPredictor,
    evaluate_persistent_self_state,
    stable_self_state_id,
)
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)


def test_persistent_self_state_keeps_activity_without_external_input():
    controller = PersistentSelfStateController(core_event_ids=(101, 202))
    controller.step(external_event_ids=(101,))
    idle_a = controller.step()
    idle_b = controller.step()

    assert idle_a["current_active_ids"]
    assert idle_b["current_active_ids"]
    assert idle_a["idle_self_state_ok"] is True
    assert 101 in idle_b["self_state_ids"] or 202 in idle_b["self_state_ids"]


def test_persistent_self_state_uses_event_memory_reactivation_hints():
    controller = PersistentSelfStateController(core_event_ids=(101,))
    memory_id = stable_self_state_id("episodic-memory-anchor")

    result = controller.step(
        reactivation_hints=(
            {
                "entry_id": "episodic-memory-anchor",
                "activation": 0.95,
                "mutates_durable_state": False,
            },
        )
    )

    assert memory_id in result["memory_event_ids"]
    assert result["current_active_ids"]


def test_sparse_internal_predictor_learns_and_recalls_transition():
    predictor = SparseInternalPredictor(max_links=8)
    predictor.observe((11,), (22,))
    predictor.observe((11,), (22,))
    predictor.observe((11,), (33,))

    predicted = predictor.predict((11,), limit=2)

    assert predicted[0] == 22
    assert 33 in predicted


def test_persistent_self_state_internal_prediction_supports_next_state():
    controller = PersistentSelfStateController(core_event_ids=(101,))
    controller.step(external_event_ids=(101,))
    controller.step(external_event_ids=(202,))
    controller.step(external_event_ids=(101,))
    result = controller.step()

    assert 202 in result["predicted_event_ids"]
    assert result["current_active_ids"]


def test_persistent_self_state_evaluation_reports_observed_metrics():
    report = evaluate_persistent_self_state()

    assert report["observed_only"] is True
    assert report["metrics"]["persistent_self_state_idle_activity"] == 1.0
    assert report["metrics"]["persistent_self_state_continuity"] == 1.0
    assert report["metrics"]["persistent_self_state_memory_reactivation"] == 1.0
    assert report["metrics"]["persistent_self_state_internal_prediction"] == 1.0


def test_persistent_self_state_accepts_real_event_memory_reactivation_hints():
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1)
    cache.admit(
        EventStateCandidate.from_verified_evidence(
            verifier_id="test-persistent-self-state",
            evidence={"entry_id": "verified-anchor", "signature": [7, 11, 13]},
            entry_id="verified-anchor",
            signature=(7, 11, 13),
            source_ref="source:verified-anchor",
            time_segment=1,
            own_latent_id="latent:anchor",
            confidence=0.95,
            uncertainty=0.05,
            source_reliability=0.95,
            resonance_score=0.95,
            metabolic_headroom=0.8,
            observed=True,
            source_backed=True,
            verified=True,
        )
    )
    retrieval = cache.retrieve((7, 11, 13), own_latent_id="latent:anchor")
    controller = PersistentSelfStateController(core_event_ids=(101,))
    result = controller.step(reactivation_hints=retrieval.reactivation_hints)

    assert retrieval.reactivation_hints
    assert stable_self_state_id("verified-anchor") in result["memory_event_ids"]


def test_self_state_alignment_scores_relation_and_concept_keys():
    source_id = stable_self_state_id("vision:visual_cluster_018")
    target_id = stable_self_state_id("audio:audio_cluster_044")
    self_state_ids = (source_id, target_id, 999)

    relation_score = relation_self_state_alignment(
        "vision:visual_cluster_018",
        "audio:audio_cluster_044",
        self_state_ids,
    )
    concept_score = concept_self_state_alignment(
        "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
        self_state_ids,
    )

    assert relation_score == 1.0
    assert concept_score == 1.0
