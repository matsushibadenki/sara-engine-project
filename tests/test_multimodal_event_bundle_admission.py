from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache
from sara_engine.memory.multimodal_event_bundle_admission import (
    build_multimodal_event_state_candidate,
)
from sara_engine.multimodal.synesthetic_binding import SparseTemporalBinder


def _bundle():
    binder = SparseTemporalBinder(window_ms=32.0)
    events = [
        binder.normalize_event(
            modality="language",
            timestamp_ms=2.0,
            source_id="language-hard",
            sparse_signature=[1, 2],
            confidence=0.9,
            label="hard",
            source_ref="fixture://hard",
        ),
        binder.normalize_event(
            modality="vision",
            timestamp_ms=10.0,
            source_id="vision-hard",
            sparse_signature=[11, 12],
            confidence=0.9,
            label="hard",
            source_ref="fixture://hard",
        ),
        binder.normalize_event(
            modality="audio",
            timestamp_ms=15.0,
            source_id="audio-hard",
            sparse_signature=[21, 22],
            confidence=0.9,
            label="hard",
            source_ref="fixture://hard",
        ),
        binder.normalize_event(
            modality="tactile",
            timestamp_ms=20.0,
            source_id="tactile-hard",
            sparse_signature=[31, 32],
            confidence=0.9,
            label="hard",
            source_ref="fixture://hard",
        ),
    ]
    return binder.bundle_events(events)[0]


def test_verified_multimodal_bundle_promotes_into_event_state_candidate():
    result = build_multimodal_event_state_candidate(_bundle(), time_segment=3)

    assert result.promotion_allowed is True
    assert result.promotion_decision == "promote_verified_multimodal_bundle"
    assert result.candidate.verified is True
    assert result.candidate.source_backed is True
    assert result.candidate.observed is True
    assert result.candidate.resonance_score > 0.0
    assert result.candidate.causal_predecessors


def test_verified_multimodal_bundle_can_enter_event_state_cache():
    result = build_multimodal_event_state_candidate(_bundle(), time_segment=3)
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")

    admission = cache.admit(result.candidate)

    assert admission.accepted is True
    retrieval = cache.retrieve(result.candidate.signature, own_latent_id=result.candidate.own_latent_id)
    assert retrieval.abstained is False
    assert retrieval.matches[0]["entry_id"] == result.candidate.entry_id


def test_bundle_without_source_backing_is_not_promoted():
    binder = SparseTemporalBinder(window_ms=32.0)
    events = [
        binder.normalize_event(
            modality="language",
            timestamp_ms=1.0,
            source_id="language-weak",
            sparse_signature=[1, 2],
            confidence=0.8,
            label="weak",
        ),
        binder.normalize_event(
            modality="vision",
            timestamp_ms=4.0,
            source_id="vision-weak",
            sparse_signature=[11, 12],
            confidence=0.8,
            label="weak",
        ),
    ]
    bundle = binder.bundle_events(events)[0]

    result = build_multimodal_event_state_candidate(bundle, time_segment=2)

    assert result.promotion_allowed is False
    assert result.promotion_decision == "freeze_material_source"
    assert result.candidate.verified is False
