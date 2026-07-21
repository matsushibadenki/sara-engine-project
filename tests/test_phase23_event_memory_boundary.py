from __future__ import annotations

from sara_engine.memory.multimodal_event_bundle_admission import build_multimodal_event_state_candidate
from sara_engine.multimodal.structural_verification import ModalityEvidence, MultimodalStructuralVerifier
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


def _decision(bundle, *, contradiction: bool = False):
    return MultimodalStructuralVerifier().verify(
        (
            ModalityEvidence(
                modality=item.modality,
                label=("conflict" if contradiction and item.modality == "audio" else item.label),
                timestamp_ms=item.timestamp_ms,
                source_ref=item.source_ref,
                observed=item.observed,
                confidence=item.confidence,
            )
            for item in bundle.child_records
        ),
        expected_modalities=bundle.modality_ids,
    )


def test_verified_structural_decision_can_reach_event_memory_boundary():
    bundle = _bundle()
    result = build_multimodal_event_state_candidate(bundle, structural_decision=_decision(bundle))

    assert result.promotion_allowed is True
    assert result.trace["structural_decision"] == "verify_cross_modal_structure"


def test_non_verified_structural_decision_is_frozen_before_event_memory():
    bundle = _bundle()
    result = build_multimodal_event_state_candidate(
        bundle, structural_decision=_decision(bundle, contradiction=True)
    )

    assert result.promotion_allowed is False
    assert result.promotion_decision == "freeze_structural_fusion_abstain_cross_modal_contradiction"
