from sara_engine.ingest import (
    ProposalVerifier,
    build_lineage_ledger_entry,
    make_candidate_event,
    make_candidate_relation,
    make_observed_event,
)


def test_make_observed_event_preserves_observed_type_and_lineage():
    event = make_observed_event(
        {
            "record_id": "obs-1",
            "modality": "audio",
            "local_time_ms": 120,
            "label": "onset_cluster_01",
            "confidence": 0.9,
            "sparse_signature": [1, 2, 2, 5],
            "lineage": {
                "source_ref": "session-1",
                "source_hash": "abc123",
                "extractor_name": "change_detection",
                "extractor_version": "v1",
            },
        }
    )

    payload = event.to_dict()
    assert payload["record_type"] == "observed_event"
    assert payload["verification"] == "observed"
    assert payload["lineage"]["source_hash"] == "abc123"
    assert payload["sparse_signature"] == [1, 2, 5]


def test_event_verifier_accepts_candidate_event_with_enough_support():
    candidate = make_candidate_event(
        {
            "record_id": "cand-1",
            "modality": "vision",
            "label": "visual_cluster_018",
            "local_time_ms": 330,
            "confidence": 0.88,
            "source_ref": "session-1",
            "source_hash": "hash-1",
            "extractor_name": "candidate_proposals",
            "extractor_version": "v1",
            "evidence_count": 5,
            "counterexample_count": 1,
            "prediction_gain": 0.02,
            "sparse_signature": [4, 8, 15],
        }
    )

    result = ProposalVerifier().verify_event(candidate)

    assert result.accepted is True
    assert result.decision == "accept_candidate_event"
    assert result.record_type == "candidate_event"
    assert result.promoted_record is None


def test_relation_verifier_promotes_only_when_prediction_gain_is_sufficient():
    candidate = make_candidate_relation(
        {
            "record_id": "rel-1",
            "relation": "predicts",
            "source_event_id": "visual_cluster_018",
            "target_event_id": "audio_cluster_044",
            "delay_lower_ms": 80,
            "delay_upper_ms": 180,
            "confidence": 0.9,
            "source_ref": "session-2",
            "source_hash": "hash-2",
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": 7,
            "counterexample_count": 1,
            "prediction_gain": 0.21,
        }
    )

    result = ProposalVerifier().verify_relation(candidate)

    assert result.accepted is True
    assert result.decision == "promote_verified_relation"
    assert result.promoted_record is not None
    assert result.promoted_record["record_type"] == "verified_relation"
    assert result.promoted_record["verification"] == "provisional"


def test_relation_verifier_rejects_low_gain_candidate_even_with_confidence():
    candidate = make_candidate_relation(
        {
            "record_id": "rel-weak",
            "relation": "predicts",
            "source_event_id": "a",
            "target_event_id": "b",
            "delay_lower_ms": 0,
            "delay_upper_ms": 50,
            "confidence": 0.95,
            "source_ref": "session-3",
            "source_hash": "hash-3",
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": 12,
            "counterexample_count": 2,
            "prediction_gain": 0.01,
        }
    )

    result = ProposalVerifier().verify_relation(candidate)

    assert result.accepted is False
    assert result.decision == "reject_low_prediction_gain"
    assert result.promoted_record is None


def test_relation_verifier_can_use_self_state_alignment_as_small_bonus():
    candidate = make_candidate_relation(
        {
            "record_id": "rel-bonus",
            "relation": "predicts",
            "source_event_id": "a",
            "target_event_id": "b",
            "delay_lower_ms": 0,
            "delay_upper_ms": 50,
            "confidence": 0.95,
            "source_ref": "session-3",
            "source_hash": "hash-3",
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": 12,
            "counterexample_count": 1,
            "prediction_gain": 0.01,
        }
    )

    verifier = ProposalVerifier(min_prediction_gain=0.05)
    blocked = verifier.verify_relation(candidate)
    recovered = verifier.verify_relation_with_self_state(candidate, self_state_alignment=1.0)

    assert blocked.accepted is False
    assert recovered.accepted is True
    assert recovered.trace["effective_prediction_gain"] >= 0.05


def test_lineage_ledger_entry_keeps_ann_proposal_metadata_separate():
    entry = build_lineage_ledger_entry(
        {
            "record_id": "cand-2",
            "record_type": "candidate_event",
            "source_ref": "video-1",
            "source_hash": "hash-video-1",
            "extractor_name": "candidate_proposals",
            "extractor_version": "v2",
            "parent_ids": ["obs-1", "obs-2"],
            "observed_anchor_ids": ["obs-1"],
            "proposal_model": "local-ann-detector",
            "proposal_config_hash": "cfg-1",
        }
    )

    payload = entry.to_dict()
    assert payload["record_type"] == "candidate_event"
    assert payload["proposal_model"] == "local-ann-detector"
    assert payload["observed_anchor_ids"] == ["obs-1"]
