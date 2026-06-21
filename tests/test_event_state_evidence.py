from sara_engine.memory.event_state_evidence import build_event_state_candidate


def _reports():
    return {
        "reasoning_prior": {
            "schema": "reason-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {
                "logic_to_state_consistency": 1.0,
                "external_event_missing_abstention": 1.0,
            },
        },
        "plan_verifier": {
            "schema": "plan-v1",
            "passed": True,
            "observed_only": True,
            "case_count": 2,
            "expected_match_count": 2,
        },
        "multimodal_binding": {
            "schema": "multi-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {
                "cross_modal_link_precision": 1.0,
                "route_traceability": 1.0,
                "missing_modality_abstention_integrity": 1.0,
            },
        },
        "dendritic_feedback": {
            "schema": "dendritic-v1",
            "passed": True,
            "observed_only": True,
            "gated_precision": 0.9,
        },
        "own_latent": {
            "schema": "latent-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {"own_latent_sample_efficiency_ok": 1.0},
        },
        "metabolic_budget": {
            "schema": "metabolic-v1",
            "observed_only": True,
            "resource_pressure": 0.2,
            "metrics": {
                "metabolic_budget_integrity": 1.0,
                "plasticity_reserve_integrity": 1.0,
            },
        },
    }


def _material():
    return {
        "schema": "sara-own-latent-manifest-row-v1",
        "manifest_id": "latent_manifest_1",
        "material_hash": "hash-1",
        "source_ref": "https://example.org/source",
        "observed_only": True,
        "compliance_level": "allow",
        "quality_score": 0.9,
        "latent_cluster_id": "latent_1",
        "sparse_signature": [3, 5, 7],
        "event_cost": 3,
    }


def test_event_state_evidence_promotes_healthy_managed_bundle():
    result = build_event_state_candidate(_material(), _reports(), time_segment=4)

    assert result.promotion_allowed is True
    assert result.promotion_decision == "promote_verified_event_state"
    assert result.candidate.source_revision == "hash-1"
    assert result.candidate.verified is True


def test_event_state_evidence_freezes_missing_report_and_predicted_material():
    reports = _reports()
    reports["own_latent"] = {}
    missing = build_event_state_candidate(_material(), reports, time_segment=4)
    predicted_material = _material()
    predicted_material["observed_only"] = False
    predicted = build_event_state_candidate(
        predicted_material,
        _reports(),
        time_segment=4,
    )

    assert missing.promotion_allowed is False
    assert missing.promotion_decision == "freeze_unverified_source"
    assert predicted.promotion_allowed is False
    assert predicted.promotion_decision == "freeze_predicted_material"
