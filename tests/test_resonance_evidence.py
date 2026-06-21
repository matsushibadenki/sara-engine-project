from sara_engine.learning.resonance_evidence import build_resonance_evidence


def _reports():
    return {
        "reasoning_prior": {
            "schema": "reasoning-v1",
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
            "case_count": 6,
            "expected_match_count": 6,
        },
        "multimodal_binding": {
            "schema": "multimodal-v1",
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
            "resource_pressure": 0.4,
            "metrics": {
                "metabolic_budget_integrity": 1.0,
                "plasticity_reserve_integrity": 1.0,
            },
        },
    }


def test_resonance_evidence_builds_trusted_multi_report_signals():
    bundle = build_resonance_evidence(_reports())

    assert bundle.signals["source_backed"] is True
    assert bundle.signals["verifier_confidence"] == 1.0
    assert bundle.signals["metabolic_headroom"] == 0.6
    assert bundle.signals["contradiction"] == 0.0
    assert bundle.trace["trusted_source_count"] == 6


def test_resonance_evidence_rejects_missing_or_failed_reports():
    reports = _reports()
    reports["plan_verifier"]["passed"] = False
    reports.pop("own_latent")
    bundle = build_resonance_evidence(reports)

    assert bundle.signals["source_backed"] is False
    assert bundle.trace["trusted_source_count"] == 4


def test_resonance_evidence_exposes_verifier_contradiction_and_abstention():
    reports = _reports()
    reports["plan_verifier"]["expected_match_count"] = 2
    reports["reasoning_prior"]["metrics"]["external_event_missing_abstention"] = 0.0
    bundle = build_resonance_evidence(reports)

    assert bundle.signals["contradiction"] > 0.55
    assert bundle.signals["abstained"] is True
