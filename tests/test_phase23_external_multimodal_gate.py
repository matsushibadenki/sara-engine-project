from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase23_external_multimodal_gate.py"
    spec = importlib.util.spec_from_file_location("phase23_external_multimodal_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(case_id, decision, domain, evidence):
    return {
        "schema": "sara-phase23-independent-multimodal-row-v1",
        "case_id": case_id,
        "source_ref": f"operator:{domain}:{case_id}",
        "source_hash": f"sha256-{case_id}",
        "source_revision": "recording-v1",
        "source_domain": domain,
        "collection_time": "2026-07-21T00:00:00Z",
        "license_hint": "operator-owned",
        "near_duplicate_signature": f"signature-{case_id}",
        "evidence_scope": "independent_external",
        "observed_only": True,
        "compliance_level": "allow",
        "expected_modalities": ["audio", "vision"],
        "expected_decision": decision,
        "evidence": evidence,
    }


def _evidence(case_id, claim_a="impact", claim_b="impact", audio_time=18.0, include_audio=True):
    rows = [
        {
            "modality": "vision",
            "label": "contact_motion",
            "claim_key": claim_a,
            "timestamp_ms": 10.0,
            "source_ref": f"operator:camera:{case_id}",
            "source_hash": f"sha256-camera-{case_id}",
        }
    ]
    if include_audio:
        rows.append(
            {
                "modality": "audio",
                "label": "impact_sound",
                "claim_key": claim_b,
                "timestamp_ms": audio_time,
                "source_ref": f"operator:microphone:{case_id}",
                "source_hash": f"sha256-microphone-{case_id}",
            }
        )
    return rows


def _valid_rows():
    return [
        _row("verified-a", "verify_cross_modal_structure", "session-a", _evidence("verified-a")),
        _row("verified-b", "verify_cross_modal_structure", "session-b", _evidence("verified-b")),
        _row(
            "missing-a",
            "provisional_missing_modality_prediction",
            "session-a",
            _evidence("missing-a", include_audio=False),
        ),
        _row(
            "contradiction-b",
            "abstain_cross_modal_contradiction",
            "session-b",
            _evidence("contradiction-b", claim_b="no-impact"),
        ),
        _row(
            "delayed-a",
            "abstain_temporal_misalignment",
            "session-a",
            _evidence("delayed-a", audio_time=80.0),
        ),
    ]


def test_phase23_external_gate_accepts_independent_decision_coverage():
    module = _load_module()
    report = module.build_report(_valid_rows())

    assert report["passed"] is True
    assert report["promotion_allowed"] is True
    assert report["metrics"]["decision_accuracy"] == 1.0
    assert report["checks"]["event_memory_admission_integrity"] is True


def test_phase23_external_gate_rejects_duplicate_and_fixture_evidence():
    module = _load_module()
    rows = deepcopy(_valid_rows())
    rows[1]["near_duplicate_signature"] = rows[0]["near_duplicate_signature"]
    rows[1]["evidence"][0]["source_ref"] = "fixture:leaked"

    report = module.build_report(rows)

    assert report["promotion_allowed"] is False
    assert report["checks"]["unique_near_duplicate_signatures"] is False
    assert report["checks"]["fixture_and_generated_refs_rejected"] is False


def test_phase23_external_gate_reports_malformed_timestamps_without_crashing():
    module = _load_module()
    rows = _valid_rows()
    rows[0]["evidence"][0]["timestamp_ms"] = "not-a-timestamp"

    report = module.build_report(rows)

    assert report["promotion_allowed"] is False
    assert report["checks"]["multimodal_evidence_shape"] is False


def test_phase23_external_gate_handles_missing_manifest():
    module = _load_module()

    assert module._load("/nonexistent/phase23.jsonl") == []
    assert module.build_report([])["promotion_allowed"] is False
