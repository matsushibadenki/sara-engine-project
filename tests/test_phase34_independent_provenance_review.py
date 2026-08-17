from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase34_independent_provenance_review.py"
    spec = importlib.util.spec_from_file_location("phase34_independent_provenance_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline(module):
    return module.build_offline_review(
        module._read_jsonl(module.DEFAULT_RAW),
        module._read_jsonl(module.DEFAULT_MANIFEST),
        module._read_json(module.DEFAULT_CASE_PLAN),
        module._read_json(module.DEFAULT_PREREGISTRATION),
        module._read_json(module.DEFAULT_BENCHMARK),
    )


def test_offline_review_binds_all_sampled_materials_to_raw_and_manifest():
    module = _load_module()
    offline = _offline(module)

    assert offline["passed"] is True
    assert offline["metrics"] == {
        "sampled_material_count": 66,
        "fetched_authoritative_count": 60,
        "transcribed_excerpt_count": 6,
    }
    assert all(offline["checks"].values())


def test_automated_review_keeps_transcribed_excerpts_for_manual_review():
    module = _load_module()
    offline = _offline(module)
    observations = [
        {
            "retrieval_succeeded": True,
            "content_hash_matches": True,
            "response_body_hash_matches": True,
        }
        for _ in offline["fetched_rows"]
    ]

    report = module.build_report(offline, observations)

    assert report["automated_provenance_passed"] is True
    assert report["provenance_review_complete"] is False
    assert report["metrics"]["manual_review_required_count"] == 6
    assert len(report["manual_review_targets"]) == 6
    assert report["promotion_ready"] is False


def test_online_content_drift_fails_automated_provenance():
    module = _load_module()
    offline = _offline(module)
    observations = [
        {
            "retrieval_succeeded": True,
            "content_hash_matches": True,
            "response_body_hash_matches": True,
        }
        for _ in offline["fetched_rows"]
    ]
    observations[0]["content_hash_matches"] = False

    report = module.build_report(offline, observations)

    assert report["automated_provenance_passed"] is False
    assert report["online_checks"]["all_normalized_content_hashes_reproduced"] is False


def test_offline_content_tampering_is_detected():
    module = _load_module()
    raw = module._read_jsonl(module.DEFAULT_RAW)
    raw = copy.deepcopy(raw)
    raw[0]["content"] += " tampered"
    offline = module.build_offline_review(
        raw,
        module._read_jsonl(module.DEFAULT_MANIFEST),
        module._read_json(module.DEFAULT_CASE_PLAN),
        module._read_json(module.DEFAULT_PREREGISTRATION),
        module._read_json(module.DEFAULT_BENCHMARK),
    )

    assert offline["passed"] is False
    assert offline["checks"]["stored_content_hashes_recompute"] is False


def test_recorded_online_review_preserves_mutable_python_docs_drift():
    module = _load_module()
    report = module._read_json(module.DEFAULT_OUTPUT)

    assert report["offline_integrity_passed"] is True
    assert report["automated_provenance_passed"] is False
    assert report["provenance_review_complete"] is False
    assert report["stable_domains"] == ["www.rfc-editor.org"]
    assert report["drifted_domains"] == ["docs.python.org"]
    assert report["domain_metrics"]["docs.python.org"]["content_hash_drift_count"] == 30
    assert report["domain_metrics"]["www.rfc-editor.org"]["content_hash_match_count"] == 30
    assert report["metrics"]["manual_review_required_count"] == 6
