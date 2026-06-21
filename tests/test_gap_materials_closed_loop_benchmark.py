import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "gap_materials_closed_loop_benchmark.py")
    )
    spec = importlib.util.spec_from_file_location("gap_materials_closed_loop_benchmark", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_accepted(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "accepted": True,
            "material_hash": "qa-hash",
            "material_type": "qa_pair",
            "prompt": "What does the source say?",
            "answer": "Sparse event routing keeps runtime state bounded.",
            "source": "web",
            "source_url": "https://example.org/sparse-routing",
            "source_type": "official_docs",
            "source_domain": "example.org",
            "source_text": "Sparse event routing keeps runtime state bounded. It helps retrieval stay source backed.",
            "quality_score": 0.9,
            "language": "en",
            "license_hint": "reference",
            "compliance_level": "allow",
        },
        {
            "accepted": True,
            "material_hash": "contrastive-hash",
            "material_type": "contrastive_pair",
            "prompt": "Pick the supported claim.",
            "answer": "Sparse event routing keeps runtime state bounded.",
            "near_miss": "Sparse event routing always needs dense matrix scans.",
            "source": "web",
            "source_url": "https://example.org/sparse-routing",
            "source_type": "official_docs",
            "source_domain": "example.org",
            "source_text": "Sparse event routing keeps runtime state bounded. It helps retrieval stay source backed.",
            "quality_score": 0.88,
            "language": "en",
            "license_hint": "reference",
            "compliance_level": "allow",
        },
        {
            "accepted": True,
            "material_hash": "claim-hash",
            "material_type": "source_claim",
            "content": "Local plasticity updates from nearby events.",
            "source": "local",
            "source_path": "data/raw/local_plasticity.txt",
            "source_type": "offline_batch",
            "source_domain": "local",
            "source_text": "Local plasticity updates from nearby events. Replay can preserve useful traces.",
            "quality_score": 0.8,
            "language": "en",
            "license_hint": "operator_supplied",
            "compliance_level": "allow",
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_targets(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-autobot-collection-targets-v1",
        "target_count": 4,
        "targets": [
            {
                "request_id": "fixture_source_diversity_gap",
                "evaluation_gaps": ["retrieval_grounding"],
                "missing_material_types": ["transcript_segment"],
                "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
            },
            {
                "request_id": "fixture_counterexample_gap",
                "evaluation_gaps": ["negative_control", "contrastive_control"],
                "missing_material_types": ["counterexample"],
                "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
            },
            {
                "request_id": "fixture_repair_support_gap",
                "evaluation_gaps": ["retrieval_grounding"],
                "missing_material_types": ["repair_note"],
                "preferred_material_types": ["repair_note", "source_claim", "qa_pair"],
            },
            {
                "request_id": "fixture_revision_conflict_gap",
                "evaluation_gaps": ["retrieval_grounding"],
                "missing_material_types": ["revision_note"],
                "preferred_material_types": ["source_claim", "revision_note", "qa_pair"],
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_fixture_feedback(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-concept-revalidation-fixture-builder-report-v1",
        "expansion_plan": [
            {
                "action": "collect_additional_distinct_sources",
                "case_type": "blocked_source_diversity",
                "priority": 5,
                "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
                "guidance": "Increase distinct source_ref coverage.",
            },
            {
                "action": "add_negative_and_contrastive_materials",
                "case_type": "blocked_counterexample_pressure",
                "priority": 4,
                "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                "guidance": "Add negative and contrastive rows.",
            },
            {
                "action": "manual_review_high_stall_candidates",
                "case_type": "blocked_attempt_budget",
                "priority": 3,
                "preferred_material_types": ["repair_note", "source_claim", "qa_pair"],
                "guidance": "Prepare support rows.",
            },
            {
                "action": "resolve_source_revision_conflicts",
                "case_type": "recoverable_revision_conflict",
                "priority": 2,
                "preferred_material_types": ["source_claim", "revision_note", "qa_pair"],
                "guidance": "Collect reconciled revisions.",
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_request_plan(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-autobot-material-request-plan-v1",
        "request_source": "fixture_feedback",
        "request_count": 4,
        "requests": [
            {"request_id": "fixture_source_diversity_gap"},
            {"request_id": "fixture_counterexample_gap"},
            {"request_id": "fixture_repair_support_gap"},
            {"request_id": "fixture_revision_conflict_gap"},
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_gap_materials_closed_loop_benchmark_reduces_coverage_gaps():
    module = _load_module()
    accepted_path = processed_data_path("autobot", "test_closed_loop_accepted.jsonl")
    targets_path = workspace_path("autobot", "test_closed_loop_targets.json")
    fixture_feedback_path = workspace_path("evaluation", "test_closed_loop_fixture_feedback.json")
    request_plan_path = workspace_path("autobot", "test_closed_loop_request_plan.json")
    report_path = workspace_path("evaluation", "test_gap_materials_closed_loop_benchmark.json")
    summary_path = workspace_path("evaluation", "test_gap_materials_closed_loop_benchmark.txt")
    _write_accepted(accepted_path)
    _write_targets(targets_path)
    _write_fixture_feedback(fixture_feedback_path)
    _write_request_plan(request_plan_path)

    exit_code = module.main(
        [
            "--accepted-path",
            accepted_path,
            "--targets-path",
            targets_path,
            "--fixture-feedback-path",
            fixture_feedback_path,
            "--request-plan-path",
            request_plan_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--width",
            "256",
            "--max-events",
            "12",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["baseline_fixture_material_coverage_gap_count"] == 4
    assert report["augmented_fixture_material_coverage_gap_count"] == 0
    assert report["coverage_gap_reduction"] == 4
    assert report["gap_material_built_count"] == 4
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Gap materials closed loop: PASS" in summary
    assert "Gap reduction: 4" in summary
