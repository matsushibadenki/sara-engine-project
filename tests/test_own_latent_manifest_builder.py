import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_manifest_builder():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "own_latent_manifest_builder.py")
    )
    spec = importlib.util.spec_from_file_location("own_latent_manifest_builder", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_materials(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "accepted": True,
            "material_hash": "qa-hash",
            "material_type": "qa_pair",
            "prompt": "What is sparse event routing?",
            "answer": "Sparse event routing keeps runtime state bounded.",
            "source": "web",
            "source_url": "https://example.org/sparse-routing",
            "source_type": "official_docs",
            "quality_score": 0.9,
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
            "quality_score": 0.8,
            "language": "en",
            "license_hint": "operator_supplied",
            "compliance_level": "allow",
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_gap_materials(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "accepted": True,
            "observed_only": True,
            "material_hash": "transcript-hash",
            "material_type": "transcript_segment",
            "prompt": "Replay the supporting source segment.",
            "content": "Sparse event routing keeps runtime state bounded.",
            "source": "web",
            "source_url": "https://example.org/sparse-routing",
            "source_type": "official_docs",
            "quality_score": 0.9,
            "language": "en",
            "license_hint": "reference",
            "compliance_level": "allow",
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_fixture_feedback(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-concept-revalidation-fixture-builder-report-v1",
        "expansion_plan": [
            {
                "action": "collect_additional_distinct_sources",
                "case_type": "blocked_source_diversity",
                "priority": 5,
                "preferred_material_types": [
                    "source_claim",
                    "qa_pair",
                    "transcript_segment",
                ],
                "guidance": "Increase distinct source_ref coverage for repeated relation candidates.",
            },
            {
                "action": "add_negative_and_contrastive_materials",
                "case_type": "blocked_counterexample_pressure",
                "priority": 4,
                "preferred_material_types": [
                    "contrastive_pair",
                    "counterexample",
                    "qa_pair",
                ],
                "guidance": "Add negative and contrastive rows that can challenge over-generalized relations.",
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_own_latent_manifest_builder_writes_source_backed_manifest():
    builder = _load_manifest_builder()
    materials_path = processed_data_path("autobot", "test_latent_learning_materials.jsonl")
    manifest_path = processed_data_path("autobot", "test_latent_manifest.jsonl")
    report_path = workspace_path("evaluation", "test_own_latent_manifest_builder.json")
    summary_path = workspace_path("evaluation", "test_own_latent_manifest_builder.txt")
    fixture_feedback_path = workspace_path("evaluation", "test_concept_revalidation_fixture_builder.json")
    request_plan_path = workspace_path("autobot", "test_fixture_material_request_plan.json")
    _write_materials(materials_path)
    _write_fixture_feedback(fixture_feedback_path)

    exit_code = builder.main(
        [
            "--materials-path",
            materials_path,
            "--manifest-path",
            manifest_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--fixture-feedback-path",
            fixture_feedback_path,
            "--request-plan-path",
            request_plan_path,
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
    assert report["observed_only"] is True
    assert report["manifest_count"] == 2
    assert report["missing_source_ref_count"] == 0
    assert report["fixture_feedback_loaded"] is True
    assert report["fixture_material_coverage_gap_count"] == 2
    assert report["fixture_material_request_count"] == 2
    assert report["fixture_material_request_plan_path"].endswith("test_fixture_material_request_plan.json")
    assert report["fixture_expansion_plan"][0]["manifest_availability"]["source_claim"] == 1
    assert "transcript_segment" in report["fixture_expansion_plan"][0]["missing_material_types_now"]
    assert os.path.exists(request_plan_path)
    with open(request_plan_path, "r", encoding="utf-8") as handle:
        request_plan = json.load(handle)
    assert request_plan["request_source"] == "fixture_feedback"
    assert request_plan["request_count"] == 2
    with open(manifest_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert {row["material_hash"] for row in rows} == {"qa-hash", "claim-hash"}
    assert all(row["observed_only"] is True for row in rows)
    assert all(row["sparse_signature"] for row in rows)
    assert all(row["source_ref"] for row in rows)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Fixture feedback loaded: True" in summary
    assert "Fixture expansion alignment:" in summary


def test_own_latent_manifest_builder_falls_back_to_type_outputs():
    builder = _load_manifest_builder()
    materials_path = processed_data_path("autobot", "test_empty_latent_learning_materials.jsonl")
    type_output_path = processed_data_path("autobot", "qa_pairs.jsonl")
    gap_output_path = processed_data_path("autobot", "transcript_segments.jsonl")
    manifest_path = processed_data_path("autobot", "test_fallback_latent_manifest.jsonl")
    report_path = workspace_path("evaluation", "test_fallback_own_latent_manifest_builder.json")
    summary_path = workspace_path("evaluation", "test_fallback_own_latent_manifest_builder.txt")
    os.makedirs(os.path.dirname(materials_path), exist_ok=True)
    with open(materials_path, "w", encoding="utf-8"):
        pass
    _write_materials(type_output_path)
    _write_gap_materials(gap_output_path)

    report = builder.run_builder(
        materials_path=materials_path,
        manifest_path=manifest_path,
        report_path=report_path,
        summary_path=summary_path,
        fixture_feedback_path=workspace_path("evaluation", "missing_fixture_feedback.json"),
        width=256,
        max_events=12,
    )

    assert report["passed"] is True
    assert report["type_output_fallback_used"] is True
    assert report["manifest_count"] >= 3
