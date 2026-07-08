import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_builder():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bot", "gap_materials_builder.py")
    )
    spec = importlib.util.spec_from_file_location("gap_materials_builder", module_path)
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
                "missing_material_types": ["transcript_segment"],
                "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
            },
            {
                "request_id": "fixture_counterexample_gap",
                "missing_material_types": ["counterexample"],
                "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
            },
            {
                "request_id": "fixture_repair_support_gap",
                "missing_material_types": ["repair_note"],
                "preferred_material_types": ["repair_note", "source_claim", "qa_pair"],
            },
            {
                "request_id": "fixture_revision_conflict_gap",
                "missing_material_types": ["revision_note"],
                "preferred_material_types": ["source_claim", "revision_note", "qa_pair"],
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_gap_materials_builder_generates_source_backed_gap_materials():
    builder = _load_builder()
    accepted_path = processed_data_path("autobot", "test_gap_materials_accepted.jsonl")
    targets_path = workspace_path("autobot", "test_gap_collection_targets.json")
    output_path = processed_data_path("autobot", "test_gap_materials.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_curriculum_manifest.jsonl")
    report_path = workspace_path("autobot", "test_gap_materials_builder_report.json")
    summary_path = workspace_path("autobot", "test_gap_materials_builder_summary.txt")
    _write_accepted(accepted_path)
    _write_targets(targets_path)

    exit_code = builder.main(
        [
            "--accepted-path",
            accepted_path,
            "--targets-path",
            targets_path,
            "--output-path",
            output_path,
            "--curriculum-path",
            curriculum_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["built_count"] == 4
    assert report["built_material_type_counts"]["transcript_segment"] == 1
    assert report["built_material_type_counts"]["counterexample"] == 1
    assert report["built_material_type_counts"]["repair_note"] == 1
    assert report["built_material_type_counts"]["revision_note"] == 1
    assert report["curriculum_distribution"]["repair"] >= 3
    assert report["curriculum_distribution"]["replay"] >= 1
    with open(output_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert {row["material_type"] for row in rows} == {
        "transcript_segment",
        "counterexample",
        "repair_note",
        "revision_note",
    }
    assert all(row["observed_only"] is True for row in rows)
    with open(curriculum_path, "r", encoding="utf-8") as handle:
        curriculum_rows = [json.loads(line) for line in handle if line.strip()]
    assert any(row["material_type"] == "counterexample" and row["curriculum_stage"] == "repair" for row in curriculum_rows)
    assert any(row["material_type"] == "transcript_segment" and row["curriculum_stage"] == "replay" for row in curriculum_rows)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Gap materials builder: PASS" in summary
    assert "Built: 4" in summary
    assert "Curriculum distribution:" in summary


def test_gap_materials_builder_blocks_flagged_requests():
    builder = _load_builder()
    accepted_path = processed_data_path("autobot", "test_gap_materials_blocked_accepted.jsonl")
    targets_path = workspace_path("autobot", "test_gap_collection_targets_blocked.json")
    output_path = processed_data_path("autobot", "test_gap_materials_blocked.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_curriculum_manifest_blocked.jsonl")
    report_path = workspace_path("autobot", "test_gap_materials_builder_blocked_report.json")
    summary_path = workspace_path("autobot", "test_gap_materials_builder_blocked_summary.txt")
    _write_accepted(accepted_path)
    _write_targets(targets_path)
    with open(targets_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["blocked_request_ids"] = ["fixture_counterexample_gap"]
    payload["blocked_request_missing_axes"] = {
        "fixture_counterexample_gap": ["source_lineage"]
    }
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    exit_code = builder.main(
        [
            "--accepted-path",
            accepted_path,
            "--targets-path",
            targets_path,
            "--output-path",
            output_path,
            "--curriculum-path",
            curriculum_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["built_count"] == 3
    assert report["blocked_request_count"] == 1
    assert report["blocked_request_ids"] == ["fixture_counterexample_gap"]
    assert report["blocked_request_missing_axes"]["fixture_counterexample_gap"] == [
        "source_lineage"
    ]
    assert report["skipped_material_type_counts"]["counterexample"] == 1
    with open(output_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert "counterexample" not in {row["material_type"] for row in rows}


def test_gap_materials_builder_clears_blocked_request_before_build():
    builder = _load_builder()
    accepted_path = processed_data_path("autobot", "test_gap_materials_clear_accepted.jsonl")
    targets_path = workspace_path("autobot", "test_gap_collection_targets_clear.json")
    output_path = processed_data_path("autobot", "test_gap_materials_clear.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_curriculum_manifest_clear.jsonl")
    report_path = workspace_path("autobot", "test_gap_materials_builder_clear_report.json")
    summary_path = workspace_path("autobot", "test_gap_materials_builder_clear_summary.txt")
    _write_accepted(accepted_path)
    _write_targets(targets_path)
    with open(targets_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["blocked_request_ids"] = ["fixture_counterexample_gap"]
    payload["blocked_request_missing_axes"] = {
        "fixture_counterexample_gap": []
    }
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    exit_code = builder.main(
        [
            "--accepted-path",
            accepted_path,
            "--targets-path",
            targets_path,
            "--output-path",
            output_path,
            "--curriculum-path",
            curriculum_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--clear-blocked-request-id",
            "fixture_counterexample_gap",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["built_count"] == 4
    assert report["blocked_request_count"] == 0
    with open(targets_path, "r", encoding="utf-8") as handle:
        updated_targets = json.load(handle)
    assert updated_targets["blocked_request_ids"] == []
