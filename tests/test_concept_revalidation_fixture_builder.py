import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_builder():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "build_concept_revalidation_fixture.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "build_concept_revalidation_fixture",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "schema": "sara-own-latent-manifest-row-v1",
            "manifest_id": "latent_manifest_000001",
            "material_hash": "hash-a",
            "material_type": "qa_pair",
            "latent_cluster_id": "latent_1",
            "sparse_signature": [1, 3, 5],
            "source_ref": "https://example.org/a",
            "quality_score": 0.9,
        },
        {
            "schema": "sara-own-latent-manifest-row-v1",
            "manifest_id": "latent_manifest_000002",
            "material_hash": "hash-b",
            "material_type": "source_claim",
            "latent_cluster_id": "latent_2",
            "sparse_signature": [2, 4, 6],
            "source_ref": "https://example.org/b",
            "quality_score": 0.88,
        },
        {
            "schema": "sara-own-latent-manifest-row-v1",
            "manifest_id": "latent_manifest_000003",
            "material_hash": "hash-c",
            "material_type": "contrastive_pair",
            "latent_cluster_id": "latent_3",
            "sparse_signature": [7, 9, 11],
            "source_ref": "https://example.org/c",
            "quality_score": 0.86,
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_concept_revalidation_fixture_builder_writes_mixed_cases():
    builder = _load_builder()
    manifest_path = processed_data_path("autobot", "test_concept_revalidation_manifest.jsonl")
    fixture_path = processed_data_path("benchmark_fixtures", "test_concept_revalidation_cases.jsonl")
    report_path = workspace_path("evaluation", "test_concept_revalidation_fixture_builder.json")
    summary_path = workspace_path("evaluation", "test_concept_revalidation_fixture_builder.txt")
    _write_manifest(manifest_path)

    exit_code = builder.main(
        [
            "--manifest-path",
            manifest_path,
            "--fixture-path",
            fixture_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--max-cases",
            "4",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["case_count"] >= 3
    assert report["case_type_counts"]["recoverable_revision_conflict"] >= 1
    assert report["case_type_counts"]["blocked_source_diversity"] >= 1
    assert report["manifest_material_type_counts"]["qa_pair"] >= 1
    assert report["next_actions"][0]["action"] == "collect_additional_distinct_sources"
    assert any(item["action"] == "add_negative_and_contrastive_materials" for item in report["next_actions"])
    assert report["expansion_plan"][0]["preferred_material_types"][0] == "source_claim"
    assert "transcript_segment" in report["expansion_plan"][0]["missing_material_types"]
    with open(fixture_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert any(row["expected_outcome"] == "admit" for row in rows)
    assert any(row["expected_outcome"] == "blocked" for row in rows)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Next actions:" in summary
    assert "Manifest material types:" in summary
    assert "Expansion plan:" in summary
    assert "collect_additional_distinct_sources" in summary
