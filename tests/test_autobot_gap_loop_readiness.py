import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_gap_loop_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bot", "run_gap_loop.py")
    )
    spec = importlib.util.spec_from_file_location("run_gap_loop_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_readiness_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "autobot_gap_loop_readiness.py")
    )
    spec = importlib.util.spec_from_file_location("autobot_gap_loop_readiness", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_records(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "source": "hot_inbox",
            "record_text": (
                "Sparse event routing is a CPU first SNN technique. "
                "It keeps runtime state bounded and supports source backed retrieval."
            ),
            "meta": {
                "quality": 0.9,
                "source_type": "offline_batch",
                "path": "data/raw/offline_batch_inbox/sparse.txt",
                "language": "en",
                "license_hint": "operator_supplied",
                "compliance_level": "allow",
            },
            "ts": "2026-06-02T00:00:00",
        },
        {
            "source": "web",
            "record_text": (
                "Local plasticity means synapses update from nearby events. "
                "Then replay can preserve useful traces without dense backpropagation."
            ),
            "meta": {
                "quality": 0.82,
                "source_type": "official_docs",
                "url": "https://example.org/local-plasticity",
                "language": "en",
                "license_hint": "reference",
                "compliance_level": "allow",
            },
            "ts": "2026-06-02T00:01:00",
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_fixture_request_plan(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-autobot-material-request-plan-v1",
        "request_count": 2,
        "request_source": "fixture_feedback",
        "requests": [
            {
                "request_id": "fixture_counterexample_gap",
                "material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                "missing_material_types": ["counterexample"],
                "evaluation_gaps": ["negative_control", "contrastive_control"],
                "priority": 0.8,
                "reason": "fixture counterexample pressure gap",
            },
            {
                "request_id": "fixture_source_diversity_gap",
                "material_types": ["source_claim", "qa_pair", "transcript_segment"],
                "missing_material_types": ["transcript_segment"],
                "evaluation_gaps": ["retrieval_grounding"],
                "priority": 1.0,
                "reason": "fixture source diversity gap",
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_autobot_gap_loop_readiness_reports_pass_on_managed_gap_cycle():
    gap_loop = _load_gap_loop_module()
    readiness = _load_readiness_module()
    records_path = processed_data_path("autobot", "test_gap_readiness_records.jsonl")
    fixture_request_plan_path = workspace_path("autobot", "test_gap_readiness_fixture_request_plan.json")
    accepted_path = processed_data_path("autobot", "test_gap_readiness_learning_materials.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_readiness_curriculum_manifest.jsonl")
    collection_targets_path = workspace_path("autobot", "test_gap_readiness_collection_targets.json")
    gap_output_path = processed_data_path("autobot", "test_gap_readiness_gap_materials.jsonl")
    gap_curriculum_path = processed_data_path("autobot", "test_gap_readiness_gap_curriculum.jsonl")
    queue_path = workspace_path("autobot", "test_gap_readiness_train_queue.json")
    loop_report_path = workspace_path("autobot", "test_gap_readiness_loop_report.json")
    loop_summary_path = workspace_path("autobot", "test_gap_readiness_loop_summary.txt")
    readiness_report_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness.json")
    readiness_summary_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness.txt")
    _write_records(records_path)
    _write_fixture_request_plan(fixture_request_plan_path)

    loop_report = gap_loop.run_gap_loop(
        records_path=records_path,
        accepted_path=accepted_path,
        curriculum_path=curriculum_path,
        fixture_request_plan_path=fixture_request_plan_path,
        collection_targets_path=collection_targets_path,
        gap_output_path=gap_output_path,
        gap_curriculum_path=gap_curriculum_path,
        queue_path=queue_path,
        report_path=loop_report_path,
        summary_path=loop_summary_path,
        evaluation_gaps=("negative_control",),
    )
    assert loop_report["passed"] is True

    report = readiness.run_readiness(
        loop_report_path=loop_report_path,
        collection_targets_path=collection_targets_path,
        report_path=readiness_report_path,
        summary_path=readiness_summary_path,
        min_accepted_count=4,
        min_gap_build_coverage=1.0,
    )

    assert report["passed"] is True
    assert report["metrics"]["requested_slot_count"] == 2
    assert report["metrics"]["fixture_request_count"] == 2
    assert report["metrics"]["fixture_requested_slot_count"] == 2
    assert report["metrics"]["gap_material_built_count"] >= 2
    assert report["metrics"]["fixture_gap_material_built_count"] >= 2
    assert report["fixture_lane"]["requested_slots_by_request"]["fixture_counterexample_gap"] == 1
    assert report["fixture_lane"]["requested_slots_by_request"]["fixture_source_diversity_gap"] == 1
    assert report["fixture_lane"]["built_by_request"]["fixture_counterexample_gap"] == 1
    assert report["fixture_lane"]["built_by_request"]["fixture_source_diversity_gap"] == 1
    assert report["metrics"]["gap_curriculum_enqueued_count"] >= 2
    assert report["checks"]["gap_material_coverage_ready"]["passed"] is True
    assert report["checks"]["gap_enqueue_ready"]["passed"] is True
    assert report["checks"]["fixture_lane_coverage_ready"]["passed"] is True
    assert report["checks"]["fixture_source_lineage_ready"]["passed"] is True
    assert report["checks"]["fixture_source_isolation_ready"]["passed"] is True
    assert report["checks"]["fixture_collection_time_ready"]["passed"] is True
    assert report["metrics"]["fixture_candidate_source_domain_count"] >= 1
    assert report["metrics"]["fixture_accepted_source_domain_count"] >= 1
    assert report["metrics"]["fixture_collection_time_coverage"] == 1.0
    assert report["fixture_isolation_audit"]["missing_axes"] == []
    assert report["fixture_request_isolation_audit"]["fixture_counterexample_gap"]["missing_axes"] == []
    assert report["fixture_request_isolation_audit"]["fixture_source_diversity_gap"]["missing_axes"] == []
    assert report["fixture_request_isolation_audit"]["fixture_counterexample_gap"]["source_hash_coverage"] == 1.0
    assert report["fixture_request_isolation_audit"]["fixture_counterexample_gap"]["source_revision_coverage"] == 1.0
    with open(readiness_summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Autobot gap loop readiness: PASS" in summary
    assert "Fixture requests: 2" in summary
    assert "Fixture build coverage: 1.000" in summary
    assert "Fixture lane by request:" in summary
    assert "fixture_counterexample_gap: requested_slots=1, built=1, skipped=0" in summary
    assert "Fixture isolation axes:" in summary
    assert "Fixture isolation missing axes: none" in summary
    assert "Fixture isolation by request:" in summary
    assert "fixture_counterexample_gap: row_count=" in summary


def test_autobot_gap_loop_readiness_marks_blocked_fixture_requests():
    readiness = _load_readiness_module()
    collection_targets_path = workspace_path("autobot", "test_gap_readiness_blocked_collection_targets.json")
    report_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness_blocked.json")
    summary_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness_blocked.txt")

    payload = {
        "schema": "sara-autobot-collection-targets-v1",
        "target_count": 2,
        "blocked_request_ids": ["fixture_counterexample_gap"],
        "blocked_request_missing_axes": {
            "fixture_counterexample_gap": ["source_lineage"]
        },
        "targets": [
            {
                "request_id": "fixture_counterexample_gap",
                "missing_material_types": ["counterexample"],
                "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                "evaluation_gaps": ["negative_control"],
                "candidate_source_domains": ["example.org"],
            },
            {
                "request_id": "fixture_source_diversity_gap",
                "missing_material_types": ["transcript_segment"],
                "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
                "evaluation_gaps": ["retrieval_grounding"],
                "candidate_source_domains": ["example.org"],
            },
        ],
    }
    os.makedirs(os.path.dirname(collection_targets_path), exist_ok=True)
    with open(collection_targets_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    report = readiness.build_report(
        loop_report={"passed": True},
        dataset_report={"accepted_count": 4, "passed": True},
        gap_report={"built_count": 1, "skipped_count": 1, "curriculum_distribution": {"repair": 1}},
        enqueue_report={"enqueued_count": 1, "queue_pending": 1, "passed": True},
        collection_targets=payload,
        accepted_rows=[],
        gap_rows=[
            {
                "request_id": "fixture_source_diversity_gap",
                "source_domain": "example.org",
                "source_url": "https://example.org/a",
                "collection_time": "2026-06-02T00:00:00",
            }
        ],
        min_accepted_count=1,
        min_gap_build_coverage=0.5,
        input_paths={"collection_targets": collection_targets_path},
    )
    readiness.write_json(report_path, report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(readiness.summarize_report(report))

    assert report["fixture_execution_policy"]["blocked_request_count"] == 1
    assert report["fixture_execution_policy"]["blocked_request_ids"] == [
        "fixture_counterexample_gap"
    ]
    blocked_action = next(
        item
        for item in report["fixture_repair_actions"]
        if item["request_id"] == "fixture_counterexample_gap"
    )
    assert blocked_action["blocked_by_isolation_review"] is True
    assert blocked_action["blocked_missing_axes"] == ["source_lineage"]
    assert blocked_action["isolation_evidence"]["request"]["missing_axes"] == []
    assert blocked_action["isolation_evidence"]["global"]["available"] is False
    assert "Review fixture isolation block" in blocked_action["command"]
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Fixture execution blocked requests: fixture_counterexample_gap" in summary


def test_autobot_gap_loop_readiness_marks_clearable_blocked_fixture_requests():
    readiness = _load_readiness_module()
    payload = {
        "schema": "sara-autobot-collection-targets-v1",
        "target_count": 1,
        "blocked_request_ids": ["fixture_counterexample_gap"],
        "blocked_request_missing_axes": {
            "fixture_counterexample_gap": []
        },
        "targets": [
            {
                "request_id": "fixture_counterexample_gap",
                "missing_material_types": ["counterexample"],
                "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                "evaluation_gaps": ["negative_control"],
                "candidate_source_domains": ["example.org"],
            }
        ],
    }
    report = readiness.build_report(
        loop_report={"passed": True},
        dataset_report={"accepted_count": 4, "passed": True},
        gap_report={"built_count": 0, "skipped_count": 0, "curriculum_distribution": {}},
        enqueue_report={"enqueued_count": 0, "queue_pending": 0, "passed": True},
        collection_targets=payload,
        accepted_rows=[],
        gap_rows=[],
        min_accepted_count=1,
        min_gap_build_coverage=0.0,
        input_paths={"collection_targets": workspace_path("autobot", "dummy_targets.json")},
    )
    assert report["fixture_execution_policy"]["clearable_blocked_request_ids"] == [
        "fixture_counterexample_gap"
    ]
    blocked_action = report["fixture_repair_actions"][0]
    assert blocked_action["blocked_by_isolation_review"] is True
    assert blocked_action["clearable_after_review"] is True
    assert "--clear-blocked-request-id" in blocked_action["command"]


def test_autobot_gap_loop_readiness_keeps_clear_release_blocked_on_global_isolation_failure():
    readiness = _load_readiness_module()
    payload = {
        "target_count": 1,
        "blocked_request_ids": ["fixture_counterexample_gap"],
        "blocked_request_missing_axes": {"fixture_counterexample_gap": []},
        "targets": [
            {
                "request_id": "fixture_counterexample_gap",
                "missing_material_types": ["counterexample"],
                "candidate_source_domains": ["example.org"],
            }
        ],
    }
    report = readiness.build_report(
        loop_report={"passed": True},
        dataset_report={"accepted_count": 1},
        gap_report={"built_count": 0, "skipped_count": 0, "curriculum_distribution": {}},
        enqueue_report={"enqueued_count": 0, "queue_pending": 0},
        collection_targets=payload,
        accepted_rows=[],
        gap_rows=[],
        min_accepted_count=1,
        min_gap_build_coverage=0.0,
        isolation_audit={
            "passed": False,
            "checks": {"source_hash_isolated": False, "time_split_isolated": True},
            "metrics": {
                "shared_source_hashes": ["hash-shared"],
                "shared_source_revisions": ["revision-shared"],
                "shared_source_domains": ["example.org"],
                "near_duplicate_pairs": [
                    {"train_signature": "aaaaaaaaaaaaaaaa", "evaluation_signature": "bbbbbbbbbbbbbbbb"}
                ],
            },
        },
    )

    assert report["phase7_global_isolation_audit"]["missing_axes"] == ["source_hash_isolated"]
    assert report["fixture_execution_policy"]["clearable_blocked_request_ids"] == []
    assert report["fixture_repair_actions"][0]["clearable_after_review"] is False
    assert report["fixture_repair_actions"][0]["isolation_evidence"]["global"]["missing_axes"] == [
        "source_hash_isolated"
    ]
    assert report["fixture_repair_actions"][0]["isolation_evidence"]["global"]["overlap_values"] == {
        "shared_source_hashes": ["hash-shared"],
        "shared_source_revisions": ["revision-shared"],
        "shared_source_domains": ["example.org"],
        "near_duplicate_pairs": [
            {"train_signature": "aaaaaaaaaaaaaaaa", "evaluation_signature": "bbbbbbbbbbbbbbbb"}
        ],
        "time_split_isolated": True,
    }
    guidance = report["fixture_repair_actions"][0]["operator_guidance"]
    assert "global=FAIL" in guidance
    assert "failed_axes=source_hash_isolated" in guidance
    assert "shared_hashes=hash-shared" in guidance
    assert "near_duplicate_pairs=1" in guidance
    rerun_commands = report["fixture_repair_actions"][0]["rerun_commands"]
    assert "eval-phase7-isolation" in rerun_commands["audit_all_axes"]
    assert "apply-phase7-isolation-block-policy" in rerun_commands["reapply_block_policy"]
    assert rerun_commands["failed_axes"] == ["source_hash_isolated"]
    assert rerun_commands["axis_repair_requirements"]["source_hash_isolated"]["required_fields"] == [
        "source_hash"
    ]
    assert "shared_source_hashes must be empty" in rerun_commands["axis_repair_requirements"][
        "source_hash_isolated"
    ]["verification"]
