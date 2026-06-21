import json
import os

from bot.curriculum_manifest import build_curriculum_manifest
from bot.dataset_builder import build_candidate_materials, run_dataset_builder
from bot.learning_material_gate import split_accepted_rejected
from bot.planner import CollectionPlanner
from bot.training_queue import TrainingQueue
from sara_engine.utils.project_paths import interim_data_path, processed_data_path, workspace_path


def _write_records(path):
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


def _write_fixture_request_plan(path):
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


def test_dataset_builder_generates_gated_materials():
    records = [
        {
            "source": "hot_inbox",
            "record_text": (
                "Sparse routing is an event based retrieval method. "
                "It avoids dense matrix runtime and keeps energy cost visible."
            ),
            "meta": {"quality": 0.9, "language": "en"},
        }
    ]

    candidates = build_candidate_materials(records)
    split = split_accepted_rejected(candidates)

    assert {item["material_type"] for item in split["accepted"]} >= {
        "summary",
        "qa_pair",
        "negative_query",
        "source_claim",
    }
    assert all(item["accepted"] is True for item in split["accepted"])


def test_run_dataset_builder_writes_managed_outputs():
    records_path = processed_data_path("autobot", "test_multimodal_records.jsonl")
    fixture_request_plan_path = workspace_path("autobot", "test_fixture_request_plan.json")
    _write_records(records_path)
    _write_fixture_request_plan(fixture_request_plan_path)

    report = run_dataset_builder(
        records_path=records_path,
        candidate_path=interim_data_path("autobot", "test_candidate_learning_materials.jsonl"),
        rejected_path=interim_data_path("autobot", "test_rejected_learning_materials.jsonl"),
        accepted_path=processed_data_path("autobot", "test_learning_materials.jsonl"),
        curriculum_path=processed_data_path("autobot", "test_curriculum_manifest.jsonl"),
        report_path=workspace_path("autobot", "test_dataset_builder_report.json"),
        summary_path=workspace_path("autobot", "test_dataset_builder_summary.txt"),
        fixture_request_plan_path=fixture_request_plan_path,
        collection_targets_path=workspace_path("autobot", "test_dataset_builder_collection_targets.json"),
        evaluation_gaps=["negative_control", "contrastive_control"],
    )

    assert report["passed"] is True
    assert report["record_count"] == 2
    assert report["accepted_count"] >= 8
    assert report["accepted_material_type_counts"]["qa_pair"] >= 2
    assert report["curriculum_distribution"]["repair"] >= 2
    assert report["fixture_request_plan_loaded"] is True
    assert report["fixture_request_count"] == 2
    assert report["collection_target_count"] == 2
    assert "retrieval_grounding" in report["evaluation_gaps"]
    assert os.path.exists(report["outputs"]["curriculum_manifest"])
    assert os.path.exists(report["outputs"]["negative_query"])
    assert os.path.exists(report["outputs"]["collection_targets"])
    with open(report["outputs"]["collection_targets"], "r", encoding="utf-8") as handle:
        collection_targets = json.load(handle)
    assert collection_targets["target_count"] == 2
    assert collection_targets["targets"][0]["candidate_source_domains"]
    assert "evaluation_gaps" in collection_targets["targets"][0]


def test_curriculum_manifest_and_training_queue_prioritize_repair_and_replay():
    materials = [
        {
            "material_hash": "neg",
            "material_type": "negative_query",
            "quality_score": 0.9,
            "source_type": "official_docs",
            "source": "web",
        },
        {
            "material_hash": "sum",
            "material_type": "summary",
            "quality_score": 0.9,
            "source_type": "official_docs",
            "source": "web",
        },
    ]
    manifest = build_curriculum_manifest(materials, evaluation_gaps=["negative_control"])
    queue_path = workspace_path("autobot", f"test_train_queue_{os.getpid()}.json")
    queue = TrainingQueue(queue_path)
    enqueued = queue.enqueue_learning_materials(manifest)
    drained = queue.drain(2)

    assert enqueued == 2
    assert drained[0]["curriculum_stage"] == "repair"
    assert drained[0]["priority"] >= 1.0


def test_gap_curriculum_materials_are_prioritized_in_training_queue():
    materials = [
        {
            "material_hash": "counter",
            "material_type": "counterexample",
            "quality_score": 0.9,
            "source_type": "official_docs",
            "source": "web",
        },
        {
            "material_hash": "transcript",
            "material_type": "transcript_segment",
            "quality_score": 0.9,
            "source_type": "offline_batch",
            "source": "local",
        },
    ]
    manifest = build_curriculum_manifest(
        materials,
        evaluation_gaps=["negative_control", "contrastive_control", "retrieval_grounding"],
    )
    queue_path = workspace_path("autobot", f"test_gap_train_queue_{os.getpid()}.json")
    queue = TrainingQueue(queue_path)
    enqueued = queue.enqueue_learning_materials(manifest)
    drained = queue.drain(2)

    assert enqueued == 2
    assert drained[0]["material_type"] == "counterexample"
    assert drained[0]["curriculum_stage"] == "repair"
    assert any(item["material_type"] == "transcript_segment" for item in drained)


def test_planner_converts_evaluation_failures_to_material_requests():
    planner = CollectionPlanner()
    plan = planner.write_material_request_plan(
        {
            "metrics": {
                "real_data_summary_keyword_coverage": 0.5,
                "negative_control_abstention_integrity": 0.0,
                "contrastive_control_accuracy": 0.5,
                "real_data_qa_accuracy": 0.7,
            },
            "language_balance": {"jp": 1, "en": 9},
        },
        output_path=workspace_path("autobot", "test_material_request_plan.json"),
    )

    request_ids = {item["request_id"] for item in plan["requests"]}
    assert "weak_summary_coverage" in request_ids
    assert "weak_negative_controls" in request_ids
    assert "weak_contrastive_controls" in request_ids
    assert "weak_retrieval_grounding" in request_ids
    assert "language_imbalance_jp" in request_ids


def test_planner_converts_fixture_feedback_to_material_requests():
    planner = CollectionPlanner()
    plan = planner.write_fixture_material_request_plan(
        {
            "fixture_expansion_plan": [
                {
                    "action": "collect_additional_distinct_sources",
                    "priority": 5,
                    "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
                    "missing_material_types_now": ["transcript_segment"],
                    "guidance": "Increase distinct source_ref coverage.",
                },
                {
                    "action": "add_negative_and_contrastive_materials",
                    "priority": 4,
                    "preferred_material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                    "missing_material_types_now": ["counterexample"],
                    "guidance": "Add contrastive evidence.",
                },
            ]
        },
        output_path=workspace_path("autobot", "test_fixture_material_request_plan.json"),
    )

    request_ids = {item["request_id"] for item in plan["requests"]}
    assert plan["request_source"] == "fixture_feedback"
    assert "fixture_source_diversity_gap" in request_ids
    assert "fixture_counterexample_gap" in request_ids
