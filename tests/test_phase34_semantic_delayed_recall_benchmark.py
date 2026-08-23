from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sara_engine.evaluation.phase34_factorial_preregistration import ARMS
from sara_engine.memory.semantic_checkpoint_adapter import (
    SemanticCheckpointLimits,
    SemanticCheckpointRuntime,
    SparseMultilingualSemanticAdapter,
    claim_stream,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_module():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_semantic_delayed_recall_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("phase34_semantic_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    module = _load_benchmark_module()
    return (
        module,
        module._read_jsonl(module.DEFAULT_FIXTURE),
        module._read_json(module.DEFAULT_PREREGISTRATION),
        module._read_json(module.DEFAULT_REQUEST),
    )


def test_sparse_semantic_adapter_maps_three_languages_to_typed_axes():
    adapter = SparseMultilingualSemanticAdapter()
    source = adapter.encode_source(
        "HTTP is a stateless application-level protocol.",
        source_ref="https://example.org/http",
        source_revision="r1",
    )

    queries = (
        "At what protocol layer does HTTP operate, and is it stateful?",
        "HTTPはどのプロトコル層で動作し、状態を保持するプロトコルですか。",
        "HTTP在哪个协议层运行，它是否是有状态协议？",
    )
    encoded = [adapter.encode_query(query) for query in queries]

    assert source.subjects == ("http",)
    assert source.axes == ("protocol_layer", "protocol_state")
    assert all(query.subjects == source.subjects for query in encoded)
    assert all(query.axes == source.axes for query in encoded)
    assert all("proposition" not in query.__dict__ for query in encoded)


def test_semantic_retention_is_query_blind_and_selection_preserves_retained_sets():
    module, _, manifest, request = _inputs()
    adapter = SparseMultilingualSemanticAdapter()
    claims, record_to_ref = module._source_claims(request, adapter)
    stream, omission = claim_stream(
        claims,
        target_source_ref=record_to_ref["arch-migration-ietf-001"],
        horizon=10,
        control_mode="none",
    )
    query = adapter.encode_query(
        "HTTP在哪个协议层运行，它是否是有状态协议？"
    )
    limits = SemanticCheckpointLimits(**{
        "max_events": manifest["budgets"]["source_events_per_case"],
        "max_attempted_checkpoints": manifest["budgets"]["attempted_checkpoints_per_case"],
        "max_checkpoints": manifest["budgets"]["max_checkpoints"],
        "selected_k": manifest["budgets"]["max_selected_checkpoints"],
        "max_claims_per_checkpoint": manifest["budgets"]["max_summary_ids_per_checkpoint"],
        "max_state_bytes": manifest["budgets"]["max_total_state_bytes"],
        "max_event_cost": manifest["budgets"]["max_local_interactions_per_case"],
    })
    results = {
        arm: SemanticCheckpointRuntime(arm, limits).evaluate(
            stream, query, horizon=10, omission_receipt=omission
        )
        for arm in ARMS
    }

    assert results[ARMS[1]]["retained_set_digest"] == results[ARMS[2]]["retained_set_digest"]
    assert results[ARMS[3]]["retained_set_digest"] == results[ARMS[4]]["retained_set_digest"]
    assert results[ARMS[0]]["decision"] == "abstain_unsupported"
    assert results[ARMS[4]]["decision"] == "retrieve_original"
    assert all(result["query_visible_during_retention"] is False for result in results.values())


def test_semantic_controls_fail_closed_from_evidence_state():
    module, rows, manifest, request = _inputs()
    adapter = SparseMultilingualSemanticAdapter()
    claims, record_to_ref = module._source_claims(request, adapter)
    limits = module._limits(manifest)
    cases = {
        row["family"]: row
        for row in rows
        if row["record_id"] == "arch-migration-python-003"
        and row["language"] == "ja"
        and row["horizon"] == 30
    }
    expected = {
        "lexical_overlap_abstention": "abstain_unsupported",
        "revision_replacement": "retrieve_revision",
        "contradiction_abstention": "abstain_contradiction",
        "missing_evidence_abstention": "abstain_missing",
    }
    for family, decision in expected.items():
        case = cases[family]
        stream, omission = claim_stream(
            claims,
            target_source_ref=record_to_ref[case["record_id"]],
            horizon=case["horizon"],
            control_mode=case["control_mode"],
        )
        result = SemanticCheckpointRuntime(ARMS[4], limits).evaluate(
            stream,
            adapter.encode_query(case["query_text"]),
            horizon=case["horizon"],
            omission_receipt=omission,
        )
        assert result["decision"] == decision
        assert result["durable_mutation"] is False


def test_semantic_benchmark_executes_frozen_conditions_without_label_leakage():
    module, rows, manifest, request = _inputs()
    frozen_environment = json.loads(
        (
            PROJECT_ROOT
            / "workspace"
            / "evaluation"
            / "phase34_semantic_delayed_recall_environment.json"
        ).read_text(encoding="utf-8")
    )
    module._environment_descriptor = lambda: frozen_environment
    report = module.build_report(rows, manifest, request)

    assert report["execution_passed"] is True
    assert report["semantic_gate_passed"] is True
    assert report["promotion_ready"] is False
    assert report["metrics"]["condition_count"] == 6750
    assert report["metrics"]["retained_set_identity"] == 1.0
    assert report["metrics"]["deterministic_replay"] == 1.0
    assert report["checks"]["expected_labels_absent_from_candidate_traces"] is True
    candidate_payload = json.dumps(
        [entry["candidate"] for entry in report["results"]], sort_keys=True
    )
    assert all(
        row["expected_proposition_id"] not in candidate_payload
        for row in rows
        if row["expected_proposition_id"] is not None
    )
