import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.operator.llm_assistant import build_readiness_report
from sara_engine.operator.llm_proposal_schema import validate_llm_proposal


def test_valid_operator_proposal_is_accepted():
    proposal = {
        "proposal_id": "summary-1",
        "proposal_type": "evaluation_summary",
        "source_refs": ["workspace/evaluation/research_benchmark_manifest.json"],
        "actions": [
            {
                "action_type": "summarize",
                "report_path": "workspace/evaluation/operator_summary.json",
            }
        ],
    }

    result = validate_llm_proposal(proposal)

    assert result.accepted is True
    assert result.rejection_reasons == []
    assert result.managed_output_count == 1


def test_invalid_json_is_rejected_without_execution():
    result = validate_llm_proposal("{invalid")

    assert result.accepted is False
    assert result.rejection_reasons == ["invalid_json"]


def test_direct_mutation_and_model_changes_are_rejected():
    proposal = {
        "proposal_id": "bad-1",
        "proposal_type": "roadmap_patch",
        "source_refs": ["doc/ROADMAP.md"],
        "actions": [
            {"action_type": "apply_patch", "target_path": "doc/ROADMAP.md"},
            {"action_type": "modify_model", "artifact_path": "models/sara.bin"},
        ],
    }

    result = validate_llm_proposal(proposal)

    assert result.accepted is False
    assert "direct_mutation_action" in result.rejection_reasons


def test_unmanaged_output_path_is_rejected():
    proposal = {
        "proposal_id": "bad-path",
        "proposal_type": "evaluation_summary",
        "source_refs": ["workspace/evaluation/research_benchmark_manifest.json"],
        "actions": [{"action_type": "summarize", "report_path": "tmp/report.json"}],
    }

    result = validate_llm_proposal(proposal)

    assert result.accepted is False
    assert "unmanaged_output_path" in result.rejection_reasons


def test_secret_like_text_is_rejected():
    proposal = {
        "proposal_id": "secret-1",
        "proposal_type": "triage_note",
        "source_refs": ["doc/policy.md"],
        "actions": [{"action_type": "triage"}],
        "note": "token=abc123456789XYZ",
    }

    result = validate_llm_proposal(proposal)

    assert result.accepted is False
    assert "secret_like_text" in result.rejection_reasons


def test_readiness_report_requires_disabled_default_and_rejection_coverage():
    proposals = [
        {
            "proposal_id": "valid",
            "proposal_type": "operator_next_action",
            "source_refs": ["doc/ROADMAP.md"],
            "actions": [{"action_type": "recommend_next_action"}],
        },
        {
            "proposal_id": "bad-path",
            "proposal_type": "evaluation_summary",
            "source_refs": ["workspace/evaluation/research_benchmark_manifest.json"],
            "actions": [{"action_type": "summarize", "report_path": "tmp/report.json"}],
        },
        {
            "proposal_id": "secret",
            "proposal_type": "triage_note",
            "source_refs": ["doc/policy.md"],
            "actions": [{"action_type": "triage"}],
            "note": "password=abc123456789XYZ",
        },
        {
            "proposal_id": "mutation",
            "proposal_type": "roadmap_patch",
            "source_refs": ["doc/ROADMAP.md"],
            "actions": [{"action_type": "apply_patch"}],
        },
    ]

    report = build_readiness_report(proposals)

    assert report["passed"] is True
    assert report["disabled_by_default"] is True
    assert report["llm_runtime_required"] is False
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 3
    assert report["proposal_acceptance_rate"] == 0.25
    assert report["latency_ms"] == 0.0
    assert report["token_budget"]["runtime"] == "not_required"
    assert "without an LLM assistant" in report["fallback_behavior"]
    assert report["rejection_counts"]["direct_mutation_action"] == 1
    json.dumps(report)
