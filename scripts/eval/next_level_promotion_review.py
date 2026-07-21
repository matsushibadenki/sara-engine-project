#!/usr/bin/env python3
"""Review next-level evidence without self-promoting runtime or roadmap state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402
from sara_engine.evaluation.promotion_approval import validate_approval  # noqa: E402

DEFAULT_REPORT = workspace_path("evaluation", "next_level_promotion_review.json")
DEFAULT_GATE = workspace_path("evaluation", "next_level_promotion_gate.json")
DEFAULT_JOURNAL = workspace_path("evaluation", "next_level_research_journal.jsonl")

REPORT_FILES = {
    "phase21_structural": "next_level_structural_benchmark.json",
    "phase22_horizon": "continual_horizon_benchmark.json",
    "phase22_external": "continual_horizon_external_gate.json",
    "phase23_multimodal": "phase23_structural_fusion_benchmark.json",
    "phase23_external": "phase23_external_multimodal_gate.json",
    "phase24_causal": "phase24_causal_benchmark.json",
    "phase25_agent": "phase25_agent_loop_benchmark.json",
}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_review(evaluation_dir: str, approval_path: str = "") -> Dict[str, Any]:
    reports = {
        key: _read_json(os.path.join(evaluation_dir, filename))
        for key, filename in REPORT_FILES.items()
    }
    internal_keys = ("phase21_structural", "phase22_horizon", "phase23_multimodal", "phase24_causal", "phase25_agent")
    internal_results = {
        key: bool(reports[key].get("passed", False)) for key in internal_keys
    }
    external = reports["phase22_external"]
    external_multimodal = reports["phase23_external"]
    approval = _read_json(approval_path) if approval_path else {}
    approval_valid = validate_approval(approval, reports)
    negative_results = []
    if not external.get("promotion_allowed", False):
        negative_results.append("independent long-horizon coverage is below 10/30/100 buckets")
    if not external_multimodal.get("promotion_allowed", False):
        negative_results.append("Phase 23 independent multimodal coverage is incomplete")
    negative_results.append("physical joule evidence remains indefinitely pending by operator decision")
    next_actions = [
        {
            "action": "collect_independent_multimodal_records",
            "command": "python scripts/sara_cli.py build-phase23-multimodal-collection-request",
            "artifact": "workspace/autobot/phase23_multimodal_collection_targets.json",
        },
        {
            "action": "collect_independent_horizon_records",
            "command": "python scripts/sara_cli.py build-continual-horizon-collection-request",
            "artifact": "workspace/autobot/continual_horizon_collection_targets.json",
        },
        {
            "action": "rerun_internal_phase_benchmarks",
            "command": "python scripts/sara_cli.py eval-phase23-structural-fusion && python scripts/sara_cli.py eval-phase24-causal && python scripts/sara_cli.py eval-phase25-agent-loop",
            "artifact": "workspace/evaluation/phase23_structural_fusion_benchmark.json",
        },
    ]
    checks = {
        "internal_phase_evidence_complete": all(internal_results.values()),
        "external_horizon_promotion_allowed": bool(external.get("promotion_allowed", False)),
        "external_multimodal_promotion_allowed": bool(
            external_multimodal.get("promotion_allowed", False)
        ),
        "human_approval_required": not approval_valid,
        "human_approval_valid": approval_valid,
        "physical_energy_excluded": True,
    }
    promotion_allowed = bool(
        checks["internal_phase_evidence_complete"]
        and checks["external_horizon_promotion_allowed"]
        and checks["external_multimodal_promotion_allowed"]
        and approval_valid
    )
    return {
        "schema": "sara-next-level-promotion-review-v1",
        "observed_only": True,
        "promotion_allowed": promotion_allowed,
        "internal_results": internal_results,
        "checks": checks,
        "negative_results": negative_results,
        "next_actions": next_actions,
        "source_artifacts": {
            key: os.path.join(evaluation_dir, filename) for key, filename in REPORT_FILES.items()
        },
        "human_approval": dict(approval),
    }


def write_outputs(review: Mapping[str, Any], report_path: str, gate_path: str, journal_path: str) -> None:
    report = dict(review)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(ensure_parent_directory(report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    gate = {
        "schema": "sara-next-level-promotion-gate-v1",
        "promotion_allowed": bool(review.get("promotion_allowed", False)),
        "human_approval_required": not bool(review.get("checks", {}).get("human_approval_valid", False)),
        "source_review": os.path.abspath(report_path),
        "negative_results": list(review.get("negative_results", [])),
    }
    with open(ensure_parent_directory(gate_path), "w", encoding="utf-8") as handle:
        json.dump(gate, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    journal_entry = {
        "schema": "sara-next-level-research-journal-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "bounded structural mechanisms can form a safer next-level research prototype",
        "evidence": dict(review.get("internal_results", {})),
        "result": "internal evidence complete; promotion blocked by unresolved external coverage",
        "confidence": 0.5,
        "negative_results": list(review.get("negative_results", [])),
        "next_tests": list(review.get("next_actions", [])),
        "promotion_allowed": bool(review.get("promotion_allowed", False)),
        "human_approval_required": not bool(review.get("checks", {}).get("human_approval_valid", False)),
    }
    with open(ensure_parent_directory(journal_path), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(journal_entry, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--gate-path", default=DEFAULT_GATE)
    parser.add_argument("--journal-path", default=DEFAULT_JOURNAL)
    parser.add_argument(
        "--approval-path",
        default=workspace_path("evaluation", "next_level_human_approval.json"),
    )
    args = parser.parse_args(argv)
    review = build_review(args.evaluation_dir, args.approval_path)
    write_outputs(review, args.report_path, args.gate_path, args.journal_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
