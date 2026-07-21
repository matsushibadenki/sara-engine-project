#!/usr/bin/env python3
"""Build a Level-2 capability matrix without promoting production defaults."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_OUTPUT = workspace_path("evaluation", "level2_capability_matrix.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "level2_capability_matrix_summary.txt")


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_matrix(evaluation_dir: str) -> Dict[str, Any]:
    reports = {
        "structural": _read(os.path.join(evaluation_dir, "next_level_structural_benchmark.json")),
        "continual": _read(os.path.join(evaluation_dir, "continual_horizon_benchmark.json")),
        "multimodal": _read(os.path.join(evaluation_dir, "phase23_structural_fusion_benchmark.json")),
        "causal": _read(os.path.join(evaluation_dir, "phase24_causal_benchmark.json")),
        "agent": _read(os.path.join(evaluation_dir, "phase25_agent_loop_benchmark.json")),
        "promotion": _read(os.path.join(evaluation_dir, "next_level_promotion_gate.json")),
        "external": _read(os.path.join(evaluation_dir, "continual_horizon_external_gate.json")),
    }
    matrix = {
        "structural_reasoning": {
            "passed": bool(reports["structural"].get("passed", False)),
            "accuracy": reports["structural"].get("metrics", {}).get("supported_composition", 0.0),
            "abstention": reports["structural"].get("metrics", {}).get("unsupported_composition_abstention", 0.0),
            "provenance": "fixture_observed_only",
        },
        "continual_revision": {
            "passed": bool(reports["continual"].get("passed", False)),
            "useful_recall": reports["continual"].get("metrics", {}).get("mean_active_useful_recall", 0.0),
            "catastrophic_interference": 1.0 - reports["continual"].get("metrics", {}).get("mean_active_protected_knowledge_retention", 0.0),
            "state_growth": reports["continual"].get("metrics", {}).get("max_state_growth", 0),
            "provenance": "fixture_plus_manifest_gate",
        },
        "multimodal_structure": {
            "passed": bool(reports["multimodal"].get("passed", False)),
            "accuracy": reports["multimodal"].get("metrics", {}).get("decision_accuracy", 0.0),
            "abstention": reports["multimodal"].get("metrics", {}).get("contradiction_abstention", 0.0),
            "provenance": "fixture_observed_only",
        },
        "causal_counterfactual": {
            "passed": bool(reports["causal"].get("passed", False)),
            "verified_case": reports["causal"].get("metrics", {}).get("verified_causal_case", 0.0),
            "abstention": reports["causal"].get("checks", {}).get("unsupported_counterfactual_abstention", False),
            "provenance": "fixture_observed_only",
        },
        "verifiable_agent": {
            "passed": bool(reports["agent"].get("passed", False)),
            "safe_plan_acceptance": reports["agent"].get("metrics", {}).get("safe_plan_acceptance", 0.0),
            "rollback_guard": reports["agent"].get("checks", {}).get("unexpected_outcome_rolls_back", False),
            "provenance": "fixture_observed_only",
        },
    }
    checks = {
        "internal_capabilities_pass": all(bool(item["passed"]) for item in matrix.values()),
        "independent_horizon_gate_pass": bool(reports["external"].get("promotion_allowed", False)),
        "human_review_gate_pass": bool(reports["promotion"].get("promotion_allowed", False)),
        "physical_energy_required_for_matrix": False,
    }
    promotion_allowed = bool(
        checks["internal_capabilities_pass"]
        and checks["independent_horizon_gate_pass"]
        and checks["human_review_gate_pass"]
    )
    unresolved_gaps = []
    if not checks["independent_horizon_gate_pass"]:
        unresolved_gaps.append("independent 10/30/100 horizon coverage")
    if not reports["multimodal"].get("independent_source_scope"):
        unresolved_gaps.append("independent multimodal workload")
    if not checks["human_review_gate_pass"]:
        unresolved_gaps.append("human promotion approval")
    return {
        "schema": "sara-level2-capability-matrix-v1",
        "observed_only": True,
        "promotion_allowed": promotion_allowed and not unresolved_gaps,
        "matrix": matrix,
        "checks": checks,
        "unresolved_gaps": unresolved_gaps,
        "physical_energy_status": "indefinitely_pending_and_excluded",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    report = build_matrix(args.evaluation_dir)
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Level-2 capability matrix: {'PASS' if report['promotion_allowed'] else 'REVIEW_REQUIRED'}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
        for gap in report["unresolved_gaps"]:
            handle.write(f"- unresolved: {gap}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
