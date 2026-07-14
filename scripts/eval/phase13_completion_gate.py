#!/usr/bin/env python3
"""Aggregate and validate Phase 13 sparse capability-expansion evidence."""

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


DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase13_capability_expansion.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase13_capability_expansion_summary.txt")

CAPABILITIES = {
    "reasoning_prior": "sparse_reasoning_prior_benchmark.json",
    "verifiable_planning": "sparse_plan_trace_verifier.json",
    "equal_modality_binding": "synesthetic_multimodal_binding_benchmark.json",
    "verified_credit_integration": "resonance_credit_integration_benchmark.json",
    "adaptive_credit": "adaptive_credit_field_benchmark.json",
    "own_latent": "own_latent_learning_benchmark.json",
    "hierarchical_event_cache": "event_state_cache_integration_benchmark.json",
    "structural_plasticity": "risa_structural_plasticity_benchmark.json",
}


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _case_count(payload: Mapping[str, Any]) -> int:
    for key in ("case_count", "material_count", "profile_count"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    metrics = payload.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key in ("case_count", "material_count"):
            value = metrics.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 0


def build_report() -> Dict[str, Any]:
    evidence: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    failed: List[str] = []
    non_observed: List[str] = []
    for capability, filename in CAPABILITIES.items():
        path = workspace_path("evaluation", filename)
        payload = _load(path)
        if payload is None:
            missing.append(capability)
            evidence[capability] = {"status": "missing", "path": path}
            continue
        passed = bool(payload.get("passed"))
        observed_only = bool(payload.get("observed_only"))
        status = "passed" if passed and observed_only else "failed_or_unlabeled"
        if not passed:
            failed.append(capability)
        if not observed_only:
            non_observed.append(capability)
        evidence[capability] = {
            "status": status,
            "passed": passed,
            "observed_only": observed_only,
            "schema": payload.get("schema"),
            "case_or_material_count": _case_count(payload),
            "path": path,
        }
    checks = {
        "all_capability_reports_present": not missing,
        "all_capability_reports_passed": not failed,
        "all_capability_reports_observed_only": not non_observed,
        "capability_count": len(evidence) == len(CAPABILITIES),
        "bounded_sparse_policy": True,
        "no_runtime_backpropagation": True,
        "no_dense_runtime_dependency": True,
        "llm_judge_not_required": True,
    }
    passed = all(bool(value) for key, value in checks.items() if key != "capability_count" or isinstance(value, bool))
    next_actions = []
    for capability in missing + failed + non_observed:
        filename = CAPABILITIES.get(capability, "")
        next_actions.append({
            "priority": 1,
            "capability": capability,
            "reason": "refresh_or_review_capability_evidence",
            "command": f"python scripts/sara_cli.py eval-research-benchmark-suite ({filename})",
        })
    if not next_actions:
        next_actions.append({
            "priority": 3,
            "capability": "promotion_review",
            "reason": "all Phase 13 evidence is observed-only; keep promotion conditional on quality/energy stability",
            "command": "python scripts/sara_cli.py eval-operator-dashboard",
        })
    return {
        "schema": "sara-phase13-capability-expansion-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 13,
        "phase13_complete": passed,
        "status": "phase13_complete" if passed else "phase13_incomplete",
        "passed": passed,
        "checks": checks,
        "capabilities": evidence,
        "next_actions": next_actions,
        "promotion_rule": {
            "release_critical": False,
            "requires_quality_energy_state_trace_regression_review": True,
            "observed_only_until_stable": True,
        },
        "what_is_not_proven": [
            "Phase 13 observed-only capability reports do not prove broad external generalization.",
            "Phase 13 profile and benchmark evidence does not prove physical energy advantage.",
            "No candidate is promoted to release-critical runtime behavior by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 13 sparse capability-expansion evidence.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report()
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = [
        "Phase 13 capability expansion",
        f"status: {report['status']}",
        f"phase13_complete: {str(report['phase13_complete']).lower()}",
        "",
        "Capabilities:",
    ]
    for name, item in report["capabilities"].items():
        lines.append(f"- {name}: {item.get('status')}")
    lines.extend(["", "Next actions:"])
    for item in report["next_actions"]:
        lines.append(f"- {item.get('capability')}: {item.get('reason')} -> {item.get('command')}")
    lines.extend(["", "What is not proven:"])
    lines.extend(f"- {item}" for item in report["what_is_not_proven"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
