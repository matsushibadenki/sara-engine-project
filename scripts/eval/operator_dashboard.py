#!/usr/bin/env python3
"""Build a compact managed operator dashboard from release evidence."""

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


DEFAULT_REPORT_PATH = workspace_path("evaluation", "operator_dashboard.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "operator_dashboard_summary.txt")

ARTIFACTS = {
    "phase6": ("workspace/evaluation/phase6_completion_gate.json", "python scripts/sara_cli.py eval-phase6-completion"),
    "phase7": ("workspace/evaluation/phase7_completion_gate.json", "python scripts/sara_cli.py eval-phase7-completion"),
    "phase8": ("workspace/evaluation/phase8_completion_gate.json", "python scripts/sara_cli.py eval-phase8-completion"),
    "phase9": ("workspace/evaluation/phase9_completion_gate.json", "python scripts/sara_cli.py eval-phase9-completion"),
    "phase10": ("workspace/evaluation/phase10_completion_gate.json", "python scripts/sara_cli.py eval-phase10-completion"),
    "phase11": ("workspace/evaluation/phase11_completion_gate.json", "python scripts/sara_cli.py eval-phase11-completion"),
    "phase13": ("workspace/evaluation/phase13_capability_expansion.json", "python scripts/sara_cli.py eval-phase13-completion"),
    "phase14": ("workspace/evaluation/phase14_completion_gate.json", "python scripts/sara_cli.py eval-phase14-completion"),
    "phase15": ("workspace/evaluation/phase15_completion_gate.json", "python scripts/sara_cli.py eval-phase15-completion"),
    "phase16": ("workspace/evaluation/phase16_completion_gate.json", "python scripts/sara_cli.py eval-phase16-completion"),
    "phase17": ("workspace/evaluation/phase17_completion_gate.json", "python scripts/sara_cli.py eval-phase17-completion"),
    "phase18": ("workspace/evaluation/phase18_completion_gate.json", "python scripts/sara_cli.py eval-phase18-completion"),
    "phase19": ("workspace/evaluation/phase19_completion_gate.json", "python scripts/sara_cli.py eval-phase19-completion"),
    "phase20": ("workspace/evaluation/phase20_completion_gate.json", "python scripts/sara_cli.py eval-phase20-completion"),
    "research_product": (
        "workspace/evaluation/research_product_completion_gate_report.json",
        "python scripts/eval/research_product_completion_gate.py",
    ),
    "release": (
        "workspace/release/v1_release_gate_report.json",
        "python scripts/eval/v1_release_gate.py",
    ),
    "energy": (
        "workspace/evaluation/energy_measurement_readiness.json",
        "python scripts/sara_cli.py eval-phase6-completion",
    ),
}


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_status(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "missing"
    if any(bool(payload.get(key)) for key in ("phase6_complete", "phase7_complete", "phase8_complete", "phase9_complete", "phase10_complete", "phase11_complete", "phase13_complete", "phase14_complete", "phase15_complete", "phase16_complete", "phase17_complete", "phase18_complete", "phase19_complete", "phase20_complete")):
        return "passed"
    if payload.get("passed") is True:
        return "passed"
    if payload.get("status") in {"phase6_complete", "phase7_complete", "phase8_complete", "phase9_complete", "phase10_complete", "phase11_complete", "phase13_complete", "phase14_complete", "phase15_complete", "phase16_complete", "phase17_complete", "phase18_complete", "phase19_complete", "phase20_complete"}:
        return "passed"
    return "failed_or_pending"


def build_dashboard() -> Dict[str, Any]:
    states: Dict[str, Dict[str, Any]] = {}
    next_actions: List[Dict[str, Any]] = []
    for name, (relative_path, command) in ARTIFACTS.items():
        path = os.path.join(PROJECT_ROOT, relative_path)
        payload = _load(path)
        status = _artifact_status(payload)
        states[name] = {
            "status": status,
            "path": path,
            "command": command,
        }
        if status != "passed":
            next_actions.append({
                "priority": 1 if name in {"phase6", "phase8", "release"} else 2,
                "artifact": name,
                "reason": "missing_or_failed_evidence",
                "command": command,
            })

    if not next_actions:
        next_actions.append({
            "priority": 3,
            "artifact": "phase6_physical_measurement",
            "reason": "physical_joule_rows_are_still_optional_and_unproven",
            "command": "python scripts/sara_cli.py run-physical-energy-session-batch",
        })
    next_actions.sort(key=lambda item: (int(item.get("priority", 9)), str(item.get("artifact", ""))))
    proven = [
        "Phase 9 research benchmark packaging is reproducible when its completion gate passes.",
        "Phase 10 Rust sparse-runtime readiness is independently gated.",
        "Phase 11 neuromorphic portability is represented as CPU-reference profile evidence.",
    ]
    not_proven = [
        "Physical joule-per-success remains unproven until paired meter rows are supplied.",
        "Neuromorphic profile compatibility is not hardware execution performance evidence.",
    ]
    return {
        "schema": "sara-operator-dashboard-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_states": states,
        "next_actions": next_actions,
        "what_is_proven": proven,
        "what_is_not_proven": not_proven,
        "operator_commands": {
            "full_research_suite": "python scripts/sara_cli.py eval-research-benchmark-suite",
            "phase6_energy_readiness": "python scripts/sara_cli.py eval-phase6-completion",
            "phase9_package_gate": "python scripts/sara_cli.py eval-phase9-completion",
            "phase10_rust_gate": "python scripts/sara_cli.py eval-phase10-completion",
            "phase11_portability_gate": "python scripts/sara_cli.py eval-phase11-completion",
            "phase13_capability_gate": "python scripts/sara_cli.py eval-phase13-completion",
            "phase14_own_latent_gate": "python scripts/sara_cli.py eval-phase14-completion",
            "phase15_dendritic_gate": "python scripts/sara_cli.py eval-phase15-completion",
            "phase16_multimodal_gate": "python scripts/sara_cli.py eval-phase16-completion",
            "phase17_resonance_gate": "python scripts/sara_cli.py eval-phase17-completion",
            "phase18_event_memory_gate": "python scripts/sara_cli.py eval-phase18-completion",
            "phase19_liquid_gate": "python scripts/sara_cli.py eval-phase19-completion",
            "phase20_semantic_echo_gate": "python scripts/sara_cli.py eval-phase20-completion",
            "refresh_dashboard": "python scripts/sara_cli.py eval-operator-dashboard",
        },
        "managed_output_policy": "All dashboard artifacts are written under workspace/evaluation.",
    }


def write_outputs(report: Mapping[str, Any], report_path: str, summary_path: str) -> None:
    resolved_report = ensure_parent_directory(report_path)
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = ["SARA operator dashboard", "", "Artifact states:"]
    for name, state in report.get("artifact_states", {}).items():
        lines.append(f"- {name}: {state.get('status')} ({state.get('path')})")
    lines.append("")
    lines.append("Next actions:")
    for action in report.get("next_actions", []):
        lines.append(f"- [{action.get('priority')}] {action.get('artifact')}: {action.get('reason')} -> {action.get('command')}")
    lines.append("")
    lines.append("What is proven:")
    lines.extend(f"- {item}" for item in report.get("what_is_proven", []))
    lines.append("")
    lines.append("What is not proven:")
    lines.extend(f"- {item}" for item in report.get("what_is_not_proven", []))
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the managed SARA operator dashboard.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_dashboard()
    write_outputs(report, args.report_path, args.summary_path)
    print(json.dumps({"report_path": os.path.abspath(args.report_path), "summary_path": os.path.abspath(args.summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
