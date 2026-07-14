#!/usr/bin/env python3
"""Validate the observed-only Phase 14 sparse own-latent learning surface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "own_latent_learning_benchmark.json")
DEFAULT_MANIFEST_PATH = workspace_path("evaluation", "own_latent_manifest_builder.json")
DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "own_latent_rhm_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase14_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase14_completion_gate_summary.txt")
MAX_STATE_BUDGET = 256


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _jsonl_count(path: str) -> int:
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    try:
                        if isinstance(json.loads(line), dict):
                            count += 1
                    except json.JSONDecodeError:
                        return 0
    except OSError:
        return 0
    return count


def build_report(
    *,
    benchmark_path: str = DEFAULT_BENCHMARK_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    fixture_path: str = DEFAULT_FIXTURE_PATH,
) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    manifest = _load_json(manifest_path)
    benchmark_metrics = benchmark.get("metrics", {}) if isinstance(benchmark, Mapping) else {}
    manifest_notes = manifest.get("policy_notes", []) if isinstance(manifest, Mapping) else []
    benchmark_notes = benchmark.get("policy_notes", []) if isinstance(benchmark, Mapping) else []
    policy_text = " ".join(str(item).lower() for item in list(manifest_notes) + list(benchmark_notes))
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "benchmark_observed_only": bool(benchmark and benchmark.get("observed_only")),
        "manifest_present": manifest is not None,
        "manifest_passed": bool(manifest and manifest.get("passed")),
        "manifest_observed_only": bool(manifest and manifest.get("observed_only")),
        "fixture_present": _jsonl_count(fixture_path) > 0,
        "source_backed_manifest": bool(manifest and _jsonl_count(str(manifest.get("manifest_path", ""))) > 0),
        "sample_efficiency_evidence": float(benchmark_metrics.get("own_latent_sample_efficiency_ok", 0.0) or 0.0) >= 1.0,
        "event_cost_bounded": float(benchmark_metrics.get("own_latent_event_cost_bounded", 0.0) or 0.0) >= 1.0,
        "state_budget_bounded": int(benchmark_metrics.get("own_latent_max_state_budget_units", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET,
        "sparse_cpu_policy_visible": "sparse" in policy_text and "dense" in policy_text and "backpropagation" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if not passed:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase14_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    else:
        next_actions.append({"priority": 3, "reason": "keep own-latent evidence observed-only until repeated quality and energy review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    return {
        "schema": "sara-phase14-completion-gate-v1",
        "phase": 14,
        "phase14_complete": passed,
        "status": "phase14_complete" if passed else "phase14_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "benchmark_case_count": int(benchmark.get("case_count", 0) or 0) if benchmark else 0,
            "manifest_count": int(manifest.get("manifest_count", 0) or 0) if manifest else 0,
            "fixture_case_count": _jsonl_count(fixture_path),
            "max_state_budget_units": int(benchmark_metrics.get("own_latent_max_state_budget_units", 0) or 0),
        },
        "evidence_paths": {"benchmark": benchmark_path, "manifest": manifest_path, "fixture": fixture_path},
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_quality_energy_abstention_regression_review": True},
        "what_is_not_proven": [
            "Phase 14 fixture evidence does not prove broad external generalization.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Own-latent behavior is not promoted to release-critical runtime behavior by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 14 sparse own-latent learning evidence.")
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(benchmark_path=args.benchmark_path, manifest_path=args.manifest_path, fixture_path=args.fixture_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = ["Phase 14 completion gate", f"status: {report['status']}", f"phase14_complete: {str(report['phase14_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
