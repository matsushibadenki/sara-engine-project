#!/usr/bin/env python3
"""Review next-level evidence without self-promoting runtime or roadmap state."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
from sara_engine.memory.verification_receipt import evidence_digest  # noqa: E402
from sara_engine.evaluation.metric_drift import (  # noqa: E402
    build_metric_snapshot,
    classify_metric_drift,
)

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

REPORT_SCRIPTS = {
    "phase21_structural": "scripts/eval/next_level_structural_benchmark.py",
    "phase22_horizon": "scripts/eval/continual_horizon_benchmark.py",
    "phase22_external": "scripts/eval/continual_horizon_external_gate.py",
    "phase23_multimodal": "scripts/eval/phase23_structural_fusion_benchmark.py",
    "phase23_external": "scripts/eval/phase23_external_multimodal_gate.py",
    "phase24_causal": "scripts/eval/phase24_causal_benchmark.py",
    "phase25_agent": "scripts/eval/phase25_agent_loop_benchmark.py",
}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_journal(path: str, *, max_entries: int = 64) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except FileNotFoundError:
        return []
    return rows[-max(1, int(max_entries)) :]


def _experiment_fingerprint(payload: Mapping[str, Any]) -> str:
    next_tests = payload.get("next_tests", payload.get("next_actions", []))
    normalized_tests = [
        {
            "action": str(item.get("action", "")),
            "command": str(item.get("command", "")),
            "artifact": str(item.get("artifact", "")),
        }
        for item in next_tests
        if isinstance(item, Mapping)
    ]
    return evidence_digest(
        {
            "hypothesis": str(
                payload.get(
                    "hypothesis",
                    "bounded structural mechanisms can form a safer next-level research prototype",
                )
            ),
            "evidence": dict(
                payload.get("evidence", payload.get("internal_results", {}))
            ),
            "negative_results": sorted(
                str(item) for item in payload.get("negative_results", [])
            ),
            "next_tests": normalized_tests,
            "promotion_allowed": bool(payload.get("promotion_allowed", False)),
            "metric_snapshot_digest": str(
                payload.get(
                    "metric_snapshot_digest",
                    payload.get("metric_snapshot", {}).get("snapshot_digest", ""),
                )
            ),
        }
    )


def _implementation_fingerprints() -> Dict[str, str]:
    fingerprints = {}
    for phase, relative_path in REPORT_SCRIPTS.items():
        path = os.path.join(PROJECT_ROOT, relative_path)
        try:
            with open(path, "rb") as handle:
                fingerprints[phase] = sha256(handle.read()).hexdigest()
        except FileNotFoundError:
            fingerprints[phase] = ""
    return fingerprints


def build_review(
    evaluation_dir: str,
    approval_path: str = "",
    journal_path: str = "",
) -> Dict[str, Any]:
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
    journal_entries = _read_journal(journal_path) if journal_path else []
    metric_snapshot = build_metric_snapshot(
        reports,
        implementation_fingerprints=_implementation_fingerprints(),
    )
    previous_metric_snapshot = next(
        (
            item.get("metric_snapshot")
            for item in reversed(journal_entries)
            if isinstance(item.get("metric_snapshot"), Mapping)
        ),
        None,
    )
    metric_drift = classify_metric_drift(
        metric_snapshot,
        previous_metric_snapshot,
    )
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
        "metric_drift_classified": True,
        "code_regression_absent": not metric_drift["code_regression_detected"],
    }
    promotion_allowed = bool(
        checks["internal_phase_evidence_complete"]
        and checks["external_horizon_promotion_allowed"]
        and checks["external_multimodal_promotion_allowed"]
        and checks["code_regression_absent"]
        and approval_valid
    )
    experiment_payload = {
        "internal_results": internal_results,
        "negative_results": negative_results,
        "next_actions": next_actions,
        "promotion_allowed": promotion_allowed,
        "metric_snapshot_digest": metric_snapshot["snapshot_digest"],
    }
    experiment_fingerprint = _experiment_fingerprint(experiment_payload)
    prior_match_count = sum(
        str(item.get("experiment_fingerprint", "")) == experiment_fingerprint
        or (
            not item.get("experiment_fingerprint")
            and _experiment_fingerprint(item) == experiment_fingerprint
        )
        for item in journal_entries
    )
    repeated_failed_experiment = bool(not promotion_allowed and prior_match_count > 0)
    suppressed_actions = []
    if repeated_failed_experiment:
        retained_actions = []
        for action in next_actions:
            if action["action"] == "rerun_internal_phase_benchmarks":
                suppressed_actions.append(
                    {
                        **action,
                        "suppression_reason": "unchanged_internal_evidence_already_passed",
                    }
                )
            else:
                retained_actions.append(action)
        next_actions = retained_actions
    checks["repeated_failed_experiment_detected"] = repeated_failed_experiment
    checks["duplicate_work_suppressed"] = bool(suppressed_actions)
    return {
        "schema": "sara-next-level-promotion-review-v1",
        "observed_only": True,
        "promotion_allowed": promotion_allowed,
        "internal_results": internal_results,
        "checks": checks,
        "negative_results": negative_results,
        "next_actions": next_actions,
        "research_memory": {
            "experiment_fingerprint": experiment_fingerprint,
            "prior_match_count": prior_match_count,
            "repeated_failed_experiment": repeated_failed_experiment,
            "duplicate_work_suppressed": bool(suppressed_actions),
            "suppressed_actions": suppressed_actions,
            "journal_entry_count_scanned": len(journal_entries),
        },
        "metric_snapshot": metric_snapshot,
        "metric_drift": metric_drift,
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
        "experiment_fingerprint": str(
            review.get("research_memory", {}).get("experiment_fingerprint", "")
        ),
        "duplicate_work_suppressed": bool(
            review.get("research_memory", {}).get("duplicate_work_suppressed", False)
        ),
        "suppressed_actions": list(
            review.get("research_memory", {}).get("suppressed_actions", [])
        ),
        "metric_snapshot": dict(review.get("metric_snapshot", {})),
        "metric_drift": dict(review.get("metric_drift", {})),
    }
    existing = _read_journal(journal_path)
    duplicate_journal_entry = bool(
        existing
        and str(existing[-1].get("experiment_fingerprint", ""))
        == journal_entry["experiment_fingerprint"]
    )
    if not duplicate_journal_entry:
        with open(ensure_parent_directory(journal_path), "a", encoding="utf-8") as handle:
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
    review = build_review(args.evaluation_dir, args.approval_path, args.journal_path)
    write_outputs(review, args.report_path, args.gate_path, args.journal_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
