#!/usr/bin/env python3
"""Run the observed-only Phase 23 structural multimodal fusion benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.multimodal.structural_verification import (  # noqa: E402
    ModalityEvidence,
    MultimodalStructuralVerifier,
)
from sara_engine.memory.multimodal_event_bundle_admission import (  # noqa: E402
    build_multimodal_event_state_candidate,
)
from sara_engine.multimodal.synesthetic_binding import SparseTemporalBinder  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "phase23_structural_fusion_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase23_structural_fusion_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "phase23_structural_fusion_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    verifier = MultimodalStructuralVerifier(max_binding_delay_ms=32.0)
    case_reports: Dict[str, Any] = {}
    case_decisions: Dict[str, Any] = {}
    for row in rows:
        evidence = tuple(
            ModalityEvidence(
                modality=str(item["modality"]),
                label=str(item.get("label", "")),
                timestamp_ms=float(item["timestamp_ms"]),
                source_ref=str(item.get("source_ref", "")),
                observed=bool(item.get("observed", True)),
                confidence=float(item.get("confidence", 1.0)),
                claim_key=str(item.get("claim_key", "")),
            )
            for item in row.get("evidence", [])
        )
        result = verifier.verify(evidence, expected_modalities=row.get("expected_modalities", []))
        case_decisions[str(row["case_id"])] = result
        case_reports[str(row["case_id"])] = {
            "expected_decision": str(row["expected_decision"]),
            "result": result.to_dict(),
        }
    decision_accuracy = sum(
        int(item["expected_decision"] == item["result"]["decision"])
        for item in case_reports.values()
    ) / float(max(1, len(case_reports)))
    contradiction_abstention = float(
        case_reports.get("contradictory_labels", {}).get("result", {}).get("decision")
        == "abstain_cross_modal_contradiction"
    )
    missing_prediction = float(
        case_reports.get("missing_audio", {}).get("result", {}).get("decision")
        == "provisional_missing_modality_prediction"
    )
    admission_boundary = {}
    binder = SparseTemporalBinder(window_ms=32.0)
    for case_id in ("verified_structure", "contradictory_labels", "missing_audio"):
        row = next(row for row in rows if str(row["case_id"]) == case_id)
        events = [
            binder.normalize_event(
                modality=str(item["modality"]),
                timestamp_ms=float(item["timestamp_ms"]),
                source_id=f"phase23-{case_id}-{item['modality']}",
                sparse_signature=(1, 2),
                label=str(item.get("label", "")),
                claim_key=str(item.get("claim_key", "")),
                source_ref=str(item.get("source_ref", "")),
            )
            for item in row.get("evidence", [])
        ]
        bundles = binder.bundle_events(events)
        structural_result = case_decisions[case_id]
        admission = build_multimodal_event_state_candidate(
            bundles[0],
            structural_decision=structural_result,
        )
        admission_boundary[case_id] = admission.to_dict()
    admission_integrity = float(
        admission_boundary["verified_structure"]["promotion_allowed"]
        and not admission_boundary["contradictory_labels"]["promotion_allowed"]
        and not admission_boundary["missing_audio"]["promotion_allowed"]
    )
    checks = {
        "decision_accuracy": decision_accuracy == 1.0,
        "contradiction_abstention": contradiction_abstention == 1.0,
        "missing_modality_is_provisional": missing_prediction == 1.0,
        "temporal_conflict_abstention": case_reports.get("delayed_audio", {}).get("result", {}).get("decision") == "abstain_temporal_misalignment",
        "event_memory_admission_boundary": admission_integrity == 1.0,
        "durable_mutation_blocked": all(
            not item["result"]["durable_mutation_allowed"] for item in case_reports.values()
        ),
    }
    return {
        "schema": "sara-phase23-structural-fusion-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "external_device_required": False,
        "metrics": {
            "case_count": len(case_reports),
            "decision_accuracy": decision_accuracy,
            "contradiction_abstention": contradiction_abstention,
            "missing_modality_provisional_prediction": missing_prediction,
            "event_memory_admission_boundary": admission_integrity,
        },
        "checks": checks,
        "cases": case_reports,
        "event_memory_admission_boundary": admission_boundary,
        "policy_notes": [
            "Modality-local evidence remains inspectable after fusion.",
            "Contradictory or temporally misaligned evidence abstains.",
            "Missing-modality predictions remain provisional.",
            "No result directly mutates durable structural state.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    report = build_report(_load(args.fixture_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 23 structural fusion benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
