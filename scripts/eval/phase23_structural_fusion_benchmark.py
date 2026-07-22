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
from sara_engine.multimodal.relation_hypothesis import (  # noqa: E402
    BoundedCrossModalHypothesisLedger,
)
from sara_engine.ingest.episode_segmentation import (  # noqa: E402
    bridge_verified_bundle_to_episode,
)
from sara_engine.risa.adapters import (  # noqa: E402
    observation_from_cross_modal_hypothesis,
    subgraph_from_bundle_admission,
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
    missing_vision_prediction = float(
        case_reports.get("missing_vision", {}).get("result", {}).get("decision")
        == "provisional_missing_modality_prediction"
    )
    asynchronous_window_boundary = float(
        case_reports.get("near_window_boundary", {}).get("result", {}).get("decision")
        == "verify_cross_modal_structure"
        and case_reports.get("outside_window_boundary", {}).get("result", {}).get("decision")
        == "abstain_temporal_misalignment"
    )
    admission_boundary = {}
    admission_results = {}
    admission_bundles = {}
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
        admission_results[case_id] = admission
        admission_bundles[case_id] = bundles[0]
        admission_boundary[case_id] = admission.to_dict()
    admission_integrity = float(
        admission_boundary["verified_structure"]["promotion_allowed"]
        and not admission_boundary["contradictory_labels"]["promotion_allowed"]
        and not admission_boundary["missing_audio"]["promotion_allowed"]
    )
    verified_episode_bridge = bridge_verified_bundle_to_episode(
        admission_bundles["verified_structure"],
        admission_results["verified_structure"],
    )
    contradicted_episode_bridge = bridge_verified_bundle_to_episode(
        admission_bundles["contradictory_labels"],
        admission_results["contradictory_labels"],
    )
    verified_subgraph = subgraph_from_bundle_admission(
        admission_results["verified_structure"]
    )
    contradicted_subgraph = subgraph_from_bundle_admission(
        admission_results["contradictory_labels"]
    )
    verified_bundle_bridges = bool(
        verified_episode_bridge.connected
        and verified_episode_bridge.episode is not None
        and len(verified_episode_bridge.episode.modalities) >= 2
        and verified_subgraph.projected
        and len(verified_subgraph.edges) >= 2
        and all(edge.verified for edge in verified_subgraph.edges)
        and not verified_episode_bridge.durable_mutation_allowed
        and not verified_subgraph.durable_mutation_allowed
    )
    rejected_bundle_isolation = bool(
        not contradicted_episode_bridge.connected
        and contradicted_episode_bridge.episode is None
        and not contradicted_subgraph.projected
        and not contradicted_subgraph.edges
    )
    verified_row = next(row for row in rows if str(row["case_id"]) == "verified_structure")
    verified_evidence = tuple(
        ModalityEvidence(
            modality=str(item["modality"]),
            label=str(item.get("label", "")),
            timestamp_ms=float(item["timestamp_ms"]),
            source_ref=str(item.get("source_ref", "")),
            observed=bool(item.get("observed", True)),
            confidence=float(item.get("confidence", 1.0)),
            claim_key=str(item.get("claim_key", "")),
        )
        for item in verified_row.get("evidence", [])
    )
    hypothesis_ledger = BoundedCrossModalHypothesisLedger()
    first_hypothesis = hypothesis_ledger.observe(
        claim_key="impact_event",
        decision=case_decisions["verified_structure"],
        evidence=verified_evidence,
        expected_modalities=verified_row.get("expected_modalities", []),
        observation_source_id="phase23-session-a",
        source_revision="fixture-v1",
    )
    independent_evidence = tuple(
        ModalityEvidence(
            modality=item.modality,
            label=item.label,
            timestamp_ms=item.timestamp_ms,
            source_ref=f"{item.source_ref}:independent-session",
            observed=item.observed,
            confidence=item.confidence,
            claim_key=item.claim_key,
        )
        for item in verified_evidence
    )
    independent_decision = verifier.verify(
        independent_evidence,
        expected_modalities=verified_row.get("expected_modalities", []),
    )
    second_hypothesis = hypothesis_ledger.observe(
        claim_key="impact_event",
        decision=independent_decision,
        evidence=independent_evidence,
        expected_modalities=verified_row.get("expected_modalities", []),
        observation_source_id="phase23-session-b",
        source_revision="fixture-v1",
    )
    contradiction_row = next(
        row for row in rows if str(row["case_id"]) == "contradictory_labels"
    )
    contradiction_evidence = tuple(
        ModalityEvidence(
            modality=str(item["modality"]),
            label=str(item.get("label", "")),
            timestamp_ms=float(item["timestamp_ms"]),
            source_ref=str(item.get("source_ref", "")),
            observed=bool(item.get("observed", True)),
            confidence=float(item.get("confidence", 1.0)),
            claim_key=str(item.get("claim_key", "")),
        )
        for item in contradiction_row.get("evidence", [])
    )
    contradiction_ledger = BoundedCrossModalHypothesisLedger()
    contradiction_ledger.observe(
        claim_key="impact_event",
        decision=case_decisions["verified_structure"],
        evidence=verified_evidence,
        expected_modalities=verified_row.get("expected_modalities", []),
        observation_source_id="phase23-session-a",
        source_revision="fixture-v1",
    )
    frozen_hypothesis = contradiction_ledger.observe(
        claim_key="impact_event",
        decision=case_decisions["contradictory_labels"],
        evidence=contradiction_evidence,
        expected_modalities=contradiction_row.get("expected_modalities", []),
        observation_source_id="phase23-session-c",
        source_revision="fixture-v1",
    )
    first_view = first_hypothesis.hypothesis
    second_view = second_hypothesis.hypothesis
    frozen_view = frozen_hypothesis.hypothesis
    risa_hypothesis_observation = (
        observation_from_cross_modal_hypothesis(second_view) if second_view else None
    )
    hypothesis_boundary = bool(
        first_view
        and first_view.state == "provisional_hypothesis"
        and not first_view.eligible_for_review
        and second_view
        and second_view.state == "eligible_for_review"
        and second_view.distinct_source_count == 2
        and not second_view.durable_mutation_allowed
    )
    hypothesis_contradiction_freeze = bool(
        frozen_view
        and frozen_view.frozen
        and frozen_view.state == "frozen_contradiction"
        and not frozen_view.eligible_for_review
    )
    risa_hypothesis_boundary = bool(
        risa_hypothesis_observation
        and not risa_hypothesis_observation.verified
        and risa_hypothesis_observation.action == "hypothesize_cross_modal_relation"
    )
    checks = {
        "decision_accuracy": decision_accuracy == 1.0,
        "contradiction_abstention": contradiction_abstention == 1.0,
        "missing_modality_is_provisional": missing_prediction == 1.0,
        "modality_dropout_is_symmetric": bool(
            missing_prediction == 1.0 and missing_vision_prediction == 1.0
        ),
        "asynchronous_window_boundary": asynchronous_window_boundary == 1.0,
        "temporal_conflict_abstention": case_reports.get("delayed_audio", {}).get("result", {}).get("decision") == "abstain_temporal_misalignment",
        "event_memory_admission_boundary": admission_integrity == 1.0,
        "verified_bundle_episode_and_subgraph_bridge": verified_bundle_bridges,
        "rejected_bundle_bridge_isolation": rejected_bundle_isolation,
        "durable_mutation_blocked": all(
            not item["result"]["durable_mutation_allowed"] for item in case_reports.values()
        ),
        "cross_modal_hypothesis_boundary": hypothesis_boundary,
        "cross_modal_hypothesis_contradiction_freeze": hypothesis_contradiction_freeze,
        "risa_hypothesis_remains_unverified": risa_hypothesis_boundary,
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
            "missing_vision_provisional_prediction": missing_vision_prediction,
            "asynchronous_window_boundary": asynchronous_window_boundary,
            "event_memory_admission_boundary": admission_integrity,
            "verified_bundle_episode_and_subgraph_bridge": float(verified_bundle_bridges),
            "rejected_bundle_bridge_isolation": float(rejected_bundle_isolation),
            "cross_modal_hypothesis_boundary": float(hypothesis_boundary),
            "cross_modal_hypothesis_contradiction_freeze": float(
                hypothesis_contradiction_freeze
            ),
            "risa_hypothesis_remains_unverified": float(risa_hypothesis_boundary),
        },
        "checks": checks,
        "cases": case_reports,
        "event_memory_admission_boundary": admission_boundary,
        "verified_bundle_episode_bridge": verified_episode_bridge.to_dict(),
        "contradicted_bundle_episode_bridge": contradicted_episode_bridge.to_dict(),
        "verified_bundle_subgraph": verified_subgraph.to_dict(),
        "contradicted_bundle_subgraph": contradicted_subgraph.to_dict(),
        "cross_modal_hypothesis_ledger": hypothesis_ledger.to_dict(),
        "cross_modal_hypothesis_contradiction": (
            frozen_hypothesis.to_dict()
        ),
        "risa_hypothesis_observation": (
            risa_hypothesis_observation.to_dict()
            if risa_hypothesis_observation is not None
            else None
        ),
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
