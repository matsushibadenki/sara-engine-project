#!/usr/bin/env python3
"""Validate independent multimodal evidence before Phase 23 promotion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.multimodal_event_bundle_admission import (  # noqa: E402
    build_multimodal_event_state_candidate,
)
from sara_engine.multimodal.structural_verification import (  # noqa: E402
    ModalityEvidence,
    MultimodalStructuralVerifier,
)
from sara_engine.multimodal.synesthetic_binding import SparseTemporalBinder  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_MANIFEST = processed_data_path(
    "autobot", "phase23_independent_multimodal_manifest.jsonl"
)
DEFAULT_REPORT = workspace_path("evaluation", "phase23_external_multimodal_gate.json")
DEFAULT_SUMMARY = workspace_path(
    "evaluation", "phase23_external_multimodal_gate_summary.txt"
)
EXPECTED_DECISIONS = {
    "verify_cross_modal_structure",
    "provisional_missing_modality_prediction",
    "abstain_cross_modal_contradiction",
    "abstain_temporal_misalignment",
}
REQUIRED_DECISION_COUNTS = {
    "verify_cross_modal_structure": 2,
    "provisional_missing_modality_prediction": 1,
    "abstain_cross_modal_contradiction": 1,
    "abstain_temporal_misalignment": 1,
}


def _load(path: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(payload)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return rows


def _is_independent_ref(value: str) -> bool:
    lowered = value.strip().lower()
    return bool(lowered) and not lowered.startswith(("fixture:", "synthetic:", "generated:"))


def _evidence(row: Mapping[str, Any]) -> tuple[ModalityEvidence, ...]:
    parsed = []
    for item in row.get("evidence", []):
        if not isinstance(item, Mapping):
            continue
        try:
            parsed.append(
                ModalityEvidence(
                    modality=str(item.get("modality", "")),
                    label=str(item.get("label", "")),
                    claim_key=str(item.get("claim_key", "")),
                    timestamp_ms=float(item["timestamp_ms"]),
                    source_ref=str(item.get("source_ref", "")),
                    observed=bool(item.get("observed", True)),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(parsed)


def _admission_allowed(
    case_id: str,
    evidence: Sequence[ModalityEvidence],
    decision: Any,
) -> bool:
    binder = SparseTemporalBinder(window_ms=32.0)
    events = [
        binder.normalize_event(
            modality=item.modality,
            timestamp_ms=item.timestamp_ms,
            source_id=f"phase23-external-{case_id}-{index}",
            sparse_signature=(index + 1, index + 2),
            label=item.label,
            claim_key=item.claim_key,
            source_ref=item.source_ref,
            observed=item.observed,
            confidence=item.confidence,
        )
        for index, item in enumerate(evidence)
    ]
    bundles = binder.bundle_events(events)
    if not bundles:
        return False
    return bool(
        build_multimodal_event_state_candidate(
            bundles[0], structural_decision=decision
        ).promotion_allowed
    )


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [
        row
        for row in rows
        if str(row.get("evidence_scope", "")) == "independent_external"
    ]
    by_domain: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_domain[str(row.get("source_domain", ""))].append(row)

    case_ids = [str(row.get("case_id", "")) for row in eligible]
    material_hashes = [
        str(row.get("material_hash", row.get("source_hash", ""))) for row in eligible
    ]
    source_refs = [str(row.get("source_ref", "")) for row in eligible]
    duplicate_signatures = [
        str(row.get("near_duplicate_signature", "")) for row in eligible
    ]
    expected_counts = Counter(str(row.get("expected_decision", "")) for row in eligible)
    modality_set = {
        str(item.get("modality", ""))
        for row in eligible
        for item in row.get("evidence", [])
        if isinstance(item, Mapping) and str(item.get("modality", ""))
    }
    evidence_refs = [
        str(item.get("source_ref", ""))
        for row in eligible
        for item in row.get("evidence", [])
        if isinstance(item, Mapping)
    ]
    evidence_hashes = [
        str(item.get("source_hash", ""))
        for row in eligible
        for item in row.get("evidence", [])
        if isinstance(item, Mapping)
    ]
    expected_modalities = [
        [str(value) for value in row.get("expected_modalities", [])]
        for row in eligible
    ]

    provenance_checks = {
        "independent_rows_present": bool(eligible),
        "minimum_case_count": len(eligible) >= sum(REQUIRED_DECISION_COUNTS.values()),
        "minimum_domains": len(by_domain) >= 2,
        "source_domains_present": bool(eligible)
        and all(str(row.get("source_domain", "")) for row in eligible),
        "minimum_records_per_domain": bool(by_domain)
        and min(len(items) for items in by_domain.values()) >= 2,
        "unique_case_ids": bool(case_ids) and all(case_ids) and len(set(case_ids)) == len(case_ids),
        "unique_material_hashes": bool(material_hashes)
        and all(material_hashes)
        and len(set(material_hashes)) == len(material_hashes),
        "unique_source_refs": bool(source_refs)
        and all(source_refs)
        and len(set(source_refs)) == len(source_refs),
        "unique_near_duplicate_signatures": bool(duplicate_signatures)
        and all(duplicate_signatures)
        and len(set(duplicate_signatures)) == len(duplicate_signatures),
        "source_revisions_present": bool(eligible)
        and all(str(row.get("source_revision", "")) for row in eligible),
        "collection_times_present": bool(eligible)
        and all(str(row.get("collection_time", "")) for row in eligible),
        "licenses_present": bool(eligible)
        and all(str(row.get("license_hint", row.get("license", ""))) for row in eligible),
        "observed_allow_only": bool(eligible)
        and all(
            bool(row.get("observed_only", False))
            and str(row.get("compliance_level", "")) == "allow"
            for row in eligible
        ),
        "fixture_and_generated_refs_rejected": bool(eligible)
        and all(_is_independent_ref(value) for value in source_refs + evidence_refs),
        "unique_evidence_source_refs": bool(evidence_refs)
        and all(evidence_refs)
        and len(set(evidence_refs)) == len(evidence_refs),
        "evidence_source_hashes_present": bool(evidence_hashes)
        and all(evidence_hashes),
        "row_schema_valid": bool(eligible)
        and all(
            str(row.get("schema", ""))
            == "sara-phase23-independent-multimodal-row-v1"
            for row in eligible
        ),
    }
    coverage_checks = {
        "required_decision_coverage": all(
            expected_counts[decision] >= count
            for decision, count in REQUIRED_DECISION_COUNTS.items()
        ),
        "expected_decisions_supported": bool(eligible)
        and all(value in EXPECTED_DECISIONS for value in expected_counts),
        "vision_audio_coverage": {"vision", "audio"}.issubset(modality_set),
        "expected_modalities_declared": bool(expected_modalities)
        and all({"vision", "audio"}.issubset(set(values)) for values in expected_modalities),
        "multimodal_evidence_shape": bool(eligible)
        and all(
            len(_evidence(row)) >= (
                1
                if str(row.get("expected_decision", ""))
                == "provisional_missing_modality_prediction"
                else 2
            )
            for row in eligible
        ),
    }

    verifier = MultimodalStructuralVerifier(max_binding_delay_ms=32.0)
    cases: Dict[str, Any] = {}
    correct = 0
    admission_integrity = True
    for row in eligible:
        case_id = str(row.get("case_id", ""))
        evidence = _evidence(row)
        expected = str(row.get("expected_decision", ""))
        decision = verifier.verify(
            evidence, expected_modalities=row.get("expected_modalities", [])
        )
        matched = decision.decision == expected
        correct += int(matched)
        admission_allowed = _admission_allowed(case_id, evidence, decision)
        should_admit = expected == "verify_cross_modal_structure"
        admission_matched = admission_allowed == should_admit
        admission_integrity = admission_integrity and admission_matched
        cases[case_id] = {
            "expected_decision": expected,
            "decision": decision.to_dict(),
            "decision_matched": matched,
            "event_memory_admission_allowed": admission_allowed,
            "admission_matched": admission_matched,
            "source_domain": str(row.get("source_domain", "")),
        }
    decision_accuracy = correct / float(max(1, len(eligible)))
    behavior_checks = {
        "decision_accuracy": bool(eligible) and decision_accuracy == 1.0,
        "event_memory_admission_integrity": bool(eligible) and admission_integrity,
    }
    checks = {**provenance_checks, **coverage_checks, **behavior_checks}
    passed = all(checks.values())
    return {
        "schema": "sara-phase23-external-multimodal-gate-v1",
        "passed": passed,
        "promotion_allowed": passed,
        "observed_only": True,
        "source_scope": "independent_external",
        "independent_source_scope": {
            "domains": sorted(by_domain),
            "domain_count": len(by_domain),
            "case_count": len(eligible),
        },
        "metrics": {
            "eligible_case_count": len(eligible),
            "source_domain_count": len(by_domain),
            "modality_count": len(modality_set),
            "decision_accuracy": decision_accuracy,
        },
        "checks": checks,
        "decision_coverage": dict(sorted(expected_counts.items())),
        "modalities": sorted(modality_set),
        "cases": cases,
        "next_actions": [
            "Build the managed Phase 23 multimodal collection request.",
            "Collect source-backed aligned, missing, contradictory, and delayed cases.",
            "Rerun eval-phase23-external-multimodal before promotion review.",
        ] if not passed else [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    report = build_report(_load(args.manifest_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(
            f"Phase 23 external multimodal gate: {'PASS' if report['passed'] else 'FAIL'}\n"
        )
        handle.write(f"Promotion allowed: {report['promotion_allowed']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
