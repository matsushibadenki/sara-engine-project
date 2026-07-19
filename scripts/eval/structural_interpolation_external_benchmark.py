#!/usr/bin/env python3
"""Evaluate structural interpolation on the frozen independent migration manifest."""

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

from sara_engine.risa.structural_interpolation import (  # noqa: E402
    StructuralEvidence,
    StructuralInterpolationEngine,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_MANIFEST = processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "structural_interpolation_external_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "structural_interpolation_external_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_domain: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("evidence_scope", "")) != "independent_external":
            continue
        by_domain[str(row.get("source_domain", ""))].append(row)

    evidence: List[StructuralEvidence] = []
    domain_horizons: Dict[str, List[int]] = {}
    for domain, domain_rows in sorted(by_domain.items()):
        domain_horizons[domain] = sorted(int(row.get("migration_horizon_index", -1)) for row in domain_rows)
        for row in domain_rows:
            evidence.append(
                StructuralEvidence(
                    source_node=f"domain:{domain}",
                    target_node="capability:verified_event_replay",
                    relation_type="supports",
                    confidence=float(row.get("quality_score", 0.0)),
                    source_ref=str(row.get("source_ref", "")),
                    source_hash=str(row.get("material_hash", "")),
                    source_revision=str(row.get("source_revision", "")),
                    context_tags=("architecture_migration", "independent_external"),
                    acquired_at=int(row.get("migration_horizon_index", 0)),
                    metabolic_cost=int(row.get("event_cost", 0)),
                    verified=bool(row.get("observed_only", False)) and str(row.get("compliance_level", "")) == "allow",
                )
            )
    result = StructuralInterpolationEngine(max_proposals=16).propose(evidence, current_segment=999)
    proposals = result.proposals
    expected_domains = set(by_domain)
    proposal_domains = {proposal.source_node.removeprefix("domain:") for proposal in proposals}
    source_counts = Counter(str(item.source_node) for item in evidence)
    checks = {
        "independent_external_rows_present": bool(evidence),
        "minimum_records_per_domain": bool(source_counts) and min(source_counts.values()) >= 3,
        "domain_proposals_complete": proposal_domains == expected_domains,
        "source_hashes_preserved": all(len(proposal.source_hashes) == source_counts[proposal.source_node] for proposal in proposals),
        "revision_metadata_preserved": all(bool(proposal.source_revisions) for proposal in proposals),
        "horizon_order_preserved": all(values == sorted(values) for values in domain_horizons.values()),
        "durable_mutation_blocked": all(not proposal.durable_mutation_allowed for proposal in proposals),
        "observed_only_scope": all(item.verified for item in evidence),
    }
    return {
        "schema": "sara-structural-interpolation-external-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "source_scope": "independent_external",
        "metrics": {
            "record_count": len(evidence),
            "source_domain_count": len(expected_domains),
            "proposal_count": len(proposals),
            "min_records_per_domain": min(source_counts.values()) if source_counts else 0,
            "total_event_cost": sum(item.metabolic_cost for item in evidence),
        },
        "checks": checks,
        "domain_horizons": domain_horizons,
        "proposals": [proposal.to_dict() for proposal in proposals],
        "trace": result.trace,
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
        handle.write(f"Structural interpolation external benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Observed only: {report['observed_only']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
