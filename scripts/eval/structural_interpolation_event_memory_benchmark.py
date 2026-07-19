#!/usr/bin/env python3
"""Evaluate structural proposals at the verified Event Memory boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache  # noqa: E402
from sara_engine.risa.structural_interpolation import StructuralEvidence, StructuralInterpolationEngine  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_MANIFEST = processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "structural_interpolation_event_memory_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "structural_interpolation_event_memory_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("evidence_scope", "")) == "independent_external":
            grouped[str(row.get("source_domain", ""))].append(row)
    evidence: List[StructuralEvidence] = []
    for domain, domain_rows in sorted(grouped.items()):
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
    interpolation = StructuralInterpolationEngine(max_proposals=16).propose(evidence, current_segment=999)
    candidates = []
    for index, proposal in enumerate(interpolation.proposals, start=1):
        signature = tuple(sorted(int(item[:8], 16) % 4096 for item in proposal.source_hashes))
        candidate = EventStateCandidate(
            entry_id=f"structural-proposal-{index}",
            signature=signature,
            source_ref=proposal.source_refs[0] if proposal.source_refs else "",
            source_revision="|".join(proposal.source_revisions),
            time_segment=proposal.acquired_at_max,
            own_latent_id=f"{proposal.relation_type}:{proposal.source_node}->{proposal.target_node}",
            causal_predecessors=(proposal.source_node, proposal.target_node),
            confidence=proposal.confidence_after,
            uncertainty=1.0 - proposal.confidence_after,
            source_reliability=1.0,
            resonance_score=proposal.confidence_after,
            sequence_support_score=0.8,
            sequence_support_count=proposal.evidence_count,
            credit_score=proposal.confidence_after,
            credit_responsibility=proposal.confidence_after,
            credit_confidence=proposal.confidence_after,
            credit_longevity=proposal.confidence_after,
            metabolic_headroom=1.0,
            observed=True,
            source_backed=True,
            verified=True,
            event_cost=proposal.metabolic_cost,
        )
        candidates.append((candidate, proposal))
    proposal_count = len(interpolation.proposals)
    profiles: Dict[str, Dict[str, Any]] = {}
    for profile in ("fixed", "linear", "logarithmic"):
        cache = VerifiedHierarchicalEventStateCache(
            retention_profile=profile,
            max_entries=4,
            retrieval_threshold=0.2,
        )
        admissions = []
        retrievals = []
        for index, (candidate, _proposal) in enumerate(candidates, start=1):
            admissions.append(cache.admit(candidate).to_dict())
            retrievals.append(cache.retrieve(candidate.signature, own_latent_id=candidate.own_latent_id, now_segment=1000).to_dict())
            blocked = EventStateCandidate(
                **{**candidate.__dict__, "entry_id": f"contradicted-{index}", "contradicted": True}
            )
            admissions.append(cache.admit(blocked).to_dict())
        accepted_count = sum(int(item["accepted"]) for item in admissions)
        contradiction_block_count = sum(int(item["decision"] == "block_contradiction") for item in admissions)
        recall_count = sum(int(bool(item["matches"]) and not item["abstained"]) for item in retrievals)
        profiles[profile] = {
            "accepted_count": accepted_count,
            "contradiction_block_count": contradiction_block_count,
            "retrieval_recall": recall_count / float(max(1, proposal_count)),
            "cache_entry_count": len(cache.entries),
            "eviction_count": cache.eviction_count,
            "admission_event_cost": sum(item["event_cost"] for item in admissions),
            "retrieval_event_cost": sum(item["event_cost"] for item in retrievals),
            "state_budget_bounded": len(cache.entries) <= 4,
            "admissions": admissions,
            "retrievals": retrievals,
        }
    fixed = profiles["fixed"]
    checks = {
        "proposal_count_present": proposal_count == len(grouped),
        "verified_proposals_admitted": fixed["accepted_count"] == proposal_count,
        "contradiction_blocked": fixed["contradiction_block_count"] == proposal_count,
        "retrieval_recall": fixed["retrieval_recall"] == 1.0,
        "state_budget_bounded": all(item["state_budget_bounded"] for item in profiles.values()),
        "durable_proposal_boundary": all(not item.durable_mutation_allowed for item in interpolation.proposals),
    }
    return {
        "schema": "sara-structural-interpolation-event-memory-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "metrics": {
            "source_domain_count": len(grouped),
            "proposal_count": proposal_count,
            "accepted_count": fixed["accepted_count"],
            "contradiction_block_count": fixed["contradiction_block_count"],
            "retrieval_recall": fixed["retrieval_recall"],
            "cache_entry_count": fixed["cache_entry_count"],
            "cache_event_cost": fixed["admission_event_cost"] + fixed["retrieval_event_cost"],
        },
        "checks": checks,
        "profiles": profiles,
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
        handle.write(f"Structural interpolation Event Memory benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Observed only: {report['observed_only']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
