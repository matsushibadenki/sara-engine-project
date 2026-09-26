#!/usr/bin/env python3
"""Retest Event Memory retention profiles on a frozen 0-100 external horizon."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.risa.structural_interpolation import StructuralEvidence, StructuralInterpolationEngine
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "structural_event_memory_long_horizon_protocol_v1.json"))
REPORT = Path(workspace_path("evaluation", "structural_event_memory_long_horizon_v1.json"))
SUMMARY = Path(workspace_path("evaluation", "structural_event_memory_long_horizon_v1_summary.txt"))
PROTOCOL_SHA256 = "dbda747b63b1b24a3e30f3de528a11516c443dd6791d7151fe2d59c9f4ba3709"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def folded_signature(source_hashes: Sequence[str], width: int) -> tuple[int, ...]:
    """Fold arbitrary evidence count into fixed sparse signature capacity."""
    if width < 1:
        raise ValueError("Signature width must be positive")
    slots = sorted({int(value, 16) % 4096 for value in source_hashes})
    if len(slots) <= width:
        return tuple(slots)
    folded = {int(sha256(f"{index}|{value}".encode()).hexdigest(), 16) % 4096
              for index, value in enumerate(sorted(source_hashes))}
    return tuple(sorted(folded)[:width])


def build_candidates(rows: Sequence[Mapping[str, Any]], width: int):
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    evidence = []
    for row in rows:
        if row.get("evidence_scope") != "independent_external":
            continue
        domain = str(row["source_domain"])
        grouped[domain].append(row)
        evidence.append(StructuralEvidence(
            source_node=f"domain:{domain}", target_node="capability:verified_event_replay",
            relation_type="supports", confidence=float(row["quality_score"]),
            source_ref=str(row["source_ref"]), source_hash=str(row["material_hash"]),
            source_revision=str(row["source_revision"]),
            context_tags=("architecture_migration", "independent_external", "long_horizon"),
            acquired_at=int(row["migration_horizon_index"]), metabolic_cost=int(row["event_cost"]),
            verified=bool(row["observed_only"]) and row["compliance_level"] == "allow"))
    interpolation = StructuralInterpolationEngine(max_proposals=16).propose(evidence, current_segment=1000)
    candidates = []
    for index, proposal in enumerate(interpolation.proposals, start=1):
        candidate = EventStateCandidate.from_verified_evidence(
            verifier_id="structural-event-memory-long-horizon-v1", evidence=proposal.to_dict(),
            entry_id=f"long-horizon-proposal-{index}",
            signature=folded_signature(proposal.source_hashes, width),
            source_ref=proposal.source_refs[0], source_revision="|".join(proposal.source_revisions),
            time_segment=proposal.acquired_at_max,
            own_latent_id=f"{proposal.relation_type}:{proposal.source_node}->{proposal.target_node}",
            causal_predecessors=(proposal.source_node, proposal.target_node),
            confidence=proposal.confidence_after, uncertainty=1.0 - proposal.confidence_after,
            source_reliability=1.0, resonance_score=proposal.confidence_after,
            sequence_support_score=1.0, sequence_support_count=proposal.evidence_count,
            credit_score=proposal.confidence_after, credit_responsibility=proposal.confidence_after,
            credit_confidence=proposal.confidence_after, credit_longevity=proposal.confidence_after,
            metabolic_headroom=1.0, observed=True, source_backed=True, verified=True,
            event_cost=proposal.metabolic_cost)
        candidates.append((candidate, proposal))
    return grouped, interpolation, candidates


def evaluate_profile(profile: str, max_entries: int, candidates, protocol: Mapping[str, Any]) -> dict[str, Any]:
    cache = VerifiedHierarchicalEventStateCache(
        retention_profile=profile, max_entries=max_entries,
        max_signature_width=protocol["signature_width"],
        retrieval_threshold=protocol["retrieval_threshold"])
    admissions = [cache.admit(candidate) for candidate, _ in candidates]
    contradiction_blocks = 0
    retrievals = []
    for index, (candidate, _) in enumerate(candidates, start=1):
        contradicted = EventStateCandidate(
            **{**candidate.__dict__, "entry_id": f"long-horizon-contradicted-{index}",
               "contradicted": True})
        contradiction_blocks += int(cache.admit(contradicted).decision == "block_contradiction")
        retrievals.append(cache.retrieve(
            candidate.signature, own_latent_id=candidate.own_latent_id,
            now_segment=protocol["query_segment"]))
    recalled = sum(int(result.matches and not result.abstained) for result in retrievals)
    total_cost = sum(result.event_cost for result in admissions) + sum(result.event_cost for result in retrievals)
    return {"max_entries": max_entries, "effective_tier_capacities": cache._tier_capacities(),
            "accepted_count": sum(int(result.accepted) for result in admissions),
            "contradiction_block_count": contradiction_blocks,
            "contradiction_recovery": recalled / max(1, contradiction_blocks),
            "retrieval_recall": recalled / max(1, len(candidates)),
            "cache_entry_count": len(cache.entries), "eviction_count": cache.eviction_count,
            "total_event_cost": total_cost,
            "state_bounded": len(cache.entries) <= max_entries}


def build_report(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    grouped, interpolation, candidates = build_candidates(rows, protocol["signature_width"])
    domain_counts = Counter({domain: len(items) for domain, items in grouped.items()})
    horizons = {domain: [min(int(row["migration_horizon_index"]) for row in items),
                         max(int(row["migration_horizon_index"]) for row in items)]
                for domain, items in grouped.items()}
    profiles = {name: evaluate_profile(name, config["max_entries"], candidates, protocol)
                for name, config in protocol["profiles"].items()}
    acceptance = protocol["acceptance"]
    checks = {
        "source_counts": dict(domain_counts) == protocol["required_domains"],
        "source_horizons": all(value == [protocol["required_horizon"]["minimum"],
                                          protocol["required_horizon"]["maximum"]]
                               for value in horizons.values()),
        "proposal_per_domain": len(candidates) == len(grouped) == 2,
        "durable_mutation_blocked": all(not proposal.durable_mutation_allowed
                                         for proposal in interpolation.proposals),
        "linear_recall": profiles["linear"]["retrieval_recall"] >= acceptance["linear_recall_minimum"],
        "logarithmic_recall": profiles["logarithmic"]["retrieval_recall"] >= acceptance["logarithmic_recall_minimum"],
        "contradiction_recovery": all(profiles[name]["contradiction_recovery"] >=
                                      acceptance["contradiction_recovery_minimum"]
                                      for name in ("linear", "logarithmic")),
        "scaled_state_bounded": all(profiles[name]["state_bounded"] and
                                    profiles[name]["cache_entry_count"] <= acceptance["scaled_profile_max_entries"]
                                    for name in ("linear", "logarithmic")),
        "scaled_evictions": all(profiles[name]["eviction_count"] <= acceptance["scaled_profile_max_evictions"]
                                for name in ("linear", "logarithmic")),
        "event_cost": all(value["total_event_cost"] <= acceptance["max_total_event_cost_per_profile"]
                          for value in profiles.values()),
    }
    return {"schema": "sara-structural-event-memory-long-horizon-result-v1",
            "passed": all(checks.values()), "observed_only": True,
            "domain_counts": dict(sorted(domain_counts.items())), "horizons": horizons,
            "proposal_count": len(candidates), "profiles": profiles, "checks": checks,
            "candidate_scored": False, "production_policy_changed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=str(REPORT))
    parser.add_argument("--summary-path", default=str(SUMMARY))
    args = parser.parse_args()
    protocol_digest = digest(PROTOCOL)
    if protocol_digest != PROTOCOL_SHA256:
        raise ValueError("Frozen long-horizon protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    manifest = Path(project_path(*protocol["manifest"].split("/")))
    if digest(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Frozen long-horizon manifest changed")
    dependencies = {
        "candidate_source": (
            "scripts/eval/structural_interpolation_event_memory_benchmark.py",
            protocol["candidate_source_sha256"],
        ),
        "cache_source": (
            "src/sara_engine/memory/event_state_cache.py",
            protocol["cache_source_sha256"],
        ),
    }
    dependency_hashes = {}
    for name, (relative_path, expected) in dependencies.items():
        path = Path(project_path(*relative_path.split("/")))
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
        dependency_hashes[name] = actual
    report = build_report(load_rows(manifest), protocol)
    report["frozen_inputs"] = {
        "protocol_sha256": protocol_digest,
        "manifest_sha256": protocol["manifest_sha256"],
        "source_dependencies": dependency_hashes,
    }
    Path(ensure_parent_directory(args.report_path)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary = [f"Structural Event Memory long horizon: {'PASS' if report['passed'] else 'FAIL'}"]
    summary.extend(f"- {name}: recall={value['retrieval_recall']}, entries={value['cache_entry_count']}, evictions={value['eviction_count']}, cost={value['total_event_cost']}"
                   for name, value in report["profiles"].items())
    Path(ensure_parent_directory(args.summary_path)).write_text("\n".join(summary) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
