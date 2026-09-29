#!/usr/bin/env python3
"""Compare Event Memory retention profiles under equal capacity pressure."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "event_memory_capacity_pressure_protocol_v1.json"))
PROTOCOL_SHA256 = "ea0daa7df763ce59a4efaace0b79334d91ebac992506433428b67142a9efe8ad"
REPORT = Path(workspace_path("evaluation", "event_memory_capacity_pressure_v1.json"))
SUMMARY = Path(workspace_path("evaluation", "event_memory_capacity_pressure_v1_summary.txt"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _signature(material_hash: str, width: int) -> tuple[int, ...]:
    return tuple(
        int(material_hash[offset:offset + 8], 16) % 65536
        for offset in range(0, width * 8, 8)
    )


def build_candidates(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]):
    external = [row for row in rows if row.get("evidence_scope") == "independent_external"]
    if len(external) < protocol["candidate_count"]:
        raise ValueError("Insufficient frozen external evidence")
    selected = [external[index * (len(external) - 1) // (protocol["candidate_count"] - 1)]
                for index in range(protocol["candidate_count"])]
    band_size = protocol["candidates_per_band"]
    band_parameters = {
        "early": (0.90, 0.80, 0.80, 0.20),
        "middle": (0.78, 0.95, 0.95, 0.05),
        "recent": (0.65, 1.00, 1.00, 0.00),
    }
    candidates = []
    for index, row in enumerate(selected):
        band = protocol["age_bands"][index // band_size]
        resonance, confidence, reliability, uncertainty = band_parameters[band]
        candidate = EventStateCandidate.from_verified_evidence(
            verifier_id="event-memory-capacity-pressure-v1",
            evidence={"material_hash": row["material_hash"], "age_band": band, "index": index},
            entry_id=f"capacity-pressure-{index}",
            signature=_signature(str(row["material_hash"]), protocol["signature_width"]),
            source_ref=str(row["source_ref"]), source_revision=str(row["source_revision"]),
            time_segment=index, own_latent_id=f"capacity-pressure:{index}",
            causal_predecessors=(f"age-band:{band}",), confidence=confidence,
            uncertainty=uncertainty, source_reliability=reliability,
            resonance_score=resonance, sequence_support_score=1.0,
            sequence_support_count=1, credit_score=1.0, credit_responsibility=1.0,
            credit_confidence=1.0, credit_longevity=1.0, metabolic_headroom=1.0,
            observed=True, source_backed=True, verified=True, event_cost=4,
        )
        candidates.append((candidate, band))
    return candidates


def evaluate_profile(profile: str, candidates, protocol: Mapping[str, Any]) -> dict[str, Any]:
    cache = VerifiedHierarchicalEventStateCache(
        retention_profile=profile, max_entries=protocol["max_entries"],
        max_signature_width=protocol["signature_width"],
        retrieval_threshold=protocol["retrieval_threshold"], top_k=1,
    )
    admissions = [cache.admit(candidate) for candidate, _ in candidates]
    band_hits = {band: 0 for band in protocol["age_bands"]}
    retrieval_cost = 0
    for candidate, band in candidates:
        result = cache.retrieve(
            candidate.signature, own_latent_id=candidate.own_latent_id,
            source_ref=candidate.source_ref, now_segment=protocol["query_segment"], top_k=1,
        )
        retrieval_cost += result.event_cost
        band_hits[band] += int(bool(result.matches) and not result.abstained)
    admission_cost = sum(result.event_cost for result in admissions)
    recalled = sum(band_hits.values())
    return {
        "entry_count": len(cache.entries),
        "eviction_count": cache.eviction_count,
        "effective_tier_capacities": cache._tier_capacities(),
        "tier_counts": cache.state_dict()["tier_counts"],
        "band_hits": band_hits,
        "represented_age_bands": sum(int(value > 0) for value in band_hits.values()),
        "aggregate_recall": recalled / len(candidates),
        "aggregate_recall_fraction": [recalled, len(candidates)],
        "admission_event_cost": admission_cost,
        "retrieval_event_cost": retrieval_cost,
        "total_event_cost": admission_cost + retrieval_cost,
    }


def build_report(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    candidates = build_candidates(rows, protocol)
    profiles = {name: evaluate_profile(name, candidates, protocol) for name in protocol["profiles"]}
    acceptance = protocol["acceptance"]
    expected_fraction = [acceptance["aggregate_recall_numerator"], acceptance["aggregate_recall_denominator"]]
    costs = [value["total_event_cost"] for value in profiles.values()]
    checks = {
        "candidate_pressure_exceeds_capacity": len(candidates) > protocol["max_entries"],
        "equal_entry_count": all(value["entry_count"] == acceptance["entry_count"] for value in profiles.values()),
        "equal_aggregate_recall": all(value["aggregate_recall_fraction"] == expected_fraction for value in profiles.values()),
        "linear_age_coverage": profiles["linear"]["represented_age_bands"] >= acceptance["minimum_linear_age_bands"],
        "logarithmic_age_coverage": profiles["logarithmic"]["represented_age_bands"] >= acceptance["minimum_logarithmic_age_bands"],
        "fixed_recent_bias": profiles["fixed"]["represented_age_bands"] <= acceptance["maximum_fixed_age_bands"],
        "equal_event_cost": max(costs) - min(costs) <= acceptance["maximum_event_cost_delta"],
        "bounded_evictions": all(value["eviction_count"] <= acceptance["maximum_evictions"] for value in profiles.values()),
    }
    return {
        "schema": "sara-event-memory-capacity-pressure-result-v1",
        "passed": all(checks.values()), "observed_only": True,
        "candidate_count": len(candidates), "max_entries": protocol["max_entries"],
        "profiles": profiles, "checks": checks,
        "interpretation": "Retention profiles redistribute equal bounded capacity; they do not increase aggregate recall.",
        "production_policy_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=str(REPORT))
    parser.add_argument("--summary-path", default=str(SUMMARY))
    args = parser.parse_args()
    protocol_hash = digest(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("Frozen capacity-pressure protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    manifest = Path(project_path(*protocol["manifest"].split("/")))
    cache_source = Path(project_path(*protocol["cache_source"].split("/")))
    if digest(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Frozen capacity-pressure manifest changed")
    if digest(cache_source) != protocol["cache_source_sha256"]:
        raise ValueError("Frozen Event Memory implementation changed")
    report = build_report(load_rows(manifest), protocol)
    report["frozen_inputs"] = {
        "protocol_sha256": protocol_hash,
        "manifest_sha256": protocol["manifest_sha256"],
        "cache_source_sha256": protocol["cache_source_sha256"],
    }
    Path(ensure_parent_directory(args.report_path)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [f"Event Memory capacity pressure: {'PASS' if report['passed'] else 'FAIL'}"]
    lines.extend(
        f"- {name}: recall={value['aggregate_recall_fraction'][0]}/{value['aggregate_recall_fraction'][1]}, bands={value['band_hits']}, cost={value['total_event_cost']}"
        for name, value in report["profiles"].items()
    )
    Path(ensure_parent_directory(args.summary_path)).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
