#!/usr/bin/env python3
"""Retest Event Memory retention across evaluator-isolated orderings."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "event_memory_retention_robustness_protocol_v1.json"))
PROTOCOL_SHA256 = "85249aa8aa305bc244f18338283fa2e730d3eb8c303770cff134a7bd538e09e5"
REPORT = Path(workspace_path("evaluation", "event_memory_retention_robustness_v1.json"))
SUMMARY = Path(workspace_path("evaluation", "event_memory_retention_robustness_v1_summary.txt"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("capacity_pressure_dependency", path)
    if spec is None or spec.loader is None:
        raise ValueError("Unable to load frozen candidate source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ordering(candidates, seed: int):
    ordered = sorted(
        candidates,
        key=lambda item: sha256(f"{seed}|{item[0].entry_id}".encode()).hexdigest(),
    )
    return [(replace(candidate, time_segment=index), band)
            for index, (candidate, band) in enumerate(ordered)]


def _query_utility(band_hits: Mapping[str, int], weights: Mapping[str, float], per_band: int) -> float:
    return sum((band_hits[band] / per_band) * float(weight) for band, weight in weights.items())


def build_report(
    rows: Sequence[Mapping[str, Any]],
    base_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    dependency,
) -> dict[str, Any]:
    candidates = dependency.build_candidates(rows, base_protocol)
    ordering_results = []
    fingerprints = set()
    for seed in protocol["ordering_seeds"]:
        ordered = _ordering(candidates, seed)
        ordering_ids = [candidate.entry_id for candidate, _ in ordered]
        fingerprint = sha256("|".join(ordering_ids).encode()).hexdigest()
        fingerprints.add(fingerprint)
        profiles = {
            name: dependency.evaluate_profile(name, ordered, base_protocol)
            for name in base_protocol["profiles"]
        }
        for values in profiles.values():
            values["query_utility"] = {
                query_name: _query_utility(
                    values["band_hits"], weights, base_protocol["candidates_per_band"]
                )
                for query_name, weights in protocol["query_profiles"].items()
            }
        ordering_results.append({
            "seed": seed, "ordering_fingerprint": fingerprint,
            "ordering_ids": ordering_ids, "profiles": profiles,
        })

    aggregate = {}
    for profile in base_protocol["profiles"]:
        aggregate[profile] = {
            "mean_query_utility": {
                query: mean(result["profiles"][profile]["query_utility"][query]
                            for result in ordering_results)
                for query in protocol["query_profiles"]
            },
            "mean_age_band_coverage": mean(
                result["profiles"][profile]["represented_age_bands"]
                for result in ordering_results
            ),
        }
    delayed_wins = sum(
        result["profiles"]["logarithmic"]["query_utility"]["delayed_heavy"]
        > result["profiles"]["fixed"]["query_utility"]["delayed_heavy"]
        for result in ordering_results
    )
    gate = protocol["promotion_gate"]
    delayed_margin = (
        aggregate["logarithmic"]["mean_query_utility"]["delayed_heavy"]
        - aggregate["fixed"]["mean_query_utility"]["delayed_heavy"]
    )
    balanced_margin = (
        aggregate["logarithmic"]["mean_query_utility"]["balanced"]
        - aggregate["fixed"]["mean_query_utility"]["balanced"]
    )
    promotion_checks = {
        "delayed_utility_margin": delayed_margin >= gate["minimum_delayed_utility_margin"],
        "delayed_ordering_wins": delayed_wins >= gate["minimum_delayed_ordering_wins"],
        "balanced_noninferiority": balanced_margin >= -gate["balanced_noninferiority_margin"],
    }
    acceptance = protocol["acceptance"]
    integrity_checks = {
        "ordering_count": len(ordering_results) == acceptance["ordering_count"],
        "distinct_orderings": len(fingerprints) >= acceptance["minimum_distinct_orderings"],
        "equal_state": all(
            values["entry_count"] == acceptance["expected_entry_count"]
            for result in ordering_results for values in result["profiles"].values()
        ),
        "equal_recall": all(
            values["aggregate_recall_fraction"] == acceptance["expected_recall_fraction"]
            for result in ordering_results for values in result["profiles"].values()
        ),
        "equal_cost": all(
            max(values["total_event_cost"] for values in result["profiles"].values())
            - min(values["total_event_cost"] for values in result["profiles"].values())
            <= acceptance["maximum_profile_event_cost_delta"]
            for result in ordering_results
        ),
        "evaluator_relevance_isolated": all(
            "query_utility" not in candidate.__dict__ and "relevance" not in candidate.__dict__
            for candidate, _ in candidates
        ),
    }
    return {
        "schema": "sara-event-memory-retention-robustness-result-v1",
        "passed": all(integrity_checks.values()),
        "promotion_gate_passed": all(promotion_checks.values()),
        "observed_only": True, "ordering_count": len(ordering_results),
        "aggregate": aggregate, "delayed_ordering_wins": delayed_wins,
        "delayed_utility_margin": delayed_margin, "balanced_utility_margin": balanced_margin,
        "integrity_checks": integrity_checks, "promotion_checks": promotion_checks,
        "orderings": ordering_results, "production_policy_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=str(REPORT))
    parser.add_argument("--summary-path", default=str(SUMMARY))
    args = parser.parse_args()
    protocol_hash = digest(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("Frozen robustness protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    base_path = Path(project_path(*protocol["base_protocol"].split("/")))
    source_path = Path(project_path(*protocol["candidate_source"].split("/")))
    if digest(base_path) != protocol["base_protocol_sha256"]:
        raise ValueError("Frozen base protocol changed")
    if digest(source_path) != protocol["candidate_source_sha256"]:
        raise ValueError("Frozen candidate source changed")
    base_protocol = json.loads(base_path.read_text())
    manifest = Path(project_path(*base_protocol["manifest"].split("/")))
    if digest(manifest) != base_protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    dependency = _load_module(source_path)
    report = build_report(dependency.load_rows(manifest), base_protocol, protocol, dependency)
    report["frozen_inputs"] = {
        "protocol_sha256": protocol_hash,
        "base_protocol_sha256": protocol["base_protocol_sha256"],
        "candidate_source_sha256": protocol["candidate_source_sha256"],
        "manifest_sha256": base_protocol["manifest_sha256"],
    }
    Path(ensure_parent_directory(args.report_path)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        f"Event Memory retention robustness: {'PASS' if report['passed'] else 'FAIL'}",
        f"Promotion gate: {'PASS' if report['promotion_gate_passed'] else 'BLOCKED'}",
        f"Delayed margin: {report['delayed_utility_margin']:.6f}",
        f"Delayed wins: {report['delayed_ordering_wins']}/{report['ordering_count']}",
    ]
    Path(ensure_parent_directory(args.summary_path)).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
