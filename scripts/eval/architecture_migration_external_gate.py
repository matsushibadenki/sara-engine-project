#!/usr/bin/env python3
"""Gate architecture migration on provenance-qualified external Event Memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.architecture_migration import (  # noqa: E402
    ArchitectureMigrationCoordinator,
    ArchitectureMigrationPolicy,
)
from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_MANIFEST_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "architecture_migration_external_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "architecture_migration_external_gate_summary.txt")
MIN_RECORDS_PER_DOMAIN = 3


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _domain(row: Mapping[str, Any]) -> str:
    return str(urlparse(str(row.get("source_url", "") or "")).hostname or "").lower()


def _source_site(row: Mapping[str, Any]) -> str:
    domain = _domain(row)
    labels = [label for label in domain.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else domain


def _eligible_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    eligible = []
    for row in rows:
        domain = _domain(row)
        signature = row.get("sparse_signature")
        if (
            str(row.get("schema", "")) not in {
                "sara-own-latent-manifest-row-v1",
                "sara-architecture-migration-source-row-v1",
            }
            or not bool(row.get("observed_only", False))
            or str(row.get("compliance_level", "")) != "allow"
            or not domain
            or domain == "example.org"
            or not str(row.get("material_hash", ""))
            or not isinstance(signature, list)
            or not signature
        ):
            continue
        eligible.append(dict(row))
    return sorted(eligible, key=lambda row: (str(row["source_url"]), str(row["material_hash"])))


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = _eligible_rows(rows)
    source_sites = sorted({_source_site(row) for row in eligible})
    site_record_counts = {
        site: sum(1 for row in eligible if _source_site(row) == site)
        for site in source_sites
    }
    hashes = {str(row["material_hash"]) for row in eligible}
    blocked_reasons: List[str] = []
    if len(eligible) < 2:
        blocked_reasons.append("insufficient_provenance_qualified_records")
    if len(source_sites) < 2:
        blocked_reasons.append("insufficient_independent_source_sites")
    if any(count < MIN_RECORDS_PER_DOMAIN for count in site_record_counts.values()):
        blocked_reasons.append("insufficient_long_horizon_records_per_site")
    if len(hashes) != len(eligible):
        blocked_reasons.append("duplicate_material_hashes")
    legacy = VerifiedHierarchicalEventStateCache(retention_profile="fixed", max_entries=32, retrieval_threshold=0.1)
    selected = eligible[:16]
    for index, row in enumerate(selected):
        manifest_id = str(row.get("manifest_id", "") or row["material_hash"])
        legacy.admit(
            EventStateCandidate.from_verified_evidence(
                verifier_id="architecture-migration-external-gate",
                evidence=dict(row),
                entry_id=manifest_id,
                signature=tuple(int(value) for value in row["sparse_signature"][:64]),
                source_ref=str(row["source_url"]),
                source_revision=str(row["material_hash"]),
                time_segment=index,
                own_latent_id=f"predicts:external:{manifest_id}->latent:{row.get('latent_cluster_id', '')}",
                confidence=float(row.get("quality_score", 0.0) or 0.0),
                uncertainty=max(0.0, 1.0 - float(row.get("quality_score", 0.0) or 0.0)),
                source_reliability=float(row.get("quality_score", 0.0) or 0.0),
                resonance_score=max(0.65, float(row.get("quality_score", 0.0) or 0.0)),
                metabolic_headroom=1.0,
                observed=True,
                source_backed=True,
                verified=True,
                event_cost=int(row.get("event_cost", 0) or 0),
                architecture_version="sara-architecture-v1",
            )
        )
    target = VerifiedHierarchicalEventStateCache(retention_profile="fixed", max_entries=32, retrieval_threshold=0.1)
    migration = ArchitectureMigrationCoordinator(
        ArchitectureMigrationPolicy("sara-architecture-v1", "sara-architecture-v2", max_replay_candidates=16)
    ).migrate(legacy, target)
    recalled = 0
    for entry in legacy.entries.values():
        result = target.retrieve(entry.signature, own_latent_id=entry.own_latent_id, source_ref=entry.source_ref)
        recalled += int(bool(result.matches) and result.matches[0]["entry_id"].endswith(entry.entry_id))
    replay_recall = float(recalled) / float(max(1, len(legacy.entries)))
    if replay_recall < 1.0:
        blocked_reasons.append("target_replay_recall_regression")
    manifest_digest = hashlib.sha256(
        json.dumps(eligible, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    promotion_eligible = not blocked_reasons and len(legacy.entries) >= 2
    return {
        "schema": "sara-architecture-migration-external-gate-v1",
        "passed": promotion_eligible,
        "promotion_eligible": promotion_eligible,
        "external_provenance_qualified": bool(len(source_sites) >= 2 and "example.org" not in source_sites),
        "manifest_sha256": manifest_digest,
        "eligible_record_count": len(eligible),
        "independent_source_sites": source_sites,
        "site_record_counts": site_record_counts,
        "blocked_reasons": blocked_reasons,
        "metrics": {
            "independent_source_site_count": len(source_sites),
            "minimum_records_per_site": MIN_RECORDS_PER_DOMAIN,
            "target_replay_recall": replay_recall,
            "migration_admission_count": migration.to_dict()["admitted_count"],
            "migration_event_cost": sum(item.event_cost for item in migration.admissions),
            "target_state_budget_units": len(target.entries),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(_read_jsonl(args.manifest_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(
            f"Architecture migration external gate: {'PASS' if report['passed'] else 'BLOCKED'}\n"
            f"Independent source sites: {report['metrics']['independent_source_site_count']}\n"
            f"Blocked reasons: {','.join(report['blocked_reasons']) or 'none'}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
