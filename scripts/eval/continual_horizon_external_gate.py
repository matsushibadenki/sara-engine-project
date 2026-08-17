#!/usr/bin/env python3
"""Validate independent source coverage before using it for horizon promotion."""

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

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MANIFEST = processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "continual_horizon_external_gate.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "continual_horizon_external_gate_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(payload)
    return rows


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [
        row for row in rows
        if str(row.get("evidence_scope", "")) == "independent_external"
    ]
    by_domain: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_domain[str(row.get("source_domain", ""))].append(row)
    hashes = [str(row.get("material_hash", "")) for row in eligible]
    refs = [str(row.get("source_ref", "")) for row in eligible]
    revisions = [str(row.get("source_revision", "")) for row in eligible]
    collection_times = [str(row.get("collection_time", "")) for row in eligible]
    horizons = {
        domain: sorted(int(row.get("migration_horizon_index", -1)) for row in domain_rows)
        for domain, domain_rows in sorted(by_domain.items())
    }
    horizon_values = sorted({index for values in horizons.values() for index in values})
    required_horizon_buckets = {10, 30, 100}
    horizon_bucket_coverage = {
        str(bucket): any(max(values, default=-1) >= bucket for values in horizons.values())
        for bucket in sorted(required_horizon_buckets)
    }
    checks = {
        "independent_rows_present": len(eligible) > 0,
        "minimum_domains": len(by_domain) >= 2,
        "minimum_records_per_domain": bool(by_domain) and min(len(items) for items in by_domain.values()) >= 3,
        "unique_material_hashes": bool(hashes) and all(hashes) and len(set(hashes)) == len(hashes),
        "unique_source_refs": bool(refs) and all(refs) and len(set(refs)) == len(refs),
        "source_revisions_present": bool(revisions) and all(revisions),
        "collection_times_present": bool(collection_times) and all(collection_times),
        "contiguous_horizons_per_domain": all(
            values == list(range(len(values))) for values in horizons.values()
        ),
        "observed_allow_only": all(
            bool(row.get("observed_only", False)) and str(row.get("compliance_level", "")) == "allow"
            for row in eligible
        ),
    }
    promotion_checks = {
        "independent_manifest_quality": all(checks.values()),
        "required_horizon_buckets_present": all(horizon_bucket_coverage.values()),
        "multi_domain_horizon_coverage": all(
            all(max(values, default=-1) >= bucket for values in horizons.values())
            for bucket in sorted(required_horizon_buckets)
        ) if horizons else False,
    }
    return {
        "schema": "sara-continual-horizon-external-gate-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "promotion_allowed": all(promotion_checks.values()),
        "source_scope": "independent_external",
        "metrics": {
            "eligible_record_count": len(eligible),
            "source_domain_count": len(by_domain),
            "min_records_per_domain": min((len(items) for items in by_domain.values()), default=0),
            "unique_source_revision_count": len(set(revisions)),
            "horizon_span": max((int(row.get("migration_horizon_index", -1)) for row in eligible), default=-1),
            "minimum_domain_horizon_span": min(
                (max(values, default=-1) for values in horizons.values()),
                default=-1,
            ),
            "observed_horizon_value_count": len(horizon_values),
        },
        "checks": checks,
        "promotion_checks": promotion_checks,
        "horizon_bucket_coverage": horizon_bucket_coverage,
        "domain_horizons": horizons,
        "source_domains": sorted(by_domain),
        "next_actions": [
            "Collect independent records through horizon 10 for every source domain.",
            "Repeat collection through horizons 30 and 100 without reusing source hashes or revisions.",
            "Rerun eval-continual-horizon-external before promotion review.",
        ] if not all(promotion_checks.values()) else [],
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
        handle.write(f"Continual horizon external gate: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Promotion allowed: {report['promotion_allowed']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
        for key, value in report["promotion_checks"].items():
            handle.write(f"- promotion_check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
