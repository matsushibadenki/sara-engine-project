#!/usr/bin/env python3
"""Gate independent Phase 34 factorial execution without fabricating evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_factorial_preregistration import (  # noqa: E402
    is_managed_preregistration_path,
    validate_preregistration,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_preregistration.json"
)
DEFAULT_FACTORIAL_REPORT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_benchmark.json"
)
DEFAULT_EXTERNAL_GATE = workspace_path(
    "evaluation", "continual_horizon_external_gate.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_gate.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def build_report(
    preregistration: Mapping[str, Any],
    factorial_report: Mapping[str, Any],
    external_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    validation = validate_preregistration(preregistration, managed_path=True)
    registration_valid = bool(validation["valid"])
    protocol_fingerprint = str(preregistration.get("protocol_fingerprint", ""))
    report_protocol_matches = (
        str(factorial_report.get("protocol_fingerprint", ""))
        == protocol_fingerprint
    )
    domain_horizons = external_gate.get("domain_horizons", {})
    if not isinstance(domain_horizons, Mapping):
        domain_horizons = {}
    required = (10, 30, 100)
    domain_coverage = {
        str(domain): {
            str(bucket): max(
                (int(value) for value in values if type(value) is int),
                default=-1,
            )
            >= bucket
            for bucket in required
        }
        for domain, values in sorted(domain_horizons.items())
        if isinstance(values, list)
    }
    missing_targets = [
        {"source_domain": domain, "required_horizon": bucket}
        for domain, coverage in domain_coverage.items()
        for bucket in required
        if not coverage[str(bucket)]
    ]
    checks = {
        "factorial_registration_valid": registration_valid,
        "factorial_protocol_matches": report_protocol_matches,
        "synthetic_factorial_execution_passed": factorial_report.get(
            "execution_passed"
        )
        is True,
        "synthetic_factorial_mechanism_passed": factorial_report.get(
            "mechanism_gate_passed"
        )
        is True,
        "factorial_production_path_unchanged": factorial_report.get(
            "production_path_changed"
        )
        is False,
        "external_manifest_quality_passed": external_gate.get("passed") is True,
        "external_horizon_promotion_allowed": external_gate.get(
            "promotion_allowed"
        )
        is True,
        "minimum_two_external_domains": len(domain_coverage) >= 2,
        "every_domain_has_10_30_100": bool(domain_coverage)
        and all(all(values.values()) for values in domain_coverage.values()),
        "selector_retuning_allowed": False,
        "query_aware_retention_allowed": False,
    }
    readiness_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"selector_retuning_allowed", "query_aware_retention_allowed"}
    }
    independent_execution_ready = all(readiness_checks.values())
    blockers = [key for key, value in readiness_checks.items() if not value]
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-gate-v1",
        "source_scope": "independent_external",
        "observed_only": True,
        "protocol_fingerprint": protocol_fingerprint,
        "independent_execution_ready": independent_execution_ready,
        "promotion_ready": False,
        "production_path_changed": False,
        "selector_retuning_allowed": False,
        "query_aware_retention_allowed": False,
        "checks": checks,
        "blockers": blockers,
        "domain_coverage": domain_coverage,
        "missing_collection_targets": missing_targets,
        "metrics": {
            "source_domain_count": len(domain_coverage),
            "missing_horizon_target_count": len(missing_targets),
            "required_horizon_bucket_count": len(required),
        },
        "next_actions": (
            [
                "Collect unique observed external records for every listed domain/horizon target.",
                "Rerun eval-continual-horizon-external before independent factorial execution.",
                "Do not tune Top-k, retention, thresholds, or fixtures on collected evidence.",
            ]
            if not independent_execution_ready
            else [
                "Execute the frozen factorial adapter on the independent manifest.",
                "Keep production promotion blocked pending provenance review and human approval.",
            ]
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--factorial-report-path", default=DEFAULT_FACTORIAL_REPORT)
    parser.add_argument("--external-gate-path", default=DEFAULT_EXTERNAL_GATE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if not is_managed_preregistration_path(args.preregistration_path):
            raise ValueError("factorial preregistration must be under workspace/")
        report = build_report(
            _read_json(args.preregistration_path),
            _read_json(args.factorial_report_path),
            _read_json(args.external_gate_path),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "independent_execution_ready": report[
                    "independent_execution_ready"
                ],
                "promotion_ready": report["promotion_ready"],
                "missing_horizon_target_count": report["metrics"][
                    "missing_horizon_target_count"
                ],
                "output_path": os.path.realpath(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
