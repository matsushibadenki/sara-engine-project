#!/usr/bin/env python3
"""Build the immutable Phase 34 independent adapter draft and case plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_factorial_preregistration import (  # noqa: E402
    ARMS,
    REPLICATE_SEEDS,
)
from sara_engine.evaluation.phase34_independent_adapter_preregistration import (  # noqa: E402
    CASE_COUNT,
    CASE_FAMILIES,
    CASE_GENERATION,
    CLAIM_BOUNDARIES,
    EXECUTION_POLICY,
    EXPERIMENT_ID,
    HORIZONS,
    PARENT_PROTOCOL_FINGERPRINT,
    PARENT_BUDGETS,
    PARENT_THRESHOLDS,
    SCHEMA,
    SOURCE_DOMAINS,
    build_registered_manifest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MANIFEST = processed_data_path(
    "autobot", "architecture_migration_latent_manifest.jsonl"
)
DEFAULT_PARENT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_preregistration.json"
)
DEFAULT_PARENT_REPORT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_benchmark.json"
)
DEFAULT_EXTERNAL_GATE = workspace_path(
    "evaluation", "continual_horizon_external_gate.json"
)
DEFAULT_READINESS_GATE = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_gate.json"
)
DEFAULT_CASE_PLAN = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_case_plan.json"
)
DEFAULT_DRAFT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_preregistration_draft.json"
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_environment.json"
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("source manifest rows must be objects")
    return rows


def validate_sources(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_domain: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = str(row.get("source_domain", ""))
        if domain in SOURCE_DOMAINS:
            by_domain[domain].append(row)
    hashes = [str(row.get("material_hash", "")) for row in rows]
    refs = [str(row.get("source_ref", "")) for row in rows]
    if set(by_domain) != set(SOURCE_DOMAINS) or len(rows) != 202:
        raise ValueError("independent source snapshot must contain the frozen two-domain corpus")
    for domain in SOURCE_DOMAINS:
        domain_rows = sorted(
            by_domain[domain], key=lambda row: int(row.get("migration_horizon_index", -1))
        )
        indices = [int(row.get("migration_horizon_index", -1)) for row in domain_rows]
        if indices != list(range(101)):
            raise ValueError(f"source horizon is not contiguous through 100: {domain}")
        if any(
            row.get("observed_only") is not True
            or str(row.get("compliance_level", "")) != "allow"
            or not isinstance(row.get("sparse_signature"), list)
            for row in domain_rows
        ):
            raise ValueError(f"source eligibility failed: {domain}")
    if not all(hashes) or len(set(hashes)) != 202:
        raise ValueError("source material hashes must be unique")
    if not all(refs) or len(set(refs)) != 202:
        raise ValueError("source references must be unique")
    return {
        "record_count": 202,
        "records_per_domain": {domain: len(by_domain[domain]) for domain in SOURCE_DOMAINS},
        "horizon_span_per_domain": {domain: 100 for domain in SOURCE_DOMAINS},
        "unique_material_hash_count": len(set(hashes)),
        "unique_source_ref_count": len(set(refs)),
        "observed_only": True,
        "compliance_level": "allow",
    }


def _positions(horizon: int) -> List[int]:
    width = min(16, horizon + 1)
    if width == 1:
        return [0]
    return [(index * horizon) // (width - 1) for index in range(width)]


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    return float(len(left_set & right_set)) / float(len(union)) if union else 0.0


def build_case_plan(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    validate_sources(rows)
    by_domain: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_domain[str(row["source_domain"])][int(row["migration_horizon_index"])] = row
    cases: List[Dict[str, Any]] = []
    for domain in SOURCE_DOMAINS:
        for horizon in HORIZONS:
            stream = [by_domain[domain][position] for position in _positions(horizon)]
            target = stream[-4]
            decoy_candidates = [row for row in stream if row is not target]
            decoy = sorted(
                decoy_candidates,
                key=lambda row: (
                    -_jaccard(target["sparse_signature"], row["sparse_signature"]),
                    str(row["material_hash"]),
                ),
            )[0]
            targets = {
                "exact_identity_selection": target,
                "signature_decoy_selection": target,
                "old_identity_retention": stream[0],
                "recent_identity_control": stream[-1],
                "missing_identity_control": None,
                "stale_digest_control": target,
                "contradiction_control": target,
            }
            focus = {
                "exact_identity_selection": "selection",
                "signature_decoy_selection": "selection",
                "old_identity_retention": "retention",
                "recent_identity_control": "retention",
                "missing_identity_control": "safety",
                "stale_digest_control": "safety",
                "contradiction_control": "safety",
            }
            negative_mode = {
                "missing_identity_control": "missing",
                "stale_digest_control": "stale_digest",
                "contradiction_control": "contradiction",
            }
            for family in CASE_FAMILIES:
                selected_target = targets[family]
                query_hash = (
                    str(selected_target["material_hash"])
                    if selected_target is not None
                    else hashlib.sha256(
                        f"{domain}|{horizon}|missing_identity_control".encode("utf-8")
                    ).hexdigest()
                )
                cases.append(
                    {
                        "case_id": f"p34-independent:{domain}:h{horizon}:{family}",
                        "family": family,
                        "factor_focus": focus[family],
                        "source_domain": domain,
                        "horizon": horizon,
                        "stream_material_hashes": [str(row["material_hash"]) for row in stream],
                        "stream_source_refs": [str(row["source_ref"]) for row in stream],
                        "query_material_hash": query_hash,
                        "signature_decoy_material_hash": str(decoy["material_hash"]),
                        "negative_mode": negative_mode.get(family, "none"),
                        "observed_source_identity_task": True,
                        "semantic_accuracy_claim_allowed": False,
                    }
                )
    if len(cases) != CASE_COUNT:
        raise ValueError("independent adapter case plan count mismatch")
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-case-plan-v1",
        "case_count": len(cases),
        "source_domains": list(SOURCE_DOMAINS),
        "horizons": list(HORIZONS),
        "case_generation": CASE_GENERATION,
        "cases": cases,
    }


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-adapter-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def build_draft(
    rows: Sequence[Mapping[str, Any]],
    parent_preregistration: Mapping[str, Any],
    parent_report: Mapping[str, Any],
    external_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
    case_plan: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    snapshot = validate_sources(rows)
    if (
        parent_preregistration.get("protocol_fingerprint") != PARENT_PROTOCOL_FINGERPRINT
        or parent_report.get("protocol_fingerprint") != PARENT_PROTOCOL_FINGERPRINT
        or parent_report.get("execution_passed") is not True
        or parent_report.get("mechanism_gate_passed") is not True
        or parent_preregistration.get("budgets") != PARENT_BUDGETS
        or parent_preregistration.get("thresholds") != PARENT_THRESHOLDS
    ):
        raise ValueError("parent factorial identity or result is not eligible")
    if external_gate.get("promotion_allowed") is not True:
        raise ValueError("external 10/30/100 horizon gate has not passed")
    if (
        readiness_gate.get("independent_execution_ready") is not True
        or readiness_gate.get("promotion_ready") is not False
        or readiness_gate.get("selector_retuning_allowed") is not False
        or readiness_gate.get("query_aware_retention_allowed") is not False
    ):
        raise ValueError("independent readiness gate has not passed unchanged")
    if case_plan.get("case_count") != CASE_COUNT:
        raise ValueError("case plan does not match the frozen adapter")
    draft = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_protocol_fingerprint": PARENT_PROTOCOL_FINGERPRINT,
        "parent_factorial_report_fingerprint": _digest(dict(parent_report)),
        "registered_before_adapter_execution": True,
        "source_manifest_fingerprint": _digest(list(rows)),
        "external_gate_fingerprint": _digest(dict(external_gate)),
        "readiness_gate_fingerprint": _digest(dict(readiness_gate)),
        "case_plan_fingerprint": _digest(dict(case_plan)),
        "environment_fingerprint": _digest(dict(environment)),
        "source_snapshot": snapshot,
        "case_plan_count": CASE_COUNT,
        "source_domains": list(SOURCE_DOMAINS),
        "required_horizons": list(HORIZONS),
        "arms": list(ARMS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "case_generation": CASE_GENERATION,
        "budgets": PARENT_BUDGETS,
        "thresholds": PARENT_THRESHOLDS,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "execution_policy": EXECUTION_POLICY,
    }
    build_registered_manifest(draft, managed_path=True)
    return draft


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--parent-preregistration-path", default=DEFAULT_PARENT_PREREGISTRATION)
    parser.add_argument("--parent-report-path", default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--external-gate-path", default=DEFAULT_EXTERNAL_GATE)
    parser.add_argument("--readiness-gate-path", default=DEFAULT_READINESS_GATE)
    parser.add_argument("--case-plan-path", default=DEFAULT_CASE_PLAN)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT)
    parser.add_argument("--environment-path", default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args(argv)
    try:
        rows = _read_jsonl(args.manifest_path)
        case_plan = build_case_plan(rows)
        environment = environment_descriptor()
        draft = build_draft(
            rows,
            _read_json(args.parent_preregistration_path),
            _read_json(args.parent_report_path),
            _read_json(args.external_gate_path),
            _read_json(args.readiness_gate_path),
            case_plan,
            environment,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for path, value in (
        (args.case_plan_path, case_plan),
        (args.environment_path, environment),
        (args.draft_path, draft),
    ):
        with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        json.dumps(
            {
                "schema": "sara-phase34-independent-adapter-draft-receipt-v1",
                "case_count": CASE_COUNT,
                "condition_count": CASE_COUNT * len(ARMS) * len(REPLICATE_SEEDS),
                "source_manifest_fingerprint": draft["source_manifest_fingerprint"],
                "case_plan_fingerprint": draft["case_plan_fingerprint"],
                "draft_path": os.path.realpath(args.draft_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
