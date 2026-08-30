#!/usr/bin/env python3
"""Check deterministic canonical sparse IR readiness without claiming Rust equivalence."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.canonical_sparse_ir import (  # noqa: E402
    canonical_json,
    canonicalize_events,
    migrate_state,
    replay_digest,
)
from sara_engine.edge.portable_decision_trace import (  # noqa: E402
    canonical_decision_json,
    decision_trace_digest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_portable_runtime_readiness.json")
DEFAULT_RUST_REPORT = workspace_path("evaluation", "rust_core_benchmark.json")
DEFAULT_TOKENIZER_REPORT = workspace_path(
    "evaluation", "phase27_tokenizer_acceleration_benchmark.json"
)
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase27_canonical_ir_cases.jsonl"
)
DEFAULT_DECISION_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase27_portable_decision_cases.jsonl"
)


def load_cases(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_conformance_cases(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    digests: Dict[str, str] = {}
    all_passed = True
    for row in rows:
        case_id = str(row["case_id"])
        expected_valid = bool(row["valid"])
        try:
            encoded = canonical_json(row["events"])
            digest = replay_digest(row["events"])
            error = ""
            observed_valid = True
        except (TypeError, ValueError) as exc:
            encoded = ""
            digest = ""
            error = str(exc)
            observed_valid = False
        expected_digest_case = str(row.get("same_digest_as", ""))
        expected_digest = str(row.get("expected_digest", ""))
        digest_matches = (
            (not expected_digest or digest == expected_digest)
            and (
                not expected_digest_case
                or (
                    observed_valid
                    and expected_digest_case in digests
                    and digest == digests[expected_digest_case]
                )
            )
        )
        error_matches = (
            expected_valid
            or str(row.get("error_contains", "")) in error
        )
        passed = (
            observed_valid == expected_valid
            and digest_matches
            and error_matches
        )
        if observed_valid:
            digests[case_id] = digest
        all_passed = all_passed and passed
        results[case_id] = {
            "passed": passed,
            "expected_valid": expected_valid,
            "observed_valid": observed_valid,
            "digest": digest or None,
            "canonical_bytes": len(encoded.encode("utf-8")),
            "error": error or None,
            "expected_digest": expected_digest or None,
            "same_digest_as": expected_digest_case or None,
        }
    return {
        "passed": all_passed and bool(rows),
        "case_count": len(rows),
        "cases": results,
    }


def load_rust_core() -> Any | None:
    for module_name in ("sara_engine.sara_rust_core", "sara_rust_core"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def evaluate_rust_canonical_equivalence(
    rows: Sequence[Mapping[str, Any]],
    rust_core: Any | None,
) -> Dict[str, Any]:
    canonical_fn = getattr(rust_core, "canonical_sparse_ir_json", None)
    digest_fn = getattr(rust_core, "canonical_sparse_ir_replay_digest", None)
    available = callable(canonical_fn) and callable(digest_fn)
    cases: Dict[str, Any] = {}
    all_passed = available and bool(rows)
    for row in rows:
        case_id = str(row["case_id"])
        expected_valid = bool(row["valid"])
        if not available:
            cases[case_id] = {"passed": False, "observed": False}
            continue
        source = json.dumps(
            row["events"], ensure_ascii=True, separators=(",", ":")
        )
        try:
            rust_json = str(canonical_fn(source))
            rust_digest = str(digest_fn(source))
            rust_valid = True
            rust_error = ""
        except (TypeError, ValueError) as exc:
            rust_json = ""
            rust_digest = ""
            rust_valid = False
            rust_error = str(exc)
        if expected_valid:
            python_json = canonical_json(row["events"])
            python_digest = replay_digest(row["events"])
            passed = (
                rust_valid
                and rust_json == python_json
                and rust_digest == python_digest
            )
        else:
            passed = not rust_valid
        all_passed = all_passed and passed
        cases[case_id] = {
            "passed": passed,
            "observed": True,
            "expected_valid": expected_valid,
            "rust_valid": rust_valid,
            "rust_digest": rust_digest or None,
            "rust_error": rust_error or None,
        }
    return {
        "available": available,
        "passed": all_passed,
        "case_count": len(rows),
        "cases": cases,
    }


def evaluate_rust_decision_equivalence(
    rows: Sequence[Mapping[str, Any]], rust_core: Any | None
) -> Dict[str, Any]:
    canonical_fn = getattr(rust_core, "canonical_portable_decision_trace_json", None)
    digest_fn = getattr(rust_core, "portable_decision_trace_digest", None)
    available = callable(canonical_fn) and callable(digest_fn)
    if not available or not rows:
        return {"available": available, "passed": False, "case_count": len(rows)}
    source = json.dumps(list(rows), ensure_ascii=True, separators=(",", ":"))
    try:
        rust_json = str(canonical_fn(source))
        rust_digest = str(digest_fn(source))
    except (TypeError, ValueError) as exc:
        return {
            "available": True,
            "passed": False,
            "case_count": len(rows),
            "error": str(exc),
        }
    python_json = canonical_decision_json(rows)
    python_digest = decision_trace_digest(rows)
    return {
        "available": True,
        "passed": rust_json == python_json and rust_digest == python_digest,
        "case_count": len(rows),
        "canonical_bytes_equivalent": rust_json == python_json,
        "digest_equivalent": rust_digest == python_digest,
        "python_digest": python_digest,
        "rust_digest": rust_digest,
    }


def build_report(
    rust_report: Optional[Mapping[str, Any]] = None,
    conformance_rows: Sequence[Mapping[str, Any]] = (),
    tokenizer_report: Optional[Mapping[str, Any]] = None,
    rust_core: Any | None = None,
    decision_rows: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    if not conformance_rows:
        conformance_rows = load_cases(DEFAULT_FIXTURE)
    if not decision_rows:
        decision_rows = load_cases(DEFAULT_DECISION_FIXTURE)
    events = [
        {
            "event_id": "b",
            "timestep": 2,
            "channel": "audio",
            "spike_id": 11,
            "modality": "audio",
            "tags": ["x"],
        },
        {
            "event_id": "a",
            "timestep": 1,
            "channel": "vision",
            "spike_id": 7,
            "modality": "vision",
            "tags": ["y"],
        },
    ]
    state = {
        "schema": "sara-canonical-ir-state-v1",
        "ir_version": "sara-canonical-ir-v1",
        "events": events,
    }
    canonical = [event.to_dict() for event in canonicalize_events(events)]
    migrated = migrate_state(
        state,
        from_version="sara-canonical-ir-v1",
        to_version="sara-canonical-ir-v1",
    )
    digest_a = replay_digest(events)
    digest_b = replay_digest(list(reversed(events)))
    conformance = evaluate_conformance_cases(conformance_rows)
    rust_canonical = evaluate_rust_canonical_equivalence(
        conformance_rows, rust_core
    )
    rust_decisions = evaluate_rust_decision_equivalence(decision_rows, rust_core)
    try:
        replay_digest(events + [dict(events[0])])
        duplicate_event_rejected = False
    except ValueError:
        duplicate_event_rejected = True
    checks = {
        "canonical_order_deterministic": [
            item["event_id"] for item in canonical
        ]
        == ["a", "b"],
        "replay_digest_deterministic": digest_a == digest_b,
        "state_migration_round_trip": migrated["events"] == canonical,
        "state_schema_preserved": migrated["schema"] == state["schema"],
        "invalid_event_rejected": duplicate_event_rejected,
        "conformance_vectors_passed": conformance["passed"],
        "rust_equivalence_evidence_consistent": (
            not rust_canonical["available"] or rust_canonical["passed"]
        ),
        "rust_decision_evidence_consistent": (
            not rust_decisions["available"] or rust_decisions["passed"]
        ),
    }
    rust_report = rust_report or {}
    tokenizer_report = tokenizer_report or {}
    sparse_primitive_equivalence = bool(
        rust_report.get("output_equivalence_passed", False)
        and rust_report.get("rust_extension_available", False)
    )
    tokenizer_checks = tokenizer_report.get("checks", {})
    tokenizer_conformance_observed = bool(
        tokenizer_report.get("passed", False)
        and tokenizer_report.get("observed_only", False)
        and tokenizer_report.get("production_path_changed") is False
        and isinstance(tokenizer_checks, Mapping)
        and tokenizer_checks.get("token_ids_equivalent", False)
        and tokenizer_checks.get("decode_round_trip_preserved", False)
        and tokenizer_checks.get("spike_event_digest_equivalent", False)
    )
    rust_scalar_tokenizer_equivalence = bool(
        tokenizer_report.get("rust_scalar_reference_available", False)
        and tokenizer_report.get("rust_scalar_reference_equivalent", False)
        and tokenizer_report.get("rust_path_observed", False)
    )
    checks["tokenizer_acceleration_not_promoted"] = (
        tokenizer_report.get("production_path_changed") is not True
    )
    return {
        "schema": "sara-phase27-portable-runtime-readiness-v2",
        "passed": all(checks.values()),
        "observed_only": True,
        "rust_equivalence_claimed": rust_canonical["passed"],
        "rust_sparse_primitive_equivalence_observed": sparse_primitive_equivalence,
        "canonical_ir_rust_equivalence_observed": rust_canonical["passed"],
        "portable_decision_rust_equivalence_observed": rust_decisions["passed"],
        "tokenizer_acceleration_conformance_observed": (
            tokenizer_conformance_observed
        ),
        "rust_scalar_tokenizer_equivalence_observed": (
            rust_scalar_tokenizer_equivalence
        ),
        "tokenizer_acceleration_production_promoted": False,
        "checks": checks,
        "conformance": conformance,
        "rust_canonical_conformance": rust_canonical,
        "rust_decision_conformance": rust_decisions,
        "metrics": {"canonical_event_count": len(canonical), "replay_digest": digest_a},
        "next_actions": [
            "Run Python/Rust replay equivalence after canonical IR is frozen.",
            "Use the frozen conformance vectors for the Rust replay implementation.",
            (
                "Keep the Rust scalar tokenizer as a correctness reference; "
                "measure an optional accelerated candidate separately."
                if rust_scalar_tokenizer_equivalence
                else "Compare the exact tokenizer snapshot with a Rust scalar reference."
            ),
            "Measure state bytes and latency under equal traces.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--rust-report-path", default=DEFAULT_RUST_REPORT)
    parser.add_argument(
        "--tokenizer-report-path", default=DEFAULT_TOKENIZER_REPORT
    )
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--decision-fixture-path", default=DEFAULT_DECISION_FIXTURE)
    args = parser.parse_args(argv)
    try:
        with open(args.rust_report_path, "r", encoding="utf-8") as handle:
            rust_report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        rust_report = {}
    try:
        with open(args.tokenizer_report_path, "r", encoding="utf-8") as handle:
            tokenizer_report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        tokenizer_report = {}
    report = build_report(
        rust_report if isinstance(rust_report, dict) else {},
        load_cases(args.fixture_path),
        tokenizer_report if isinstance(tokenizer_report, dict) else {},
        load_rust_core(),
        load_cases(args.decision_fixture_path),
    )
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
