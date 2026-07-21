#!/usr/bin/env python3
"""Check deterministic canonical sparse IR readiness without claiming Rust equivalence."""

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

from sara_engine.edge.canonical_sparse_ir import canonicalize_events, migrate_state, replay_digest  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_portable_runtime_readiness.json")
DEFAULT_RUST_REPORT = workspace_path("evaluation", "rust_core_benchmark.json")


def build_report(rust_report: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    events = [
        {"event_id": "b", "timestep": 2, "channel": "audio", "spike_id": 11, "modality": "audio", "tags": ["x"]},
        {"event_id": "a", "timestep": 1, "channel": "vision", "spike_id": 7, "modality": "vision", "tags": ["y"]},
    ]
    state = {"schema": "sara-canonical-ir-state-v1", "ir_version": "sara-canonical-ir-v1", "events": events}
    canonical = [event.to_dict() for event in canonicalize_events(events)]
    migrated = migrate_state(state, from_version="sara-canonical-ir-v1", to_version="sara-canonical-ir-v1")
    digest_a = replay_digest(events)
    digest_b = replay_digest(list(reversed(events)))
    checks = {
        "canonical_order_deterministic": [item["event_id"] for item in canonical] == ["a", "b"],
        "replay_digest_deterministic": digest_a == digest_b,
        "state_migration_round_trip": migrated["events"] == canonical,
        "state_schema_preserved": migrated["schema"] == state["schema"],
        "rust_equivalence_not_claimed": True,
    }
    rust_report = rust_report or {}
    sparse_primitive_equivalence = bool(
        rust_report.get("output_equivalence_passed", False)
        and rust_report.get("rust_extension_available", False)
    )
    return {
        "schema": "sara-phase27-portable-runtime-readiness-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "rust_equivalence_claimed": False,
        "rust_sparse_primitive_equivalence_observed": sparse_primitive_equivalence,
        "canonical_ir_rust_equivalence_observed": False,
        "checks": checks,
        "metrics": {"canonical_event_count": len(canonical), "replay_digest": digest_a},
        "next_actions": [
            "Run Python/Rust replay equivalence after canonical IR is frozen.",
            "Add incompatible-version migration rejection cases.",
            "Measure state bytes and latency under equal traces.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--rust-report-path", default=DEFAULT_RUST_REPORT)
    args = parser.parse_args(argv)
    try:
        with open(args.rust_report_path, "r", encoding="utf-8") as handle:
            rust_report = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        rust_report = {}
    report = build_report(rust_report if isinstance(rust_report, dict) else {})
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
