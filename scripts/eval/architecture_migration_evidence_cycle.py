#!/usr/bin/env python3
"""Run qualification, external migration gating, and collection-request handoff."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


def _load(name: str):
    path = os.path.join(os.path.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location(f"architecture_migration_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cycle(input_path: str) -> Dict[str, Any]:
    builder = _load("architecture_migration_manifest_builder.py")
    gate = _load("architecture_migration_external_gate.py")
    request = _load("architecture_migration_collection_request.py")
    source_rows = builder._read_jsonl(input_path)
    qualified = builder.build_manifest(source_rows)
    gate_report = gate.build_report(qualified)
    targets = request.build_targets(gate_report)
    return {
        "schema": "sara-architecture-migration-evidence-cycle-v1",
        "observed_only": True,
        "input_record_count": len(source_rows),
        "qualified_record_count": len(qualified),
        "gate": gate_report,
        "collection_targets": targets,
        "promotion_eligible": bool(gate_report.get("promotion_eligible", False)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=processed_data_path("autobot", "latent_manifest.jsonl"))
    parser.add_argument("--report-path", default=workspace_path("evaluation", "architecture_migration_evidence_cycle.json"))
    args = parser.parse_args(argv)
    report = run_cycle(args.input_path)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
