#!/usr/bin/env python3
"""Register the immutable commit-pinned CPython source snapshot contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_cpython_snapshot import (  # noqa: E402
    build_preregistration,
    validate_preregistration,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_CASE_PLAN = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_case_plan.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_cpython_v3_14_6_snapshot_preregistration.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def register(case_plan_path: str, output_path: str) -> Dict[str, Any]:
    candidate = build_preregistration(_read_json(case_plan_path))
    resolved = ensure_parent_directory(output_path)
    if os.path.exists(resolved):
        existing = _read_json(resolved)
        if existing != candidate:
            raise ValueError(
                "existing CPython snapshot preregistration is immutable; use a new experiment identity"
            )
        status = "already_registered_identical"
    else:
        try:
            with open(resolved, "x", encoding="utf-8") as handle:
                json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError:
            if _read_json(resolved) != candidate:
                raise ValueError(
                    "concurrent CPython snapshot preregistration does not match"
                )
        status = "registered_new"
    validation = validate_preregistration(candidate)
    if not validation["valid"]:
        raise ValueError("generated CPython snapshot preregistration failed validation")
    return {
        "schema": "sara-phase34-cpython-source-snapshot-registration-receipt-v1",
        "registered": True,
        "status": status,
        "source_count": candidate["source_count"],
        "commit": candidate["commit"],
        "protocol_fingerprint": candidate["protocol_fingerprint"],
        "output_path": os.path.realpath(resolved),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan-path", default=DEFAULT_CASE_PLAN)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        receipt = register(args.case_plan_path, args.output_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
