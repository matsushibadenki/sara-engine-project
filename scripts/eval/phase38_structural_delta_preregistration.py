#!/usr/bin/env python3
"""Register the immutable Phase 38 structural-delta protocol."""

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

from sara_engine.evaluation.phase38_preregistration import build_registered_manifest, compare_existing_registration, is_managed_preregistration_path  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_OUTPUT = workspace_path("evaluation", "phase38_structural_delta_preregistration.json")


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("preregistration must be an object")
    return value


def register_manifest(draft_path: str, output_path: str) -> Dict[str, Any]:
    if not is_managed_preregistration_path(output_path):
        raise ValueError("preregistration output must be under workspace/")
    candidate = build_registered_manifest(_read(draft_path), managed_path=True)
    try:
        existing = _read(output_path)
    except FileNotFoundError:
        existing = {}
    allowed, status = compare_existing_registration(existing, candidate)
    if not allowed:
        raise ValueError("existing preregistration is immutable; use a new experiment identity")
    if not existing:
        with open(ensure_parent_directory(output_path), "x", encoding="utf-8") as handle:
            json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return {"schema": "sara-phase38-registration-receipt-v1", "registered": True, "status": status, "experiment_id": candidate["experiment_id"], "protocol_fingerprint": candidate["protocol_fingerprint"], "output_path": os.path.realpath(output_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", required=True)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        receipt = register_manifest(args.draft_path, args.output_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
