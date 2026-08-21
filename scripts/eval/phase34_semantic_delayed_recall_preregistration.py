#!/usr/bin/env python3
"""Register the immutable Phase 34 semantic delayed-recall contract."""

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

from sara_engine.evaluation.phase34_semantic_preregistration import (  # noqa: E402
    build_registered_manifest,
    compare_existing_registration,
    is_managed_preregistration_path,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_semantic_delayed_recall_preregistration.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read preregistration JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("preregistration draft must be a JSON object")
    return value


def register_manifest(draft_path: str, output_path: str) -> Dict[str, Any]:
    if not is_managed_preregistration_path(output_path):
        raise ValueError("preregistration output must be under workspace/")
    candidate = build_registered_manifest(_read_json(draft_path), managed_path=True)
    try:
        existing = _read_json(output_path)
    except ValueError:
        if os.path.exists(output_path):
            raise
        existing = {}
    allowed, status = compare_existing_registration(existing, candidate)
    if not allowed:
        raise ValueError("existing preregistration is immutable; use a new experiment identity")
    if not existing:
        resolved = ensure_parent_directory(output_path)
        try:
            with open(resolved, "x", encoding="utf-8") as handle:
                json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError:
            concurrent = _read_json(resolved)
            allowed, status = compare_existing_registration(concurrent, candidate)
            if not allowed:
                raise ValueError(
                    "existing preregistration is immutable; use a new experiment identity"
                )
    return {
        "schema": "sara-phase34-semantic-delayed-recall-registration-receipt-v1",
        "registered": True,
        "status": status,
        "experiment_id": candidate["experiment_id"],
        "protocol_fingerprint": candidate["protocol_fingerprint"],
        "output_path": os.path.realpath(output_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", required=True)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        receipt = register_manifest(args.draft_path, args.output_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
