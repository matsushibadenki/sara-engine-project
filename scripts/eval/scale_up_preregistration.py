#!/usr/bin/env python3
"""Register an immutable managed Phase 29 scale-up protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EVAL_PATH = os.path.dirname(os.path.abspath(__file__))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if EVAL_PATH not in sys.path:
    sys.path.insert(0, EVAL_PATH)

from scale_up_experiment_readiness import (  # noqa: E402
    DEFAULT_PREREGISTRATION,
    _is_managed_preregistration_path,
    preregistration_fingerprint,
    validate_preregistration,
)
from sara_engine.utils.project_paths import ensure_parent_directory  # noqa: E402


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read preregistration JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("preregistration draft must be a JSON object")
    return value


def build_registered_manifest(
    draft: Mapping[str, Any],
    *,
    managed_path: bool,
) -> Dict[str, Any]:
    candidate = dict(draft)
    candidate.pop("protocol_fingerprint", None)
    candidate["protocol_fingerprint"] = preregistration_fingerprint(candidate)
    validation = validate_preregistration(
        candidate,
        managed_path=managed_path,
    )
    if not validation["valid"]:
        raise ValueError(
            "invalid scale-up preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


def compare_existing_registration(
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Tuple[bool, str]:
    if not existing:
        return True, "new_registration"
    if dict(existing) == dict(candidate):
        return True, "identical_registration_preserved"
    return False, "existing_registration_is_immutable"


def register_manifest(
    draft_path: str,
    output_path: str,
) -> Dict[str, Any]:
    if not _is_managed_preregistration_path(output_path):
        raise ValueError("preregistration output must be under workspace/")
    candidate = build_registered_manifest(
        _read_json(draft_path),
        managed_path=True,
    )
    try:
        existing = _read_json(output_path)
    except ValueError:
        if os.path.exists(output_path):
            raise
        existing = {}
    allowed, status = compare_existing_registration(existing, candidate)
    if not allowed:
        raise ValueError(
            "existing preregistration is immutable; use a new experiment identity"
        )
    if not existing:
        resolved_output = ensure_parent_directory(output_path)
        try:
            with open(resolved_output, "x", encoding="utf-8") as handle:
                json.dump(
                    candidate,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
        except FileExistsError:
            concurrent_existing = _read_json(resolved_output)
            allowed, status = compare_existing_registration(
                concurrent_existing,
                candidate,
            )
            if not allowed:
                raise ValueError(
                    "existing preregistration is immutable; "
                    "use a new experiment identity"
                )
    return {
        "schema": "sara-scale-up-preregistration-receipt-v1",
        "registered": True,
        "status": status,
        "protocol_fingerprint": candidate["protocol_fingerprint"],
        "output_path": os.path.realpath(output_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", required=True)
    parser.add_argument("--output-path", default=DEFAULT_PREREGISTRATION)
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
