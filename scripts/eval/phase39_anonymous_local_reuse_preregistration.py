#!/usr/bin/env python3
"""Register the immutable Phase 39 anonymous local-reuse protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.phase39_preregistration import build_registered_manifest, compare_existing_registration, is_managed_preregistration_path  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_OUTPUT = workspace_path("evaluation", "phase39_anonymous_structure_reuse_preregistration.json")


def _read(path: str):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preregistration_must_be_object")
    return value


def register_manifest(draft_path: str, output_path: str):
    if not is_managed_preregistration_path(output_path):
        raise ValueError("preregistration_output_must_be_managed")
    candidate = build_registered_manifest(_read(draft_path), managed_path=True)
    existing = _read(output_path) if os.path.exists(output_path) else {}
    allowed, status = compare_existing_registration(existing, candidate)
    if not allowed:
        raise ValueError("existing_preregistration_is_immutable")
    if not existing:
        with open(ensure_parent_directory(output_path), "x", encoding="utf-8") as handle:
            json.dump(candidate, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return {"registered": True, "status": status, "experiment_id": candidate["experiment_id"], "protocol_fingerprint": candidate["protocol_fingerprint"], "output_path": os.path.realpath(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-path", required=True)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        print(json.dumps(register_manifest(args.draft_path, args.output_path), ensure_ascii=False, indent=2, sort_keys=True))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
