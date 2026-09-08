#!/usr/bin/env python3
"""Register the immutable R1 temporal-learning protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sara_engine.evaluation.r1_temporal_learning_preregistration import (
    build_registered_manifest,
    compare_existing_registration,
    is_managed_preregistration_path,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_DRAFT = workspace_path("evaluation", "r1_temporal_learning_preregistration_draft.json")
DEFAULT_OUTPUT = workspace_path("evaluation", "r1_temporal_learning_preregistration.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--overrides")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    draft_path = Path(args.draft)
    output_path = Path(ensure_parent_directory(args.output))
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft.pop("protocol_fingerprint", None)
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
        draft.update(overrides)
    manifest = build_registered_manifest(
        draft,
        managed_path=is_managed_preregistration_path(str(output_path)),
    )

    existing = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    allowed, reason = compare_existing_registration(existing, manifest)
    if not allowed:
        raise RuntimeError(reason)
    if reason == "new_registration":
        output_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": reason, "output": str(output_path), "protocol_fingerprint": manifest["protocol_fingerprint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
