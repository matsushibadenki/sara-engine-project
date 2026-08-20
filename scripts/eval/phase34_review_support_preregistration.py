#!/usr/bin/env python3
"""Register the immutable Phase 34 human-review support snapshot."""

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

from sara_engine.evaluation.phase34_review_support import build_preregistration  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REQUEST = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_review_support_preregistration.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        registration = build_preregistration(_read_json(args.request_path))
        output = ensure_parent_directory(args.output_path)
        if os.path.exists(output):
            if _read_json(output) != registration:
                raise ValueError("existing review-support preregistration differs")
            status = "identical_registration_preserved"
        else:
            with open(output, "x", encoding="utf-8") as handle:
                json.dump(registration, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            status = "registered_new"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": registration["schema"],
                "status": status,
                "source_count": registration["source_count"],
                "protocol_fingerprint": registration["protocol_fingerprint"],
                "output_path": os.path.realpath(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
