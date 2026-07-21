#!/usr/bin/env python3
"""Record explicit human approval bound to the current next-level evidence."""

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

from sara_engine.evaluation.promotion_approval import build_approval  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

from next_level_promotion_review import REPORT_FILES  # noqa: E402


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_evidence(evaluation_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        key: _read(os.path.join(evaluation_dir, filename))
        for key, filename in REPORT_FILES.items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--output-path",
        default=workspace_path("evaluation", "next_level_human_approval.json"),
    )
    args = parser.parse_args(argv)
    reports = load_evidence(args.evaluation_dir)
    approval = build_approval(reports, reviewer=args.reviewer, note=args.note)
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(approval, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
