#!/usr/bin/env python3
"""Freeze evaluator-isolated Phase 39 execution histories."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.phase39_fixtures import build_fixtures, write_jsonl  # noqa: E402
from sara_engine.utils.project_paths import processed_data_path, workspace_path  # noqa: E402


INPUTS = processed_data_path("benchmark_fixtures", "phase39_anonymous_structure_reuse_cases.jsonl")
KEY = processed_data_path("benchmark_fixtures", "phase39_anonymous_structure_reuse_evaluator_key.jsonl")
MANIFEST = workspace_path("evaluation", "phase39_anonymous_structure_reuse_fixture_freeze.json")


def main() -> int:
    inputs, keys, manifest = build_fixtures()
    write_jsonl(Path(INPUTS), inputs)
    write_jsonl(Path(KEY), keys)
    output = Path(MANIFEST)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
