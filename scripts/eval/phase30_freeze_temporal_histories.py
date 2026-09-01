#!/usr/bin/env python3
"""Freeze evaluator-isolated Phase 30 temporal histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.phase30_fixtures import build_fixtures, write_jsonl  # noqa: E402
from sara_engine.utils.project_paths import processed_data_path, workspace_path  # noqa: E402


DEFAULT_INPUTS = processed_data_path("benchmark_fixtures", "phase30_temporal_effective_interaction_cases.jsonl")
DEFAULT_KEY = processed_data_path("benchmark_fixtures", "phase30_temporal_effective_interaction_evaluator_key.jsonl")
DEFAULT_MANIFEST = workspace_path("evaluation", "phase30_temporal_effective_interaction_fixture_freeze.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--evaluator-key", default=DEFAULT_KEY)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    inputs, keys, manifest = build_fixtures()
    write_jsonl(Path(args.inputs), inputs)
    write_jsonl(Path(args.evaluator_key), keys)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
