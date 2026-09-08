#!/usr/bin/env python3
"""Execute the frozen R1 temporal-learning evaluation once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sara_engine.evaluation.r1_temporal_learning_benchmark import load_manifest, run_benchmark
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_MANIFEST = workspace_path("evaluation", "r1_temporal_learning_preregistration.json")
DEFAULT_OUTPUT = workspace_path("evaluation", "r1_temporal_learning_result.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = Path(ensure_parent_directory(args.output))
    if output.exists():
        raise RuntimeError("Frozen R1 evaluation already exists; a second run is forbidden")
    result = run_benchmark(load_manifest(args.manifest))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
