#!/usr/bin/env python3
"""Run the frozen revision-gain policy benchmark once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.revision_gain_benchmark import PROTOCOL_SHA256, load_protocol, run_benchmark
from sara_engine.utils.project_paths import ensure_parent_directory, project_path, workspace_path


def main() -> int:
    protocol = load_protocol()
    marker = Path(ensure_parent_directory(workspace_path("evaluation", "revision_gain_policy_v1_attempt.json")))
    output = Path(ensure_parent_directory(workspace_path("evaluation", "revision_gain_policy_v1_result.json")))
    if output.exists():
        raise FileExistsError("Frozen revision gain result already exists")
    sources = [
        "src/sara_engine/neuro/neuron.py",
        "src/sara_engine/evaluation/local_outcome_snn_integration.py",
        "src/sara_engine/learning/observable_revision.py",
        "src/sara_engine/learning/revision_gain.py",
        "src/sara_engine/evaluation/revision_gain_benchmark.py",
        "scripts/eval/revision_gain_benchmark.py",
    ]
    provenance = {
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": {name: hashlib.sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources},
        "python": sys.version,
        "platform": platform.platform(),
    }
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    result = run_benchmark(protocol)
    result["provenance"] = provenance
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
