#!/usr/bin/env python3
"""Run the single registered R2 Beijing development attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

from sara_engine.evaluation.r2_beijing_development import PROTOCOL_SHA256, load_protocol, run_development
from sara_engine.utils.project_paths import ensure_parent_directory, project_path, workspace_path


def main() -> int:
    protocol = load_protocol()
    marker = Path(ensure_parent_directory(workspace_path("evaluation", "r2_beijing_development_v1_attempt.json")))
    output = Path(ensure_parent_directory(workspace_path("evaluation", "r2_beijing_development_v1_result.json")))
    if output.exists():
        raise FileExistsError("R2 Beijing development result already exists")
    sources = [
        "src/sara_engine/neuro/neuron.py", "src/sara_engine/learning/local_outcome.py",
        "src/sara_engine/learning/observable_revision.py",
        "src/sara_engine/evaluation/r2_beijing_development.py", "scripts/eval/r2_beijing_development.py",
    ]
    provenance = {"protocol_sha256": PROTOCOL_SHA256,
                  "sources": {name: hashlib.sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources},
                  "python": sys.version, "platform": platform.platform()}
    if marker.exists():
        recovery = Path(ensure_parent_directory(workspace_path("evaluation", "r2_beijing_development_v1_recovery.json")))
        recovery_record = dict(provenance)
        recovery_record["reason"] = "Same-protocol recovery after AttributeError before any result artifact or decision was emitted"
        with recovery.open("x", encoding="utf-8") as handle:
            json.dump(recovery_record, handle, indent=2); handle.write("\n")
    else:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2); handle.write("\n")
    result = run_development(protocol); result["provenance"] = provenance
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2); handle.write("\n")
    print(json.dumps({"output": str(output), "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
