#!/usr/bin/env python3
"""Run the frozen sparse predictive-memory component comparison on CPU."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.sparse_predictive_memory import PROTOCOL_SHA256, run_seed
from sara_engine.utils.project_paths import (
    ensure_parent_directory, processed_data_path, project_path, workspace_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=workspace_path("evaluation", "sparse_predictive_memory_v1.json"))
    args = parser.parse_args()
    output = Path(ensure_parent_directory(args.output))
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_predictive_memory_v1.json")).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed; register a new experiment version")
    protocol = json.loads(raw)
    runs = [run_seed(seed, protocol) for seed in protocol["seeds"]]
    sources = ("src/sara_engine/evaluation/sparse_predictive_memory.py",
               "scripts/eval/sparse_predictive_memory.py")
    report = {"schema": "sara-sparse-predictive-memory-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "protocol": protocol,
              "source_sha256": {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources},
              "python": platform.python_version(), "platform": platform.platform(),
              "runs": runs, "component_gate_passed": all(all(run["gates"].values()) for run in runs),
              "promotion_authorized": False, "anonymous_factor_discovery": False,
              "spike_advantage_tested": False, "physical_energy_measured": False,
              "real_data_read": False}
    # Exclusive creation retains previous evidence; choose a fresh managed path for reruns.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"component_gate_passed": report["component_gate_passed"],
                      "seeds": len(runs), "report": str(output)}, sort_keys=True))
    return 0 if report["component_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
