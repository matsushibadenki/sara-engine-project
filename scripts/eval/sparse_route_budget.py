#!/usr/bin/env python3
"""Run the fixed actual-update-matched candidate-pruning study."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.sparse_route_budget import PROTOCOL_SHA256, run_case, summarize
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, project_path, workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=workspace_path("evaluation", "sparse_route_budget_v1.json"))
    args = parser.parse_args()
    output = Path(ensure_parent_directory(args.output))
    if output.exists():
        raise FileExistsError("Refusing to overwrite experiment evidence")
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_route_budget_v1.json")).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(raw)
    sources = ("src/sara_engine/evaluation/sparse_route_budget.py",
               "src/sara_engine/evaluation/sparse_factor_routing.py",
               "src/sara_engine/evaluation/sparse_predictive_memory.py",
               "scripts/eval/sparse_route_budget.py")
    source_hashes = {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources}
    runs = []
    for task in protocol["tasks"]:
        for seed in protocol["seeds"]:
            run = run_case(seed, task, protocol)
            runs.append(run)
            print(json.dumps({"task": task, "seed": seed, "integrity_passed": all(run["integrity"].values()),
                              "failed_component_gates": [k for k, v in run["component_gates"].items() if not v]}, sort_keys=True), flush=True)
    if any(sha256(Path(project_path(name)).read_bytes()).hexdigest() != value for name, value in source_hashes.items()):
        raise RuntimeError("Study source changed during execution")
    summary = summarize(runs, protocol)
    report = {"schema": "sara-sparse-route-budget-result-v1", "protocol": protocol,
              "protocol_sha256": PROTOCOL_SHA256, "source_sha256": source_hashes,
              "python": platform.python_version(), "platform": platform.platform(), "runs": runs,
              **summary, "promotion_authorized": False, "physical_energy_measured": False,
              "spike_advantage_tested": False, "real_data_read": False,
              "anonymous_concept_emergence": False, "equal_total_cpu_claimed": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"report": str(output), **summary}, sort_keys=True))
    return 0 if summary["execution_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
