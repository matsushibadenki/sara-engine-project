#!/usr/bin/env python3
"""Execute the registered route-selection study and preserve negative gates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

from sara_engine.evaluation.sparse_factor_routing import PROTOCOL_SHA256, run_case
from sara_engine.utils.project_paths import (
    ensure_parent_directory, processed_data_path, project_path, workspace_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=workspace_path("evaluation", "sparse_factor_routing_v1.json"))
    args = parser.parse_args()
    output = Path(ensure_parent_directory(args.output))
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment report")
    raw = Path(processed_data_path("benchmark_fixtures", "sparse_factor_routing_v1.json")).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Protocol hash changed; register a new version")
    protocol = json.loads(raw)
    sources = ("src/sara_engine/evaluation/sparse_factor_routing.py",
               "src/sara_engine/evaluation/sparse_predictive_memory.py",
               "scripts/eval/sparse_factor_routing.py")
    source_hashes = {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in sources}
    runs = []
    for task in protocol["tasks"]:
        for seed in protocol["seeds"]:
            result = run_case(seed, task, protocol)
            runs.append(result)
            print(json.dumps({"task": task, "seed": seed,
                              "integrity_passed": all(result["integrity"].values()),
                              "specialization_passed": result["specialization_passed"],
                              "failed_gates": [key for key, value in result["specialization_gates"].items() if not value]}, sort_keys=True), flush=True)
    if any(sha256(Path(project_path(name)).read_bytes()).hexdigest() != digest
           for name, digest in source_hashes.items()):
        raise RuntimeError("Study source changed during execution")
    report = {"schema": "sara-sparse-factor-routing-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "protocol": protocol,
              "source_sha256": source_hashes, "python": platform.python_version(),
              "platform": platform.platform(), "runs": runs,
              "execution_integrity_passed": all(all(run["integrity"].values()) for run in runs),
              "specialization_passed": all(run["specialization_passed"] for run in runs),
              "promotion_authorized": False, "anonymous_concept_emergence": False,
              "spike_advantage_tested": False, "physical_energy_measured": False,
              "real_data_read": False, "confirmatory_real_world_evidence": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"report": str(output), "execution_integrity_passed": report["execution_integrity_passed"],
                      "specialization_passed": report["specialization_passed"]}, sort_keys=True))
    # A scientifically negative result is a completed run, not execution failure.
    return 0 if report["execution_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
