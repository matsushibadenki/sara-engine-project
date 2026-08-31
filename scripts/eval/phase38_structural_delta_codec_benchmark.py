#!/usr/bin/env python3
"""Evaluate the default-off Phase 38 codec on frozen execution identities."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)

from sara_engine.risa.structural_delta import CanonicalStructuralDeltaCodec  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

INPUTS = processed_data_path("benchmark_fixtures", "phase38_structural_delta_execution_inputs.jsonl")
KEY = processed_data_path("benchmark_fixtures", "phase38_structural_delta_evaluator_key.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase38_structural_delta_codec_benchmark.json")
EXPECTED_INPUT = "7dee3c55c7e5291bac8a3f7fc034740c84296dbcf8021f620c4761ba522398c2"
EXPECTED_KEY = "79b85c6d54ac1eeec8a1dc04711f9d2cacfe99fdb139425c6671b135790cec1a"


def _rows(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def _sha(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""): digest.update(block)
    return digest.hexdigest()


def build_report(inputs: Sequence[Dict[str, Any]], keys: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    codec = CanonicalStructuralDeltaCodec()
    key_by_id = {row["case_id"]: row for row in keys}
    valid_total = invalid_total = exact = digest = rollback = abstain = evidence = tombstone_total = tombstone_ok = 0
    results = []
    start = time.perf_counter(); event_cost = max_state = 0
    for row in inputs:
        candidate = codec.apply(row["base"], row["visible_delta"])
        repeated = codec.apply(row["base"], row["visible_delta"])
        key = key_by_id[row["case_id"]]
        valid = key["expected_decision"] == "materialize"
        exact_match = bool(valid and candidate.accepted and candidate.target == key["exact_target"])
        rollback_value = codec.rollback(candidate)
        rollback_match = bool(valid and rollback_value == row["base"])
        if valid:
            valid_total += 1; exact += int(exact_match); digest += int(exact_match and candidate.target["canonical_digest"] == key["exact_target"]["canonical_digest"]); rollback += int(rollback_match)
            evidence += int(bool(candidate.target and set(row["visible_delta"]["evidence_ids"]) <= set(candidate.target["evidence_ids"])))
            if key["operator"] in {"REMOVE_NODE", "REMOVE_RELATION"}:
                tombstone_total += 1; tombstone_ok += int(bool(candidate.target and candidate.target["tombstones"]))
        else:
            invalid_total += 1; abstain += int(not candidate.accepted)
        event_cost += candidate.event_cost; max_state = max(max_state, candidate.state_bytes)
        results.append({"case_id":row["case_id"],"candidate":candidate.to_dict(),"evaluation":{"exact_match":exact_match,"rollback_match":rollback_match,"deterministic":candidate.to_dict()==repeated.to_dict()}})
    latency = (time.perf_counter()-start)*1000.0
    metrics = {
        "exact_reconstruction_rate": exact/max(1,valid_total), "digest_match_rate":digest/max(1,valid_total), "rollback_fidelity":rollback/max(1,valid_total),
        "provenance_tombstone_preservation":tombstone_ok/max(1,tombstone_total), "justified_abstention_accuracy":abstain/max(1,invalid_total), "evidence_traceability":evidence/max(1,valid_total),
        "deterministic_replay":float(all(item["evaluation"]["deterministic"] for item in results)), "event_cost":float(event_cost), "max_state_bytes":float(max_state), "cpu_latency_ms":latency,
    }
    checks = {"valid_exact_gate":metrics["exact_reconstruction_rate"]==metrics["digest_match_rate"]==metrics["rollback_fidelity"]==1.0,"tombstone_gate":metrics["provenance_tombstone_preservation"]==1.0,"malformed_abstention_gate":metrics["justified_abstention_accuracy"]==1.0,"resource_gate":event_cost<=8192 and max_state<=131072,"all_provisional":all(not item["candidate"]["durable_mutation_allowed"] for item in results),"evaluator_labels_absent_from_candidate":True}
    passed = all(checks.values())
    return {"schema":"sara-phase38-structural-delta-codec-benchmark-v1","protocol_fingerprint":"9dfafe9ed01d80c0eadf1d59620391332a24d40a7029bbccbc919f98e67080dd","single_registered_attempt_consumed":True,"passed":passed,"retained_negative_result":not passed,"promotion_ready":False,"transformation_sharing_executed":False,"metrics":metrics,"checks":checks,"results":results,"claim_boundary":"Codec mechanics are synthetic and default-off. A failed malformed-control gate blocks transformation sharing and production claims; frozen labels are not retuned."}


def main(argv: Optional[Sequence[str]]=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-path",default=DEFAULT_REPORT); args=parser.parse_args(argv)
    if _sha(INPUTS)!=EXPECTED_INPUT or _sha(KEY)!=EXPECTED_KEY: print("Phase 38 frozen identity mismatch",file=sys.stderr); return 2
    report=build_report(_rows(INPUTS),_rows(KEY))
    with open(ensure_parent_directory(args.report_path),"w",encoding="utf-8") as handle: json.dump(report,handle,ensure_ascii=False,indent=2,sort_keys=True); handle.write("\n")
    print(json.dumps({"passed":report["passed"],"retained_negative_result":report["retained_negative_result"],"metrics":report["metrics"],"checks":report["checks"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
