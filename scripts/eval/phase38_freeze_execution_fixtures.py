#!/usr/bin/env python3
"""Freeze evaluator-isolated Phase 38 synthetic structural histories."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase38_preregistration import CASE_FAMILIES, OPERATORS  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

PROTOCOL = "9dfafe9ed01d80c0eadf1d59620391332a24d40a7029bbccbc919f98e67080dd"
DEFAULT_SOURCES = processed_data_path("autobot", "phase38_structural_history_manifest.jsonl")
DEFAULT_TRAIN = processed_data_path("benchmark_fixtures", "phase38_structural_delta_train.jsonl")
DEFAULT_INPUTS = processed_data_path("benchmark_fixtures", "phase38_structural_delta_execution_inputs.jsonl")
DEFAULT_KEY = processed_data_path("benchmark_fixtures", "phase38_structural_delta_evaluator_key.jsonl")
DEFAULT_RECEIPT = workspace_path("evaluation", "phase38_execution_fixture_freeze_receipt.json")


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _base(identity: str, revision: int = 1) -> Dict[str, Any]:
    value = {
        "schema": "phase38-canonical-structure-v1",
        "structure_id": identity,
        "revision": revision,
        "nodes": [
            {"role": "role:source", "type": "entity", "value": f"{identity}:source"},
            {"role": "role:target", "type": "entity", "value": f"{identity}:target"},
        ],
        "relations": [{"source_role": "role:source", "relation_type": "observes", "target_role": "role:target", "order": 0, "evidence_ids": [f"evidence::{identity}"]}],
        "tombstones": [],
        "evidence_ids": [f"evidence::{identity}"],
    }
    value["canonical_digest"] = _digest(value)
    return value


def _operation(operator: str, index: int) -> Dict[str, Any]:
    common = {"operator": operator, "role_path": "role:target", "evidence_ids": [f"evidence::delta-{index}"], "precondition_revision": 1}
    values = {
        "ADD_NODE": {"node": {"role": "role:mediator", "type": "event", "value": f"mediator-{index}"}},
        "REMOVE_NODE": {"role": "role:target"},
        "ADD_RELATION": {"relation": {"source_role": "role:source", "relation_type": "supports", "target_role": "role:target", "order": 1, "evidence_ids": [f"evidence::delta-{index}"]}},
        "REMOVE_RELATION": {"relation_type": "observes"},
        "CHANGE_ROLE": {"from_role": "role:target", "to_role": "role:mediator"},
        "CHANGE_VALUE": {"role": "role:target", "value": f"revised-{index}"},
        "GENERALIZE": {"role": "role:target", "type": "concept"},
        "SPECIALIZE": {"role": "role:target", "type": "specialized_entity"},
        "REORDER_TIME": {"relation_type": "observes", "order": 2},
        "MERGE": {"roles": ["role:source", "role:target"], "merged_role": "role:source"},
        "SPLIT": {"role": "role:target", "new_roles": ["role:mediator", "role:target"]},
    }
    return common | values[operator]


def _reference_apply(base: Mapping[str, Any], operation: Mapping[str, Any]) -> Dict[str, Any]:
    target = deepcopy(dict(base))
    target.pop("canonical_digest", None)
    op = operation["operator"]
    if op == "ADD_NODE": target["nodes"].append(deepcopy(operation["node"]))
    elif op == "REMOVE_NODE":
        removed = [node for node in target["nodes"] if node["role"] == operation["role"]]
        target["nodes"] = [node for node in target["nodes"] if node["role"] != operation["role"]]
        target["relations"] = [rel for rel in target["relations"] if operation["role"] not in (rel["source_role"], rel["target_role"])]
        target["tombstones"].append({"operator": op, "removed": removed, "evidence_ids": list(operation["evidence_ids"])})
    elif op == "ADD_RELATION": target["relations"].append(deepcopy(operation["relation"]))
    elif op == "REMOVE_RELATION":
        removed = [rel for rel in target["relations"] if rel["relation_type"] == operation["relation_type"]]
        target["relations"] = [rel for rel in target["relations"] if rel["relation_type"] != operation["relation_type"]]
        target["tombstones"].append({"operator": op, "removed": removed, "evidence_ids": list(operation["evidence_ids"])})
    elif op == "CHANGE_ROLE":
        for node in target["nodes"]:
            if node["role"] == operation["from_role"]: node["role"] = operation["to_role"]
        for rel in target["relations"]:
            if rel["source_role"] == operation["from_role"]: rel["source_role"] = operation["to_role"]
            if rel["target_role"] == operation["from_role"]: rel["target_role"] = operation["to_role"]
    elif op == "CHANGE_VALUE":
        for node in target["nodes"]:
            if node["role"] == operation["role"]: node["value"] = operation["value"]
    elif op in {"GENERALIZE", "SPECIALIZE"}:
        for node in target["nodes"]:
            if node["role"] == operation["role"]: node["type"] = operation["type"]
    elif op == "REORDER_TIME":
        for rel in target["relations"]:
            if rel["relation_type"] == operation["relation_type"]: rel["order"] = operation["order"]
    elif op == "MERGE":
        target["nodes"] = [node for node in target["nodes"] if node["role"] == operation["merged_role"]]
        target["relations"] = []
    elif op == "SPLIT":
        original = next(node for node in target["nodes"] if node["role"] == operation["role"])
        target["nodes"] = [node for node in target["nodes"] if node["role"] != operation["role"]]
        target["nodes"].extend({**deepcopy(original), "role": role, "value": f"{original['value']}:{role}"} for role in operation["new_roles"])
    target["revision"] = int(base["revision"]) + 1
    target["evidence_ids"] = sorted(set([*target["evidence_ids"], *operation["evidence_ids"]]))
    target["nodes"] = sorted(target["nodes"], key=lambda x: (x["role"], x["type"], x["value"]))
    target["relations"] = sorted(target["relations"], key=lambda x: (x["source_role"], x["relation_type"], x["target_role"], x["order"]))
    target["canonical_digest"] = _digest(target)
    return target


def build_artifacts() -> Dict[str, str]:
    source_rows = []
    for index in range(10):
        partition = "train" if index < 5 else "evaluation"
        source_rows.append({"schema": "sara-phase38-synthetic-history-source-v1", "source_id": f"phase38-history-{index:02d}", "partition": partition, "structure_family": f"structure-family-{partition}-{index:02d}", "transformation_family": f"transformation-family-{partition}-{index:02d}", "evidence_scope": "registered_synthetic_control", "observed_only": True, "production_claim_allowed": False})
    train_rows = []
    for index, operator in enumerate(OPERATORS):
        base = _base(f"train-{index}")
        operation = _operation(operator, index)
        train_rows.append({"schema": "sara-phase38-train-example-v1", "example_id": f"phase38-train-{index:02d}", "source_id": source_rows[index % 5]["source_id"], "base": base, "delta": {"base_digest": base["canonical_digest"], "operations": [operation]}, "target": _reference_apply(base, operation), "evaluator_labels_present": False})
    invalid_families = {"ambiguous_base", "branch_merge_conflict", "duplicated_evidence", "stale_revision", "contradiction", "missing_base", "corrupted_delta", "invalid_inverse", "cycle", "budget_exceeded"}
    inputs, keys = [], []
    for index, family in enumerate(CASE_FAMILIES):
        operator = OPERATORS[index % len(OPERATORS)]
        source = source_rows[5 + (index % 5)]
        base = _base(f"evaluation-{index}")
        operation = _operation(operator, index + 100)
        valid = family not in invalid_families
        visible_delta = {"schema": "phase38-typed-delta-v1", "base_digest": base["canonical_digest"], "operations": [deepcopy(operation)], "evidence_ids": list(operation["evidence_ids"])}
        if family == "stale_revision": visible_delta["operations"][0]["precondition_revision"] = 0
        elif family == "missing_base": visible_delta["base_digest"] = "missing"
        elif family == "corrupted_delta": visible_delta["operations"][0]["operator"] = "CORRUPTED"
        elif family == "cycle": visible_delta["operations"].append({**_operation("ADD_RELATION", index + 200), "relation": {"source_role":"role:target","relation_type":"cycles_to","target_role":"role:source","order":2,"evidence_ids":[f"evidence::cycle-{index}"]}})
        elif family == "budget_exceeded": visible_delta["operations"] = [deepcopy(operation) for _ in range(17)]
        case_id = f"phase38-execution-{index:02d}"
        inputs.append({"schema": "sara-phase38-execution-input-v1", "case_id": case_id, "source_id": source["source_id"], "base": None if family == "missing_base" else base, "visible_delta": visible_delta, "prediction_context": {"visible_roles":["role:source","role:target"],"evidence_ids":list(operation["evidence_ids"])}, "max_operations":16, "max_chain_depth":8, "durable_mutation_allowed":False})
        target = _reference_apply(base, operation) if valid else None
        keys.append({"schema":"sara-phase38-evaluator-key-v1","case_id":case_id,"case_family":family,"expected_decision":"materialize" if valid else "abstain","exact_target":target,"withheld_delta":{"base_digest":base["canonical_digest"],"operations":[operation]},"operator":operator,"synthetic_control":True})
    def jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
        return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return {"sources": jsonl(source_rows), "train": jsonl(train_rows), "inputs": jsonl(inputs), "key": jsonl(keys)}


def _write(path: str, payload: str) -> None:
    resolved = ensure_parent_directory(path)
    if os.path.exists(resolved):
        with open(resolved, encoding="utf-8") as handle:
            if handle.read() != payload: raise ValueError(f"frozen Phase 38 artifact is immutable: {path}")
        return
    with open(resolved, "x", encoding="utf-8") as handle: handle.write(payload)


def freeze(paths: Mapping[str, str], receipt_path: str) -> Dict[str, Any]:
    payloads = build_artifacts()
    for key, path in paths.items(): _write(path, payloads[key])
    hashes = {key: sha256(payload.encode()).hexdigest() for key, payload in payloads.items()}
    receipt = {"schema":"sara-phase38-execution-fixture-freeze-receipt-v1","protocol_fingerprint":PROTOCOL,"artifact_hashes":hashes,"artifact_paths":{key:os.path.realpath(path) for key,path in paths.items()},"source_count":10,"train_example_count":len(OPERATORS),"execution_case_count":len(CASE_FAMILIES),"candidate_evaluator_isolated":True,"evidence_scope":"registered_synthetic_control","candidate_implementation_allowed":True,"production_promotion_allowed":False}
    _write(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=DEFAULT_SOURCES); parser.add_argument("--train", default=DEFAULT_TRAIN); parser.add_argument("--inputs", default=DEFAULT_INPUTS); parser.add_argument("--key", default=DEFAULT_KEY); parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try: receipt = freeze({"sources":args.sources,"train":args.train,"inputs":args.inputs,"key":args.key}, args.receipt)
    except ValueError as exc: print(str(exc), file=sys.stderr); return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
