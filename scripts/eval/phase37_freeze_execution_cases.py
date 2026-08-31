#!/usr/bin/env python3
"""Freeze Phase 37 evaluator-isolated execution cases from approved bases."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase37_preregistration import REQUIRED_CASE_FAMILIES  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

TRAIN_BASE = processed_data_path("benchmark_fixtures", "phase37_structural_train_base.jsonl")
EVALUATION_BASE = processed_data_path("benchmark_fixtures", "phase37_structural_evaluation_base.jsonl")
SOURCE_MANIFEST = processed_data_path("autobot", "phase37_structural_source_manifest.jsonl")
DEFAULT_CANDIDATE = processed_data_path("benchmark_fixtures", "phase37_structural_execution_inputs_v2.jsonl")
DEFAULT_KEY = processed_data_path("benchmark_fixtures", "phase37_structural_execution_evaluator_key_v2.jsonl")
DEFAULT_RECEIPT = workspace_path("evaluation", "phase37_execution_fixture_freeze_receipt_v2.json")
EXPECTED_BASE_HASHES = {
    "source_manifest": "6971a874bc86422b38ed279c5cfc1b4fc75081ace46dcdbec9c1c5fb60172096",
    "train_fixture": "a4f7728695d0f08ca43fda02720f7aba4bbaaee52586acfe2d5d7dc465166d7c",
    "evaluation_fixture": "705f134ddebe6ff262df8eddef6f255bc83b1e5959d671d2a4b40a4226ea1873",
}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file_sha(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def _renamed(edges: Sequence[Mapping[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    names: Dict[str, str] = {}
    def role(value: str) -> str:
        if value not in names:
            names[value] = f"anonymous:{prefix}:node-{len(names)}"
        return names[value]
    return [{**edge, "source": role(str(edge["source"])), "target": role(str(edge["target"]))} for edge in edges]


def _reverse(edges: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{**edge, "source": edge["target"], "target": edge["source"]} for edge in edges]


def build_execution_artifacts(train: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(train) != 4 or len(evaluation) != 4 or len(sources) != 8:
        raise ValueError("frozen Phase 37 base cardinality mismatch")
    train_ids = {row["case_id"] for row in train}
    evaluation_ids = {row["case_id"] for row in evaluation}
    if train_ids & evaluation_ids:
        raise ValueError("train/evaluation source identity overlap")
    plans: Tuple[Tuple[str, int, str, bool], ...] = (
        ("label_renamed_isomorph", 0, "rename", True),
        ("same_relations_different_topology", 0, "reverse", False),
        ("unseen_nodes", 1, "rename", True),
        ("heldout_domain", 2, "identity", True),
        ("multi_edge_role_transfer", 1, "identity", True),
        ("temporal_order_reversal", 1, "order_reverse", False),
        ("causal_direction_reversal", 1, "reverse", False),
        ("context_change", 2, "context_mismatch", False),
        ("rare_exception", 3, "exception", False),
        ("revised_evidence", 2, "revision", False),
        ("contradiction", 0, "contradiction", False),
        ("missing_role", 3, "identity", False),
        ("adversarial_hub", 0, "hub", False),
        ("no_transfer", 3, "unrelated", False),
    )
    inputs: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    for index, (family, base_index, transform, should_propose) in enumerate(plans):
        base = evaluation[base_index]
        visible = [dict(edge) for edge in base["visible_edges"]]
        answer = dict(base["withheld_edge"])
        if transform == "rename":
            combined = _renamed([*visible, answer], str(index))
            visible, answer = combined[:-1], combined[-1]
        elif transform in {"reverse", "order_reverse"}:
            visible = _reverse(visible)
        if transform == "hub":
            visible.extend({"source": "anonymous:hub", "relation_type": "unrelated", "target": f"anonymous:noise-{n}", "evidence_id": base["evidence_ids"][0], "verified": True} for n in range(4))
        if transform == "unrelated":
            visible = [{"source": "anonymous:unrelated-a", "relation_type": "unrelated", "target": "anonymous:unrelated-b", "evidence_id": base["evidence_ids"][0], "verified": True}]
        context = "heldout"
        if transform == "context_mismatch":
            context = "mismatched"
        case_id = f"phase37-execution-{index:02d}"
        candidate = {
            "schema": "sara-phase37-execution-input-v1",
            "case_id": case_id,
            "source_record_id": base["case_id"],
            "visible_edges": visible,
            "query": {"source_role": "role:source", "target_role": "role:target", "context": context},
            "max_candidate_patterns": 8,
            "max_proposals": 4,
            "durable_mutation_allowed": False,
        }
        key = {
            "schema": "sara-phase37-execution-evaluator-key-v1",
            "case_id": case_id,
            "case_family": family,
            "source_record_id": base["case_id"],
            "expected_decision": "propose" if should_propose else "abstain",
            "withheld_edge": answer,
            "control_transform": transform,
            "synthetic_control": transform not in {"identity", "rename"},
        }
        inputs.append(candidate)
        keys.append(key)
    return {"inputs": inputs, "keys": keys}


def _write_new_or_identical(path: str, payload: str) -> None:
    resolved = ensure_parent_directory(path)
    if os.path.exists(resolved):
        with open(resolved, encoding="utf-8") as handle:
            if handle.read() != payload:
                raise ValueError(f"frozen execution artifact is immutable: {path}")
        return
    with open(resolved, "x", encoding="utf-8") as handle:
        handle.write(payload)


def freeze(candidate_path: str, key_path: str, receipt_path: str) -> Dict[str, Any]:
    actual = {"source_manifest": _file_sha(SOURCE_MANIFEST), "train_fixture": _file_sha(TRAIN_BASE), "evaluation_fixture": _file_sha(EVALUATION_BASE)}
    if actual != EXPECTED_BASE_HASHES:
        raise ValueError("frozen Phase 37 base hash mismatch")
    artifacts = build_execution_artifacts(_read_jsonl(TRAIN_BASE), _read_jsonl(EVALUATION_BASE), _read_jsonl(SOURCE_MANIFEST))
    input_payload = _jsonl(artifacts["inputs"])
    key_payload = _jsonl(artifacts["keys"])
    input_hash = sha256(input_payload.encode()).hexdigest()
    key_hash = sha256(key_payload.encode()).hexdigest()
    _write_new_or_identical(candidate_path, input_payload)
    _write_new_or_identical(key_path, key_payload)
    receipt = {
        "schema": "sara-phase37-execution-fixture-freeze-receipt-v2",
        "protocol_fingerprint": "e77d34460bfc2ae2440d765616a65ce7dad734d07ef6cca3b0d17b1532cfe704",
        "base_hashes": actual,
        "case_count": len(artifacts["inputs"]),
        "case_families": list(REQUIRED_CASE_FAMILIES),
        "candidate_input_hash": input_hash,
        "evaluator_key_hash": key_hash,
        "candidate_input_path": os.path.realpath(candidate_path),
        "evaluator_key_path": os.path.realpath(key_path),
        "evaluator_labels_isolated": True,
        "candidate_implementation_allowed": True,
        "production_promotion_allowed": False,
    }
    receipt_payload = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_new_or_identical(receipt_path, receipt_payload)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE)
    parser.add_argument("--key-path", default=DEFAULT_KEY)
    parser.add_argument("--receipt-path", default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        receipt = freeze(args.candidate_path, args.key_path, args.receipt_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
