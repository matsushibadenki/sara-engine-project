"""Default-off canonical structural delta codec for Phase 38 research."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from sara_engine.evaluation.phase38_preregistration import OPERATORS


def canonical_digest(structure: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(structure))
    payload.pop("canonical_digest", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _canonicalize(structure: Dict[str, Any]) -> Dict[str, Any]:
    structure["nodes"] = sorted(structure.get("nodes", []), key=lambda x: (x["role"], x["type"], x["value"]))
    structure["relations"] = sorted(structure.get("relations", []), key=lambda x: (x["source_role"], x["relation_type"], x["target_role"], x["order"]))
    structure["evidence_ids"] = sorted(set(structure.get("evidence_ids", [])))
    structure["canonical_digest"] = canonical_digest(structure)
    return structure


def _has_cycle(relations: Sequence[Mapping[str, Any]]) -> bool:
    outgoing: Dict[str, list[str]] = {}
    for relation in relations:
        outgoing.setdefault(str(relation["source_role"]), []).append(str(relation["target_role"]))
    visiting: set[str] = set()
    visited: set[str] = set()
    def walk(node: str) -> bool:
        if node in visiting: return True
        if node in visited: return False
        visiting.add(node)
        if any(walk(target) for target in outgoing.get(node, ())): return True
        visiting.remove(node); visited.add(node); return False
    return any(walk(node) for node in tuple(outgoing))


@dataclass(frozen=True)
class DeltaApplicationResult:
    accepted: bool
    reason: str
    target: Optional[Dict[str, Any]]
    rollback_receipt: Optional[Dict[str, Any]]
    operation_count: int
    event_cost: int
    state_bytes: int
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"schema":"sara-phase38-delta-application-v1","accepted":self.accepted,"reason":self.reason,"target":deepcopy(self.target),"rollback_receipt":deepcopy(self.rollback_receipt),"operation_count":self.operation_count,"event_cost":self.event_cost,"state_bytes":self.state_bytes,"durable_mutation_allowed":False}


class CanonicalStructuralDeltaCodec:
    def __init__(self, *, max_operations: int = 16, max_nodes: int = 64, max_relations: int = 128) -> None:
        self.max_operations = max(1, int(max_operations)); self.max_nodes = max(1, int(max_nodes)); self.max_relations = max(1, int(max_relations))

    def apply(self, base: Optional[Mapping[str, Any]], delta: Mapping[str, Any]) -> DeltaApplicationResult:
        operations = tuple(delta.get("operations", ()))
        if base is None: return self._reject("missing_base", len(operations))
        if canonical_digest(base) != base.get("canonical_digest") or delta.get("base_digest") != base.get("canonical_digest"): return self._reject("base_digest_mismatch", len(operations))
        if not operations or len(operations) > self.max_operations: return self._reject("operation_budget_exceeded", len(operations))
        if any(operation.get("operator") not in OPERATORS for operation in operations): return self._reject("unknown_or_corrupted_operator", len(operations))
        if any(operation.get("precondition_revision") != base.get("revision") for operation in operations): return self._reject("stale_revision", len(operations))
        target = deepcopy(dict(base)); target.pop("canonical_digest", None)
        receipt = {"schema":"sara-phase38-rollback-receipt-v1","base":deepcopy(dict(base)),"base_digest":base["canonical_digest"],"delta_digest":sha256(json.dumps(delta,sort_keys=True,separators=(",", ":")).encode()).hexdigest()}
        try:
            for operation in operations: self._apply_one(target, operation)
        except (KeyError, StopIteration, ValueError): return self._reject("precondition_failed", len(operations))
        if len(target.get("nodes", ())) > self.max_nodes or len(target.get("relations", ())) > self.max_relations: return self._reject("state_budget_exceeded", len(operations))
        if _has_cycle(target.get("relations", ())): return self._reject("cycle_detected", len(operations))
        target["revision"] = int(base["revision"]) + 1
        target["evidence_ids"] = sorted(set([*target.get("evidence_ids", ()), *(e for op in operations for e in op.get("evidence_ids", ())) ]))
        target = _canonicalize(target)
        state_bytes = len(json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()) + len(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        return DeltaApplicationResult(True, "materialized", target, receipt, len(operations), len(operations) + len(target.get("nodes", ())) + len(target.get("relations", ())), state_bytes)

    def rollback(self, result: DeltaApplicationResult) -> Optional[Dict[str, Any]]:
        if not result.accepted or not result.rollback_receipt: return None
        base = deepcopy(result.rollback_receipt["base"])
        return base if canonical_digest(base) == result.rollback_receipt["base_digest"] == base.get("canonical_digest") else None

    def _reject(self, reason: str, operation_count: int) -> DeltaApplicationResult:
        return DeltaApplicationResult(False, reason, None, None, operation_count, operation_count, 0)

    def _apply_one(self, target: Dict[str, Any], operation: Mapping[str, Any]) -> None:
        op = operation["operator"]
        if op == "ADD_NODE": target["nodes"].append(deepcopy(operation["node"]))
        elif op == "REMOVE_NODE":
            removed = [node for node in target["nodes"] if node["role"] == operation["role"]]
            if not removed: raise ValueError("missing node")
            target["nodes"] = [node for node in target["nodes"] if node["role"] != operation["role"]]
            relations = [rel for rel in target["relations"] if operation["role"] in (rel["source_role"], rel["target_role"])]
            target["relations"] = [rel for rel in target["relations"] if rel not in relations]
            target["tombstones"].append({"operator":op,"removed":removed,"evidence_ids":list(operation["evidence_ids"])})
        elif op == "ADD_RELATION": target["relations"].append(deepcopy(operation["relation"]))
        elif op == "REMOVE_RELATION":
            removed = [rel for rel in target["relations"] if rel["relation_type"] == operation["relation_type"]]
            if not removed: raise ValueError("missing relation")
            target["relations"] = [rel for rel in target["relations"] if rel not in removed]
            target["tombstones"].append({"operator":op,"removed":removed,"evidence_ids":list(operation["evidence_ids"])})
        elif op == "CHANGE_ROLE":
            matched = False
            for node in target["nodes"]:
                if node["role"] == operation["from_role"]: node["role"] = operation["to_role"]; matched = True
            if not matched: raise ValueError("missing role")
            for rel in target["relations"]:
                if rel["source_role"] == operation["from_role"]: rel["source_role"] = operation["to_role"]
                if rel["target_role"] == operation["from_role"]: rel["target_role"] = operation["to_role"]
        elif op == "CHANGE_VALUE":
            node = next(node for node in target["nodes"] if node["role"] == operation["role"]); node["value"] = operation["value"]
        elif op in {"GENERALIZE", "SPECIALIZE"}:
            node = next(node for node in target["nodes"] if node["role"] == operation["role"]); node["type"] = operation["type"]
        elif op == "REORDER_TIME":
            relation = next(rel for rel in target["relations"] if rel["relation_type"] == operation["relation_type"]); relation["order"] = operation["order"]
        elif op == "MERGE":
            if operation["merged_role"] not in operation["roles"]: raise ValueError("invalid merge")
            target["nodes"] = [node for node in target["nodes"] if node["role"] == operation["merged_role"]]; target["relations"] = []
        elif op == "SPLIT":
            original = next(node for node in target["nodes"] if node["role"] == operation["role"])
            target["nodes"] = [node for node in target["nodes"] if node["role"] != operation["role"]]
            target["nodes"].extend({**deepcopy(original),"role":role,"value":f"{original['value']}:{role}"} for role in operation["new_roles"])


__all__ = ["CanonicalStructuralDeltaCodec", "DeltaApplicationResult", "canonical_digest"]
