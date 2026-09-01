"""Deterministic evaluator-isolated fixtures for Phase 39 anonymous reuse."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .phase39_preregistration import CASE_FAMILIES


INPUT_SCHEMA = "sara-phase39-anonymous-reuse-input-v1"
KEY_SCHEMA = "sara-phase39-anonymous-reuse-evaluator-key-v1"
MANIFEST_SCHEMA = "sara-phase39-anonymous-reuse-fixture-freeze-v1"
PARTITIONS: Tuple[str, ...] = ("train", "evaluation")
REPLICATE_SEEDS: Tuple[int, ...] = (601, 709, 811, 907, 1009)
EVENTS_PER_CASE = 256
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "case_family",
        "hidden_factor_id",
        "hidden_factor_ids",
        "task_label",
        "human_concept_name",
        "offline_cluster_id",
        "expected_outcome",
        "source_partition_label",
        "partition",
        "replicate_seed",
        "generator_seed",
        "generator_id",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _seed(family: str, replicate_seed: int, partition: str) -> int:
    return int.from_bytes(sha256(f"phase39|{partition}|{family}|{replicate_seed}".encode()).digest()[:8], "big")


def _opaque(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{digest(list(parts))[:20]}"


def _event_stream(family: str, generator_seed: int, partition: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(generator_seed)
    factor_count = 2 if family in {"overlapping_hidden_factors", "spatial_role_interaction", "unnamed_multiscale_factor"} else 1
    hidden_factors = [_opaque("factor", partition, generator_seed, index) for index in range(factor_count)]
    base_fragments = [f"fragment:{rng.randrange(12):02d}" for _ in range(4)]
    outcome = rng.choice(("outcome:left", "outcome:right", "outcome:hold"))
    timestamp = rng.randrange(0, 1_000_000, 1_000)
    events: List[Dict[str, Any]] = []
    for order in range(EVENTS_PER_CASE):
        interval = rng.choice((1_000, 2_000, 4_000, 8_000))
        if family == "interval_shift" and order >= 192:
            interval *= 4
        timestamp += interval
        fragment = base_fragments[order % len(base_fragments)]
        role = ("source", "mediator", "target")[order % 3]
        phase = order % 8
        relation = ("precedes", "supports", "inhibits", "cooccurs")[order % 4]
        revision_state = "current"

        if family == "surface_variant_same_generator":
            fragment = f"surface:{(order // 16) % 4}:{order % 4}"
        elif family == "similar_surface_different_generator":
            fragment = f"surface:shared:{order % 4}"
            relation = "supports" if generator_seed % 2 else "inhibits"
        elif family == "repeated_local_fragments":
            fragment = base_fragments[order % 2]
        elif family == "unseen_fragment_composition" and order >= 192:
            fragment = f"composition:{order % 4}:{(order + 1) % 4}"
        elif family == "overlapping_hidden_factors":
            fragment = f"overlap:{order % 3}:{(order // 3) % 2}"
        elif family == "unnamed_multiscale_factor":
            fragment = f"multiscale:{order % 5}:{(order // 8) % 3}"
            phase = (order + order // 16) % 8
        elif family == "temporal_order_reversal" and order >= 192:
            role = ("target", "mediator", "source")[order % 3]
        elif family == "causal_direction_reversal" and order >= 192:
            relation = "inhibits" if relation == "supports" else "supports"
        elif family == "spatial_role_interaction":
            fragment = f"spatial:{('near', 'far')[order % 2]}:{role}"
        elif family == "rare_exception" and order in {191, 223, 255}:
            relation, revision_state = "inhibits", "rare_exception"
        elif family == "abrupt_context_shift" and order >= 192:
            fragment = f"shifted:{order % 4}"
        elif family == "irrelevant_burst" and 160 <= order < 192:
            fragment, relation = f"noise:{rng.randrange(64):02d}", "cooccurs"
        elif family == "forced_hash_collision":
            fragment = f"collision:{order % 2}"
            relation = ("supports", "inhibits")[order % 2]
        elif family == "dominant_frequency_pressure":
            fragment = "fragment:dominant" if order % 10 else f"fragment:rare:{order % 3}"
        elif family == "random_noncompressible_stream":
            fragment, relation, role, phase = f"random:{rng.getrandbits(48):012x}", rng.choice(("precedes", "supports", "inhibits", "cooccurs")), rng.choice(("source", "mediator", "target")), rng.randrange(8)
        elif family == "all_new_no_reuse":
            fragment = f"unique:{order:03d}:{rng.getrandbits(24):06x}"
        elif family == "capacity_saturation":
            fragment = f"capacity:{order % 192:03d}"
        elif family == "dead_unit_recovery" and order >= 224:
            fragment = base_fragments[order % 2]
        elif family == "revision_contradiction_expiry":
            if order == 176:
                revision_state = "revised"
            elif order == 208:
                revision_state, relation = "contradicted", "inhibits"
            elif order == 240:
                revision_state = "expired"
        elif family == "source_replacement" and order == 208:
            revision_state = "source_replaced"

        events.append(
            {
                "event_id": _opaque("event", partition, generator_seed, order),
                "typed_fragment": {"type": "relation_fragment", "value": fragment, "role": role, "relation": relation},
                "timestamp_us": timestamp,
                "order": order,
                "phase": phase,
                "context_local_id": _opaque("context", partition, generator_seed, order // 64),
                "recent_activity": round((1 + (order % 7)) / 8.0, 6),
                "local_neighbor_ids": [_opaque("neighbor", partition, generator_seed, order % 8)],
                "support_reference": _opaque("support", partition, generator_seed, order),
                "revision_state": revision_state,
            }
        )
    evaluator = {
        "hidden_factor_ids": hidden_factors,
        "expected_outcome": outcome,
        "random_noncompressible": family == "random_noncompressible_stream",
        "expected_reuse": family not in {"random_noncompressible_stream", "all_new_no_reuse"},
        "requires_revision_retraction": family in {"revision_contradiction_expiry", "source_replacement"},
    }
    return events, evaluator


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(key in FORBIDDEN_INPUT_FIELDS for key in value):
            return True
        return any(_contains_forbidden(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item) for item in value)
    return False


def build_fixtures() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    for partition in PARTITIONS:
        for family in CASE_FAMILIES:
            for replicate_seed in REPLICATE_SEEDS:
                generator_seed = _seed(family, replicate_seed, partition)
                case_id = _opaque("case", partition, family, replicate_seed, generator_seed)
                source_id = _opaque("source", partition, family, replicate_seed, generator_seed)
                generator_id = _opaque("generator", partition, family, generator_seed)
                events, evaluator = _event_stream(family, generator_seed, partition)
                inputs.append({"schema": INPUT_SCHEMA, "case_id": case_id, "events": events})
                keys.append(
                    {
                        "schema": KEY_SCHEMA,
                        "case_id": case_id,
                        "partition": partition,
                        "case_family": family,
                        "replicate_seed": replicate_seed,
                        "source_id": source_id,
                        "generator_id": generator_id,
                        **evaluator,
                    }
                )
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "case_count": len(inputs),
        "event_count": sum(len(row["events"]) for row in inputs),
        "events_per_case": EVENTS_PER_CASE,
        "partitions": list(PARTITIONS),
        "case_families": list(CASE_FAMILIES),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "input_digest": digest(inputs),
        "evaluator_key_digest": digest(keys),
        "forbidden_input_fields": sorted(FORBIDDEN_INPUT_FIELDS),
        "evaluator_fields_absent_from_inputs": not _contains_forbidden(inputs),
        "source_generator_disjoint": _disjoint(keys, "source_id") and _disjoint(keys, "generator_id"),
    }
    manifest["freeze_fingerprint"] = digest(manifest)
    validate_fixtures(inputs, keys, manifest)
    return inputs, keys, manifest


def _disjoint(keys: Sequence[Mapping[str, Any]], field: str) -> bool:
    train = {str(row[field]) for row in keys if row["partition"] == "train"}
    evaluation = {str(row[field]) for row in keys if row["partition"] == "evaluation"}
    return train.isdisjoint(evaluation)


def validate_fixtures(inputs: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    expected_count = len(PARTITIONS) * len(CASE_FAMILIES) * len(REPLICATE_SEEDS)
    if len(inputs) != expected_count or len(keys) != expected_count:
        raise ValueError("phase39_case_count_mismatch")
    if any(row.get("schema") != INPUT_SCHEMA or len(row.get("events", ())) != EVENTS_PER_CASE for row in inputs):
        raise ValueError("phase39_input_contract_mismatch")
    if _contains_forbidden(inputs):
        raise ValueError("phase39_evaluator_field_leakage")
    input_ids = [str(row["case_id"]) for row in inputs]
    key_ids = [str(row["case_id"]) for row in keys]
    if len(set(input_ids)) != expected_count or set(input_ids) != set(key_ids):
        raise ValueError("phase39_case_identity_mismatch")
    if digest(list(inputs)) != manifest.get("input_digest") or digest(list(keys)) != manifest.get("evaluator_key_digest"):
        raise ValueError("phase39_fixture_digest_mismatch")
    check = dict(manifest)
    fingerprint = check.pop("freeze_fingerprint", None)
    if fingerprint != digest(check):
        raise ValueError("phase39_freeze_fingerprint_mismatch")
    if manifest.get("source_generator_disjoint") is not True or manifest.get("evaluator_fields_absent_from_inputs") is not True:
        raise ValueError("phase39_isolation_gate_failed")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


__all__ = ["EVENTS_PER_CASE", "FORBIDDEN_INPUT_FIELDS", "build_fixtures", "digest", "validate_fixtures", "write_jsonl"]
