"""Deterministic Phase 30 temporal-history fixture freezing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .phase30_preregistration import CASE_FAMILIES


INPUT_SCHEMA = "sara-phase30-temporal-history-input-v1"
KEY_SCHEMA = "sara-phase30-temporal-history-evaluator-key-v1"
MANIFEST_SCHEMA = "sara-phase30-temporal-history-freeze-v1"
PARTITIONS: Tuple[str, ...] = ("train", "evaluation")
EVENTS_PER_CASE = 256
REPLICATE_SEEDS: Tuple[int, ...] = (101, 211, 307, 401, 509)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _generator_seed(family: str, replicate_seed: int, partition: str) -> int:
    token = f"phase30|{family}|{replicate_seed}|{partition}".encode("utf-8")
    return int.from_bytes(sha256(token).digest()[:8], "big")


def _family_parameters(family: str, rng: random.Random, partition: str) -> Dict[str, Any]:
    sign = 1 if rng.random() >= 0.5 else -1
    base = {
        "target_sign": sign,
        "interval_us": rng.choice((2_000, 4_000, 8_000, 16_000)),
        "delay_us": rng.choice((1_000, 3_000, 7_000, 15_000)),
        "phase_bucket": rng.randrange(8),
        "revision_index": rng.randrange(176, 224),
    }
    if family in {"unseen_context", "no_reuse"}:
        base["target_sign"] = 0
    if family == "contradiction":
        base["target_sign"] = -sign
    if family in {"shuffled_time", "phase_shifted", "stale_cache"}:
        base["expected_abstain"] = True
    else:
        base["expected_abstain"] = base["target_sign"] == 0
    base["partition_marker"] = "source_a" if partition == "train" else "source_b"
    return base


def _events(family: str, seed: int, partition: str, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    timestamp = rng.randrange(0, 1_000_000, 1_000)
    result: List[Dict[str, Any]] = []
    revision_index = int(params["revision_index"])
    for index in range(EVENTS_PER_CASE):
        interval = int(params["interval_us"])
        if family == "irregular_event_gaps":
            interval = rng.choice((1_000, 2_000, 5_000, 11_000, 23_000))
        elif family == "equal_count_different_interval" and index >= EVENTS_PER_CASE // 2:
            interval *= 3
        timestamp += interval
        edge = "edge_a" if index % 2 == 0 else "edge_b"
        phase = (int(params["phase_bucket"]) + index) % 8
        kind = "spike"
        polarity = int(params["target_sign"]) or (1 if index % 2 == 0 else -1)

        if family == "firing_order_reversal" and index >= EVENTS_PER_CASE // 2:
            edge = "edge_b" if index % 2 == 0 else "edge_a"
        elif family == "delayed_response" and index % 8 == 7:
            timestamp += int(params["delay_us"])
        elif family == "phase_synchrony_discrimination":
            phase = int(params["phase_bucket"]) if index % 4 < 2 else (int(params["phase_bucket"]) + 4) % 8
        elif family == "context_revision" and index == revision_index:
            kind, polarity = "context_revision", -polarity
        elif family == "shuffled_time" and index >= EVENTS_PER_CASE - 32:
            timestamp -= rng.choice((2, 4, 8)) * interval
        elif family == "phase_shifted" and index >= EVENTS_PER_CASE - 32:
            phase = (phase + 3) % 8
        elif family == "duplicate_event" and index % 32 == 31:
            kind = "duplicate"
        elif family == "stale_cache" and index == revision_index:
            kind, polarity = "expiry", 0
        elif family == "contradiction" and index == revision_index:
            kind, polarity = "contradiction", -polarity
        elif family == "unseen_context" and index >= EVENTS_PER_CASE - 32:
            kind, polarity = "unknown_context", 0
        elif family == "no_reuse":
            edge = f"edge_unique_{index:03d}"

        result.append(
            {
                "source_event_id": f"{partition}-{seed:016x}-{index:03d}",
                "edge_id": edge,
                "timestamp_us": timestamp,
                "order": index,
                "phase_bucket": phase,
                "polarity": polarity,
                "kind": kind,
                "context_id": f"{params['partition_marker']}-{family}",
                "provenance_reference": f"phase30:{partition}:{seed:016x}:{index:03d}",
            }
        )
    return result


def build_fixtures() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    generator_identities = set()
    source_identities = set()
    for partition in PARTITIONS:
        for family in CASE_FAMILIES:
            for replicate_seed in REPLICATE_SEEDS:
                generator_seed = _generator_seed(family, replicate_seed, partition)
                rng = random.Random(generator_seed)
                params = _family_parameters(family, rng, partition)
                case_id = f"phase30-{partition}-{family}-{replicate_seed}"
                generator_id = f"generator-{generator_seed:016x}"
                source_id = f"{partition}-source-{digest([family, replicate_seed, generator_seed])[:16]}"
                generator_identities.add((partition, generator_id))
                source_identities.add((partition, source_id))
                events = _events(family, generator_seed, partition, params)
                inputs.append(
                    {
                        "schema": INPUT_SCHEMA,
                        "case_id": case_id,
                        "partition": partition,
                        "case_family": family,
                        "replicate_seed": replicate_seed,
                        "generator_id": generator_id,
                        "source_id": source_id,
                        "events": events,
                    }
                )
                keys.append(
                    {
                        "schema": KEY_SCHEMA,
                        "case_id": case_id,
                        "partition": partition,
                        "expected_decision": "abstain" if params["expected_abstain"] else ("positive" if int(params["target_sign"]) > 0 else "negative"),
                        "timing_required": family not in {"duplicate_event", "unseen_context", "no_reuse"},
                        "required_invalidation": family in {"context_revision", "stale_cache", "contradiction"},
                    }
                )

    input_digest = digest(inputs)
    key_digest = digest(keys)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "case_count": len(inputs),
        "event_count": sum(len(case["events"]) for case in inputs),
        "events_per_case": EVENTS_PER_CASE,
        "partitions": list(PARTITIONS),
        "case_families": list(CASE_FAMILIES),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "input_digest": input_digest,
        "evaluator_key_digest": key_digest,
        "source_generator_disjoint": _partition_disjoint(source_identities) and _partition_disjoint(generator_identities),
        "evaluator_labels_absent_from_inputs": all("expected_decision" not in case and "timing_required" not in case for case in inputs),
    }
    manifest["freeze_fingerprint"] = digest(manifest)
    validate_fixtures(inputs, keys, manifest)
    return inputs, keys, manifest


def _partition_disjoint(identities: Iterable[Tuple[str, str]]) -> bool:
    grouped = {partition: set() for partition in PARTITIONS}
    for partition, identity in identities:
        grouped[partition].add(identity)
    return grouped["train"].isdisjoint(grouped["evaluation"])


def validate_fixtures(inputs: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if len(inputs) != len(PARTITIONS) * len(CASE_FAMILIES) * len(REPLICATE_SEEDS):
        raise ValueError("case_count_mismatch")
    if len(keys) != len(inputs):
        raise ValueError("evaluator_key_count_mismatch")
    if any(case.get("schema") != INPUT_SCHEMA or len(case.get("events", ())) != EVENTS_PER_CASE for case in inputs):
        raise ValueError("input_contract_mismatch")
    if any("expected_decision" in case or "timing_required" in case for case in inputs):
        raise ValueError("evaluator_label_leakage")
    input_ids = [str(case["case_id"]) for case in inputs]
    key_ids = [str(case["case_id"]) for case in keys]
    if len(set(input_ids)) != len(input_ids) or set(input_ids) != set(key_ids):
        raise ValueError("case_identity_mismatch")
    if digest(list(inputs)) != manifest.get("input_digest") or digest(list(keys)) != manifest.get("evaluator_key_digest"):
        raise ValueError("fixture_digest_mismatch")
    check = dict(manifest)
    fingerprint = check.pop("freeze_fingerprint", None)
    if fingerprint != digest(check):
        raise ValueError("freeze_fingerprint_mismatch")
    if manifest.get("source_generator_disjoint") is not True or manifest.get("evaluator_labels_absent_from_inputs") is not True:
        raise ValueError("isolation_gate_failed")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


__all__ = [
    "EVENTS_PER_CASE",
    "INPUT_SCHEMA",
    "KEY_SCHEMA",
    "MANIFEST_SCHEMA",
    "PARTITIONS",
    "REPLICATE_SEEDS",
    "build_fixtures",
    "digest",
    "validate_fixtures",
    "write_jsonl",
]
