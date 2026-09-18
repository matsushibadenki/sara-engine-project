"""One-shot synthetic refractory mechanism study with no CartPole environment."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from sara_engine.evaluation.cartpole_refractory_mechanism import (
    ARMS, evaluate_arm, generate_episodes,
)
from sara_engine.utils.project_paths import ensure_allowed_output_path, project_path


PROTOCOL_PATH = "data/processed/benchmark_fixtures/cartpole_refractory_mechanism_v1.json"
OUTPUT_PATH = "workspace/evaluation/cartpole_refractory_mechanism_v1.json"
SEEDS = (0, 1, 2, 3, 4)
SOURCE_PATHS = {
    "src/sara_engine/evaluation/cartpole_refractory_mechanism.py",
    "src/sara_engine/evaluation/cartpole_refractory_runner.py",
    "src/sara_engine/evaluation/event_unit_causal_isolation.py",
    "src/sara_engine/neuro/neuron.py",
}
GATE = {
    "minimum_compact_accuracy": 0.90,
    "minimum_spiking_accuracy": 0.90,
    "minimum_spiking_per_seed_accuracy": 0.875,
    "maximum_compact_spiking_gap": 0.10,
    "maximum_negative_control_accuracy": 0.60,
    "minimum_spiking_minus_mask_accuracy": 0.35,
    "minimum_spiking_minus_refractory_disabled_accuracy": 0.35,
    "minimum_spiking_minus_timing_removed_accuracy": 0.35,
    "require_rate_matched_feature_count": True,
    "require_bounded_local_updates": True,
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def load_protocol(expected_sha256: str) -> dict:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("Frozen mechanism protocol digest is required")
    raw = Path(project_path(PROTOCOL_PATH)).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Mechanism protocol digest mismatch")
    protocol = json.loads(raw)
    if (raw != _canonical_bytes(protocol)
            or protocol.get("schema") != "sara-cartpole-refractory-mechanism-v1"):
        raise ValueError("Mechanism protocol is not canonical")
    if (set(protocol["source_sha256"]) != SOURCE_PATHS
            or any(sha256(Path(project_path(path)).read_bytes()).hexdigest() != digest
                   for path, digest in protocol["source_sha256"].items())):
        raise ValueError("Mechanism source changed")
    if (protocol["seeds"] != list(SEEDS)
            or protocol["arms"] != list(ARMS)
            or protocol["training_per_seed"] != 64
            or protocol["development_per_seed"] != 32
            or protocol["events_per_episode"] != 3
            or protocol["development_output"] != OUTPUT_PATH
            or protocol["development_scoring_authorized"] is not True
            or protocol["heldout_exists"] is not False
            or protocol["gate"] != GATE):
        raise ValueError("Mechanism experiment boundaries changed")
    return protocol


def audit_generated_splits() -> dict:
    """Check labels, nuisance/time separation and label-blind mask balance."""
    identities = set()
    for seed in SEEDS:
        training = generate_episodes(split="training", seed=seed)
        development = generate_episodes(split="development", seed=seed)
        if len(training) != 64 or len(development) != 32:
            raise ValueError("Mechanism split size changed")
        train_patterns = set()
        dev_patterns = set()
        for split, episodes, patterns in (
            ("training", training, train_patterns),
            ("development", development, dev_patterns),
        ):
            expected_per_label = len(episodes) // 2
            if Counter(episode.target for episode in episodes) != {0: expected_per_label,
                                                                  1: expected_per_label}:
                raise ValueError("Mechanism labels are not balanced")
            for label in (0, 1):
                subset = [episode for episode in episodes if episode.target == label]
                if Counter(episode.index % 2 for episode in subset) != {
                    0: expected_per_label // 2, 1: expected_per_label // 2,
                }:
                    raise ValueError("Feature-loss mask is label-associated")
            for episode in episodes:
                if (episode.identity in identities or len(episode.events) != 3
                        or episode.split != split):
                    raise ValueError("Mechanism episode identity or input changed")
                identities.add(episode.identity)
                patterns.add(tuple((event.route, event.time, event.branch)
                                   for event in episode.events))
        if train_patterns & dev_patterns:
            raise ValueError("Mechanism training/development patterns overlap")
        if ({event.time for episode in training for event in episode.events}
                & {event.time for episode in development for event in episode.events}):
            raise ValueError("Mechanism absolute times overlap")
    return {"seed_count": len(SEEDS), "training_episodes": 64 * len(SEEDS),
            "development_episodes": 32 * len(SEEDS),
            "episode_identity_overlap": 0, "exact_pattern_overlap": 0}


def development_decision(results: list[dict]) -> dict:
    if len(results) != len(SEEDS) * len(ARMS):
        raise ValueError("Incomplete mechanism result")
    by_arm = {arm: [] for arm in ARMS}
    for result in results:
        if (result["arm"] not in by_arm or result["seed"] not in SEEDS
                or result["development_count"] != 32
                or len(result["development_predictions"]) != 32):
            raise ValueError("Invalid mechanism arm result")
        by_arm[result["arm"]].append(result)
    if any(sorted(result["seed"] for result in rows) != list(SEEDS)
           for rows in by_arm.values()):
        raise ValueError("Mechanism seed coverage changed")
    accuracy = {arm: sum(result["development_correct"] for result in rows) / 160
                for arm, rows in by_arm.items()}
    spiking = accuracy["spiking_refractory"]
    negatives = ("scalar_count", "refractory_disabled", "rate_matched_mask",
                 "timing_removed")
    spiking_feature_counts = {
        row["seed"]: row["development_feature_one_count"]
        for row in by_arm["spiking_refractory"]
    }
    mask_feature_counts = {
        row["seed"]: row["development_feature_one_count"]
        for row in by_arm["rate_matched_mask"]
    }
    checks = {
        "compact_accuracy": accuracy["compact_gap"] >= 0.90,
        "spiking_accuracy": spiking >= 0.90,
        "spiking_every_seed": all(result["development_correct"] / 32 >= 0.875
                                  for result in by_arm["spiking_refractory"]),
        "compact_spiking_equivalence": abs(accuracy["compact_gap"] - spiking) <= 0.10,
        "negative_controls": all(accuracy[arm] <= 0.60 for arm in negatives),
        "spiking_minus_mask": spiking - accuracy["rate_matched_mask"] >= 0.35,
        "spiking_minus_refractory_disabled": (
            spiking - accuracy["refractory_disabled"] >= 0.35
        ),
        "spiking_minus_timing_removed": spiking - accuracy["timing_removed"] >= 0.35,
        "rate_matched_feature_count": (
            spiking_feature_counts == mask_feature_counts == {seed: 16 for seed in SEEDS}
        ),
        "bounded_local_updates": all(
            0 <= result["training_updates"] <= 64
            and all(-4 <= weight <= 4 for weight in result["final_weights"])
            for result in results
        ),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "development_accuracy": accuracy, "heldout_opening_authorized": False}


def run_registered_study(*, expected_protocol_sha256: str) -> dict:
    """Reserve and score only the frozen synthetic development task once."""
    protocol = load_protocol(expected_protocol_sha256)
    output = Path(ensure_allowed_output_path(protocol["development_output"]))
    if output.exists():
        raise ValueError("Mechanism result already exists")
    split_audit = audit_generated_splits()
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".lock").open("xb") as lock:
        lock.write((expected_protocol_sha256 + "\n").encode("ascii"))
    results = [evaluate_arm(arm=arm, seed=seed) for seed in SEEDS for arm in ARMS]
    result = {
        "schema": "sara-cartpole-refractory-mechanism-result-v1",
        "protocol_sha256": expected_protocol_sha256,
        "split_audit": split_audit,
        "results": results,
        "decision": development_decision(results),
    }
    payload = _canonical_bytes(result)
    with output.open("xb") as stream:
        stream.write(payload)
    return result
