#!/usr/bin/env python3
"""Two-command, development-only capture of a preregistered event-unit pair."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from sara_engine.evaluation.event_unit_causal_isolation import (
    generate_episodes, run_v2_development_arm,
)
from sara_engine.research import (
    CaptureIntent, CaptureOutcome, DiscoveryCaptureLog,
    complete_prepared_development_action,
)
from sara_engine.utils.project_paths import project_path


PROTOCOL_PATH = Path(project_path(
    "data", "processed", "benchmark_fixtures", "discovery_capture_pair_v1.json"))
PROTOCOL_SHA256 = "dbe0d1999a0af71ff4d058cc484c81d0bcdc96a00ee6425a899f5ae8b3043221"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict:
    if _sha(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("Frozen pair protocol hash changed")
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = Path(project_path(value["candidate_source"]["path"]))
    if _sha(source) != value["candidate_source"]["sha256"]:
        raise ValueError("Frozen candidate source hash changed")
    return value


def _intent(config: dict, sequence: int) -> CaptureIntent:
    actions = config["fixed_action_order"]
    if sequence < 0 or sequence > len(actions):
        raise ValueError("Pair action sequence is out of range")
    node = config["capture"]["root_node_id"] if sequence == 0 else actions[sequence - 1]["node_id"]
    return CaptureIntent(
        sequence=sequence, node_id=node,
        parent_id=None if sequence == 0 else config["capture"]["root_node_id"],
        hypothesis_id="temporal-state-pair-v1",
        policy_id=config["capture"]["policy_id"],
        candidate_sha256=config["candidate_source"]["sha256"],
        preregistration_sha256=PROTOCOL_SHA256,
        evaluator_id=config["evaluator"]["evaluator_id"],
    )


def _verified_view(log: DiscoveryCaptureLog, config: dict):
    view = log.load()
    intents = [event for event in view.events if isinstance(event, CaptureIntent)]
    outcomes = [event for event in view.events if isinstance(event, CaptureOutcome)]
    if any(event != _intent(config, index) for index, event in enumerate(intents)):
        raise ValueError("Capture intent differs from frozen pair protocol")
    if len(intents) > len(config["fixed_action_order"]) + 1:
        raise ValueError("Capture has extra actions")
    if any(event.status != "valid" for event in outcomes):
        raise ValueError("Pair capture contains a non-valid outcome; stop for review")
    return view, intents, outcomes


def prepare_next(capture_path: str) -> dict:
    """Persist the next intent; return its head without invoking the evaluator."""
    config = _protocol()
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _verified_view(log, config)
    if not intents:
        if log.path.exists():
            raise ValueError("Existing empty capture path is not accepted")
        root_head = log.begin(_intent(config, 0), expected_head_sha256=view.head_sha256)
        view, intents, outcomes = _verified_view(log, config)
        if view.head_sha256 != root_head:
            raise ValueError("Root capture head changed")
    if view.pending_node_ids:
        raise ValueError("Complete or review the pending intent before preparing another")
    sequence = len(intents)
    if sequence > len(config["fixed_action_order"]):
        raise ValueError("All fixed pair actions are already captured")
    if len(outcomes) != sequence - 1:
        raise ValueError("Capture has incomplete prior actions")
    intent = _intent(config, sequence)
    head = log.begin(intent, expected_head_sha256=view.head_sha256)
    return {
        "schema": "sara-discovery-pair-prepared-v1",
        "action": intent.node_id,
        "intent_head_sha256": head,
        "protocol_sha256": PROTOCOL_SHA256,
        "evaluator_invoked": False,
    }


def execute_pending(capture_path: str, *, published_intent_head_sha256: str) -> dict:
    """Run only the exact previously prepared action and close its outcome."""
    config = _protocol()
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _verified_view(log, config)
    if len(intents) != len(outcomes) + 2 or len(view.pending_node_ids) != 1:
        raise ValueError("Capture has no sole pending pair action")
    sequence = len(intents) - 1
    intent = _intent(config, sequence)
    action = config["fixed_action_order"][sequence - 1]
    generator = config["generator"]
    observed = {}

    def evaluate() -> CaptureOutcome:
        started = time.process_time_ns()
        training = generate_episodes(
            seeds=generator["training_seeds"],
            count_per_family=generator["training_count_per_family_per_seed"],
            split="training", namespace=generator["namespace"],
        )
        development = generate_episodes(
            seeds=generator["development_seeds"],
            count_per_family=generator["development_count_per_family_per_seed"],
            split="development", namespace=generator["namespace"],
        )
        result = run_v2_development_arm(
            action["arm"], training, development,
            intervention=action["intervention"],
        )
        elapsed_ns = time.process_time_ns() - started
        labels = {episode.identity: episode.label for episode in development}
        per_seed = {str(seed): [0, 0] for seed in generator["development_seeds"]}
        for identity, predicted in result["prediction_rows"]:
            seed = identity.split(":")[2]
            per_seed[seed][0] += int(predicted == labels[identity])
            per_seed[seed][1] += 1
        observed.update({
            "prediction_trace_sha256": result["prediction_trace_sha256"],
            "per_seed_accuracy": {seed: correct / total
                                  for seed, (correct, total) in per_seed.items()},
            "maximum_event_work": result["maximum_event_work"],
            "development_predictions": result["predictions"],
        })
        return CaptureOutcome(
            sequence=sequence, node_id=intent.node_id, status="valid",
            score=float(result["accuracy"]),
            cpu_ms=max(1, (elapsed_ns + 999_999) // 1_000_000),
            event_count=sum(len(row.events) for row in (*training, *development)),
            state_bytes=int(result["state_bytes"]),
        )

    captured = complete_prepared_development_action(
        log, intent, evaluate,
        published_intent_head_sha256=published_intent_head_sha256,
    )
    return {
        "schema": "sara-discovery-pair-observed-v1",
        "action": intent.node_id,
        "intent_head_sha256": captured.intent_head_sha256,
        "outcome_head_sha256": captured.outcome_head_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "score": captured.outcome.score,
        "cpu_ms": captured.outcome.cpu_ms,
        "input_event_count": captured.outcome.event_count,
        "state_bytes": captured.outcome.state_bytes,
        "unapproved": True,
        **observed,
    }


def verify_capture(capture_path: str, *, expected_capture_head_sha256: str,
                   expected_traces: dict[str, str]) -> dict:
    """Replay fixed development predictions against separately recorded digests."""
    config = _protocol()
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _verified_view(log, config)
    actions = config["fixed_action_order"]
    if view.head_sha256 != expected_capture_head_sha256:
        raise ValueError("Published final capture head does not match")
    if view.pending_node_ids or len(intents) != len(actions) + 1 or len(outcomes) != len(actions):
        raise ValueError("Pair capture is incomplete")
    if [outcome.node_id for outcome in outcomes] != [action["node_id"] for action in actions]:
        raise ValueError("Pair outcomes differ from the fixed action order")
    if set(expected_traces) != {action["node_id"] for action in actions}:
        raise ValueError("Expected traces must cover the four fixed actions")
    generator = config["generator"]
    training = generate_episodes(
        seeds=generator["training_seeds"],
        count_per_family=generator["training_count_per_family_per_seed"],
        split="training", namespace=generator["namespace"],
    )
    development = generate_episodes(
        seeds=generator["development_seeds"],
        count_per_family=generator["development_count_per_family_per_seed"],
        split="development", namespace=generator["namespace"],
    )
    event_count = sum(len(row.events) for row in (*training, *development))
    labels = {row.identity: row.label for row in development}
    gate = config["diagnostic_gate"]
    replayed = {}
    for action, outcome in zip(actions, outcomes):
        result = run_v2_development_arm(
            action["arm"], training, development,
            intervention=action["intervention"],
        )
        node = action["node_id"]
        trace = result["prediction_trace_sha256"]
        if (trace != expected_traces[node] or result["accuracy"] != outcome.score
                or result["state_bytes"] != outcome.state_bytes
                or outcome.event_count != event_count or outcome.cpu_ms < 1):
            raise ValueError(f"Captured outcome or prediction trace differs: {node}")
        per_seed = {str(seed): [0, 0] for seed in generator["development_seeds"]}
        for identity, predicted in result["prediction_rows"]:
            seed = identity.split(":")[2]
            per_seed[seed][0] += int(predicted == labels[identity])
            per_seed[seed][1] += 1
        replayed[node] = {
            "accuracy": result["accuracy"],
            "per_seed_accuracy": {seed: correct / total
                                  for seed, (correct, total) in per_seed.items()},
            "prediction_trace_sha256": trace,
            "maximum_event_work": result["maximum_event_work"],
            "state_bytes": result["state_bytes"],
            "cpu_ms_recorded": outcome.cpu_ms,
        }
    b = replayed["B_compact_event"]
    c = replayed["C_temporal_state"]
    shuffled = replayed["C_time_shuffle"]
    reset = replayed["C_state_reset"]
    checks = {
        "C_minus_B": c["accuracy"] - b["accuracy"] >= gate["C_minus_B_minimum"],
        "all_five_C_minus_B_positive": all(
            c["per_seed_accuracy"][seed] > b["per_seed_accuracy"][seed]
            for seed in c["per_seed_accuracy"]),
        "C_minus_time_shuffle": c["accuracy"] - shuffled["accuracy"] >= gate["C_minus_time_shuffle_minimum"],
        "C_minus_state_reset": c["accuracy"] - reset["accuracy"] >= gate["C_minus_state_reset_minimum"],
        "state_budget": all(row["state_bytes"] <= gate["maximum_retained_state_bytes"]
                            for row in replayed.values()),
        "event_work_budget": all(row["maximum_event_work"] <= gate["maximum_event_work_per_prediction"]
                                 for row in replayed.values()),
        "exact_prediction_trace_replay": True,
    }
    projection = log.project_source_snapshot(expected_head_sha256=view.head_sha256)
    return {
        "schema": "sara-discovery-pair-verification-v1",
        "capture_head_sha256": view.head_sha256,
        "unapproved_snapshot_sha256": projection.snapshot_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "replayed": replayed,
        "development_only": True,
        "export_approved": False,
    }


def audit_pair_lineage(capture_path: str, *, expected_capture_file_sha256: str,
                       expected_capture_head_sha256: str,
                       expected_snapshot_sha256: str,
                       expected_intent_heads: list[str]) -> dict:
    """Read-only lineage and exact-pattern overlap audit; never approve export."""
    config = _protocol()
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _verified_view(log, config)
    actions = config["fixed_action_order"]
    if view.pending_node_ids or len(intents) != len(actions) + 1 or len(outcomes) != len(actions):
        raise ValueError("Pair capture is incomplete")
    if view.head_sha256 != expected_capture_head_sha256:
        raise ValueError("Capture head differs from external record")
    raw = log.path.read_bytes()
    if sha256(raw).hexdigest() != expected_capture_file_sha256:
        raise ValueError("Capture file digest differs from external record")
    lines = [json.loads(line) for line in raw.splitlines()]
    intent_heads = [line["entry_sha256"] for line, event in zip(lines, view.events)
                    if isinstance(event, CaptureIntent) and event.sequence > 0]
    if intent_heads != expected_intent_heads:
        raise ValueError("Intent heads differ from external record")
    projection = log.project_source_snapshot(expected_head_sha256=view.head_sha256)
    if projection.snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError("Projected snapshot differs from external record")
    generator = config["generator"]
    training = generate_episodes(
        seeds=generator["training_seeds"],
        count_per_family=generator["training_count_per_family_per_seed"],
        split="training", namespace=generator["namespace"],
    )
    development = generate_episodes(
        seeds=generator["development_seeds"],
        count_per_family=generator["development_count_per_family_per_seed"],
        split="development", namespace=generator["namespace"],
    )
    pattern = lambda row: (row.family, row.events, row.label)
    training_patterns = {pattern(row) for row in training}
    overlapping = sum(pattern(row) in training_patterns for row in development)
    identity_overlap = len({row.identity for row in training} &
                           {row.identity for row in development})
    return {
        "schema": "sara-discovery-pair-lineage-audit-v1",
        "capture_file_sha256": expected_capture_file_sha256,
        "capture_head_sha256": view.head_sha256,
        "snapshot_sha256": projection.snapshot_sha256,
        "record_count": projection.record_count,
        "intent_heads_match_external_record": True,
        "training_development_identity_overlap": identity_overlap,
        "training_unique_labeled_patterns": len(training_patterns),
        "development_count": len(development),
        "development_examples_with_training_labeled_pattern": overlapping,
        "source_pattern_isolation_passed": overlapping == 0,
        "independent_head_attestation": False,
        "export_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "execute", "verify", "audit"))
    parser.add_argument("--capture", required=True)
    parser.add_argument("--published-intent-head")
    parser.add_argument("--expected-capture-head")
    parser.add_argument("--expected-trace", action="append", default=[])
    parser.add_argument("--expected-capture-file-sha256")
    parser.add_argument("--expected-snapshot-sha256")
    parser.add_argument("--expected-intent-head", action="append", default=[])
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.published_intent_head is not None:
            parser.error("prepare does not accept a published head")
        result = prepare_next(args.capture)
    elif args.operation == "execute":
        if args.published_intent_head is None:
            parser.error("execute requires a separately published intent head")
        result = execute_pending(args.capture, published_intent_head_sha256=args.published_intent_head)
    elif args.operation == "verify":
        if args.expected_capture_head is None or len(args.expected_trace) != 4:
            parser.error("verify requires the final capture head and four recorded traces")
        try:
            traces = dict(item.split("=", 1) for item in args.expected_trace)
        except ValueError:
            parser.error("expected traces must be NODE=SHA256")
        result = verify_capture(args.capture, expected_capture_head_sha256=args.expected_capture_head,
                                expected_traces=traces)
    else:
        if (args.expected_capture_head is None or args.expected_capture_file_sha256 is None
                or args.expected_snapshot_sha256 is None or len(args.expected_intent_head) != 4):
            parser.error("audit requires final/file/snapshot hashes and four recorded intent heads")
        result = audit_pair_lineage(
            args.capture,
            expected_capture_file_sha256=args.expected_capture_file_sha256,
            expected_capture_head_sha256=args.expected_capture_head,
            expected_snapshot_sha256=args.expected_snapshot_sha256,
            expected_intent_heads=args.expected_intent_head,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
