#!/usr/bin/env python3
"""Two-stage capture of the frozen temporal-transfer development experiment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from sara_engine.evaluation.temporal_transfer import (
    DEVELOPMENT_NUISANCE_ROUTES, TRAIN_NUISANCE_ROUTES,
    generate_temporal_transfer, labeled_pattern, run_frozen_development_arm,
)
from sara_engine.research import (
    CaptureIntent, CaptureOutcome, DiscoveryCaptureLog,
    complete_prepared_development_action,
)
from sara_engine.utils.project_paths import project_path


PROTOCOL_PATH = Path(project_path(
    "data", "processed", "benchmark_fixtures", "temporal_transfer_v1.json"))
PROTOCOL_SHA256 = "10d186195b7c2fff6d441f7d8e9ff254e5a8d71198216cea060d91ca7e9789cd"
ROOT_ID = "temporal-transfer-root"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict:
    if _digest(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("Frozen temporal-transfer protocol hash changed")
    config = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    sources = config["candidate_sources"]
    for path_key, hash_key in (
        ("generator_and_evaluator_path", "generator_and_evaluator_sha256"),
        ("event_unit_path", "event_unit_sha256"),
    ):
        if _digest(Path(project_path(sources[path_key]))) != sources[hash_key]:
            raise ValueError("Frozen temporal-transfer candidate source hash changed")
    return config


def preflight(config: dict) -> tuple[list, list, dict]:
    """Verify source and split identity without calling any model evaluator."""
    generator = config["generator"]
    training_seeds = generator["training_seeds"]
    development_seeds = generator["development_seeds"]
    if (len(training_seeds) != 5 or len(development_seeds) != 5
            or len(set(training_seeds + development_seeds)) != 10):
        raise ValueError("Temporal-transfer seeds are not the fixed disjoint split")
    if (tuple(generator["training_nuisance_routes"]) != TRAIN_NUISANCE_ROUTES
            or tuple(generator["development_nuisance_routes"]) != DEVELOPMENT_NUISANCE_ROUTES
            or set(TRAIN_NUISANCE_ROUTES) & set(DEVELOPMENT_NUISANCE_ROUTES)):
        raise ValueError("Temporal-transfer nuisance routes are not isolated")
    training = generate_temporal_transfer(
        seeds=training_seeds, count_per_seed=generator["training_count_per_seed"],
        split="training", namespace=generator["namespace"],
    )
    development = generate_temporal_transfer(
        seeds=development_seeds, count_per_seed=generator["development_count_per_seed"],
        split="development", namespace=generator["namespace"],
    )
    if len(training) != 200 or len(development) != 100:
        raise ValueError("Temporal-transfer episode count changed")
    training_patterns = {labeled_pattern(row) for row in training}
    pattern_overlap = sum(labeled_pattern(row) in training_patterns for row in development)
    identity_overlap = len({row.identity for row in training} &
                           {row.identity for row in development})
    training_routes = {row.events[-1].route for row in training}
    development_routes = {row.events[-1].route for row in development}
    route_overlap = len(training_routes & development_routes)
    if pattern_overlap or identity_overlap or route_overlap:
        raise ValueError("Temporal-transfer pre-execution identity gate failed")
    if any(row.label not in (0, 1) for row in (*training, *development)):
        raise ValueError("Temporal-transfer labels are invalid")
    for rows, seeds, count in ((training, training_seeds, 40),
                               (development, development_seeds, 20)):
        for seed in seeds:
            selected = [row for row in rows if row.identity.split(":")[2] == str(seed)]
            if len(selected) != count or sum(row.label for row in selected) * 2 != count:
                raise ValueError("Temporal-transfer per-seed label balance changed")
    return training, development, {
        "training_count": len(training),
        "development_count": len(development),
        "exact_labeled_pattern_overlap": pattern_overlap,
        "episode_identity_overlap": identity_overlap,
        "nuisance_route_overlap": route_overlap,
        "input_event_count": sum(len(row.events) for row in (*training, *development)),
    }


def _intent(config: dict, sequence: int) -> CaptureIntent:
    actions = config["fixed_action_order"]
    if sequence < 0 or sequence > len(actions):
        raise ValueError("Temporal-transfer action sequence is invalid")
    return CaptureIntent(
        sequence=sequence,
        node_id=ROOT_ID if sequence == 0 else actions[sequence - 1]["node_id"],
        parent_id=None if sequence == 0 else ROOT_ID,
        hypothesis_id="temporal-transfer-v1",
        policy_id="manual-fixed-order-v1",
        candidate_sha256=config["candidate_sources"]["generator_and_evaluator_sha256"],
        preregistration_sha256=PROTOCOL_SHA256,
        evaluator_id="frozen-temporal-transfer-v1",
    )


def _checked_view(log: DiscoveryCaptureLog, config: dict):
    view = log.load()
    intents = [event for event in view.events if isinstance(event, CaptureIntent)]
    outcomes = [event for event in view.events if isinstance(event, CaptureOutcome)]
    if len(intents) > len(config["fixed_action_order"]) + 1:
        raise ValueError("Temporal-transfer capture has extra actions")
    if any(event != _intent(config, index) for index, event in enumerate(intents)):
        raise ValueError("Temporal-transfer capture intent identity changed")
    if any(event.status != "valid" for event in outcomes):
        raise ValueError("Temporal-transfer capture has a non-valid outcome")
    return view, intents, outcomes


def prepare_next(capture_path: str) -> dict:
    config = _protocol()
    _, _, audit = preflight(config)
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _checked_view(log, config)
    if not intents:
        if log.path.exists():
            raise ValueError("Existing empty capture path is not accepted")
        root_head = log.begin(_intent(config, 0), expected_head_sha256=view.head_sha256)
        view, intents, outcomes = _checked_view(log, config)
        if view.head_sha256 != root_head:
            raise ValueError("Temporal-transfer root head changed")
    if view.pending_node_ids:
        raise ValueError("Complete or review the pending intent first")
    sequence = len(intents)
    if sequence > len(config["fixed_action_order"]):
        raise ValueError("All temporal-transfer actions are complete")
    if len(outcomes) != sequence - 1:
        raise ValueError("Prior temporal-transfer action is incomplete")
    intent = _intent(config, sequence)
    head = log.begin(intent, expected_head_sha256=view.head_sha256)
    return {
        "schema": "sara-temporal-transfer-prepared-v1",
        "action": intent.node_id,
        "intent_head_sha256": head,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight": audit,
        "evaluator_invoked": False,
    }


def execute_pending(capture_path: str, *, published_intent_head_sha256: str) -> dict:
    config = _protocol()
    training, development, audit = preflight(config)
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _checked_view(log, config)
    if len(intents) != len(outcomes) + 2 or len(view.pending_node_ids) != 1:
        raise ValueError("Capture has no sole pending temporal-transfer action")
    sequence = len(intents) - 1
    intent = _intent(config, sequence)
    action = config["fixed_action_order"][sequence - 1]
    observed = {}

    def evaluate() -> CaptureOutcome:
        started = time.process_time_ns()
        result = run_frozen_development_arm(
            action["arm"], training, development,
            intervention=action["intervention"],
        )
        elapsed_ns = time.process_time_ns() - started
        labels = {row.identity: row.label for row in development}
        per_seed = {str(seed): [0, 0] for seed in config["generator"]["development_seeds"]}
        for identity, predicted in result["prediction_rows"]:
            seed = identity.split(":")[2]
            per_seed[seed][0] += int(predicted == labels[identity])
            per_seed[seed][1] += 1
        if any(total != 20 for _, total in per_seed.values()):
            raise ValueError("Temporal-transfer prediction trace is incomplete")
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
            event_count=audit["input_event_count"],
            state_bytes=int(result["state_bytes"]),
        )

    captured = complete_prepared_development_action(
        log, intent, evaluate,
        published_intent_head_sha256=published_intent_head_sha256,
    )
    return {
        "schema": "sara-temporal-transfer-observed-v1",
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
    """Replay frozen development predictions against first-run trace digests."""
    config = _protocol()
    training, development, audit = preflight(config)
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _checked_view(log, config)
    actions = config["fixed_action_order"]
    if view.head_sha256 != expected_capture_head_sha256:
        raise ValueError("Published final capture head does not match")
    if view.pending_node_ids or len(intents) != len(actions) + 1 or len(outcomes) != len(actions):
        raise ValueError("Temporal-transfer capture is incomplete")
    if [row.node_id for row in outcomes] != [action["node_id"] for action in actions]:
        raise ValueError("Temporal-transfer outcomes differ from fixed action order")
    if set(expected_traces) != {action["node_id"] for action in actions}:
        raise ValueError("Expected traces must cover all fixed actions")
    labels = {row.identity: row.label for row in development}
    replayed = {}
    for action, outcome in zip(actions, outcomes):
        result = run_frozen_development_arm(
            action["arm"], training, development,
            intervention=action["intervention"],
        )
        node = action["node_id"]
        if (result["prediction_trace_sha256"] != expected_traces[node]
                or result["accuracy"] != outcome.score
                or result["state_bytes"] != outcome.state_bytes
                or outcome.event_count != audit["input_event_count"]
                or outcome.cpu_ms < 1):
            raise ValueError(f"Captured outcome or prediction trace differs: {node}")
        per_seed = {str(seed): [0, 0] for seed in config["generator"]["development_seeds"]}
        for identity, predicted in result["prediction_rows"]:
            seed = identity.split(":")[2]
            per_seed[seed][0] += int(predicted == labels[identity])
            per_seed[seed][1] += 1
        if any(total != 20 for _, total in per_seed.values()):
            raise ValueError("Replayed temporal-transfer predictions are incomplete")
        replayed[node] = {
            "accuracy": result["accuracy"],
            "per_seed_accuracy": {seed: correct / total
                                  for seed, (correct, total) in per_seed.items()},
            "prediction_trace_sha256": result["prediction_trace_sha256"],
            "maximum_event_work": result["maximum_event_work"],
            "state_bytes": result["state_bytes"],
            "cpu_ms_recorded": outcome.cpu_ms,
        }
    b = replayed["B_compact_event"]
    c = replayed["C_temporal_state"]
    shuffled = replayed["C_time_shuffle"]
    reset = replayed["C_state_reset"]
    gate = config["diagnostic_gate"]
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
        "pre_execution_identity_gate": audit["exact_labeled_pattern_overlap"] == 0,
    }
    projection = log.project_source_snapshot(expected_head_sha256=view.head_sha256)
    return {
        "schema": "sara-temporal-transfer-verification-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "capture_head_sha256": view.head_sha256,
        "unapproved_snapshot_sha256": projection.snapshot_sha256,
        "preflight": audit,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "replayed": replayed,
        "export_approved": False,
        "real_event_transfer_claim_allowed": False,
    }


def audit_lineage(capture_path: str, *, expected_capture_file_sha256: str,
                  expected_capture_head_sha256: str,
                  expected_snapshot_sha256: str,
                  expected_intent_heads: list[str]) -> dict:
    """Reconcile caller-recorded heads and snapshot without granting approval."""
    config = _protocol()
    _, _, split_audit = preflight(config)
    log = DiscoveryCaptureLog(capture_path)
    view, intents, outcomes = _checked_view(log, config)
    actions = config["fixed_action_order"]
    if view.pending_node_ids or len(intents) != len(actions) + 1 or len(outcomes) != len(actions):
        raise ValueError("Temporal-transfer capture is incomplete")
    if [row.node_id for row in outcomes] != [action["node_id"] for action in actions]:
        raise ValueError("Temporal-transfer outcome order changed")
    if view.head_sha256 != expected_capture_head_sha256:
        raise ValueError("Capture head differs from external record")
    raw = log.path.read_bytes()
    if sha256(raw).hexdigest() != expected_capture_file_sha256:
        raise ValueError("Capture file differs from external record")
    lines = [json.loads(line) for line in raw.splitlines()]
    intent_heads = [line["entry_sha256"] for line, event in zip(lines, view.events)
                    if isinstance(event, CaptureIntent) and event.sequence > 0]
    if intent_heads != expected_intent_heads:
        raise ValueError("Intent heads differ from external record")
    projection = log.project_source_snapshot(expected_head_sha256=view.head_sha256)
    if projection.snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError("Snapshot differs from external record")
    return {
        "schema": "sara-temporal-transfer-lineage-audit-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "capture_file_sha256": expected_capture_file_sha256,
        "capture_head_sha256": view.head_sha256,
        "snapshot_sha256": projection.snapshot_sha256,
        "record_count": projection.record_count,
        "intent_heads_match_caller_record": True,
        "split_audit": split_audit,
        "independent_head_attestation": False,
        "export_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "prepare", "execute", "verify", "audit"))
    parser.add_argument("--capture")
    parser.add_argument("--published-intent-head")
    parser.add_argument("--expected-capture-head")
    parser.add_argument("--expected-trace", action="append", default=[])
    parser.add_argument("--expected-capture-file-sha256")
    parser.add_argument("--expected-snapshot-sha256")
    parser.add_argument("--expected-intent-head", action="append", default=[])
    args = parser.parse_args()
    if args.operation == "preflight":
        result = {"schema": "sara-temporal-transfer-preflight-v1",
                  "protocol_sha256": PROTOCOL_SHA256,
                  "audit": preflight(_protocol())[2], "evaluator_invoked": False}
    elif args.operation == "prepare":
        if args.capture is None or args.published_intent_head is not None:
            parser.error("prepare requires a capture path and no published head")
        result = prepare_next(args.capture)
    elif args.operation == "execute":
        if args.capture is None or args.published_intent_head is None:
            parser.error("execute requires capture path and separately published head")
        result = execute_pending(args.capture,
                                 published_intent_head_sha256=args.published_intent_head)
    elif args.operation == "verify":
        if args.capture is None or args.expected_capture_head is None or len(args.expected_trace) != 4:
            parser.error("verify requires capture path, final head, and four recorded traces")
        try:
            traces = dict(item.split("=", 1) for item in args.expected_trace)
        except ValueError:
            parser.error("expected traces must be NODE=SHA256")
        result = verify_capture(args.capture,
                                expected_capture_head_sha256=args.expected_capture_head,
                                expected_traces=traces)
    else:
        if (args.capture is None or args.expected_capture_file_sha256 is None
                or args.expected_capture_head is None or args.expected_snapshot_sha256 is None
                or len(args.expected_intent_head) != 4):
            parser.error("audit requires capture/file/snapshot hashes and four intent heads")
        result = audit_lineage(
            args.capture,
            expected_capture_file_sha256=args.expected_capture_file_sha256,
            expected_capture_head_sha256=args.expected_capture_head,
            expected_snapshot_sha256=args.expected_snapshot_sha256,
            expected_intent_heads=args.expected_intent_head,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
