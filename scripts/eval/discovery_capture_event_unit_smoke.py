#!/usr/bin/env python3
"""Development-only smoke run for intent-before-outcome discovery capture."""
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
    run_captured_development_action,
)
from sara_engine.utils.project_paths import project_path


PROTOCOL_PATH = project_path("doc", "DISCOVERY_CAPTURE_SMOKE_PROTOCOL.md")
CANDIDATE_PATH = project_path("src", "sara_engine", "evaluation",
                              "event_unit_causal_isolation.py")
SEED = 92017
ARM = "B_compact_event"
ROOT_ID = "capture-smoke-root"
ACTION_ID = "capture-smoke-compact-event"


def _digest(path: str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def run_pilot(capture_path: str) -> dict:
    """Run one fixed synthetic development action without promotion or export."""
    log = DiscoveryCaptureLog(capture_path)
    if log.path.exists():
        raise FileExistsError("Capture path already exists")
    candidate_hash = _digest(CANDIDATE_PATH)
    protocol_hash = _digest(PROTOCOL_PATH)
    shared = {
        "hypothesis_id": "capture-integration-smoke-v1",
        "policy_id": "manual-fixed-single-arm-v1",
        "candidate_sha256": candidate_hash,
        "preregistration_sha256": protocol_hash,
        "evaluator_id": "event-unit-v2-development-smoke-v1",
    }
    root_head = log.begin(CaptureIntent(0, ROOT_ID, None, **shared),
                          expected_head_sha256="0" * 64)
    observed: dict = {}

    def evaluate() -> CaptureOutcome:
        if log.load().pending_node_ids != (ACTION_ID,):
            raise ValueError("Evaluator did not observe a pending intent")
        started = time.process_time_ns()
        training = generate_episodes(
            seeds=(SEED,), count_per_family=4, split="training",
            namespace="capture-smoke-v1",
        )
        development = generate_episodes(
            seeds=(SEED,), count_per_family=2, split="development",
            namespace="capture-smoke-v1",
        )
        result = run_v2_development_arm(ARM, training, development)
        elapsed_ns = time.process_time_ns() - started
        observed.update(result)
        observed["input_event_count"] = sum(
            len(episode.events) for episode in (*training, *development)
        )
        return CaptureOutcome(
            sequence=1, node_id=ACTION_ID, status="valid",
            score=float(result["accuracy"]),
            cpu_ms=max(1, (elapsed_ns + 999_999) // 1_000_000),
            event_count=observed["input_event_count"],
            state_bytes=int(result["state_bytes"]),
        )

    captured = run_captured_development_action(
        log, CaptureIntent(1, ACTION_ID, ROOT_ID, **shared), evaluate,
        expected_head_sha256=root_head,
    )
    view = log.load()
    if view.pending_node_ids or view.head_sha256 != captured.outcome_head_sha256:
        raise ValueError("Capture did not close cleanly")
    projection = log.project_source_snapshot(
        expected_head_sha256=captured.outcome_head_sha256)
    return {
        "schema": "sara-discovery-capture-smoke-result-v1",
        "observed_only": True,
        "promotion_authorized": False,
        "candidate_sha256": candidate_hash,
        "protocol_sha256": protocol_hash,
        "capture_head_sha256": captured.outcome_head_sha256,
        "unapproved_snapshot_sha256": projection.snapshot_sha256,
        "accuracy": captured.outcome.score,
        "input_event_count": captured.outcome.event_count,
        "state_bytes": captured.outcome.state_bytes,
        "cpu_ms": captured.outcome.cpu_ms,
        "prediction_trace_sha256": observed["prediction_trace_sha256"],
        "development_predictions": observed["predictions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-path", required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.capture_path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
