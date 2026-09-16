#!/usr/bin/env python3
"""Build and verify hash-pinned research-only event-stream checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from sara_engine.evaluation.bpi2012_data import load_traces as load_bpi
from sara_engine.evaluation.bpi2012_data import split_name as bpi_split
from sara_engine.evaluation.sepsis_data import load_traces as load_sepsis
from sara_engine.evaluation.sepsis_data import split_name as sepsis_split
from sara_engine.learning.event_stream_engine import BoundedEventStreamEngine, EventRouteConfig
from sara_engine.learning.normalized_hybrid import NormalizedHybridConfig
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path


SCHEMA = "sara-research-event-stream-checkpoint-release-v1"
MANIFEST_NAME = "research-checkpoints-v1-manifest.json"
DATASETS = ("bpi2012", "sepsis")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _configuration(dataset: str) -> tuple[Callable, Callable, EventRouteConfig, NormalizedHybridConfig, str, str]:
    if dataset == "bpi2012":
        return (
            load_bpi,
            bpi_split,
            EventRouteConfig(include_calendar_route=True),
            NormalizedHybridConfig(
                routing_mode="absolute",
                base_probability_cap=0.75,
                local_score_margin=0.10,
                minimum_base_support=16,
                max_active=8,
                max_classes=24,
                explicit_neurons=True,
                compact_event_units=True,
            ),
            "r2_bpi2012_hybrid_final_v1_result.json",
            "snn_hybrid",
        )
    if dataset == "sepsis":
        return (
            load_sepsis,
            sepsis_split,
            EventRouteConfig(include_calendar_route=False),
            NormalizedHybridConfig(
                routing_mode="normalized",
                normalized_threshold=0.6223091976516634,
                minimum_base_support=16,
                max_active=7,
                max_classes=16,
                explicit_neurons=True,
                compact_event_units=True,
            ),
            "r2_sepsis_normalized_final_v1_result.json",
            "snn_normalized",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _accepted_result(result_name: str, arm: str) -> tuple[str, str]:
    path = Path(workspace_path("evaluation", result_name))
    document = json.loads(path.read_text())
    return document["arms"][arm]["frozen_test"]["prediction_trace_sha256"], _sha256(path)


def _prediction_digest(rows: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _train_to_boundary(dataset: str) -> tuple[BoundedEventStreamEngine, int]:
    loader, split, route_config, hybrid_config, _, _ = _configuration(dataset)
    traces = loader()
    labels = tuple(sorted({event.activity for trace in traces if split(trace) != "frozen_test" for event in trace.events}))
    engine = BoundedEventStreamEngine(labels, route_config=route_config, hybrid_config=hybrid_config)
    observed = 0
    frozen_seen = False
    for trace in traces:
        if split(trace) == "frozen_test":
            frozen_seen = True
            continue
        if frozen_seen:
            raise ValueError("Non-test trace appeared after the frozen-test boundary")
        for index in range(len(trace.events) - 1):
            prediction = engine.predict(trace.events, index)
            engine.observe(prediction, trace.events[index + 1].activity)
            observed += 1
    if not frozen_seen:
        raise ValueError("Frozen-test boundary was not found")
    return engine, observed


def _replay_frozen(dataset: str, engine: BoundedEventStreamEngine) -> tuple[int, str]:
    loader, split, _, _, _, _ = _configuration(dataset)
    rows = []
    for trace in loader():
        if split(trace) != "frozen_test":
            continue
        for index in range(len(trace.events) - 1):
            prediction = engine.predict(trace.events, index)
            target = trace.events[index + 1].activity
            rows.append((target, prediction.predicted))
            engine.observe(prediction, target)
    return len(rows), _prediction_digest(rows)


def _durable_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace research release manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def build() -> dict:
    planned = [Path(model_path("event_stream_engine", f"{dataset}-pretest-v1.json")) for dataset in DATASETS]
    planned.append(Path(model_path("event_stream_engine", MANIFEST_NAME)))
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace published research evidence: {existing}")
    release_rows = {}
    for dataset in DATASETS:
        _, _, route_config, hybrid_config, result_name, arm = _configuration(dataset)
        accepted_trace, result_sha256 = _accepted_result(result_name, arm)
        engine, training_predictions = _train_to_boundary(dataset)
        filename = f"{dataset}-pretest-v1.json"
        receipt = engine.publish(filename, expected_generation=0)
        artifact = Path(receipt.path)
        release_rows[dataset] = {
            "artifact": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "payload_sha256": receipt.payload_sha256,
            "generation": receipt.generation,
            "training_predictions": training_predictions,
            "accepted_prediction_trace_sha256": accepted_trace,
            "accepted_result": result_name,
            "accepted_result_sha256": result_sha256,
            "configuration_sha256": _canonical_sha256(
                {"route": asdict(route_config), "hybrid": asdict(hybrid_config)}
            ),
        }
    manifest = {
        "schema": SCHEMA,
        "production_authorized": False,
        "artifacts_are_pretest_only": True,
        "datasets": release_rows,
    }
    path = Path(ensure_parent_directory(model_path("event_stream_engine", MANIFEST_NAME)))
    _durable_write(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return {"manifest": str(path), "manifest_sha256": _sha256(path), "datasets": release_rows}


def verify() -> dict:
    manifest_path = Path(model_path("event_stream_engine", MANIFEST_NAME))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("production_authorized") is not False:
        raise ValueError("Research checkpoint manifest boundary is invalid")
    if manifest.get("artifacts_are_pretest_only") is not True or set(manifest.get("datasets", {})) != set(DATASETS):
        raise ValueError("Research checkpoint manifest dataset set is invalid")
    results = {}
    for dataset in DATASETS:
        row = manifest["datasets"][dataset]
        _, _, route_config, hybrid_config, result_name, arm = _configuration(dataset)
        accepted_trace, accepted_result_sha256 = _accepted_result(result_name, arm)
        artifact = Path(model_path("event_stream_engine", row["artifact"]))
        checks = {
            "artifact_hash": _sha256(artifact) == row["artifact_sha256"],
            "accepted_result_hash": accepted_result_sha256 == row["accepted_result_sha256"],
            "accepted_trace_binding": accepted_trace == row["accepted_prediction_trace_sha256"],
            "configuration_binding": row["configuration_sha256"]
            == _canonical_sha256({"route": asdict(route_config), "hybrid": asdict(hybrid_config)}),
        }
        if not all(checks.values()):
            raise ValueError(f"Hash-pinned research inputs changed for {dataset}")
        engine, generation = BoundedEventStreamEngine.load_with_generation(row["artifact"])
        loaded_payload_sha256 = _canonical_sha256(engine.state_dict())
        predictions, trace_sha256 = _replay_frozen(dataset, engine)
        checks.update(
            {
                "generation": generation == row["generation"] == 1,
                "payload_hash": loaded_payload_sha256 == row["payload_sha256"],
                "private_replay_changed_state": _canonical_sha256(engine.state_dict()) != loaded_payload_sha256,
                "frozen_prediction_trace": trace_sha256 == accepted_trace,
                "nonempty_frozen_partition": predictions > 0,
            }
        )
        # Replay updates private in-memory state. The changed state hash proves that the artifact itself
        # stopped at the pre-test boundary instead of storing post-test updates.
        results[dataset] = {"predictions": predictions, "prediction_trace_sha256": trace_sha256, "checks": checks}
    passed = all(all(result["checks"].values()) for result in results.values())
    report = {
        "schema": "sara-research-event-stream-checkpoint-verification-v1",
        "manifest": MANIFEST_NAME,
        "manifest_sha256": _sha256(manifest_path),
        "load_only": True,
        "production_authorized": False,
        "results": results,
        "passed": passed,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "research_checkpoint_release_v1.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    result = build() if args.command == "build" else verify()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
