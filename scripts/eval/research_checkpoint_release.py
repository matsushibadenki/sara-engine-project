#!/usr/bin/env python3
"""Build and verify hash-pinned research-only event-stream checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from sara_engine.evaluation.bpi2012_data import load_traces as load_bpi
from sara_engine.evaluation.bpi2012_data import split_name as bpi_split
from sara_engine.evaluation.sepsis_data import load_traces as load_sepsis
from sara_engine.evaluation.sepsis_data import split_name as sepsis_split
from sara_engine.learning.event_stream_engine import BoundedEventStreamEngine, EventRouteConfig
from sara_engine.learning.normalized_hybrid import NormalizedHybridConfig
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, processed_data_path, workspace_path


SCHEMA = "sara-research-event-stream-checkpoint-release-v1"
MANIFEST_NAME = "research-checkpoints-v1-manifest.json"
DATASETS = ("bpi2012", "sepsis")
PACKAGE_SCHEMA = "sara-event-stream-research-package-v1"
PACKAGE_FILES = {
    "bpi2012": "sara-bpi2012-pretest-v1.sara",
    "sepsis": "sara-sepsis-pretest-v1.sara",
}


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


def _durable_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace research package artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            os.chmod(temporary, 0o600)
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(dataset: str) -> dict:
    name = "source_manifest_v1.json"
    directory = "r2_bpi2012" if dataset == "bpi2012" else "r2_sepsis"
    path = Path(processed_data_path(directory, name))
    document = json.loads(path.read_text())
    return {
        "manifest": f"data/processed/{directory}/{name}",
        "manifest_sha256": _sha256(path),
        "source_sha256": document["source_sha256"],
        "case_counts": document["case_counts"],
        "prediction_counts": document["prediction_counts"],
        "case_overlap": document["case_overlap"],
    }


def _negative_results(dataset: str) -> list[dict]:
    shared = [
        {
            "id": "no-spike-specific-quality-advantage",
            "result": "Explicit-neuron and matched scalar prediction traces are exactly identical.",
        },
        {
            "id": "physical-energy-unmeasured",
            "result": "CPU time and state size are proxies; no joule or hardware-energy advantage is claimed.",
        },
    ]
    if dataset == "bpi2012":
        return shared + [{
            "id": "explicit-neuron-cpu-overhead",
            "result": "The explicit-neuron implementation used 13.11% more median CPU time than scalar mode.",
        }]
    return shared + [
        {
            "id": "inherited-absolute-router-negative",
            "result": "The inherited absolute router failed the Sepsis development gate before normalization.",
        },
        {
            "id": "explicit-neuron-cpu-overhead",
            "result": "The explicit-neuron implementation used 14.97% more median CPU time than scalar mode.",
        },
    ]


def package() -> dict:
    release_path = Path(model_path("event_stream_engine", MANIFEST_NAME))
    release = json.loads(release_path.read_text())
    if release.get("schema") != SCHEMA or set(release.get("datasets", {})) != set(DATASETS):
        raise ValueError("Source release manifest is invalid")
    planned = [Path(model_path(dataset, name)) for dataset, name in PACKAGE_FILES.items()]
    planned.extend(Path(model_path(dataset, name)) for dataset in DATASETS for name in ("manifest.json", "SHA256SUMS"))
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace formal research packages: {existing}")
    results = {}
    for dataset in DATASETS:
        row = release["datasets"][dataset]
        _, _, route_config, hybrid_config, result_name, arm = _configuration(dataset)
        accepted_trace, accepted_result_sha256 = _accepted_result(result_name, arm)
        source_artifact = Path(model_path("event_stream_engine", row["artifact"]))
        package_dir = Path(model_path(dataset))
        artifact = package_dir / PACKAGE_FILES[dataset]
        _durable_copy(source_artifact, artifact)
        engine, generation = BoundedEventStreamEngine.load_from_model_path(artifact)
        state = engine.state_dict()
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "dataset": dataset,
            "research_only": True,
            "production_authorized": False,
            "algorithm": {
                "engine_schema": BoundedEventStreamEngine.SCHEMA,
                "artifact_schema": BoundedEventStreamEngine.ARTIFACT_SCHEMA,
                "learning": "bounded normalized local hybrid",
                "global_gradient_backpropagation": False,
                "dense_matrix_training": False,
                "gpu_required": False,
                "route_configuration": asdict(route_config),
                "hybrid_configuration": asdict(hybrid_config),
                "configuration_sha256": row["configuration_sha256"],
            },
            "data": _source_identity(dataset),
            "checkpoint": {
                "file": artifact.name,
                "generation": generation,
                "artifact_sha256": _sha256(artifact),
                "payload_sha256": row["payload_sha256"],
                "pretest_only": True,
                "training_predictions": row["training_predictions"],
                "routes": len(state["encoder"]["routes"]),
                "weights": len(state["hybrid"]["weights"]),
                "contexts": len(state["hybrid"]["contexts"]),
            },
            "evaluation": {
                "method": "Load the immutable pre-test checkpoint, predict each frozen event in chronological order, then apply its observed outcome only to the private in-memory replay state.",
                "command": f"python3 scripts/eval/research_checkpoint_release.py verify-package --dataset {dataset}",
                "accepted_result": f"workspace/evaluation/{result_name}",
                "accepted_result_sha256": accepted_result_sha256,
                "prediction_digest_encoding": "SHA-256 of compact JSON [target,predicted] pairs in frozen chronological order",
                "frozen_predictions": _source_identity(dataset)["prediction_counts"]["frozen_test"],
                "frozen_prediction_sha256": accepted_trace,
                "online_updates_during_frozen_replay": True,
                "checkpoint_refitting": False,
            },
            "negative_results": _negative_results(dataset),
        }
        manifest_path = package_dir / "manifest.json"
        _durable_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        sums = f"{_sha256(artifact)}  {artifact.name}\n{_sha256(manifest_path)}  manifest.json\n"
        sums_path = package_dir / "SHA256SUMS"
        _durable_write(sums_path, sums.encode())
        results[dataset] = {
            "directory": str(package_dir),
            "artifact_sha256": _sha256(artifact),
            "manifest_sha256": _sha256(manifest_path),
            "sha256sums_sha256": _sha256(sums_path),
        }
    return {"schema": PACKAGE_SCHEMA, "packages": results}


def _read_sums(path: Path, artifact_name: str) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or Path(name).name != name or name in result:
            raise ValueError("SHA256SUMS is invalid")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("SHA256SUMS digest is invalid") from error
        result[name] = digest
    if set(result) != {"manifest.json", artifact_name}:
        raise ValueError("SHA256SUMS file set is invalid")
    return result


def verify_packages(selected: str | None = None) -> dict:
    datasets = (selected,) if selected else DATASETS
    if any(dataset not in DATASETS for dataset in datasets):
        raise ValueError("Unsupported research package dataset")
    results = {}
    for dataset in datasets:
        package_dir = Path(model_path(dataset))
        artifact = package_dir / PACKAGE_FILES[dataset]
        manifest_path = package_dir / "manifest.json"
        sums = _read_sums(package_dir / "SHA256SUMS", artifact.name)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("dataset") != dataset:
            raise ValueError("Research package manifest identity is invalid")
        if manifest.get("research_only") is not True or manifest.get("production_authorized") is not False:
            raise ValueError("Research package authorization boundary is invalid")
        _, _, route_config, hybrid_config, result_name, arm = _configuration(dataset)
        accepted_trace, result_sha256 = _accepted_result(result_name, arm)
        source = _source_identity(dataset)
        checkpoint = manifest["checkpoint"]
        evaluation = manifest["evaluation"]
        checks = {
            "artifact_sum": _sha256(artifact) == sums[artifact.name] == checkpoint["artifact_sha256"],
            "manifest_sum": _sha256(manifest_path) == sums["manifest.json"],
            "source_manifest": manifest["data"] == source,
            "configuration": manifest["algorithm"]["configuration_sha256"]
            == _canonical_sha256({"route": asdict(route_config), "hybrid": asdict(hybrid_config)}),
            "accepted_result": evaluation["accepted_result_sha256"] == result_sha256,
            "accepted_trace": evaluation["frozen_prediction_sha256"] == accepted_trace,
            "no_checkpoint_refit": evaluation["checkpoint_refitting"] is False,
        }
        if not all(checks.values()):
            raise ValueError(f"Research package binding failed for {dataset}")
        engine, generation = BoundedEventStreamEngine.load_from_model_path(artifact)
        state = engine.state_dict()
        loaded_payload_sha256 = _canonical_sha256(state)
        predictions, trace_sha256 = _replay_frozen(dataset, engine)
        checks.update({
            "generation": generation == checkpoint["generation"] == 1,
            "payload": loaded_payload_sha256 == checkpoint["payload_sha256"],
            "resource_identity": (
                len(state["encoder"]["routes"]), len(state["hybrid"]["weights"]), len(state["hybrid"]["contexts"])
            ) == (checkpoint["routes"], checkpoint["weights"], checkpoint["contexts"]),
            "prediction_count": predictions == evaluation["frozen_predictions"],
            "prediction_digest": trace_sha256 == accepted_trace,
            "artifact_immutable": _sha256(artifact) == sums[artifact.name],
        })
        results[dataset] = {"checks": checks, "predictions": predictions, "prediction_trace_sha256": trace_sha256}
    passed = all(all(row["checks"].values()) for row in results.values())
    report = {
        "schema": "sara-event-stream-research-package-verification-v1",
        "load_only": True,
        "production_authorized": False,
        "results": results,
        "passed": passed,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "research_package_verification_v1.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


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
    parser.add_argument("command", choices=("build", "verify", "package", "verify-package"))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.dataset and args.command != "verify-package":
        parser.error("--dataset is valid only with verify-package")
    actions = {"build": build, "verify": verify, "package": package}
    result = verify_packages(args.dataset) if args.command == "verify-package" else actions[args.command]()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
