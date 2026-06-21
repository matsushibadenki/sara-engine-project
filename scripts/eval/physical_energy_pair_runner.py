#!/usr/bin/env python3
"""Prepare and execute one fair SARA/ANN physical-energy measurement pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path, workspace_path  # noqa: E402


DEFAULT_CORPUS_PATH = processed_data_path("corpus.txt")
DEFAULT_MEASUREMENT_PATH = raw_data_path("energy_measurements.jsonl")
DEFAULT_MANIFEST_PATH = workspace_path("evaluation", "physical_energy_pair_manifest.json")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "physical_energy_pair_trace.jsonl")


def _load_energy_module():
    path = os.path.join(PROJECT_ROOT, "scripts", "eval", "energy_measurement_readiness.py")
    spec = importlib.util.spec_from_file_location("physical_pair_energy_readiness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load energy measurement readiness module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_fixture_hash(
    *,
    corpus_path: str,
    max_docs: int,
    max_cases: int,
    success_criterion_id: str,
) -> str:
    payload = {
        "corpus_sha256": _sha256_file(corpus_path),
        "max_docs": int(max_docs),
        "max_cases": int(max_cases),
        "success_criterion_id": success_criterion_id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cpu_model() -> str:
    command = ["sysctl", "-n", "machdep.cpu.brand_string"]
    result = subprocess.run(command, capture_output=True, text=True)
    value = result.stdout.strip()
    if value:
        return value
    profiler = subprocess.run(
        ["system_profiler", "SPHardwareDataType"],
        capture_output=True,
        text=True,
    )
    chip = ""
    model_identifier = ""
    for line in profiler.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chip:"):
            chip = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Model Identifier:"):
            model_identifier = stripped.split(":", 1)[1].strip()
    if chip or model_identifier:
        return " ".join(part for part in (chip, model_identifier) if part)
    return platform.processor() or platform.machine()


def build_manifest(
    *,
    corpus_path: str,
    replicate_index: int,
    repetitions: int,
    warmup_count: int,
    thread_count: int,
    process_affinity: str,
    power_mode: str,
    measurement_tool: str,
    pair_id: str,
    max_docs: int = 256,
    max_cases: int = 24,
) -> Dict[str, Any]:
    energy = _load_energy_module()
    cpu_model = _cpu_model()
    environment_fingerprint = energy.default_environment_fingerprint(
        cpu_model=cpu_model,
        thread_count=thread_count,
        process_affinity=process_affinity,
        power_mode=power_mode,
    )
    order = ["sara", "ann"] if replicate_index % 2 else ["ann", "sara"]
    return {
        "schema": "sara-physical-energy-pair-manifest-v1",
        "protocol_version": energy.MEASUREMENT_PROTOCOL_VERSION,
        "pair_id": pair_id,
        "replicate_index": int(replicate_index),
        "task": "paired_retrieval",
        "task_fixture_hash": _task_fixture_hash(
            corpus_path=corpus_path,
            max_docs=max_docs,
            max_cases=max_cases,
            success_criterion_id="retrieval-exact-document-index-v1",
        ),
        "success_criterion_id": "retrieval-exact-document-index-v1",
        "measurement_boundary": "warm-index-repeated-query-process-v1",
        "measurement_tool": measurement_tool,
        "cpu_model": cpu_model,
        "thread_count": int(thread_count),
        "process_affinity": process_affinity,
        "power_mode": power_mode,
        "environment_fingerprint": environment_fingerprint,
        "warmup_count": int(warmup_count),
        "measured_repetitions": int(repetitions),
        "run_order": order,
        "corpus_path": os.path.abspath(corpus_path),
        "max_docs": int(max_docs),
        "max_cases": int(max_cases),
        "created_at_unix": time.time(),
    }


def _workload_command(
    *,
    system: str,
    manifest: Mapping[str, Any],
    output_path: str,
) -> List[str]:
    return [
        sys.executable,
        "scripts/eval/energy_pair_workload.py",
        "--system",
        system,
        "--task",
        "paired_retrieval",
        "--corpus-path",
        str(manifest["corpus_path"]),
        "--repetitions",
        str(manifest["measured_repetitions"]),
        "--max-docs",
        str(manifest["max_docs"]),
        "--max-cases",
        str(manifest["max_cases"]),
        "--warmup-count",
        str(manifest["warmup_count"]),
        "--output-path",
        output_path,
    ]


def execute_pair(
    manifest: Mapping[str, Any],
    *,
    trace_path: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for order_index, system in enumerate(manifest["run_order"], start=1):
        output_path = workspace_path(
            "evaluation",
            f"physical_energy_{manifest['pair_id']}_{system}.json",
        )
        command = _workload_command(
            system=system,
            manifest=manifest,
            output_path=output_path,
        )
        trace: Dict[str, Any] = {
            "system": system,
            "run_order": order_index,
            "command": command,
            "output_path": output_path,
            "status": "planned" if dry_run else "pending",
        }
        if not dry_run:
            started = time.perf_counter()
            environment = dict(os.environ)
            thread_count = str(manifest["thread_count"])
            environment.update(
                {
                    "OMP_NUM_THREADS": thread_count,
                    "OPENBLAS_NUM_THREADS": thread_count,
                    "MKL_NUM_THREADS": thread_count,
                    "VECLIB_MAXIMUM_THREADS": thread_count,
                }
            )
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=environment,
            )
            trace["wall_duration_seconds"] = time.perf_counter() - started
            trace["returncode"] = result.returncode
            trace["status"] = "passed" if result.returncode == 0 else "failed"
            trace["stderr_tail"] = result.stderr[-1000:]
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as handle:
                    trace["workload_result"] = json.load(handle)
        traces.append(trace)
    resolved = ensure_parent_directory(trace_path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    return traces


def append_measured_rows(
    manifest: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    sara_joules: float,
    ann_joules: float,
    measurement_path: str,
) -> List[Dict[str, Any]]:
    energy = _load_energy_module()
    joules = {"sara": float(sara_joules), "ann": float(ann_joules)}
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        system = str(trace.get("system", ""))
        result = trace.get("workload_result", {})
        if not isinstance(result, Mapping) or not bool(result.get("passed", False)):
            raise ValueError(f"Cannot record failed workload for {system}.")
        row = energy.build_measurement_row(
            run_id=f"{manifest['pair_id']}-{system}-r{manifest['replicate_index']}",
            system=system,
            task=str(manifest["task"]),
            success_count=int(result["success_count"]),
            trial_count=int(result["trial_count"]),
            joules=joules[system],
            source="physical_energy_pair_runner",
            duration_seconds=float(result["duration_seconds"]),
            protocol_version=str(manifest["protocol_version"]),
            pair_id=str(manifest["pair_id"]),
            replicate_index=int(manifest["replicate_index"]),
            environment_fingerprint=str(manifest["environment_fingerprint"]),
            task_fixture_hash=str(manifest["task_fixture_hash"]),
            success_criterion_id=str(manifest["success_criterion_id"]),
            measurement_boundary=str(manifest["measurement_boundary"]),
            measurement_tool=str(manifest["measurement_tool"]),
            cpu_model=str(manifest["cpu_model"]),
            thread_count=int(manifest["thread_count"]),
            process_affinity=str(manifest["process_affinity"]),
            power_mode=str(manifest["power_mode"]),
            warmup_count=int(manifest["warmup_count"]),
            measured_repetitions=int(manifest["measured_repetitions"]),
            run_order=int(trace["run_order"]),
        )
        energy.append_measurement(measurement_path, row)
        rows.append(row)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fair physical-energy pair.")
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--process-affinity", default="unbound-single-process")
    parser.add_argument("--power-mode", default="ac-power-default")
    parser.add_argument("--measurement-tool", default="external-meter-manual-v1")
    parser.add_argument("--sara-joules", type=float, default=0.0)
    parser.add_argument("--ann-joules", type=float, default=0.0)
    parser.add_argument("--measurement-path", default=DEFAULT_MEASUREMENT_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        corpus_path=args.corpus_path,
        replicate_index=args.replicate_index,
        repetitions=args.repetitions,
        warmup_count=args.warmup_count,
        thread_count=args.thread_count,
        process_affinity=args.process_affinity,
        power_mode=args.power_mode,
        measurement_tool=args.measurement_tool,
        pair_id=args.pair_id,
        max_docs=args.max_docs,
        max_cases=args.max_cases,
    )
    manifest_path = ensure_parent_directory(args.manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    traces = execute_pair(manifest, trace_path=args.trace_path, dry_run=args.dry_run)
    recorded_rows: List[Dict[str, Any]] = []
    if not args.dry_run and args.sara_joules > 0.0 and args.ann_joules > 0.0:
        recorded_rows = append_measured_rows(
            manifest,
            traces,
            sara_joules=args.sara_joules,
            ann_joules=args.ann_joules,
            measurement_path=args.measurement_path,
        )
    report = {
        "passed": all(trace["status"] in {"planned", "passed"} for trace in traces),
        "dry_run": args.dry_run,
        "manifest_path": manifest_path,
        "trace_path": os.path.abspath(args.trace_path),
        "recorded_row_count": len(recorded_rows),
        "measurement_pending": len(recorded_rows) == 0,
        "run_order": manifest["run_order"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
