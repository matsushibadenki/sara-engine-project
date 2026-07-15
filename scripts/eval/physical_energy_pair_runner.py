#!/usr/bin/env python3
"""Prepare and execute one fair SARA/ANN physical-energy measurement pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
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
DEFAULT_REPORT_PATH = workspace_path("evaluation", "physical_energy_pair_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "physical_energy_pair_summary.txt")
DEFAULT_METER_TEMPLATE_PATH = workspace_path(
    "evaluation",
    "physical_energy_pair_meter_template.json",
)
DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH = workspace_path(
    "evaluation",
    "internal_maintenance_efficiency_benchmark.json",
)
DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH = workspace_path(
    "evaluation",
    "event_memory_maintenance_coupling_benchmark.json",
)


def _positive_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return parsed


def _load_optional_json(path: str) -> Optional[Dict[str, Any]]:
    candidate = str(path or "").strip()
    if not candidate or not os.path.exists(candidate):
        return None
    with open(candidate, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _bundle_contribution_warning(reference: Mapping[str, Any] | None) -> str:
    payload = reference if isinstance(reference, Mapping) else {}
    if not bool(payload.get("available", False)):
        return ""
    try:
        contribution = float(
            payload.get("best_profile_multimodal_bundle_compression_contribution", 0.0)
            or 0.0
        )
    except (TypeError, ValueError):
        contribution = 0.0
    if contribution >= 0.5:
        return ""
    return (
        "Bundle-backed compression contribution is weak; rerun the Event Memory maintenance coupling "
        "benchmark with stronger verified multimodal bundle fixtures before treating physical compression "
        "wins as SARA-native binding gains."
    )


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


def _macos_system_power_sample() -> Optional[Dict[str, Any]]:
    """Read macOS system power telemetry as an explicitly non-physical estimate."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-k", "BatteryData", "-d", "1"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r'"SystemPowerIn"\s*=\s*([0-9]+)', result.stdout)
    if not match:
        return None
    milliwatts = float(match.group(1))
    if milliwatts <= 0.0:
        return None
    return {
        "watts": milliwatts / 1000.0,
        "source": "macos_ioreg_system_power",
        "measurement_quality": "system_estimate",
        "physical_evidence": False,
    }


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


def _resume_append_command(
    manifest: Mapping[str, Any],
    *,
    measurement_path: str,
    manifest_path: str,
    trace_path: str,
    report_path: str,
    summary_path: str,
) -> str:
    corpus_path = str(manifest.get("corpus_path", "<corpus-path>") or "<corpus-path>")
    max_docs = int(manifest.get("max_docs", 256) or 256)
    max_cases = int(manifest.get("max_cases", 24) or 24)
    repetitions = int(manifest.get("measured_repetitions", 1) or 1)
    warmup_count = int(manifest.get("warmup_count", 0) or 0)
    thread_count = int(manifest.get("thread_count", 1) or 1)
    process_affinity = str(manifest.get("process_affinity", "<affinity>") or "<affinity>")
    power_mode = str(manifest.get("power_mode", "<power-mode>") or "<power-mode>")
    measurement_tool = str(
        manifest.get("measurement_tool", "<measurement-tool>") or "<measurement-tool>"
    )
    return (
        "python scripts/sara_cli.py run-physical-energy-pair "
        f"--pair-id {manifest['pair_id']} "
        f"--replicate-index {int(manifest['replicate_index'])} "
        f"--corpus-path {corpus_path} "
        f"--max-docs {max_docs} "
        f"--max-cases {max_cases} "
        f"--repetitions {repetitions} "
        f"--warmup-count {warmup_count} "
        f"--thread-count {thread_count} "
        f"--process-affinity {process_affinity} "
        f"--power-mode {power_mode} "
        f"--measurement-tool {measurement_tool} "
        "--sara-joules <J> --ann-joules <J> "
        f"--measurement-path {measurement_path} "
        f"--manifest-path {manifest_path} "
        f"--trace-path {trace_path} "
        f"--report-path {report_path} "
        f"--summary-path {summary_path}"
    )


def _record_measurement_commands(
    manifest: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    measurement_path: str,
) -> List[Dict[str, str]]:
    commands: List[Dict[str, str]] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        system = str(trace.get("system", "") or "")
        result = trace.get("workload_result", {})
        if system not in {"sara", "ann"} or not isinstance(result, Mapping) or not result:
            continue
        success_count = int(result.get("success_count", 0) or 0)
        trial_count = int(result.get("trial_count", success_count) or success_count)
        duration_seconds = float(result.get("duration_seconds", 0.0) or 0.0)
        command = (
            "python scripts/sara_cli.py record-energy-measurement "
            f"--measurement-path {measurement_path} "
            f"--run-id {manifest['pair_id']}-{system}-r{int(manifest['replicate_index'])} "
            f"--system {system} "
            f"--task {manifest['task']} "
            f"--success-count {success_count} "
            "--joules <J> "
            "--source real_energy_session "
            f"--duration-seconds {duration_seconds:.6f} "
            f"--pair-id {manifest['pair_id']} "
            f"--replicate-index {int(manifest['replicate_index'])} "
            f"--environment-fingerprint {manifest.get('environment_fingerprint', '<env-sha256>')} "
            f"--task-fixture-hash {manifest.get('task_fixture_hash', '<fixture-sha256>')} "
            f"--success-criterion-id {manifest.get('success_criterion_id', '<criterion-id>')} "
            f"--measurement-boundary {manifest.get('measurement_boundary', '<boundary-id>')} "
            f"--measurement-tool {manifest.get('measurement_tool', '<measurement-tool>')} "
            f"--cpu-model {manifest.get('cpu_model', '<cpu-model>')} "
            f"--thread-count {int(manifest.get('thread_count', 1) or 1)} "
            f"--process-affinity {manifest.get('process_affinity', '<affinity>')} "
            f"--power-mode {manifest.get('power_mode', '<power-mode>')} "
            f"--warmup-count {int(manifest.get('warmup_count', 0) or 0)} "
            f"--measured-repetitions {int(manifest.get('measured_repetitions', 1) or 1)} "
            f"--trial-count {trial_count} "
            f"--run-order {int(trace.get('run_order', 0) or 0)} "
            f"--maintenance-selected-count {int(result.get('maintenance_selected_count', 0) or 0)} "
            f"--maintenance-phase-count {int(result.get('maintenance_phase_count', 0) or 0)} "
            f"--maintenance-refresh-count {int(result.get('maintenance_refresh_count', 0) or 0)} "
            f"--maintenance-event-cost {float(result.get('maintenance_event_cost', 0.0) or 0.0):.6f}"
        )
        commands.append({"system": system, "command": command})
    return commands


def execute_pair(
    manifest: Mapping[str, Any],
    *,
    trace_path: str,
    dry_run: bool,
    auto_system_energy_estimate: bool = False,
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
            power_before = (
                _macos_system_power_sample() if auto_system_energy_estimate else None
            )
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
            power_after = (
                _macos_system_power_sample() if auto_system_energy_estimate else None
            )
            if power_before and power_after:
                average_watts = (float(power_before["watts"]) + float(power_after["watts"])) / 2.0
                trace["system_energy_estimate"] = {
                    "average_watts": average_watts,
                    "duration_seconds": trace["wall_duration_seconds"],
                    "joules": average_watts * trace["wall_duration_seconds"],
                    "source": "macos_ioreg_system_power",
                    "measurement_quality": "system_estimate",
                    "physical_evidence": False,
                }
            elif auto_system_energy_estimate:
                trace["system_energy_estimate"] = {
                    "available": False,
                    "source": "macos_ioreg_system_power",
                    "measurement_quality": "system_estimate",
                    "physical_evidence": False,
                    "notes": "macOS SystemPowerIn telemetry was unavailable before or after the workload.",
                }
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
    source: str = "physical_energy_pair_runner",
    measurement_quality: str = "physical_meter",
    physical_evidence: bool = True,
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
            source=source,
            measurement_quality=measurement_quality,
            physical_evidence=physical_evidence,
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
            maintenance_selected_count=int(result.get("maintenance_selected_count", 0) or 0),
            maintenance_phase_count=int(result.get("maintenance_phase_count", 0) or 0),
            maintenance_refresh_count=int(result.get("maintenance_refresh_count", 0) or 0),
            maintenance_event_cost=float(result.get("maintenance_event_cost", 0.0) or 0.0),
        )
        energy.append_measurement(measurement_path, row)
        rows.append(row)
    return rows


def _load_meter_reading_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Meter reading file must contain one JSON object.")
    return payload


def _meter_joules_for_system(reading: Mapping[str, Any], *, system: str) -> float:
    if "joules" in reading:
        return _positive_float(reading.get("joules"), field_name=f"{system}.joules")
    average_watts = _positive_float(
        reading.get("average_watts"),
        field_name=f"{system}.average_watts",
    )
    duration_seconds = _positive_float(
        reading.get("duration_seconds"),
        field_name=f"{system}.duration_seconds",
    )
    return average_watts * duration_seconds


def load_meter_joules(
    path: str,
    *,
    manifest: Mapping[str, Any],
) -> Dict[str, float]:
    payload = _load_meter_reading_payload(path)
    pair_id = str(payload.get("pair_id", "") or "")
    if pair_id and pair_id != str(manifest.get("pair_id", "") or ""):
        raise ValueError(
            f"Meter reading pair_id mismatch: expected {manifest.get('pair_id')}, got {pair_id}."
        )
    if "replicate_index" in payload and int(payload["replicate_index"]) != int(
        manifest.get("replicate_index", 0) or 0
    ):
        raise ValueError(
            "Meter reading replicate_index mismatch: "
            f"expected {manifest.get('replicate_index')}, got {payload['replicate_index']}."
        )
    readings = payload.get("readings", {})
    if not isinstance(readings, Mapping):
        raise ValueError("Meter reading file must contain a readings object.")
    joules: Dict[str, float] = {}
    for system in ("sara", "ann"):
        reading = readings.get(system, {})
        if not isinstance(reading, Mapping):
            raise ValueError(f"Meter reading for {system} must be an object.")
        joules[system] = _meter_joules_for_system(reading, system=system)
    return joules


def build_meter_reading_template(
    manifest: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    readings: Dict[str, Dict[str, Any]] = {}
    for system in ("sara", "ann"):
        trace = next(
            (
                item
                for item in traces
                if isinstance(item, Mapping) and str(item.get("system", "") or "") == system
            ),
            {},
        )
        result = trace.get("workload_result", {}) if isinstance(trace, Mapping) else {}
        if not isinstance(result, Mapping):
            result = {}
        readings[system] = {
            "joules": None,
            "average_watts": None,
            "duration_seconds": (
                float(result.get("duration_seconds", 0.0) or 0.0)
                if "duration_seconds" in result
                else None
            ),
            "run_order": int(trace.get("run_order", 0) or 0) if isinstance(trace, Mapping) else 0,
            "trial_count": int(result.get("trial_count", 0) or 0) if result else 0,
            "success_count": int(result.get("success_count", 0) or 0) if result else 0,
            "notes": "",
        }
    return {
        "schema": "sara-physical-meter-readings-v1",
        "pair_id": str(manifest.get("pair_id", "") or ""),
        "replicate_index": int(manifest.get("replicate_index", 0) or 0),
        "measurement_tool": str(manifest.get("measurement_tool", "") or ""),
        "measurement_boundary": str(manifest.get("measurement_boundary", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "readings": readings,
    }


def build_pair_report(
    manifest: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    measurement_path: str,
    meter_reading_path: Optional[str],
    meter_template_path: str,
    recorded_rows: Sequence[Mapping[str, Any]],
    manifest_path: str,
    trace_path: str,
    report_path: str,
    summary_path: str,
    internal_maintenance_report: Optional[Mapping[str, Any]] = None,
    event_memory_maintenance_coupling_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    workload_results: Dict[str, Mapping[str, Any]] = {}
    maintenance_by_system: Dict[str, Dict[str, float]] = {}
    maintenance_alignment: Dict[str, Dict[str, float]] = {}
    internal_reference: Dict[str, float] = {}
    event_memory_maintenance_coupling_reference: Dict[str, float | int | str | bool] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        system = str(trace.get("system", "") or "")
        result = trace.get("workload_result", {})
        if isinstance(result, Mapping) and result:
            workload_results[system] = result
            maintenance_by_system[system] = {
                "maintenance_selected_count": float(result.get("maintenance_selected_count", 0) or 0),
                "maintenance_phase_count": float(result.get("maintenance_phase_count", 0) or 0),
                "maintenance_refresh_count": float(result.get("maintenance_refresh_count", 0) or 0),
                "maintenance_event_cost": float(result.get("maintenance_event_cost", 0.0) or 0.0),
                "maintenance_idle_self_state_ok_count": float(
                    result.get("maintenance_idle_self_state_ok_count", 0) or 0
                ),
                "maintenance_spontaneous_event_count": float(
                    result.get("maintenance_spontaneous_event_count", 0) or 0
                ),
                "maintenance_predicted_event_count": float(
                    result.get("maintenance_predicted_event_count", 0) or 0
                ),
            }
    if isinstance(internal_maintenance_report, Mapping):
        counts = internal_maintenance_report.get("counts", {})
        normalized = internal_maintenance_report.get("normalized_metrics", {})
        if isinstance(counts, Mapping) and isinstance(normalized, Mapping):
            internal_reference = {
                "maintenance_selected_count": float(
                    counts.get("maintenance_selected_count", 0) or 0
                ),
                "maintenance_phase_count": float(
                    counts.get("maintenance_phase_count", 0) or 0
                ),
                "maintenance_refresh_count": float(
                    counts.get("maintenance_refresh_count", 0) or 0
                ),
                "maintenance_idle_self_state_ok_count": float(
                    counts.get("maintenance_idle_self_state_ok_count", 0) or 0
                ),
                "maintenance_spontaneous_event_count": float(
                    counts.get("maintenance_spontaneous_event_count", 0) or 0
                ),
                "maintenance_predicted_event_count": float(
                    counts.get("maintenance_predicted_event_count", 0) or 0
                ),
                "maintenance_event_cost": float(
                    normalized.get("maintenance_event_cost", 0.0) or 0.0
                ),
                "maintenance_event_cost_per_selected": float(
                    normalized.get("maintenance_event_cost_per_selected", 0.0) or 0.0
                ),
                "maintenance_event_cost_per_refresh": float(
                    normalized.get("maintenance_event_cost_per_refresh", 0.0) or 0.0
                ),
            }
    if internal_reference and "sara" in maintenance_by_system:
        actual = maintenance_by_system["sara"]
        selected = max(1.0, float(actual.get("maintenance_selected_count", 0.0) or 0.0))
        refresh = max(1.0, float(actual.get("maintenance_refresh_count", 0.0) or 0.0))
        actual_per_selected = float(actual.get("maintenance_event_cost", 0.0) or 0.0) / selected
        actual_per_refresh = float(actual.get("maintenance_event_cost", 0.0) or 0.0) / refresh
        reference_per_selected = float(
            internal_reference.get("maintenance_event_cost_per_selected", 0.0) or 0.0
        )
        reference_per_refresh = float(
            internal_reference.get("maintenance_event_cost_per_refresh", 0.0) or 0.0
        )
        maintenance_alignment["sara"] = {
            "actual_event_cost_per_selected": actual_per_selected,
            "reference_event_cost_per_selected": reference_per_selected,
            "event_cost_per_selected_delta": actual_per_selected - reference_per_selected,
            "event_cost_per_selected_ratio": (
                actual_per_selected / reference_per_selected
                if reference_per_selected > 0.0
                else 0.0
            ),
            "actual_event_cost_per_refresh": actual_per_refresh,
            "reference_event_cost_per_refresh": reference_per_refresh,
            "event_cost_per_refresh_delta": actual_per_refresh - reference_per_refresh,
            "event_cost_per_refresh_ratio": (
                actual_per_refresh / reference_per_refresh
                if reference_per_refresh > 0.0
                else 0.0
            ),
            "selected_count_delta": float(actual.get("maintenance_selected_count", 0.0) or 0.0)
            - float(internal_reference.get("maintenance_selected_count", 0.0) or 0.0),
            "refresh_count_delta": float(actual.get("maintenance_refresh_count", 0.0) or 0.0)
            - float(internal_reference.get("maintenance_refresh_count", 0.0) or 0.0),
            "idle_self_state_ok_delta": float(
                actual.get("maintenance_idle_self_state_ok_count", 0.0) or 0.0
            )
            - float(
                internal_reference.get("maintenance_idle_self_state_ok_count", 0.0) or 0.0
            ),
            "spontaneous_event_delta": float(
                actual.get("maintenance_spontaneous_event_count", 0.0) or 0.0
            )
            - float(internal_reference.get("maintenance_spontaneous_event_count", 0.0) or 0.0),
            "predicted_event_delta": float(
                actual.get("maintenance_predicted_event_count", 0.0) or 0.0
            )
            - float(internal_reference.get("maintenance_predicted_event_count", 0.0) or 0.0),
        }
    if isinstance(event_memory_maintenance_coupling_report, Mapping):
        metrics = (
            event_memory_maintenance_coupling_report.get("metrics", {})
            if isinstance(event_memory_maintenance_coupling_report.get("metrics", {}), Mapping)
            else {}
        )
        best_profile = (
            event_memory_maintenance_coupling_report.get("best_profile", {})
            if isinstance(event_memory_maintenance_coupling_report.get("best_profile", {}), Mapping)
            else {}
        )
        event_memory_maintenance_coupling_reference = {
            "available": True,
            "passed": bool(event_memory_maintenance_coupling_report.get("passed", False)),
            "observed_only": bool(
                event_memory_maintenance_coupling_report.get("observed_only", False)
            ),
            "profile_count": int(
                event_memory_maintenance_coupling_report.get("profile_count", 0) or 0
            ),
            "best_profile_id": str(best_profile.get("profile_id", "") or ""),
            "compression_to_maintenance_correlation": float(
                metrics.get("compression_to_maintenance_correlation", 0.0) or 0.0
            ),
            "best_profile_compression_efficiency_per_maintenance": float(
                metrics.get("best_profile_compression_efficiency_per_maintenance", 0.0)
                or 0.0
            ),
            "best_profile_self_state_continuity": float(
                metrics.get("best_profile_self_state_continuity", 0.0) or 0.0
            ),
            "best_profile_episode_compression_ratio": float(
                metrics.get("best_profile_episode_compression_ratio", 0.0) or 0.0
            ),
            "best_profile_multimodal_bundle_compression_contribution": float(
                metrics.get("best_profile_multimodal_bundle_compression_contribution", 0.0)
                or 0.0
            ),
        }
    bundle_contribution_warning = _bundle_contribution_warning(
        event_memory_maintenance_coupling_reference
    )
    pending_measurement = bool(dry_run or len(recorded_rows) == 0)
    resume_command = _resume_append_command(
        manifest,
        measurement_path=measurement_path,
        manifest_path=manifest_path,
        trace_path=trace_path,
        report_path=report_path,
        summary_path=summary_path,
    )
    record_commands = _record_measurement_commands(
        manifest,
        traces,
        measurement_path=measurement_path,
    )
    return {
        "schema": "sara-physical-energy-pair-report-v1",
        "passed": all(trace.get("status") in {"planned", "passed"} for trace in traces if isinstance(trace, Mapping)),
        "dry_run": bool(dry_run),
        "measurement_pending": pending_measurement,
        "manifest_path": os.path.abspath(manifest_path),
        "trace_path": os.path.abspath(trace_path),
        "measurement_path": str(measurement_path),
        "meter_reading_path": os.path.abspath(meter_reading_path) if meter_reading_path else "",
        "meter_template_path": os.path.abspath(meter_template_path),
        "recorded_row_count": len(recorded_rows),
        "pair_id": str(manifest.get("pair_id", "") or ""),
        "replicate_index": int(manifest.get("replicate_index", 0) or 0),
        "task": str(manifest.get("task", "") or ""),
        "run_order": list(manifest.get("run_order", ())),
        "measurement_tool": str(manifest.get("measurement_tool", "") or ""),
        "maintenance_by_system": maintenance_by_system,
        "internal_maintenance_reference": internal_reference,
        "event_memory_maintenance_coupling_reference": event_memory_maintenance_coupling_reference,
        "bundle_contribution_warning": bundle_contribution_warning,
        "maintenance_alignment": maintenance_alignment,
        "workload_results": {key: dict(value) for key, value in workload_results.items()},
        "resume_append_command_template": resume_command,
        "record_measurement_commands": record_commands,
        "next_step": (
            "Fill the generated meter template with exact SARA and ANN joules, then rerun the pair command with --meter-reading-path or both joule values, or execute the per-system record-energy-measurement commands."
            if pending_measurement
            else "Rows already recorded. Refresh energy_measurement_readiness for updated pair evidence."
        )
        + (f" {bundle_contribution_warning}" if bundle_contribution_warning else ""),
    }


def format_pair_summary(report: Mapping[str, Any]) -> str:
    maintenance = (
        report.get("maintenance_by_system", {})
        if isinstance(report.get("maintenance_by_system", {}), Mapping)
        else {}
    )
    lines = [
        "# SARA Physical Energy Pair",
        f"- passed: {bool(report.get('passed', False))}",
        f"- dry_run: {bool(report.get('dry_run', False))}",
        f"- measurement_pending: {bool(report.get('measurement_pending', False))}",
        f"- pair_id: {report.get('pair_id', '')}",
        f"- replicate_index: {int(report.get('replicate_index', 0) or 0)}",
        f"- task: {report.get('task', '')}",
        f"- run_order: {', '.join(str(item) for item in report.get('run_order', []))}",
        f"- measurement_tool: {report.get('measurement_tool', '')}",
        f"- recorded_row_count: {int(report.get('recorded_row_count', 0) or 0)}",
        f"- manifest_path: {report.get('manifest_path', '')}",
        f"- trace_path: {report.get('trace_path', '')}",
        f"- measurement_path: {report.get('measurement_path', '')}",
        f"- meter_reading_path: {report.get('meter_reading_path', '')}",
        f"- meter_template_path: {report.get('meter_template_path', '')}",
        "Maintenance:",
    ]
    if maintenance:
        for system in ("sara", "ann"):
            metrics = maintenance.get(system, {})
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                "- "
                f"{system}: "
                f"selected={int(metrics.get('maintenance_selected_count', 0) or 0)}, "
                f"phases={int(metrics.get('maintenance_phase_count', 0) or 0)}, "
                f"refresh={int(metrics.get('maintenance_refresh_count', 0) or 0)}, "
                f"event_cost={float(metrics.get('maintenance_event_cost', 0.0) or 0.0):.3f}, "
                f"idle_ok={int(metrics.get('maintenance_idle_self_state_ok_count', 0) or 0)}, "
                f"spontaneous={int(metrics.get('maintenance_spontaneous_event_count', 0) or 0)}, "
                f"predicted={int(metrics.get('maintenance_predicted_event_count', 0) or 0)}"
            )
    else:
        lines.append("- none")
    internal_reference = (
        report.get("internal_maintenance_reference", {})
        if isinstance(report.get("internal_maintenance_reference", {}), Mapping)
        else {}
    )
    event_memory_maintenance_coupling_reference = (
        report.get("event_memory_maintenance_coupling_reference", {})
        if isinstance(report.get("event_memory_maintenance_coupling_reference", {}), Mapping)
        else {}
    )
    bundle_contribution_warning = str(report.get("bundle_contribution_warning", "") or "")
    lines.append("Internal Maintenance Reference:")
    if internal_reference:
        lines.append(
            "- "
            f"selected={int(internal_reference.get('maintenance_selected_count', 0) or 0)}, "
            f"refresh={int(internal_reference.get('maintenance_refresh_count', 0) or 0)}, "
            f"idle_ok={int(internal_reference.get('maintenance_idle_self_state_ok_count', 0) or 0)}, "
            f"spontaneous={int(internal_reference.get('maintenance_spontaneous_event_count', 0) or 0)}, "
            f"predicted={int(internal_reference.get('maintenance_predicted_event_count', 0) or 0)}, "
            f"event_cost={float(internal_reference.get('maintenance_event_cost', 0.0) or 0.0):.3f}, "
            f"event_cost_per_selected={float(internal_reference.get('maintenance_event_cost_per_selected', 0.0) or 0.0):.3f}, "
            f"event_cost_per_refresh={float(internal_reference.get('maintenance_event_cost_per_refresh', 0.0) or 0.0):.3f}"
        )
    else:
        lines.append("- none")
    lines.append("Event Memory Maintenance Coupling Reference:")
    if event_memory_maintenance_coupling_reference:
        lines.append(
            "- "
            f"available={bool(event_memory_maintenance_coupling_reference.get('available', False))}, "
            f"passed={bool(event_memory_maintenance_coupling_reference.get('passed', False))}, "
            f"best_profile={event_memory_maintenance_coupling_reference.get('best_profile_id', '')}, "
            f"best_efficiency={float(event_memory_maintenance_coupling_reference.get('best_profile_compression_efficiency_per_maintenance', 0.0) or 0.0):.3f}, "
            f"best_bundle_contribution={float(event_memory_maintenance_coupling_reference.get('best_profile_multimodal_bundle_compression_contribution', 0.0) or 0.0):.3f}, "
            f"best_continuity={float(event_memory_maintenance_coupling_reference.get('best_profile_self_state_continuity', 0.0) or 0.0):.3f}, "
            f"correlation={float(event_memory_maintenance_coupling_reference.get('compression_to_maintenance_correlation', 0.0) or 0.0):.3f}"
        )
    if bundle_contribution_warning:
        lines.append(f"Bundle Contribution Warning: {bundle_contribution_warning}")
    else:
        lines.append("- none")
    alignment = (
        report.get("maintenance_alignment", {})
        if isinstance(report.get("maintenance_alignment", {}), Mapping)
        else {}
    )
    lines.append("Maintenance Alignment:")
    if alignment:
        for system, metrics in alignment.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                "- "
                f"{system}: "
                f"event_cost_per_selected_delta={float(metrics.get('event_cost_per_selected_delta', 0.0) or 0.0):.3f}, "
                f"event_cost_per_selected_ratio={float(metrics.get('event_cost_per_selected_ratio', 0.0) or 0.0):.3f}, "
                f"event_cost_per_refresh_delta={float(metrics.get('event_cost_per_refresh_delta', 0.0) or 0.0):.3f}, "
                f"selected_delta={int(metrics.get('selected_count_delta', 0) or 0)}, "
                f"refresh_delta={int(metrics.get('refresh_count_delta', 0) or 0)}, "
                f"idle_ok_delta={int(metrics.get('idle_self_state_ok_delta', 0) or 0)}, "
                f"spontaneous_delta={int(metrics.get('spontaneous_event_delta', 0) or 0)}, "
                f"predicted_delta={int(metrics.get('predicted_event_delta', 0) or 0)}"
            )
    else:
        lines.append("- none")
    resume_command = str(report.get("resume_append_command_template", "") or "")
    if resume_command:
        lines.append("Resume Append Command:")
        lines.append(f"- {resume_command}")
    record_commands = (
        report.get("record_measurement_commands", [])
        if isinstance(report.get("record_measurement_commands", []), list)
        else []
    )
    lines.append("Record Commands:")
    if record_commands:
        for item in record_commands:
            if not isinstance(item, Mapping):
                continue
            lines.append(f"- {item.get('system', '')}: {item.get('command', '')}")
    else:
        lines.append("- none")
    lines.append(f"Next Step: {report.get('next_step', '')}")
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--auto-system-energy-estimate",
        action="store_true",
        help="Estimate workload energy from macOS ioreg telemetry; this is not physical-meter evidence.",
    )
    parser.add_argument("--measurement-path", default=DEFAULT_MEASUREMENT_PATH)
    parser.add_argument(
        "--meter-reading-path",
        default="",
        help="Optional JSON file with measured SARA/ANN joules or average watts and duration.",
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--meter-template-path", default=DEFAULT_METER_TEMPLATE_PATH)
    parser.add_argument(
        "--internal-maintenance-report-path",
        default=DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH,
        help="Optional internal maintenance efficiency benchmark report for pair-alignment summaries.",
    )
    parser.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH,
        help="Optional Event Memory compression-versus-maintenance coupling report for pair summaries.",
    )
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
    traces = execute_pair(
        manifest,
        trace_path=args.trace_path,
        dry_run=args.dry_run,
        auto_system_energy_estimate=args.auto_system_energy_estimate,
    )
    meter_template = build_meter_reading_template(manifest, traces)
    meter_template_path = ensure_parent_directory(args.meter_template_path)
    with open(meter_template_path, "w", encoding="utf-8") as handle:
        json.dump(meter_template, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    recorded_rows: List[Dict[str, Any]] = []
    sara_joules = float(args.sara_joules)
    ann_joules = float(args.ann_joules)
    if not args.dry_run and args.meter_reading_path:
        meter_joules = load_meter_joules(args.meter_reading_path, manifest=manifest)
        sara_joules = meter_joules["sara"]
        ann_joules = meter_joules["ann"]
    measurement_quality = "physical_meter"
    physical_evidence = True
    measurement_source = "physical_energy_pair_runner"
    if (
        not args.dry_run
        and args.auto_system_energy_estimate
        and sara_joules <= 0.0
        and ann_joules <= 0.0
    ):
        estimates = {
            str(trace.get("system", "")): trace.get("system_energy_estimate", {})
            for trace in traces
            if isinstance(trace, Mapping)
        }
        sara_estimate = estimates.get("sara", {})
        ann_estimate = estimates.get("ann", {})
        if (
            isinstance(sara_estimate, Mapping)
            and isinstance(ann_estimate, Mapping)
            and float(sara_estimate.get("joules", 0.0) or 0.0) > 0.0
            and float(ann_estimate.get("joules", 0.0) or 0.0) > 0.0
        ):
            sara_joules = float(sara_estimate["joules"])
            ann_joules = float(ann_estimate["joules"])
            measurement_quality = "system_estimate"
            physical_evidence = False
            measurement_source = "macos_ioreg_system_power"
    if not args.dry_run and sara_joules > 0.0 and ann_joules > 0.0:
        recorded_rows = append_measured_rows(
            manifest,
            traces,
            sara_joules=sara_joules,
            ann_joules=ann_joules,
            measurement_path=args.measurement_path,
            source=measurement_source,
            measurement_quality=measurement_quality,
            physical_evidence=physical_evidence,
        )
    report = build_pair_report(
        manifest,
        traces,
        dry_run=args.dry_run,
        measurement_path=args.measurement_path,
        meter_reading_path=args.meter_reading_path,
        meter_template_path=meter_template_path,
        recorded_rows=recorded_rows,
        manifest_path=manifest_path,
        trace_path=args.trace_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        internal_maintenance_report=_load_optional_json(args.internal_maintenance_report_path),
        event_memory_maintenance_coupling_report=_load_optional_json(
            args.event_memory_maintenance_coupling_report_path
        ),
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_pair_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
