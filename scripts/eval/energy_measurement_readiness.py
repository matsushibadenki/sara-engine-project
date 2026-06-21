#!/usr/bin/env python3
"""Validate real energy-measurement readiness and optional joule evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path  # noqa: E402


DEFAULT_MEASUREMENT_PATH = raw_data_path("energy_measurements.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "energy_measurement_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "energy_measurement_readiness_summary.txt")
DEFAULT_SESSION_PLAN_PATH = workspace_path("evaluation", "energy_measurement_session_plan.json")
DEFAULT_SESSION_PLAN_SUMMARY_PATH = workspace_path("evaluation", "energy_measurement_session_plan.txt")
REQUIRED_FIELDS = {
    "run_id",
    "system",
    "task",
    "success_count",
    "joules",
}
FAIRNESS_FIELDS = {
    "protocol_version",
    "pair_id",
    "replicate_index",
    "environment_fingerprint",
    "task_fixture_hash",
    "success_criterion_id",
    "measurement_boundary",
    "measurement_tool",
    "cpu_model",
    "thread_count",
    "process_affinity",
    "power_mode",
    "warmup_count",
    "measured_repetitions",
    "trial_count",
    "run_order",
}
PAIR_MATCH_FIELDS = (
    "protocol_version",
    "environment_fingerprint",
    "task_fixture_hash",
    "success_criterion_id",
    "measurement_boundary",
    "measurement_tool",
    "cpu_model",
    "thread_count",
    "process_affinity",
    "power_mode",
    "warmup_count",
    "measured_repetitions",
    "trial_count",
)
MEASUREMENT_PROTOCOL_VERSION = "sara-energy-fair-comparison-v2"
CANONICAL_MEASUREMENT_TASKS = (
    "real_data_external_validity",
    "energy_efficiency_benchmark",
)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def default_environment_fingerprint(
    *,
    cpu_model: str,
    thread_count: int,
    process_affinity: str,
    power_mode: str,
) -> str:
    payload = {
        "cpu_model": str(cpu_model),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "process_affinity": str(process_affinity),
        "power_mode": str(power_mode),
        "thread_count": int(thread_count),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_measurements(path: str | None) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    rows.append(payload)
        else:
            payload = json.load(handle)
            if isinstance(payload, list):
                rows.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                measurements = payload.get("measurements", [])
                if isinstance(measurements, list):
                    rows.extend(item for item in measurements if isinstance(item, dict))
    return rows


def build_measurement_row(
    *,
    run_id: str,
    system: str,
    task: str,
    success_count: int,
    joules: float,
    source: str = "manual",
    duration_seconds: float | None = None,
    average_watts: float | None = None,
    notes: str = "",
    protocol_version: str = MEASUREMENT_PROTOCOL_VERSION,
    pair_id: str = "",
    replicate_index: int = 1,
    environment_fingerprint: str = "",
    task_fixture_hash: str = "",
    success_criterion_id: str = "",
    measurement_boundary: str = "",
    measurement_tool: str = "",
    cpu_model: str = "",
    thread_count: int = 1,
    process_affinity: str = "",
    power_mode: str = "",
    warmup_count: int = 0,
    measured_repetitions: int = 1,
    trial_count: int | None = None,
    run_order: int = 1,
) -> Dict[str, Any]:
    resolved_joules = float(joules)
    if resolved_joules <= 0.0 and average_watts is not None and duration_seconds is not None:
        resolved_joules = float(average_watts) * float(duration_seconds)
    row: Dict[str, Any] = {
        "run_id": str(run_id),
        "system": str(system).lower(),
        "task": str(task),
        "success_count": int(success_count),
        "joules": float(resolved_joules),
        "source": str(source),
        "protocol_version": str(protocol_version),
        "pair_id": str(pair_id),
        "replicate_index": int(replicate_index),
        "environment_fingerprint": str(environment_fingerprint),
        "task_fixture_hash": str(task_fixture_hash),
        "success_criterion_id": str(success_criterion_id),
        "measurement_boundary": str(measurement_boundary),
        "measurement_tool": str(measurement_tool),
        "cpu_model": str(cpu_model),
        "thread_count": int(thread_count),
        "process_affinity": str(process_affinity),
        "power_mode": str(power_mode),
        "warmup_count": int(warmup_count),
        "measured_repetitions": int(measured_repetitions),
        "trial_count": int(trial_count if trial_count is not None else success_count),
        "run_order": int(run_order),
    }
    if not row["environment_fingerprint"] and cpu_model and process_affinity and power_mode:
        row["environment_fingerprint"] = default_environment_fingerprint(
            cpu_model=cpu_model,
            thread_count=thread_count,
            process_affinity=process_affinity,
            power_mode=power_mode,
        )
    if duration_seconds is not None:
        row["duration_seconds"] = float(duration_seconds)
    if average_watts is not None:
        row["average_watts"] = float(average_watts)
        row["joules_derivation"] = "average_watts_x_duration_seconds"
    if notes:
        row["notes"] = str(notes)
    errors = _validate_measurement(row)
    if errors:
        raise ValueError("Invalid energy measurement row: " + ", ".join(errors))
    return row


def append_measurement(path: str, row: Mapping[str, Any]) -> str:
    errors = _validate_measurement(row)
    if errors:
        raise ValueError("Invalid energy measurement row: " + ", ".join(errors))
    resolved = ensure_parent_directory(path)
    with open(resolved, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def _validate_measurement(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    missing_fairness = sorted(
        field
        for field in FAIRNESS_FIELDS
        if field not in row or row.get(field) in {"", None}
    )
    if missing_fairness:
        errors.append("missing_fairness_fields:" + ",".join(missing_fairness))
    if str(row.get("system", "")).lower() not in {"sara", "ann"}:
        errors.append("system_must_be_sara_or_ann")
    if _safe_int(row.get("success_count")) <= 0:
        errors.append("success_count_must_be_positive")
    if _safe_int(row.get("trial_count")) < _safe_int(row.get("success_count")):
        errors.append("trial_count_must_cover_success_count")
    if _safe_int(row.get("replicate_index")) <= 0:
        errors.append("replicate_index_must_be_positive")
    if _safe_int(row.get("thread_count")) <= 0:
        errors.append("thread_count_must_be_positive")
    if _safe_int(row.get("measured_repetitions")) <= 0:
        errors.append("measured_repetitions_must_be_positive")
    if _safe_int(row.get("warmup_count")) < 0:
        errors.append("warmup_count_must_be_nonnegative")
    if _safe_int(row.get("run_order")) <= 0:
        errors.append("run_order_must_be_positive")
    if _safe_float(row.get("joules")) <= 0.0:
        errors.append("joules_must_be_positive")
    if "average_watts" in row and _safe_float(row.get("average_watts")) <= 0.0:
        errors.append("average_watts_must_be_positive")
    if "duration_seconds" in row and _safe_float(row.get("duration_seconds")) <= 0.0:
        errors.append("duration_seconds_must_be_positive")
    return errors


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("task", "")),
        str(row.get("pair_id", "")),
        _safe_int(row.get("replicate_index")),
    )


def _pair_fairness_errors(
    sara: Mapping[str, Any],
    ann: Mapping[str, Any],
) -> List[str]:
    errors = [
        f"mismatch:{field}"
        for field in PAIR_MATCH_FIELDS
        if sara.get(field) != ann.get(field)
    ]
    if _safe_int(sara.get("run_order")) == _safe_int(ann.get("run_order")):
        errors.append("run_order_must_differ_within_pair")
    return errors


def _aggregate_measurements(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_success_rate_delta: float,
) -> Dict[str, Any]:
    rows = list(rows)
    by_system: Dict[str, Dict[str, float]] = {
        "sara": {"success_count": 0.0, "joules": 0.0, "row_count": 0.0},
        "ann": {"success_count": 0.0, "joules": 0.0, "row_count": 0.0},
    }
    by_task: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        system = str(row.get("system", "")).lower()
        if system not in by_system:
            continue
        task = str(row.get("task", "") or "").strip() or "unspecified"
        by_system[system]["success_count"] += max(_safe_int(row.get("success_count")), 0)
        by_system[system]["joules"] += max(_safe_float(row.get("joules")), 0.0)
        by_system[system]["row_count"] += 1.0
        task_bucket = by_task.setdefault(
            task,
            {
                "sara": {"success_count": 0.0, "joules": 0.0, "row_count": 0.0},
                "ann": {"success_count": 0.0, "joules": 0.0, "row_count": 0.0},
            },
        )
        task_bucket[system]["success_count"] += max(_safe_int(row.get("success_count")), 0)
        task_bucket[system]["joules"] += max(_safe_float(row.get("joules")), 0.0)
        task_bucket[system]["row_count"] += 1.0

    sara = by_system["sara"]
    ann = by_system["ann"]
    sara_joule_per_success = sara["joules"] / max(sara["success_count"], 1e-9)
    ann_joule_per_success = ann["joules"] / max(ann["success_count"], 1e-9)
    paired_task_ratios: Dict[str, float] = {}
    unpaired_tasks: List[str] = []
    for task, task_bucket in sorted(by_task.items()):
        task_sara = task_bucket["sara"]
        task_ann = task_bucket["ann"]
        task_has_pair = task_sara["row_count"] > 0 and task_ann["row_count"] > 0
        if not task_has_pair:
            unpaired_tasks.append(task)
            continue
    pair_buckets: Dict[tuple[str, str, int], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        system = str(row.get("system", "")).lower()
        if system in {"sara", "ann"}:
            pair_buckets.setdefault(_pair_key(row), {})[system] = row
    valid_pairs: List[Dict[str, Any]] = []
    pair_errors: List[Dict[str, Any]] = []
    for key, systems in sorted(pair_buckets.items()):
        if set(systems) != {"sara", "ann"}:
            pair_errors.append({"pair_key": list(key), "errors": ["missing_system_pair"]})
            continue
        sara_row = systems["sara"]
        ann_row = systems["ann"]
        errors = _pair_fairness_errors(sara_row, ann_row)
        sara_rate = _safe_int(sara_row.get("success_count")) / max(
            _safe_int(sara_row.get("trial_count")), 1
        )
        ann_rate = _safe_int(ann_row.get("success_count")) / max(
            _safe_int(ann_row.get("trial_count")), 1
        )
        quality_delta = abs(sara_rate - ann_rate)
        if quality_delta > max_success_rate_delta:
            errors.append("success_rate_parity_failed")
        if errors:
            pair_errors.append({"pair_key": list(key), "errors": errors})
            continue
        sara_jps = _safe_float(sara_row.get("joules")) / max(
            _safe_int(sara_row.get("success_count")), 1
        )
        ann_jps = _safe_float(ann_row.get("joules")) / max(
            _safe_int(ann_row.get("success_count")), 1
        )
        valid_pairs.append(
            {
                "task": key[0],
                "pair_id": key[1],
                "replicate_index": key[2],
                "sara_joule_per_success": sara_jps,
                "ann_joule_per_success": ann_jps,
                "ann_to_sara_ratio": ann_jps / max(sara_jps, 1e-9),
                "sara_success_rate": sara_rate,
                "ann_success_rate": ann_rate,
                "success_rate_delta": quality_delta,
                "run_order": {
                    "sara": _safe_int(sara_row.get("run_order")),
                    "ann": _safe_int(ann_row.get("run_order")),
                },
            }
        )
    task_pair_metrics: Dict[str, Dict[str, Any]] = {}
    for task in sorted({pair["task"] for pair in valid_pairs}):
        task_pairs = [pair for pair in valid_pairs if pair["task"] == task]
        sara_values = [float(pair["sara_joule_per_success"]) for pair in task_pairs]
        ann_values = [float(pair["ann_joule_per_success"]) for pair in task_pairs]
        ratio_values = [float(pair["ann_to_sara_ratio"]) for pair in task_pairs]
        task_pair_metrics[task] = {
            "valid_pair_count": len(task_pairs),
            "sara_median_joule_per_success": _median(sara_values),
            "sara_joule_per_success_mad": _mad(sara_values),
            "ann_median_joule_per_success": _median(ann_values),
            "ann_joule_per_success_mad": _mad(ann_values),
            "median_ann_to_sara_ratio": _median(ratio_values),
            "ratio_mad": _mad(ratio_values),
            "max_success_rate_delta": max(
                (float(pair["success_rate_delta"]) for pair in task_pairs),
                default=0.0,
            ),
        }
        paired_task_ratios[task] = task_pair_metrics[task][
            "median_ann_to_sara_ratio"
        ]
    run_order_balance = {
        "sara_first": sum(
            1
            for pair in valid_pairs
            if pair["run_order"]["sara"] < pair["run_order"]["ann"]
        ),
        "ann_first": sum(
            1
            for pair in valid_pairs
            if pair["run_order"]["ann"] < pair["run_order"]["sara"]
        ),
    }
    return {
        "systems": by_system,
        "tasks": by_task,
        "sara_joule_per_success": float(sara_joule_per_success),
        "ann_joule_per_success": float(ann_joule_per_success),
        "ann_to_sara_joule_efficiency_ratio": float(
            ann_joule_per_success / max(sara_joule_per_success, 1e-9)
        ),
        "paired_task_count": len(paired_task_ratios),
        "unpaired_task_count": len(unpaired_tasks),
        "unpaired_tasks": unpaired_tasks,
        "paired_task_ann_to_sara_ratios": paired_task_ratios,
        "valid_pairs": valid_pairs,
        "valid_pair_count": len(valid_pairs),
        "invalid_pair_count": len(pair_errors),
        "pair_errors": pair_errors,
        "task_pair_statistics": task_pair_metrics,
        "run_order_balance": run_order_balance,
        "min_paired_task_ann_to_sara_ratio": min(paired_task_ratios.values())
        if paired_task_ratios
        else 0.0,
        "has_sara_measurements": sara["row_count"] > 0,
        "has_ann_measurements": ann["row_count"] > 0,
    }


def _record_command_template(*, task: str, system: str) -> str:
    return (
        "python scripts/sara_cli.py record-energy-measurement "
        f"--run-id <run-id> --system {system} --task {task} "
        "--success-count <count> --joules <J>"
    )


def _session_run_id_template(*, session_id: str, task: str, system: str) -> str:
    safe_session = str(session_id or "energy-session").strip().replace(" ", "-")
    safe_task = str(task or "task").strip().replace(" ", "-")
    safe_system = str(system or "system").strip().replace(" ", "-")
    return f"{safe_session}-{safe_task}-{safe_system}-<replicate>"


def _record_command_for_session(*, session_id: str, task: str, system: str) -> str:
    run_id = _session_run_id_template(session_id=session_id, task=task, system=system)
    return (
        "python scripts/sara_cli.py record-energy-measurement "
        f"--run-id {run_id} --system {system} --task {task} "
        "--success-count <count> --trial-count <trials> --joules <J> "
        "--source real_energy_session --pair-id <pair-id> "
        "--replicate-index <replicate> --environment-fingerprint <sha256> "
        "--task-fixture-hash <sha256> --success-criterion-id <criterion-id> "
        "--measurement-boundary <boundary-id> --measurement-tool <tool-id> "
        "--cpu-model <cpu> --thread-count <threads> --process-affinity <affinity> "
        "--power-mode <mode> --warmup-count <count> "
        "--measured-repetitions <count> --run-order <1-or-2>"
    )


def _build_measurement_session_plan(
    measurement_plan: Mapping[str, Any],
    *,
    measurement_path: str,
    min_ann_to_sara_ratio: float,
    session_id: str,
) -> Dict[str, Any]:
    pending_pairs = (
        measurement_plan.get("pending_pairs", [])
        if isinstance(measurement_plan.get("pending_pairs"), list)
        else []
    )
    weak_pairs = (
        measurement_plan.get("weak_pairs", [])
        if isinstance(measurement_plan.get("weak_pairs"), list)
        else []
    )
    planned_runs: List[Dict[str, Any]] = []

    for item in pending_pairs:
        if not isinstance(item, Mapping):
            continue
        task = str(item.get("task", "") or "").strip()
        system = str(item.get("missing_system", "") or "").strip().lower()
        if not task or system not in {"sara", "ann"}:
            continue
        planned_runs.append(
            {
                "category": "collect_missing_pair",
                "priority": str(item.get("priority", "high")),
                "task": task,
                "system": system,
                "run_id_template": _session_run_id_template(
                    session_id=session_id,
                    task=task,
                    system=system,
                ),
                "command_template": _record_command_for_session(
                    session_id=session_id,
                    task=task,
                    system=system,
                ),
            }
        )

    for item in weak_pairs:
        if not isinstance(item, Mapping):
            continue
        task = str(item.get("task", "") or "").strip()
        if not task:
            continue
        for system in ("sara", "ann"):
            planned_runs.append(
                {
                    "category": "repeat_weak_pair",
                    "priority": str(item.get("priority", "medium")),
                    "task": task,
                    "system": system,
                    "observed_ratio": _safe_float(item.get("ann_to_sara_joule_efficiency_ratio")),
                    "required_min": _safe_float(item.get("required_min")),
                    "run_id_template": _session_run_id_template(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                    "command_template": _record_command_for_session(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                }
            )

    return {
        "schema": "sara-energy-measurement-session-plan-v2",
        "session_id": str(session_id or "energy-session"),
        "status": "ready_for_real_joule_claim" if not planned_runs else "pending_measurement",
        "measurement_path": str(measurement_path),
        "min_ann_to_sara_ratio": float(min_ann_to_sara_ratio),
        "planned_run_count": len(planned_runs),
        "planned_runs": planned_runs,
        "pairing_matrix": {
            "tasks": list(CANONICAL_MEASUREMENT_TASKS),
            "systems": ["sara", "ann"],
            "required_rows_per_task": 2,
            "required_paired_replicates_per_task": 3,
        },
        "fair_comparison_contract": {
            "protocol_version": MEASUREMENT_PROTOCOL_VERSION,
            "required_pair_match_fields": list(PAIR_MATCH_FIELDS),
            "quality_parity_metric": "absolute_success_rate_delta",
            "run_order_policy": "alternate_or_randomized_block",
            "aggregation": "per-task median joule_per_success with MAD",
        },
        "operator_notes": [
            "Use the same hardware power source and sampling method for both systems in a task pair.",
            "Replace <replicate>, <count>, and <J> with observed values; do not record proxy event costs as joules.",
            "If only average power is available, replace --joules with --average-watts and --duration-seconds.",
        ],
    }


def _build_measurement_plan(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    min_ann_to_sara_ratio: float,
) -> Dict[str, Any]:
    tasks = aggregate.get("tasks", {}) if isinstance(aggregate.get("tasks"), Mapping) else {}
    pending_pairs: List[Dict[str, Any]] = []
    if not rows:
        for task in CANONICAL_MEASUREMENT_TASKS:
            for system in ("sara", "ann"):
                pending_pairs.append(
                    {
                        "task": task,
                        "missing_system": system,
                        "priority": "high",
                        "command_template": _record_command_template(task=task, system=system),
                    }
                )
    else:
        for task, task_bucket in sorted(tasks.items()):
            if not isinstance(task_bucket, Mapping):
                continue
            missing_systems = []
            for system in ("sara", "ann"):
                bucket = task_bucket.get(system, {}) if isinstance(task_bucket.get(system), Mapping) else {}
                if _safe_float(bucket.get("row_count")) <= 0.0:
                    missing_systems.append(system)
            for system in missing_systems:
                pending_pairs.append(
                    {
                        "task": str(task),
                        "missing_system": system,
                        "priority": "high",
                        "command_template": _record_command_template(task=str(task), system=system),
                    }
                )

    paired_ratios = (
        aggregate.get("paired_task_ann_to_sara_ratios", {})
        if isinstance(aggregate.get("paired_task_ann_to_sara_ratios"), Mapping)
        else {}
    )
    weak_pairs = [
        {
            "task": str(task),
            "ann_to_sara_joule_efficiency_ratio": float(ratio),
            "required_min": float(min_ann_to_sara_ratio),
            "priority": "medium",
            "next_action": "Repeat paired measurement or inspect the SARA sparse-event trace for this task.",
        }
        for task, ratio in sorted(paired_ratios.items())
        if _safe_float(ratio) < float(min_ann_to_sara_ratio)
    ]
    ready_for_real_claim = bool(not pending_pairs and not weak_pairs and int(aggregate.get("paired_task_count", 0) or 0) > 0)
    return {
        "schema": "sara-energy-measurement-plan-v2",
        "ready_for_real_joule_claim": ready_for_real_claim,
        "required_task_pair_count": len(CANONICAL_MEASUREMENT_TASKS),
        "observed_paired_task_count": int(aggregate.get("paired_task_count", 0) or 0),
        "pending_pair_count": len(pending_pairs),
        "weak_pair_count": len(weak_pairs),
        "pending_pairs": pending_pairs,
        "weak_pairs": weak_pairs,
        "recommended_tasks": list(CANONICAL_MEASUREMENT_TASKS),
        "operator_notes": [
            "Use the same task label for SARA and ANN rows.",
            "Record success_count with the same scoring rule for both systems.",
            "Use direct joules or average_watts plus duration_seconds; do not mix unrelated workloads under one task.",
        ],
    }


def build_energy_measurement_readiness_report(
    measurements: Iterable[Mapping[str, Any]],
    *,
    min_ann_to_sara_ratio: float = 1.0,
    measurement_path: str = "data/raw/energy_measurements.jsonl",
    session_id: str = "ann-efficiency-real-joule",
    max_success_rate_delta: float = 0.0,
    min_paired_replicates_per_task: int = 3,
) -> Dict[str, Any]:
    rows = [dict(row) for row in measurements]
    row_errors = [
        {"index": index, "errors": _validate_measurement(row)}
        for index, row in enumerate(rows)
    ]
    row_errors = [item for item in row_errors if item["errors"]]
    valid_rows = [
        row for index, row in enumerate(rows)
        if not any(item["index"] == index for item in row_errors)
    ]
    aggregate = _aggregate_measurements(
        valid_rows,
        max_success_rate_delta=max_success_rate_delta,
    )
    has_real_measurements = bool(
        aggregate["has_sara_measurements"] and aggregate["has_ann_measurements"]
    )
    ratio = float(aggregate["ann_to_sara_joule_efficiency_ratio"])
    paired_task_count = int(aggregate.get("paired_task_count", 0) or 0)
    unpaired_task_count = int(aggregate.get("unpaired_task_count", 0) or 0)
    min_paired_task_ratio = float(aggregate.get("min_paired_task_ann_to_sara_ratio", 0.0) or 0.0)
    valid_pair_count = int(aggregate.get("valid_pair_count", 0) or 0)
    invalid_pair_count = int(aggregate.get("invalid_pair_count", 0) or 0)
    task_pair_statistics = aggregate.get("task_pair_statistics", {})
    replicate_floor_passed = bool(
        task_pair_statistics
        and all(
            _safe_int(metrics.get("valid_pair_count"))
            >= int(min_paired_replicates_per_task)
            for metrics in task_pair_statistics.values()
            if isinstance(metrics, Mapping)
        )
    )
    order_balance = aggregate.get("run_order_balance", {})
    sara_first = _safe_int(order_balance.get("sara_first")) if isinstance(order_balance, Mapping) else 0
    ann_first = _safe_int(order_balance.get("ann_first")) if isinstance(order_balance, Mapping) else 0
    run_order_balanced = bool(
        valid_pair_count <= 1 or abs(sara_first - ann_first) <= 1
    )
    measurement_plan = _build_measurement_plan(
        valid_rows,
        aggregate,
        min_ann_to_sara_ratio=min_ann_to_sara_ratio,
    )
    measurement_session_plan = _build_measurement_session_plan(
        measurement_plan,
        measurement_path=measurement_path,
        min_ann_to_sara_ratio=min_ann_to_sara_ratio,
        session_id=session_id,
    )
    checks = {
        "schema_ready": True,
        "rows_valid": len(row_errors) == 0,
        "sara_measurements_present": bool(aggregate["has_sara_measurements"]),
        "ann_measurements_present": bool(aggregate["has_ann_measurements"]),
        "joule_efficiency_ratio_passed": has_real_measurements and ratio >= min_ann_to_sara_ratio,
        "paired_task_measurements_present": has_real_measurements and paired_task_count > 0,
        "paired_task_rows_balanced": (not rows) or (has_real_measurements and unpaired_task_count == 0),
        "paired_task_efficiency_ratio_passed": (
            has_real_measurements
            and paired_task_count > 0
            and min_paired_task_ratio >= min_ann_to_sara_ratio
        ),
        "fair_pair_contract_passed": (not rows)
        or (valid_pair_count > 0 and invalid_pair_count == 0),
        "quality_parity_passed": (not rows)
        or (
            valid_pair_count > 0
            and all(
                _safe_float(pair.get("success_rate_delta"))
                <= max_success_rate_delta
                for pair in aggregate.get("valid_pairs", [])
            )
        ),
        "paired_replicate_floor_passed": (not rows) or replicate_floor_passed,
        "run_order_balance_passed": (not rows) or run_order_balanced,
    }
    protocol_ready = bool(checks["schema_ready"] and checks["rows_valid"])
    return {
        "schema": "sara-energy-measurement-readiness-v2",
        "passed": bool(protocol_ready and (not rows or all(checks.values()))),
        "status": "real_joule_evidence_passed"
        if all(checks.values())
        else ("protocol_ready_pending_measurements" if protocol_ready and not rows else "needs_measurement_repair"),
        "measurement_count": len(rows),
        "valid_measurement_count": len(valid_rows),
        "real_joule_measurements_present": has_real_measurements,
        "min_ann_to_sara_ratio": float(min_ann_to_sara_ratio),
        "max_success_rate_delta": float(max_success_rate_delta),
        "min_paired_replicates_per_task": int(min_paired_replicates_per_task),
        "checks": checks,
        "row_errors": row_errors,
        "metrics": aggregate,
        "measurement_plan": measurement_plan,
        "measurement_session_plan": measurement_session_plan,
        "measurement_protocol": {
            "required_fields": sorted(REQUIRED_FIELDS | FAIRNESS_FIELDS),
            "systems": ["sara", "ann"],
            "units": {"average_watts": "W", "duration_seconds": "s", "joules": "J", "success_count": "count"},
            "recommended_path": str(measurement_path),
            "accepted_energy_inputs": [
                "direct_joules",
                "average_watts_x_duration_seconds",
            ],
            "pairing_rule": "Rows must include matching SARA and ANN measurements for each task before real joule evidence is accepted.",
            "fair_pair_rule": "pair_id and replicate_index identify one SARA/ANN pair; all configured environment, task, criterion, boundary, and tool fields must match.",
            "aggregation_rule": "Task claims use the median paired joule_per_success ratio and report median absolute deviation.",
            "quality_rule": "Energy advantage is credited only when paired success-rate delta is within the configured tolerance.",
        },
    }


def format_energy_measurement_summary(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), Mapping) else {}
    plan = report.get("measurement_plan", {}) if isinstance(report.get("measurement_plan"), Mapping) else {}
    session_plan = (
        report.get("measurement_session_plan", {})
        if isinstance(report.get("measurement_session_plan"), Mapping)
        else {}
    )
    lines = [
        "# SARA Energy Measurement Readiness",
        f"- passed: {bool(report.get('passed', False))}",
        f"- status: {report.get('status', '')}",
        f"- measurement_count: {int(report.get('measurement_count', 0) or 0)}",
        f"- real_joule_measurements_present: {bool(report.get('real_joule_measurements_present', False))}",
        f"- sara_joule_per_success: {_safe_float(metrics.get('sara_joule_per_success')):.6f}",
        f"- ann_joule_per_success: {_safe_float(metrics.get('ann_joule_per_success')):.6f}",
        f"- ann_to_sara_joule_efficiency_ratio: {_safe_float(metrics.get('ann_to_sara_joule_efficiency_ratio')):.3f}",
        f"- paired_task_count: {_safe_int(metrics.get('paired_task_count'))}",
        f"- min_paired_task_ann_to_sara_ratio: {_safe_float(metrics.get('min_paired_task_ann_to_sara_ratio')):.3f}",
        f"- unpaired_task_count: {_safe_int(metrics.get('unpaired_task_count'))}",
        f"- valid_pair_count: {_safe_int(metrics.get('valid_pair_count'))}",
        f"- invalid_pair_count: {_safe_int(metrics.get('invalid_pair_count'))}",
        f"- measurement_pending_pair_count: {_safe_int(plan.get('pending_pair_count'))}",
        f"- measurement_weak_pair_count: {_safe_int(plan.get('weak_pair_count'))}",
        f"- measurement_session_planned_run_count: {_safe_int(session_plan.get('planned_run_count'))}",
        "Checks:",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if bool(checks[name]) else 'FAIL'}")
    pending_pairs = plan.get("pending_pairs", []) if isinstance(plan.get("pending_pairs"), list) else []
    weak_pairs = plan.get("weak_pairs", []) if isinstance(plan.get("weak_pairs"), list) else []
    lines.append("Measurement Plan:")
    if pending_pairs:
        lines.append("- pending_pairs:")
        for item in pending_pairs[:8]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - "
                f"task={item.get('task', '')}, "
                f"missing_system={item.get('missing_system', '')}, "
                f"command={item.get('command_template', '')}"
            )
    else:
        lines.append("- pending_pairs: none")
    if weak_pairs:
        lines.append("- weak_pairs:")
        for item in weak_pairs[:8]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - "
                f"task={item.get('task', '')}, "
                f"ratio={_safe_float(item.get('ann_to_sara_joule_efficiency_ratio')):.3f}, "
                f"required_min={_safe_float(item.get('required_min')):.3f}"
            )
    else:
        lines.append("- weak_pairs: none")
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    lines.append("Measurement Session Plan:")
    if planned_runs:
        for item in planned_runs[:8]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - "
                f"category={item.get('category', '')}, "
                f"task={item.get('task', '')}, "
                f"system={item.get('system', '')}, "
                f"run_id_template={item.get('run_id_template', '')}, "
                f"command={item.get('command_template', '')}"
            )
    else:
        lines.append("- planned_runs: none")
    return "\n".join(lines) + "\n"


def format_measurement_session_plan_summary(session_plan: Mapping[str, Any]) -> str:
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    pairing_matrix = (
        session_plan.get("pairing_matrix", {})
        if isinstance(session_plan.get("pairing_matrix"), Mapping)
        else {}
    )
    lines = [
        "# SARA Energy Measurement Session Plan",
        f"- schema: {session_plan.get('schema', '')}",
        f"- status: {session_plan.get('status', '')}",
        f"- session_id: {session_plan.get('session_id', '')}",
        f"- measurement_path: {session_plan.get('measurement_path', '')}",
        f"- min_ann_to_sara_ratio: {_safe_float(session_plan.get('min_ann_to_sara_ratio')):.3f}",
        f"- planned_run_count: {_safe_int(session_plan.get('planned_run_count'))}",
        f"- required_rows_per_task: {_safe_int(pairing_matrix.get('required_rows_per_task'))}",
        f"- required_paired_replicates_per_task: {_safe_int(pairing_matrix.get('required_paired_replicates_per_task'))}",
    ]
    systems = pairing_matrix.get("systems", []) if isinstance(pairing_matrix.get("systems"), list) else []
    tasks = pairing_matrix.get("tasks", []) if isinstance(pairing_matrix.get("tasks"), list) else []
    lines.append("- systems: " + ", ".join(str(item) for item in systems))
    lines.append("- tasks: " + ", ".join(str(item) for item in tasks))
    lines.append("Planned Runs:")
    if planned_runs:
        for item in planned_runs:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"priority={item.get('priority', '')}, "
                f"category={item.get('category', '')}, "
                f"task={item.get('task', '')}, "
                f"system={item.get('system', '')}, "
                f"run_id_template={item.get('run_id_template', '')}"
            )
            lines.append(f"  command: {item.get('command_template', '')}")
    else:
        lines.append("- none")
    notes = session_plan.get("operator_notes", []) if isinstance(session_plan.get("operator_notes"), list) else []
    lines.append("Operator Notes:")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real joule measurement readiness.")
    parser.add_argument("--measurement-path", default=DEFAULT_MEASUREMENT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--session-plan-path", default=DEFAULT_SESSION_PLAN_PATH)
    parser.add_argument("--session-plan-summary-path", default=DEFAULT_SESSION_PLAN_SUMMARY_PATH)
    parser.add_argument("--min-ann-to-sara-ratio", type=float, default=1.0)
    parser.add_argument("--session-id", default="ann-efficiency-real-joule")
    parser.add_argument("--append-measurement", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--system", choices=["sara", "ann"], default="sara")
    parser.add_argument("--task", default="")
    parser.add_argument("--success-count", type=int, default=0)
    parser.add_argument("--joules", type=float, default=0.0)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--average-watts", type=float, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--protocol-version", default=MEASUREMENT_PROTOCOL_VERSION)
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--replicate-index", type=int, default=1)
    parser.add_argument("--environment-fingerprint", default="")
    parser.add_argument("--task-fixture-hash", default="")
    parser.add_argument("--success-criterion-id", default="")
    parser.add_argument("--measurement-boundary", default="")
    parser.add_argument("--measurement-tool", default="")
    parser.add_argument("--cpu-model", default="")
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--process-affinity", default="")
    parser.add_argument("--power-mode", default="")
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--measured-repetitions", type=int, default=1)
    parser.add_argument("--trial-count", type=int, default=None)
    parser.add_argument("--run-order", type=int, default=1)
    parser.add_argument("--max-success-rate-delta", type=float, default=0.0)
    parser.add_argument("--min-paired-replicates-per-task", type=int, default=3)
    args = parser.parse_args(argv)

    if args.append_measurement:
        row = build_measurement_row(
            run_id=args.run_id,
            system=args.system,
            task=args.task,
            success_count=args.success_count,
            joules=args.joules,
            source=args.source,
            duration_seconds=args.duration_seconds,
            average_watts=args.average_watts,
            notes=args.notes,
            protocol_version=args.protocol_version,
            pair_id=args.pair_id,
            replicate_index=args.replicate_index,
            environment_fingerprint=args.environment_fingerprint,
            task_fixture_hash=args.task_fixture_hash,
            success_criterion_id=args.success_criterion_id,
            measurement_boundary=args.measurement_boundary,
            measurement_tool=args.measurement_tool,
            cpu_model=args.cpu_model,
            thread_count=args.thread_count,
            process_affinity=args.process_affinity,
            power_mode=args.power_mode,
            warmup_count=args.warmup_count,
            measured_repetitions=args.measured_repetitions,
            trial_count=args.trial_count,
            run_order=args.run_order,
        )
        append_measurement(args.measurement_path, row)

    measurements = load_measurements(args.measurement_path)
    report = build_energy_measurement_readiness_report(
        measurements,
        min_ann_to_sara_ratio=args.min_ann_to_sara_ratio,
        measurement_path=args.measurement_path,
        session_id=args.session_id,
        max_success_rate_delta=args.max_success_rate_delta,
        min_paired_replicates_per_task=args.min_paired_replicates_per_task,
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    session_plan_path = ensure_parent_directory(args.session_plan_path)
    session_plan_summary_path = ensure_parent_directory(args.session_plan_summary_path)
    session_plan = report.get("measurement_session_plan", {})
    if not isinstance(session_plan, Mapping):
        session_plan = {}
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_energy_measurement_summary(report))
    with open(session_plan_path, "w", encoding="utf-8") as handle:
        json.dump(session_plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(session_plan_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_measurement_session_plan_summary(session_plan))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
