#!/usr/bin/env python3
"""Validate real energy-measurement readiness and optional joule evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_SESSION_PROGRESS_PATH = workspace_path("evaluation", "physical_energy_session_progress.json")
DEFAULT_SESSION_PROGRESS_SUMMARY_PATH = workspace_path("evaluation", "physical_energy_session_progress.txt")
DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH = workspace_path(
    "evaluation", "internal_maintenance_efficiency_benchmark.json"
)
DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH = workspace_path(
    "evaluation", "event_memory_maintenance_coupling_benchmark.json"
)
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
MAINTENANCE_FIELDS = (
    "maintenance_selected_count",
    "maintenance_phase_count",
    "maintenance_refresh_count",
    "maintenance_event_cost",
)


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_optional_json(path: str) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _internal_maintenance_reference_summary(report: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = report if isinstance(report, Mapping) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), Mapping) else {}
    normalized = (
        payload.get("normalized_metrics", {})
        if isinstance(payload.get("normalized_metrics"), Mapping)
        else {}
    )
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    return {
        "available": bool(payload),
        "observed_only": bool(payload.get("observed_only", False)),
        "passed": bool(payload.get("passed", False)),
        "maintenance_selected_count": _safe_int(counts.get("maintenance_selected_count")),
        "maintenance_refresh_count": _safe_int(counts.get("maintenance_refresh_count")),
        "maintenance_idle_self_state_ok_count": _safe_int(
            counts.get("maintenance_idle_self_state_ok_count")
        ),
        "maintenance_event_cost": _safe_float(normalized.get("maintenance_event_cost")),
        "maintenance_event_cost_per_selected": _safe_float(
            normalized.get("maintenance_event_cost_per_selected")
        ),
        "maintenance_self_state_continuity_observed": _safe_float(
            metrics.get("maintenance_self_state_continuity_observed")
        ),
        "maintenance_event_cost_efficiency_observed": _safe_float(
            metrics.get("maintenance_event_cost_efficiency_observed")
        ),
    }


def _maintenance_alignment_summary(
    aggregate: Mapping[str, Any],
    internal_maintenance_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    if not internal_maintenance_reference or not bool(
        internal_maintenance_reference.get("available", False)
    ):
        return {"available": False}
    actual_selected = _safe_float(aggregate.get("sara_maintenance_event_cost_per_selected"))
    actual_refresh = _safe_float(aggregate.get("sara_maintenance_event_cost_per_refresh"))
    reference_selected = _safe_float(
        internal_maintenance_reference.get("maintenance_event_cost_per_selected")
    )
    reference_refresh = _safe_float(
        internal_maintenance_reference.get("maintenance_event_cost_per_refresh")
    )
    return {
        "available": _safe_int(aggregate.get("valid_pair_count")) > 0 and actual_selected > 0.0,
        "valid_pair_count": _safe_int(aggregate.get("valid_pair_count")),
        "sara_physical_maintenance_event_cost_per_selected": actual_selected,
        "reference_maintenance_event_cost_per_selected": reference_selected,
        "maintenance_event_cost_per_selected_delta": actual_selected - reference_selected,
        "maintenance_event_cost_per_selected_ratio": (
            actual_selected / reference_selected if reference_selected > 0.0 else 0.0
        ),
        "sara_physical_maintenance_event_cost_per_refresh": actual_refresh,
        "reference_maintenance_event_cost_per_refresh": reference_refresh,
        "maintenance_event_cost_per_refresh_delta": actual_refresh - reference_refresh,
        "maintenance_event_cost_per_refresh_ratio": (
            actual_refresh / reference_refresh if reference_refresh > 0.0 else 0.0
        ),
    }


def _event_memory_maintenance_coupling_reference_summary(
    report: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    payload = report if isinstance(report, Mapping) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    best_profile = (
        payload.get("best_profile", {})
        if isinstance(payload.get("best_profile"), Mapping)
        else {}
    )
    return {
        "available": bool(payload),
        "passed": bool(payload.get("passed", False)),
        "observed_only": bool(payload.get("observed_only", False)),
        "profile_count": _safe_int(payload.get("profile_count")),
        "best_profile_id": str(best_profile.get("profile_id", "") or ""),
        "compression_to_maintenance_correlation": _safe_float(
            metrics.get("compression_to_maintenance_correlation")
        ),
        "best_profile_compression_efficiency_per_maintenance": _safe_float(
            metrics.get("best_profile_compression_efficiency_per_maintenance")
        ),
        "best_profile_self_state_continuity": _safe_float(
            metrics.get("best_profile_self_state_continuity")
        ),
        "best_profile_episode_compression_ratio": _safe_float(
            metrics.get("best_profile_episode_compression_ratio")
        ),
        "best_profile_multimodal_bundle_compression_contribution": _safe_float(
            metrics.get("best_profile_multimodal_bundle_compression_contribution")
        ),
    }


def _bundle_contribution_warning(reference: Mapping[str, Any] | None) -> str:
    payload = reference if isinstance(reference, Mapping) else {}
    if not bool(payload.get("available", False)):
        return ""
    contribution = _safe_float(
        payload.get("best_profile_multimodal_bundle_compression_contribution")
    )
    if contribution >= 0.5:
        return ""
    return (
        "Bundle-backed compression contribution is weak; expand verified multimodal bundle fixtures "
        "before treating this compression win as a strong SARA-native advantage."
    )


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
    maintenance_selected_count: int | None = None,
    maintenance_phase_count: int | None = None,
    maintenance_refresh_count: int | None = None,
    maintenance_event_cost: float | None = None,
    measurement_quality: str = "physical_meter",
    physical_evidence: bool = True,
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
        "measurement_quality": str(measurement_quality),
        "physical_evidence": bool(physical_evidence),
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
    if maintenance_selected_count is not None:
        row["maintenance_selected_count"] = int(maintenance_selected_count)
    if maintenance_phase_count is not None:
        row["maintenance_phase_count"] = int(maintenance_phase_count)
    if maintenance_refresh_count is not None:
        row["maintenance_refresh_count"] = int(maintenance_refresh_count)
    if maintenance_event_cost is not None:
        row["maintenance_event_cost"] = float(maintenance_event_cost)
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
    if "maintenance_selected_count" in row and _safe_int(row.get("maintenance_selected_count")) < 0:
        errors.append("maintenance_selected_count_must_be_nonnegative")
    if "maintenance_phase_count" in row and _safe_int(row.get("maintenance_phase_count")) < 0:
        errors.append("maintenance_phase_count_must_be_nonnegative")
    if "maintenance_refresh_count" in row and _safe_int(row.get("maintenance_refresh_count")) < 0:
        errors.append("maintenance_refresh_count_must_be_nonnegative")
    if "maintenance_event_cost" in row and _safe_float(row.get("maintenance_event_cost")) < 0.0:
        errors.append("maintenance_event_cost_must_be_nonnegative")
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


def _classify_pair_fairness_errors(errors: Sequence[str]) -> Dict[str, Any]:
    mismatch_fields: List[str] = []
    run_order_conflict = False
    other_errors: List[str] = []
    for item in errors:
        token = str(item).strip()
        if token.startswith("mismatch:"):
            mismatch_fields.append(token.split(":", 1)[1].strip())
        elif token == "run_order_must_differ_within_pair":
            run_order_conflict = True
        elif token:
            other_errors.append(token)
    mismatch_fields = sorted({field for field in mismatch_fields if field})
    if mismatch_fields and run_order_conflict:
        category = "fairness_and_run_order_conflict"
        priority = "high"
        remediation = "Repair fairness-field mismatches and rerun the pair with alternating run order."
    elif mismatch_fields:
        category = "fairness_field_mismatch"
        priority = "high"
        remediation = "Repair fairness-field mismatches before repeating this physical pair."
    elif run_order_conflict:
        category = "run_order_conflict"
        priority = "medium"
        remediation = "Rerun this physical pair with alternating or explicitly differentiated run order."
    elif other_errors:
        category = "unclassified_fairness_error"
        priority = "medium"
        remediation = "Inspect pair fairness errors and rerun the physical pair after correcting the mismatch."
    else:
        category = "none"
        priority = "low"
        remediation = ""
    return {
        "category": category,
        "priority": priority,
        "mismatch_fields": mismatch_fields,
        "run_order_conflict": run_order_conflict,
        "other_errors": other_errors,
        "remediation": remediation,
    }


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
    sara_maintenance_event_cost_per_success = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "sara"
    ) / max(sara["success_count"], 1e-9)
    ann_maintenance_event_cost_per_success = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "ann"
    ) / max(ann["success_count"], 1e-9)
    sara_maintenance_event_cost_per_selected = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "sara"
    ) / max(
        sum(
            max(_safe_int(row.get("maintenance_selected_count")), 0)
            for row in rows
            if str(row.get("system", "")).lower() == "sara"
        ),
        1e-9,
    )
    ann_maintenance_event_cost_per_selected = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "ann"
    ) / max(
        sum(
            max(_safe_int(row.get("maintenance_selected_count")), 0)
            for row in rows
            if str(row.get("system", "")).lower() == "ann"
        ),
        1e-9,
    )
    sara_maintenance_event_cost_per_refresh = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "sara"
    ) / max(
        sum(
            max(_safe_int(row.get("maintenance_refresh_count")), 0)
            for row in rows
            if str(row.get("system", "")).lower() == "sara"
        ),
        1e-9,
    )
    ann_maintenance_event_cost_per_refresh = sum(
        max(_safe_float(row.get("maintenance_event_cost")), 0.0)
        for row in rows
        if str(row.get("system", "")).lower() == "ann"
    ) / max(
        sum(
            max(_safe_int(row.get("maintenance_refresh_count")), 0)
            for row in rows
            if str(row.get("system", "")).lower() == "ann"
        ),
        1e-9,
    )
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
                "sara_maintenance_event_cost_per_success": _safe_float(
                    sara_row.get("maintenance_event_cost")
                )
                / max(_safe_int(sara_row.get("success_count")), 1),
                "ann_maintenance_event_cost_per_success": _safe_float(
                    ann_row.get("maintenance_event_cost")
                )
                / max(_safe_int(ann_row.get("success_count")), 1),
                "sara_maintenance_selected_per_success": _safe_float(
                    sara_row.get("maintenance_selected_count")
                )
                / max(_safe_int(sara_row.get("success_count")), 1),
                "ann_maintenance_selected_per_success": _safe_float(
                    ann_row.get("maintenance_selected_count")
                )
                / max(_safe_int(ann_row.get("success_count")), 1),
                "sara_maintenance_event_cost_per_selected": _safe_float(
                    sara_row.get("maintenance_event_cost")
                )
                / max(_safe_int(sara_row.get("maintenance_selected_count")), 1),
                "ann_maintenance_event_cost_per_selected": _safe_float(
                    ann_row.get("maintenance_event_cost")
                )
                / max(_safe_int(ann_row.get("maintenance_selected_count")), 1),
                "sara_maintenance_event_cost_per_refresh": _safe_float(
                    sara_row.get("maintenance_event_cost")
                )
                / max(_safe_int(sara_row.get("maintenance_refresh_count")), 1),
                "ann_maintenance_event_cost_per_refresh": _safe_float(
                    ann_row.get("maintenance_event_cost")
                )
                / max(_safe_int(ann_row.get("maintenance_refresh_count")), 1),
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
        sara_maintenance_cost_values = [
            float(pair["sara_maintenance_event_cost_per_success"])
            for pair in task_pairs
        ]
        ann_maintenance_cost_values = [
            float(pair["ann_maintenance_event_cost_per_success"])
            for pair in task_pairs
        ]
        sara_maintenance_selected_values = [
            float(pair["sara_maintenance_selected_per_success"])
            for pair in task_pairs
        ]
        ann_maintenance_selected_values = [
            float(pair["ann_maintenance_selected_per_success"])
            for pair in task_pairs
        ]
        sara_maintenance_selected_cost_values = [
            float(pair["sara_maintenance_event_cost_per_selected"])
            for pair in task_pairs
        ]
        ann_maintenance_selected_cost_values = [
            float(pair["ann_maintenance_event_cost_per_selected"])
            for pair in task_pairs
        ]
        sara_maintenance_refresh_cost_values = [
            float(pair["sara_maintenance_event_cost_per_refresh"])
            for pair in task_pairs
        ]
        ann_maintenance_refresh_cost_values = [
            float(pair["ann_maintenance_event_cost_per_refresh"])
            for pair in task_pairs
        ]
        task_pair_metrics[task] = {
            "valid_pair_count": len(task_pairs),
            "sara_median_joule_per_success": _median(sara_values),
            "sara_joule_per_success_mad": _mad(sara_values),
            "ann_median_joule_per_success": _median(ann_values),
            "ann_joule_per_success_mad": _mad(ann_values),
            "sara_median_maintenance_event_cost_per_success": _median(
                sara_maintenance_cost_values
            ),
            "ann_median_maintenance_event_cost_per_success": _median(
                ann_maintenance_cost_values
            ),
            "sara_median_maintenance_selected_per_success": _median(
                sara_maintenance_selected_values
            ),
            "ann_median_maintenance_selected_per_success": _median(
                ann_maintenance_selected_values
            ),
            "sara_median_maintenance_event_cost_per_selected": _median(
                sara_maintenance_selected_cost_values
            ),
            "ann_median_maintenance_event_cost_per_selected": _median(
                ann_maintenance_selected_cost_values
            ),
            "sara_median_maintenance_event_cost_per_refresh": _median(
                sara_maintenance_refresh_cost_values
            ),
            "ann_median_maintenance_event_cost_per_refresh": _median(
                ann_maintenance_refresh_cost_values
            ),
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
        "sara_maintenance_event_cost_per_success": float(
            sara_maintenance_event_cost_per_success
        ),
        "ann_maintenance_event_cost_per_success": float(
            ann_maintenance_event_cost_per_success
        ),
        "sara_maintenance_event_cost_per_selected": float(
            sara_maintenance_event_cost_per_selected
        ),
        "ann_maintenance_event_cost_per_selected": float(
            ann_maintenance_event_cost_per_selected
        ),
        "sara_maintenance_event_cost_per_refresh": float(
            sara_maintenance_event_cost_per_refresh
        ),
        "ann_maintenance_event_cost_per_refresh": float(
            ann_maintenance_event_cost_per_refresh
        ),
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
        "maintenance_trace_rows_present": any(
            any(field in row for field in MAINTENANCE_FIELDS)
            for row in rows
        ),
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


def _session_pair_id_template(*, session_id: str, task: str) -> str:
    safe_session = str(session_id or "energy-session").strip().replace(" ", "-")
    safe_task = str(task or "task").strip().replace(" ", "-")
    return f"{safe_session}-{safe_task}-pair-<replicate>"


def _session_artifact_prefix(*, session_id: str, task: str) -> str:
    safe_session = str(session_id or "energy-session").strip().replace(" ", "-")
    safe_task = str(task or "task").strip().replace(" ", "-")
    return f"workspace/evaluation/{safe_session}_{safe_task}_r<replicate>"


def _session_manifest_path(*, session_id: str, task: str) -> str:
    return _session_artifact_prefix(session_id=session_id, task=task) + "_manifest.json"


def _session_trace_path(*, session_id: str, task: str) -> str:
    return _session_artifact_prefix(session_id=session_id, task=task) + "_trace.jsonl"


def _session_report_path(*, session_id: str, task: str) -> str:
    return _session_artifact_prefix(session_id=session_id, task=task) + "_report.json"


def _session_summary_path(*, session_id: str, task: str) -> str:
    return _session_artifact_prefix(session_id=session_id, task=task) + "_summary.txt"


def _session_meter_template_path(*, session_id: str, task: str) -> str:
    return _session_artifact_prefix(session_id=session_id, task=task) + "_meter_template.json"


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
        "--measured-repetitions <count> --run-order <1-or-2> "
        "--maintenance-selected-count <count> --maintenance-phase-count <count> "
        "--maintenance-refresh-count <count> --maintenance-event-cost <cost>"
    )


def _replace_replicate_placeholder(value: str, replicate_index: int) -> str:
    return str(value).replace("<replicate>", str(replicate_index))


def _physical_pair_command_for_session(*, session_id: str, task: str) -> str:
    pair_id = _session_pair_id_template(session_id=session_id, task=task)
    meter_template_path = _session_meter_template_path(
        session_id=session_id,
        task=task,
    )
    manifest_path = _session_manifest_path(session_id=session_id, task=task)
    trace_path = _session_trace_path(session_id=session_id, task=task)
    report_path = _session_report_path(session_id=session_id, task=task)
    summary_path = _session_summary_path(session_id=session_id, task=task)
    return (
        "python scripts/sara_cli.py run-physical-energy-pair "
        f"--pair-id {pair_id} --replicate-index <replicate> "
        "--measurement-tool <tool-id> --thread-count <threads> "
        "--process-affinity <affinity> --power-mode <mode> "
        "--event-memory-maintenance-coupling-report-path "
        "workspace/evaluation/event_memory_maintenance_coupling_benchmark.json "
        f"--manifest-path {manifest_path} --trace-path {trace_path} "
        f"--report-path {report_path} --summary-path {summary_path} "
        f"--meter-template-path {meter_template_path}"
    )


def _build_measurement_session_progress(
    session_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in planned_runs:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category", "") or "").strip()
        task = str(item.get("task", "") or "").strip()
        pair_id_template = str(item.get("pair_id_template", "") or "").strip()
        pair_command_template = str(item.get("pair_command_template", "") or "").strip()
        if not category or not task or not pair_id_template:
            continue
        key = (category, task)
        entry = grouped.setdefault(
            key,
            {
                "category": category,
                "task": task,
                "priority": str(item.get("priority", "") or ""),
                "replicate_count": 0,
                "fixed_replicate_index": 0,
                "pair_id_template": pair_id_template,
                "pair_command_template": pair_command_template,
                "meter_template_path": str(item.get("meter_template_path", "") or ""),
                "report_path_template": str(item.get("report_path_template", "") or ""),
                "summary_path_template": str(item.get("summary_path_template", "") or ""),
            },
        )
        entry["replicate_count"] = max(
            int(entry.get("replicate_count", 0) or 0),
            _safe_int(item.get("replicate_count")) or 1,
        )
        fixed_replicate_index = _safe_int(item.get("fixed_replicate_index"))
        if fixed_replicate_index > 0:
            entry["fixed_replicate_index"] = fixed_replicate_index

    expected_pairs: List[Dict[str, Any]] = []
    for _, item in sorted(grouped.items()):
        fixed_replicate_index = _safe_int(item.get("fixed_replicate_index"))
        replicate_indexes = (
            [fixed_replicate_index]
            if fixed_replicate_index > 0
            else list(range(1, max(_safe_int(item.get("replicate_count")), 1) + 1))
        )
        for replicate_index in replicate_indexes:
            expected_pairs.append(
                {
                    "category": str(item.get("category", "") or ""),
                    "task": str(item.get("task", "") or ""),
                    "priority": str(item.get("priority", "") or ""),
                    "pair_id": _replace_replicate_placeholder(
                        str(item.get("pair_id_template", "") or ""),
                        replicate_index,
                    ),
                    "replicate_index": int(replicate_index),
                    "pair_command": _replace_replicate_placeholder(
                        str(item.get("pair_command_template", "") or ""),
                        replicate_index,
                    ),
                    "meter_template_path": _replace_replicate_placeholder(
                        str(item.get("meter_template_path", "") or ""),
                        replicate_index,
                    ),
                    "report_path": _replace_replicate_placeholder(
                        str(item.get("report_path_template", "") or ""),
                        replicate_index,
                    ),
                    "summary_path": _replace_replicate_placeholder(
                        str(item.get("summary_path_template", "") or ""),
                        replicate_index,
                    ),
                }
            )

    pair_index: Dict[tuple[str, str, int], Dict[str, Mapping[str, Any]]] = {}
    row_errors: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        errors = _validate_measurement(row)
        if errors:
            row_errors.append({"index": index, "errors": errors})
            continue
        system = str(row.get("system", "")).lower()
        if system not in {"sara", "ann"}:
            continue
        pair_index.setdefault(_pair_key(row), {})[system] = row

    expected_keys: set[tuple[str, str, int]] = set()
    pair_statuses: List[Dict[str, Any]] = []
    for item in expected_pairs:
        task = str(item.get("task", "") or "")
        pair_id = str(item.get("pair_id", "") or "")
        replicate_index = _safe_int(item.get("replicate_index"))
        key = (task, pair_id, replicate_index)
        expected_keys.add(key)
        systems = pair_index.get(key, {})
        present_systems = sorted(systems.keys())
        status = "missing_pair"
        errors: List[str] = []
        fairness_classification: Dict[str, Any] = {}
        ann_to_sara_ratio = 0.0
        if set(present_systems) == {"ann", "sara"}:
            sara_row = systems["sara"]
            ann_row = systems["ann"]
            errors = _pair_fairness_errors(sara_row, ann_row)
            fairness_classification = _classify_pair_fairness_errors(errors)
            if errors:
                status = "invalid_pair"
            else:
                status = "complete_valid_pair"
                sara_jps = _safe_float(sara_row.get("joules")) / max(
                    _safe_int(sara_row.get("success_count")),
                    1,
                )
                ann_jps = _safe_float(ann_row.get("joules")) / max(
                    _safe_int(ann_row.get("success_count")),
                    1,
                )
                ann_to_sara_ratio = ann_jps / max(sara_jps, 1e-9)
        elif present_systems:
            status = "partial_pair"
            missing = sorted({"sara", "ann"} - set(present_systems))
            errors = [f"missing_system:{name}" for name in missing]
        pair_statuses.append(
            {
                "category": str(item.get("category", "") or ""),
                "task": task,
                "priority": str(item.get("priority", "") or ""),
                "pair_id": pair_id,
                "replicate_index": replicate_index,
                "status": status,
                "present_systems": present_systems,
                "errors": errors,
                "invalid_reason_category": str(
                    fairness_classification.get("category", "") if errors else ""
                ),
                "invalid_reason_priority": str(
                    fairness_classification.get("priority", "") if errors else ""
                ),
                "invalid_reason_fields": list(
                    fairness_classification.get("mismatch_fields", []) if errors else []
                ),
                "invalid_reason_remediation": str(
                    fairness_classification.get("remediation", "") if errors else ""
                ),
                "ann_to_sara_joule_efficiency_ratio": float(ann_to_sara_ratio),
                "pair_command": str(item.get("pair_command", "") or ""),
                "meter_template_path": str(item.get("meter_template_path", "") or ""),
                "report_path": str(item.get("report_path", "") or ""),
                "summary_path": str(item.get("summary_path", "") or ""),
            }
        )

    orphan_pairs: List[Dict[str, Any]] = []
    for key, systems in sorted(pair_index.items()):
        if key in expected_keys:
            continue
        orphan_pairs.append(
            {
                "task": key[0],
                "pair_id": key[1],
                "replicate_index": key[2],
                "present_systems": sorted(systems.keys()),
            }
        )

    task_progress: Dict[str, Dict[str, int]] = {}
    for item in pair_statuses:
        task_bucket = task_progress.setdefault(
            str(item["task"]),
            {
                "planned_pair_count": 0,
                "complete_valid_pair_count": 0,
                "invalid_pair_count": 0,
                "partial_pair_count": 0,
                "missing_pair_count": 0,
            },
        )
        task_bucket["planned_pair_count"] += 1
        task_bucket[f"{item['status']}_count"] += 1

    complete_valid_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "complete_valid_pair"
    )
    invalid_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "invalid_pair"
    )
    partial_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "partial_pair"
    )
    missing_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "missing_pair"
    )
    planned_pair_count = len(pair_statuses)
    return {
        "schema": "sara-physical-energy-session-progress-v1",
        "session_id": str(session_plan.get("session_id", "") or ""),
        "status": (
            "complete"
            if planned_pair_count > 0 and complete_valid_pair_count == planned_pair_count
            else ("in_progress" if complete_valid_pair_count or partial_pair_count else "pending")
        ),
        "planned_pair_count": planned_pair_count,
        "complete_valid_pair_count": complete_valid_pair_count,
        "invalid_pair_count": invalid_pair_count,
        "partial_pair_count": partial_pair_count,
        "missing_pair_count": missing_pair_count,
        "orphan_pair_count": len(orphan_pairs),
        "invalid_measurement_row_count": len(row_errors),
        "task_progress": task_progress,
        "pair_statuses": pair_statuses,
        "orphan_pairs": orphan_pairs,
        "invalid_measurement_rows": row_errors,
    }


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
    invalid_pairs = (
        measurement_plan.get("invalid_pairs", [])
        if isinstance(measurement_plan.get("invalid_pairs"), list)
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
                "replicate_count": _safe_int(item.get("replicate_count")) or 1,
                "pair_id_template": _session_pair_id_template(
                    session_id=session_id,
                    task=task,
                ),
                "run_id_template": _session_run_id_template(
                    session_id=session_id,
                    task=task,
                    system=system,
                ),
                "manifest_path_template": _session_manifest_path(
                    session_id=session_id,
                    task=task,
                ),
                "trace_path_template": _session_trace_path(
                    session_id=session_id,
                    task=task,
                ),
                "report_path_template": _session_report_path(
                    session_id=session_id,
                    task=task,
                ),
                "summary_path_template": _session_summary_path(
                    session_id=session_id,
                    task=task,
                ),
                "meter_template_path": _session_meter_template_path(
                    session_id=session_id,
                    task=task,
                ),
                "pair_command_template": _physical_pair_command_for_session(
                    session_id=session_id,
                    task=task,
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
                    "replicate_count": _safe_int(item.get("replicate_count")) or 1,
                    "observed_ratio": _safe_float(item.get("ann_to_sara_joule_efficiency_ratio")),
                    "required_min": _safe_float(item.get("required_min")),
                    "pair_id_template": _session_pair_id_template(
                        session_id=session_id,
                        task=task,
                    ),
                    "run_id_template": _session_run_id_template(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                    "manifest_path_template": _session_manifest_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "trace_path_template": _session_trace_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "report_path_template": _session_report_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "summary_path_template": _session_summary_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "meter_template_path": _session_meter_template_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "pair_command_template": _physical_pair_command_for_session(
                        session_id=session_id,
                        task=task,
                    ),
                    "command_template": _record_command_for_session(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                }
            )

    for item in invalid_pairs:
        if not isinstance(item, Mapping):
            continue
        task = str(item.get("task", "") or "").strip()
        pair_id = str(item.get("pair_id", "") or "").strip()
        replicate_index = _safe_int(item.get("replicate_index")) or 1
        if not task or not pair_id:
            continue
        for system in ("sara", "ann"):
            planned_runs.append(
                {
                    "category": "repair_invalid_pair",
                    "priority": str(item.get("priority", "high")),
                    "task": task,
                    "system": system,
                    "replicate_count": 1,
                    "fixed_replicate_index": int(replicate_index),
                    "pair_id_template": pair_id,
                    "run_id_template": _session_run_id_template(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                    "manifest_path_template": _session_manifest_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "trace_path_template": _session_trace_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "report_path_template": _session_report_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "summary_path_template": _session_summary_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "meter_template_path": _session_meter_template_path(
                        session_id=session_id,
                        task=task,
                    ),
                    "pair_command_template": _physical_pair_command_for_session(
                        session_id=session_id,
                        task=task,
                    ),
                    "command_template": _record_command_for_session(
                        session_id=session_id,
                        task=task,
                        system=system,
                    ),
                    "invalid_reason_category": str(item.get("reason_category", "") or ""),
                    "invalid_reason_fields": list(item.get("reason_fields", []))
                    if isinstance(item.get("reason_fields", []), list)
                    else [],
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
            "If idle maintenance or spontaneous replay is active, record maintenance counts and event cost in the optional maintenance fields.",
        ],
    }


def _build_measurement_plan(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    min_ann_to_sara_ratio: float,
    min_paired_replicates_per_task: int,
) -> Dict[str, Any]:
    tasks = aggregate.get("tasks", {}) if isinstance(aggregate.get("tasks"), Mapping) else {}
    pair_errors = (
        aggregate.get("pair_errors", [])
        if isinstance(aggregate.get("pair_errors"), list)
        else []
    )
    pending_pairs: List[Dict[str, Any]] = []
    if not rows:
        for task in CANONICAL_MEASUREMENT_TASKS:
            for system in ("sara", "ann"):
                pending_pairs.append(
                    {
                        "task": task,
                        "missing_system": system,
                        "priority": "high",
                        "replicate_count": int(min_paired_replicates_per_task),
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
                        "replicate_count": 1,
                        "command_template": _record_command_template(task=str(task), system=system),
                    }
                )

    paired_ratios = (
        aggregate.get("paired_task_ann_to_sara_ratios", {})
        if isinstance(aggregate.get("paired_task_ann_to_sara_ratios"), Mapping)
        else {}
    )
    weak_pairs: List[Dict[str, Any]] = []
    for task, ratio in sorted(paired_ratios.items()):
        observed_ratio = _safe_float(ratio)
        required_min = float(min_ann_to_sara_ratio)
        if observed_ratio >= required_min:
            continue
        relative_ratio = observed_ratio / required_min if required_min > 0.0 else 0.0
        ratio_gap = required_min - observed_ratio
        severity = "moderate"
        priority = "medium"
        next_action = "Repeat paired measurement or inspect the SARA sparse-event trace for this task."
        if relative_ratio < 0.5:
            severity = "critical"
            priority = "high"
            next_action = (
                "Repeat this paired measurement with more replicates and inspect both fairness fields "
                "and the SARA sparse-event trace before treating the energy result as credible."
            )
        elif relative_ratio < 0.8:
            severity = "high"
            priority = "high"
            next_action = (
                "Repeat this paired measurement and inspect the SARA sparse-event trace before "
                "promoting the current ratio into roadmap evidence."
            )
        weak_pairs.append(
            {
                "task": str(task),
                "ann_to_sara_joule_efficiency_ratio": float(observed_ratio),
                "required_min": required_min,
                "relative_ratio": float(relative_ratio),
                "ratio_gap": float(ratio_gap),
                "severity": severity,
                "priority": priority,
                "replicate_count": 1,
                "next_action": next_action,
            }
        )
    invalid_pairs: List[Dict[str, Any]] = []
    for item in pair_errors:
        if not isinstance(item, Mapping):
            continue
        pair_key = item.get("pair_key", [])
        if not isinstance(pair_key, list) or len(pair_key) != 3:
            continue
        errors = [str(value).strip() for value in item.get("errors", []) if str(value).strip()]
        if "missing_system_pair" in errors:
            continue
        task = str(pair_key[0] or "")
        pair_id = str(pair_key[1] or "")
        replicate_index = _safe_int(pair_key[2])
        fairness_classification = _classify_pair_fairness_errors(errors)
        quality_parity_failed = "success_rate_parity_failed" in errors
        reason_category = str(fairness_classification.get("category", "") or "")
        priority = str(fairness_classification.get("priority", "medium") or "medium")
        next_action = str(
            fairness_classification.get(
                "remediation",
                "Inspect invalid pair conditions and rerun the physical pair.",
            )
            or "Inspect invalid pair conditions and rerun the physical pair."
        )
        if quality_parity_failed and reason_category in {"none", "unclassified_fairness_error"}:
            reason_category = "success_rate_parity_failure"
            priority = "high"
            next_action = (
                "Rerun this physical pair with a verified shared success criterion before treating "
                "the result as comparable."
            )
        invalid_pairs.append(
            {
                "task": task,
                "pair_id": pair_id,
                "replicate_index": replicate_index,
                "errors": errors,
                "reason_category": reason_category,
                "reason_fields": list(fairness_classification.get("mismatch_fields", [])),
                "priority": priority,
                "replicate_count": 1,
                "next_action": next_action,
            }
        )
    ready_for_real_claim = bool(
        not pending_pairs
        and not weak_pairs
        and not invalid_pairs
        and int(aggregate.get("paired_task_count", 0) or 0) > 0
    )
    return {
        "schema": "sara-energy-measurement-plan-v2",
        "ready_for_real_joule_claim": ready_for_real_claim,
        "required_task_pair_count": len(CANONICAL_MEASUREMENT_TASKS),
        "observed_paired_task_count": int(aggregate.get("paired_task_count", 0) or 0),
        "pending_pair_count": len(pending_pairs),
        "weak_pair_count": len(weak_pairs),
        "invalid_pair_repair_count": len(invalid_pairs),
        "pending_pairs": pending_pairs,
        "weak_pairs": weak_pairs,
        "invalid_pairs": invalid_pairs,
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
    internal_maintenance_report: Mapping[str, Any] | None = None,
    event_memory_maintenance_coupling_report: Mapping[str, Any] | None = None,
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
    physical_rows = [row for row in valid_rows if row.get("physical_evidence", True) is not False]
    system_estimate_rows = [row for row in valid_rows if row.get("physical_evidence", True) is False]
    has_real_measurements = bool(
        any(str(row.get("system", "")).lower() == "sara" for row in physical_rows)
        and any(str(row.get("system", "")).lower() == "ann" for row in physical_rows)
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
        min_paired_replicates_per_task=min_paired_replicates_per_task,
    )
    measurement_session_plan = _build_measurement_session_plan(
        measurement_plan,
        measurement_path=measurement_path,
        min_ann_to_sara_ratio=min_ann_to_sara_ratio,
        session_id=session_id,
    )
    measurement_session_progress = _build_measurement_session_progress(
        measurement_session_plan,
        valid_rows,
    )
    session_pair_completion_passed = bool(
        (not rows)
        or _safe_int(measurement_session_progress.get("planned_pair_count")) == 0
        or (
            _safe_int(measurement_session_progress.get("complete_valid_pair_count"))
            == _safe_int(measurement_session_progress.get("planned_pair_count"))
            and _safe_int(measurement_session_progress.get("invalid_pair_count")) == 0
            and _safe_int(measurement_session_progress.get("partial_pair_count")) == 0
            and _safe_int(measurement_session_progress.get("orphan_pair_count")) == 0
            and _safe_int(measurement_session_progress.get("invalid_measurement_row_count")) == 0
        )
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
        "session_pair_completion_passed": session_pair_completion_passed,
    }
    protocol_ready = bool(checks["schema_ready"] and checks["rows_valid"])
    internal_maintenance_reference = _internal_maintenance_reference_summary(
        internal_maintenance_report
    )
    event_memory_maintenance_coupling_reference = (
        _event_memory_maintenance_coupling_reference_summary(
            event_memory_maintenance_coupling_report
        )
    )
    measurement_session_progress = dict(measurement_session_progress)
    measurement_session_progress["internal_maintenance_reference"] = internal_maintenance_reference
    measurement_session_progress["event_memory_maintenance_coupling_reference"] = (
        event_memory_maintenance_coupling_reference
    )
    bundle_contribution_warning = _bundle_contribution_warning(
        event_memory_maintenance_coupling_reference
    )
    maintenance_alignment = _maintenance_alignment_summary(
        aggregate,
        internal_maintenance_reference,
    )
    physical_checks_passed = bool(all(checks.values()) and has_real_measurements)
    return {
        "schema": "sara-energy-measurement-readiness-v2",
        "passed": bool(protocol_ready and (not rows or physical_checks_passed)),
        "status": "real_joule_evidence_passed"
        if physical_checks_passed
        else ("system_estimate_pending_physical_measurement" if system_estimate_rows and protocol_ready else ("protocol_ready_pending_measurements" if protocol_ready and not rows else "needs_measurement_repair")),
        "measurement_count": len(rows),
        "valid_measurement_count": len(valid_rows),
        "physical_measurement_count": len(physical_rows),
        "system_estimate_measurement_count": len(system_estimate_rows),
        "real_joule_measurements_present": has_real_measurements,
        "min_ann_to_sara_ratio": float(min_ann_to_sara_ratio),
        "max_success_rate_delta": float(max_success_rate_delta),
        "min_paired_replicates_per_task": int(min_paired_replicates_per_task),
        "checks": checks,
        "row_errors": row_errors,
        "metrics": aggregate,
        "measurement_plan": measurement_plan,
        "measurement_session_plan": measurement_session_plan,
        "measurement_session_progress": measurement_session_progress,
        "internal_maintenance_reference": internal_maintenance_reference,
        "event_memory_maintenance_coupling_reference": event_memory_maintenance_coupling_reference,
        "bundle_contribution_warning": bundle_contribution_warning,
        "maintenance_alignment": maintenance_alignment,
        "measurement_protocol": {
            "required_fields": sorted(REQUIRED_FIELDS | FAIRNESS_FIELDS),
            "systems": ["sara", "ann"],
            "units": {"average_watts": "W", "duration_seconds": "s", "joules": "J", "success_count": "count"},
            "recommended_path": str(measurement_path),
            "accepted_energy_inputs": [
                "direct_joules",
                "average_watts_x_duration_seconds",
            ],
            "optional_maintenance_fields": list(MAINTENANCE_FIELDS),
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
    session_progress = (
        report.get("measurement_session_progress", {})
        if isinstance(report.get("measurement_session_progress"), Mapping)
        else {}
    )
    internal_maintenance_reference = (
        report.get("internal_maintenance_reference", {})
        if isinstance(report.get("internal_maintenance_reference"), Mapping)
        else {}
    )
    maintenance_alignment = (
        report.get("maintenance_alignment", {})
        if isinstance(report.get("maintenance_alignment"), Mapping)
        else {}
    )
    event_memory_maintenance_coupling_reference = (
        report.get("event_memory_maintenance_coupling_reference", {})
        if isinstance(report.get("event_memory_maintenance_coupling_reference"), Mapping)
        else {}
    )
    bundle_contribution_warning = str(report.get("bundle_contribution_warning", "") or "")
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
        f"- maintenance_trace_rows_present: {bool(metrics.get('maintenance_trace_rows_present', False))}",
        f"- measurement_pending_pair_count: {_safe_int(plan.get('pending_pair_count'))}",
        f"- measurement_weak_pair_count: {_safe_int(plan.get('weak_pair_count'))}",
        f"- measurement_session_planned_run_count: {_safe_int(session_plan.get('planned_run_count'))}",
        f"- measurement_session_complete_valid_pair_count: {_safe_int(session_progress.get('complete_valid_pair_count'))}",
        f"- measurement_session_partial_pair_count: {_safe_int(session_progress.get('partial_pair_count'))}",
        f"- measurement_session_invalid_pair_count: {_safe_int(session_progress.get('invalid_pair_count'))}",
        f"- measurement_session_orphan_pair_count: {_safe_int(session_progress.get('orphan_pair_count'))}",
        f"- internal_maintenance_reference_available: {bool(internal_maintenance_reference.get('available', False))}",
        f"- internal_maintenance_event_cost_per_selected: {_safe_float(internal_maintenance_reference.get('maintenance_event_cost_per_selected')):.3f}",
        f"- internal_maintenance_self_state_continuity_observed: {_safe_float(internal_maintenance_reference.get('maintenance_self_state_continuity_observed')):.3f}",
        f"- event_memory_maintenance_coupling_reference_available: {bool(event_memory_maintenance_coupling_reference.get('available', False))}",
        f"- event_memory_maintenance_best_profile: {event_memory_maintenance_coupling_reference.get('best_profile_id', '')}",
        f"- event_memory_maintenance_best_efficiency: {_safe_float(event_memory_maintenance_coupling_reference.get('best_profile_compression_efficiency_per_maintenance')):.3f}",
        f"- event_memory_maintenance_best_bundle_contribution: {_safe_float(event_memory_maintenance_coupling_reference.get('best_profile_multimodal_bundle_compression_contribution')):.3f}",
        f"- maintenance_alignment_available: {bool(maintenance_alignment.get('available', False))}",
        f"- physical_internal_maintenance_event_cost_per_selected: {_safe_float(maintenance_alignment.get('sara_physical_maintenance_event_cost_per_selected')):.3f}",
        f"- maintenance_event_cost_per_selected_alignment_ratio: {_safe_float(maintenance_alignment.get('maintenance_event_cost_per_selected_ratio')):.3f}",
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
    pair_statuses = (
        session_progress.get("pair_statuses", [])
        if isinstance(session_progress.get("pair_statuses"), list)
        else []
    )
    lines.append("Measurement Session Progress:")
    if pair_statuses:
        for item in pair_statuses[:8]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - "
                f"task={item.get('task', '')}, "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"status={item.get('status', '')}"
            )
    else:
        lines.append("- pair_statuses: none")
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
            lines.append(
                "    "
                f"pair_command={item.get('pair_command_template', '')}"
            )
    else:
        lines.append("- planned_runs: none")
    if internal_maintenance_reference:
        lines.append("Internal Maintenance Reference:")
        lines.append(
            "- "
            f"available={bool(internal_maintenance_reference.get('available', False))}, "
            f"observed_only={bool(internal_maintenance_reference.get('observed_only', False))}, "
            f"passed={bool(internal_maintenance_reference.get('passed', False))}, "
            f"selected={_safe_int(internal_maintenance_reference.get('maintenance_selected_count'))}, "
            f"refresh={_safe_int(internal_maintenance_reference.get('maintenance_refresh_count'))}, "
            f"event_cost_per_selected={_safe_float(internal_maintenance_reference.get('maintenance_event_cost_per_selected')):.3f}"
        )
    if event_memory_maintenance_coupling_reference:
        lines.append("Event Memory Maintenance Coupling Reference:")
        lines.append(
            "- "
            f"available={bool(event_memory_maintenance_coupling_reference.get('available', False))}, "
            f"passed={bool(event_memory_maintenance_coupling_reference.get('passed', False))}, "
            f"best_profile={event_memory_maintenance_coupling_reference.get('best_profile_id', '')}, "
            f"best_efficiency={_safe_float(event_memory_maintenance_coupling_reference.get('best_profile_compression_efficiency_per_maintenance')):.3f}, "
            f"best_bundle_contribution={_safe_float(event_memory_maintenance_coupling_reference.get('best_profile_multimodal_bundle_compression_contribution')):.3f}, "
            f"best_continuity={_safe_float(event_memory_maintenance_coupling_reference.get('best_profile_self_state_continuity')):.3f}"
        )
    if bundle_contribution_warning:
        lines.append(f"Bundle Contribution Warning: {bundle_contribution_warning}")
    if maintenance_alignment:
        lines.append("Maintenance Alignment:")
        lines.append(
            "- "
            f"available={bool(maintenance_alignment.get('available', False))}, "
            f"physical_event_cost_per_selected={_safe_float(maintenance_alignment.get('sara_physical_maintenance_event_cost_per_selected')):.3f}, "
            f"reference_event_cost_per_selected={_safe_float(maintenance_alignment.get('reference_maintenance_event_cost_per_selected')):.3f}, "
            f"ratio={_safe_float(maintenance_alignment.get('maintenance_event_cost_per_selected_ratio')):.3f}, "
            f"delta={_safe_float(maintenance_alignment.get('maintenance_event_cost_per_selected_delta')):.3f}"
        )
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
            lines.append(f"  pair_command: {item.get('pair_command_template', '')}")
            lines.append(f"  meter_template_path: {item.get('meter_template_path', '')}")
    else:
        lines.append("- none")
    notes = session_plan.get("operator_notes", []) if isinstance(session_plan.get("operator_notes"), list) else []
    lines.append("Operator Notes:")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def format_measurement_session_progress_summary(
    session_progress: Mapping[str, Any],
    internal_maintenance_reference: Mapping[str, Any] | None = None,
    event_memory_maintenance_coupling_reference: Mapping[str, Any] | None = None,
) -> str:
    task_progress = (
        session_progress.get("task_progress", {})
        if isinstance(session_progress.get("task_progress"), Mapping)
        else {}
    )
    pair_statuses = (
        session_progress.get("pair_statuses", [])
        if isinstance(session_progress.get("pair_statuses"), list)
        else []
    )
    orphan_pairs = (
        session_progress.get("orphan_pairs", [])
        if isinstance(session_progress.get("orphan_pairs"), list)
        else []
    )
    lines = [
        "# SARA Physical Energy Session Progress",
        f"- schema: {session_progress.get('schema', '')}",
        f"- session_id: {session_progress.get('session_id', '')}",
        f"- status: {session_progress.get('status', '')}",
        f"- planned_pair_count: {_safe_int(session_progress.get('planned_pair_count'))}",
        f"- complete_valid_pair_count: {_safe_int(session_progress.get('complete_valid_pair_count'))}",
        f"- invalid_pair_count: {_safe_int(session_progress.get('invalid_pair_count'))}",
        f"- partial_pair_count: {_safe_int(session_progress.get('partial_pair_count'))}",
        f"- missing_pair_count: {_safe_int(session_progress.get('missing_pair_count'))}",
        f"- orphan_pair_count: {_safe_int(session_progress.get('orphan_pair_count'))}",
        f"- invalid_measurement_row_count: {_safe_int(session_progress.get('invalid_measurement_row_count'))}",
        "Task Progress:",
    ]
    if task_progress:
        for task, metrics in sorted(task_progress.items()):
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                "- "
                f"task={task}, "
                f"planned={_safe_int(metrics.get('planned_pair_count'))}, "
                f"complete={_safe_int(metrics.get('complete_valid_pair_count'))}, "
                f"partial={_safe_int(metrics.get('partial_pair_count'))}, "
                f"invalid={_safe_int(metrics.get('invalid_pair_count'))}, "
                f"missing={_safe_int(metrics.get('missing_pair_count'))}"
            )
    else:
        lines.append("- none")
    lines.append("Pair Statuses:")
    if pair_statuses:
        for item in pair_statuses:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"task={item.get('task', '')}, "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"status={item.get('status', '')}"
            )
    else:
        lines.append("- none")
    lines.append("Orphan Pairs:")
    if orphan_pairs:
        for item in orphan_pairs:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"task={item.get('task', '')}, "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}"
            )
    else:
        lines.append("- none")
    maintenance_reference = (
        internal_maintenance_reference
        if isinstance(internal_maintenance_reference, Mapping)
        else (
            session_progress.get("internal_maintenance_reference", {})
            if isinstance(session_progress.get("internal_maintenance_reference"), Mapping)
            else {}
        )
    )
    coupling_reference = (
        event_memory_maintenance_coupling_reference
        if isinstance(event_memory_maintenance_coupling_reference, Mapping)
        else (
            session_progress.get("event_memory_maintenance_coupling_reference", {})
            if isinstance(
                session_progress.get("event_memory_maintenance_coupling_reference"), Mapping
            )
            else {}
        )
    )
    if maintenance_reference:
        lines.append("Internal Maintenance Reference:")
        lines.append(
            "- "
            f"available={bool(maintenance_reference.get('available', False))}, "
            f"passed={bool(maintenance_reference.get('passed', False))}, "
            f"event_cost_per_selected={_safe_float(maintenance_reference.get('maintenance_event_cost_per_selected')):.3f}, "
            f"continuity={_safe_float(maintenance_reference.get('maintenance_self_state_continuity_observed')):.3f}"
        )
    if coupling_reference:
        lines.append("Event Memory Maintenance Coupling Reference:")
        lines.append(
            "- "
            f"available={bool(coupling_reference.get('available', False))}, "
            f"passed={bool(coupling_reference.get('passed', False))}, "
            f"best_profile={coupling_reference.get('best_profile_id', '')}, "
            f"best_efficiency={_safe_float(coupling_reference.get('best_profile_compression_efficiency_per_maintenance')):.3f}, "
            f"best_bundle_contribution={_safe_float(coupling_reference.get('best_profile_multimodal_bundle_compression_contribution')):.3f}, "
            f"best_continuity={_safe_float(coupling_reference.get('best_profile_self_state_continuity')):.3f}"
        )
        warning = _bundle_contribution_warning(coupling_reference)
        if warning:
            lines.append(f"Bundle Contribution Warning: {warning}")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real joule measurement readiness.")
    parser.add_argument("--measurement-path", default=DEFAULT_MEASUREMENT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--session-plan-path", default=DEFAULT_SESSION_PLAN_PATH)
    parser.add_argument("--session-plan-summary-path", default=DEFAULT_SESSION_PLAN_SUMMARY_PATH)
    parser.add_argument("--session-progress-path", default=DEFAULT_SESSION_PROGRESS_PATH)
    parser.add_argument("--session-progress-summary-path", default=DEFAULT_SESSION_PROGRESS_SUMMARY_PATH)
    parser.add_argument("--internal-maintenance-report-path", default=DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH)
    parser.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=DEFAULT_EVENT_MEMORY_MAINTENANCE_COUPLING_REPORT_PATH,
    )
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
    parser.add_argument("--maintenance-selected-count", type=int, default=None)
    parser.add_argument("--maintenance-phase-count", type=int, default=None)
    parser.add_argument("--maintenance-refresh-count", type=int, default=None)
    parser.add_argument("--maintenance-event-cost", type=float, default=None)
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
            maintenance_selected_count=args.maintenance_selected_count,
            maintenance_phase_count=args.maintenance_phase_count,
            maintenance_refresh_count=args.maintenance_refresh_count,
            maintenance_event_cost=args.maintenance_event_cost,
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
        internal_maintenance_report=_load_optional_json(args.internal_maintenance_report_path),
        event_memory_maintenance_coupling_report=_load_optional_json(
            args.event_memory_maintenance_coupling_report_path
        ),
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    session_plan_path = ensure_parent_directory(args.session_plan_path)
    session_plan_summary_path = ensure_parent_directory(args.session_plan_summary_path)
    session_progress_path = ensure_parent_directory(args.session_progress_path)
    session_progress_summary_path = ensure_parent_directory(args.session_progress_summary_path)
    session_plan = report.get("measurement_session_plan", {})
    if not isinstance(session_plan, Mapping):
        session_plan = {}
    session_progress = report.get("measurement_session_progress", {})
    if not isinstance(session_progress, Mapping):
        session_progress = {}
    internal_maintenance_reference = report.get("internal_maintenance_reference", {})
    if not isinstance(internal_maintenance_reference, Mapping):
        internal_maintenance_reference = {}
    event_memory_maintenance_coupling_reference = report.get(
        "event_memory_maintenance_coupling_reference", {}
    )
    if not isinstance(event_memory_maintenance_coupling_reference, Mapping):
        event_memory_maintenance_coupling_reference = {}
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_energy_measurement_summary(report))
    with open(session_plan_path, "w", encoding="utf-8") as handle:
        json.dump(session_plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(session_plan_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_measurement_session_plan_summary(session_plan))
    with open(session_progress_path, "w", encoding="utf-8") as handle:
        json.dump(session_progress, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(session_progress_summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            format_measurement_session_progress_summary(
                session_progress,
                internal_maintenance_reference=internal_maintenance_reference,
                event_memory_maintenance_coupling_reference=(
                    event_memory_maintenance_coupling_reference
                ),
            )
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
