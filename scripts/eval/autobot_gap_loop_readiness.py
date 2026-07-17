#!/usr/bin/env python3
"""Evaluate whether the managed autobot gap loop is producing usable repair evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_LOOP_REPORT_PATH = workspace_path("autobot", "gap_loop_report.json")
DEFAULT_COLLECTION_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness_summary.txt")
DEFAULT_ISOLATION_AUDIT_PATH = workspace_path("evaluation", "phase7_isolation_audit.json")

ISOLATION_AXIS_REPAIR_REQUIREMENTS = {
    "metadata_complete": {
        "required_fields": ["source_hash", "source_revision", "source_domain", "collection_time", "near_duplicate_signature"],
        "verification": "all train and evaluation rows must contain non-empty provenance fields",
    },
    "source_hash_isolated": {
        "required_fields": ["source_hash"],
        "verification": "shared_source_hashes must be empty after the full split audit",
    },
    "source_revision_isolated": {
        "required_fields": ["source_revision"],
        "verification": "shared_source_revisions must be empty after the full split audit",
    },
    "source_domain_isolated": {
        "required_fields": ["source_domain"],
        "verification": "shared_source_domains must be empty after the full split audit",
    },
    "time_split_isolated": {
        "required_fields": ["collection_time"],
        "verification": "the latest train timestamp must be earlier than the earliest evaluation timestamp",
    },
    "independent_evidence_scope_valid": {
        "required_fields": ["evidence_scope"],
        "verification": "all rows must declare independent_external evidence scope",
    },
    "near_duplicate_signature_format_valid": {
        "required_fields": ["near_duplicate_signature"],
        "verification": "all signatures must be valid 16-digit lowercase hexadecimal values",
    },
    "near_duplicate_signature_isolated": {
        "required_fields": ["near_duplicate_signature"],
        "verification": "all train/evaluation signature pairs must exceed the configured Hamming threshold",
    },
}


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_jsonl(path: str) -> Sequence[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def write_json(path: str, payload: Mapping[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_output_path(loop_report: Optional[Dict[str, Any]], key: str) -> str:
    if not isinstance(loop_report, dict):
        return ""
    outputs = loop_report.get("outputs", {})
    if not isinstance(outputs, dict):
        return ""
    return str(outputs.get(key, "") or "")


def _count_requested_slots(targets_payload: Optional[Dict[str, Any]]) -> int:
    if not isinstance(targets_payload, dict):
        return 0
    targets = targets_payload.get("targets", [])
    if not isinstance(targets, list):
        return 0
    total = 0
    for item in targets:
        if not isinstance(item, dict):
            continue
        total += len([value for value in item.get("missing_material_types", []) if str(value)])
    return total


def _fixture_targets(targets_payload: Optional[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    if not isinstance(targets_payload, dict):
        return []
    targets = targets_payload.get("targets", [])
    if not isinstance(targets, list):
        return []
    return [
        item
        for item in targets
        if isinstance(item, dict) and str(item.get("request_id", "") or "").startswith("fixture_")
    ]


def _count_fixture_requested_slots(targets_payload: Optional[Dict[str, Any]]) -> int:
    total = 0
    for item in _fixture_targets(targets_payload):
        total += len([value for value in item.get("missing_material_types", []) if str(value)])
    return total


def _check(condition: bool, value: Any, detail: str = "") -> Dict[str, Any]:
    return {
        "passed": bool(condition),
        "value": value,
        "detail": detail,
    }


def _build_fixture_repair_actions(
    *,
    fixture_targets: Sequence[Dict[str, Any]],
    requested_slots_by_request: Mapping[str, int],
    built_by_request: Mapping[str, int],
    skipped_by_request: Mapping[str, int],
    input_paths: Optional[Mapping[str, str]] = None,
    blocked_request_ids: Optional[Sequence[str]] = None,
    blocked_request_missing_axes: Optional[Mapping[str, Sequence[str]]] = None,
    clearable_blocked_request_ids: Optional[Sequence[str]] = None,
    request_isolation_audit: Optional[Mapping[str, Any]] = None,
    global_isolation_audit: Optional[Mapping[str, Any]] = None,
) -> Sequence[Dict[str, Any]]:
    targets_by_request = {
        str(item.get("request_id", "") or ""): item
        for item in fixture_targets
        if isinstance(item, dict) and str(item.get("request_id", "") or "")
    }
    gap_report_path = ""
    collection_targets_path = ""
    if isinstance(input_paths, Mapping):
        gap_report_path = str(input_paths.get("gap_report", "") or "")
        collection_targets_path = str(input_paths.get("collection_targets", "") or "")
    blocked_request_id_set = {str(item) for item in (blocked_request_ids or []) if str(item)}
    clearable_blocked_request_id_set = {
        str(item) for item in (clearable_blocked_request_ids or []) if str(item)
    }
    request_isolation_audit = request_isolation_audit if isinstance(request_isolation_audit, Mapping) else {}
    global_isolation_audit = global_isolation_audit if isinstance(global_isolation_audit, Mapping) else {}
    actions = []
    for request_id in sorted(requested_slots_by_request):
        requested = _safe_int(requested_slots_by_request.get(request_id))
        built = _safe_int(built_by_request.get(request_id))
        skipped = _safe_int(skipped_by_request.get(request_id))
        if requested <= 0:
            continue
        missing_slots = max(requested - built, 0)
        if missing_slots <= 0 and skipped <= 0:
            continue
        target = targets_by_request.get(request_id, {})
        missing_material_types = [
            str(value)
            for value in (
                target.get("missing_material_types", [])
                if isinstance(target.get("missing_material_types", []), list)
                else []
            )
            if str(value)
        ]
        evaluation_gaps = [
            str(value)
            for value in (
                target.get("evaluation_gaps", [])
                if isinstance(target.get("evaluation_gaps", []), list)
                else []
            )
            if str(value)
        ]
        candidate_source_domains = [
            str(value)
            for value in (
                target.get("candidate_source_domains", [])
                if isinstance(target.get("candidate_source_domains", []), list)
                else []
            )
            if str(value)
        ]
        blocked_missing_axes = []
        if isinstance(blocked_request_missing_axes, Mapping):
            blocked_missing_axes = [
                str(value)
                for value in (
                    blocked_request_missing_axes.get(request_id, [])
                    if isinstance(blocked_request_missing_axes.get(request_id, []), list)
                    else []
                )
                if str(value)
            ]
        is_blocked = request_id in blocked_request_id_set
        is_clearable = request_id in clearable_blocked_request_id_set
        request_audit = request_isolation_audit.get(request_id, {})
        request_audit = request_audit if isinstance(request_audit, Mapping) else {}
        isolation_evidence = {
            "schema": "sara-phase7-repair-action-isolation-evidence-v1",
            "global": {
                "available": bool(global_isolation_audit.get("available", False)),
                "passed": global_isolation_audit.get("passed"),
                "missing_axes": [
                    str(axis)
                    for axis in global_isolation_audit.get("missing_axes", [])
                    if str(axis)
                ],
                "overlap_values": {
                    "shared_source_hashes": [
                        str(value)
                        for value in global_isolation_audit.get("shared_source_hashes", [])
                        if str(value)
                    ],
                    "shared_source_revisions": [
                        str(value)
                        for value in global_isolation_audit.get("shared_source_revisions", [])
                        if str(value)
                    ],
                    "shared_source_domains": [
                        str(value)
                        for value in global_isolation_audit.get("shared_source_domains", [])
                        if str(value)
                    ],
                    "near_duplicate_pairs": [
                        dict(value)
                        for value in global_isolation_audit.get("near_duplicate_pairs", [])
                        if isinstance(value, Mapping)
                    ],
                    "time_split_isolated": global_isolation_audit.get("time_split_isolated"),
                },
            },
            "request": {
                "row_count": _safe_int(request_audit.get("row_count")),
                "axis_status": dict(request_audit.get("axis_status", {}))
                if isinstance(request_audit.get("axis_status"), Mapping)
                else {},
                "missing_axes": [
                    str(axis) for axis in request_audit.get("missing_axes", []) if str(axis)
                ],
                "source_hash_coverage": _safe_float(request_audit.get("source_hash_coverage")),
                "source_revision_coverage": _safe_float(
                    request_audit.get("source_revision_coverage")
                ),
                "source_domain_candidate_count": _safe_int(
                    request_audit.get("candidate_source_domain_count")
                ),
                "collection_time_coverage": _safe_float(
                    request_audit.get("collection_time_coverage")
                ),
            },
        }
        overlap_values = isolation_evidence["global"]["overlap_values"]
        guidance_parts = [
            "Phase 7 isolation evidence",
            f"global={'PASS' if isolation_evidence['global']['passed'] is True else 'FAIL' if isolation_evidence['global']['available'] else 'UNAVAILABLE'}",
            "failed_axes="
            + (",".join(isolation_evidence["global"]["missing_axes"]) or "none"),
            "time_split="
            + (
                str(overlap_values["time_split_isolated"])
                if overlap_values["time_split_isolated"] is not None
                else "unknown"
            ),
        ]
        for label, key in (
            ("shared_hashes", "shared_source_hashes"),
            ("shared_revisions", "shared_source_revisions"),
            ("shared_domains", "shared_source_domains"),
        ):
            values = overlap_values[key]
            if values:
                guidance_parts.append(f"{label}={','.join(values)}")
        if overlap_values["near_duplicate_pairs"]:
            guidance_parts.append(
                f"near_duplicate_pairs={len(overlap_values['near_duplicate_pairs'])}"
            )
        operator_guidance = "; ".join(guidance_parts)
        command = ""
        if is_blocked and is_clearable and collection_targets_path:
            command = (
                f"Clear fixture isolation block for {request_id} "
                f"(missing_axes={','.join(blocked_missing_axes) or 'none'}) and rerun "
                "python bot/gap_materials_builder.py "
                f"--targets-path {json.dumps(collection_targets_path)} "
                f"--clear-blocked-request-id {json.dumps(request_id)}"
            )
            if gap_report_path:
                command += f" --report-path {json.dumps(gap_report_path)}"
        elif is_blocked:
            command = (
                f"Review fixture isolation block for {request_id} "
                f"(missing_axes={','.join(blocked_missing_axes) or 'none'}) before rerunning "
                "python bot/gap_materials_builder.py"
            )
        elif collection_targets_path:
            command = (
                f"Review fixture request {request_id} "
                f"(missing_types={','.join(missing_material_types) or 'none'}) and rerun "
                "python bot/gap_materials_builder.py "
                f"--targets-path {json.dumps(collection_targets_path)}"
            )
        if gap_report_path:
            command = (
                f"{command} --report-path {json.dumps(gap_report_path)}"
                if command
                else (
                    "python bot/gap_materials_builder.py "
                    f"--report-path {json.dumps(gap_report_path)}"
                )
            )
        isolation_audit_command = (
            "python scripts/sara_cli.py eval-phase7-isolation "
            "--report-path workspace/evaluation/phase7_isolation_audit.json "
            "--summary-path workspace/evaluation/phase7_isolation_audit_summary.txt"
        )
        isolation_policy_command = (
            "python scripts/sara_cli.py apply-phase7-isolation-block-policy "
            "--audit-path workspace/evaluation/phase7_isolation_audit.json "
            "--targets-path workspace/autobot/dataset_builder_collection_targets.json "
            "--report-path workspace/evaluation/phase7_isolation_block_policy.json"
        )
        repair_command = command or "python bot/gap_materials_builder.py"
        actions.append(
            {
                "request_id": request_id,
                "priority": "high" if missing_slots > 0 else "medium",
                "missing_slots": missing_slots,
                "skipped_slots": skipped,
                "missing_material_types": missing_material_types,
                "evaluation_gaps": evaluation_gaps,
                "candidate_source_domains": candidate_source_domains,
                "command": command,
                "reason": (
                    f"fixture_request={request_id}; missing_slots={missing_slots}; "
                    f"skipped_slots={skipped}; "
                    f"blocked_by_isolation_review={str(is_blocked).lower()}; "
                    f"clearable_after_review={str(is_clearable).lower()}"
                ),
                "blocked_by_isolation_review": is_blocked,
                "clearable_after_review": is_clearable,
                "blocked_missing_axes": blocked_missing_axes,
                "isolation_evidence": isolation_evidence,
                "operator_guidance": operator_guidance,
                "rerun_commands": {
                    "audit_all_axes": isolation_audit_command,
                    "reapply_block_policy": isolation_policy_command,
                    "repair_request": repair_command,
                    "failed_axes": list(isolation_evidence["global"]["missing_axes"]),
                    "axis_repair_requirements": {
                        axis: dict(ISOLATION_AXIS_REPAIR_REQUIREMENTS.get(axis, {}))
                        for axis in isolation_evidence["global"]["missing_axes"]
                        if axis in ISOLATION_AXIS_REPAIR_REQUIREMENTS
                    },
                },
                "affected_checks": [
                    "autobot_gap_loop_readiness",
                    "event_memory_ingest_pipeline",
                ],
            }
        )
    return actions


def _fixture_request_isolation_audit(
    *,
    fixture_targets: Sequence[Dict[str, Any]],
    fixture_gap_rows: Sequence[Dict[str, Any]],
    accepted_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    accepted_source_domains = {
        str(row.get("source_domain", "") or "local")
        for row in accepted_rows
        if isinstance(row, dict) and str(row.get("source_domain", "") or "local")
    }
    rows_by_request: Dict[str, list[Dict[str, Any]]] = {}
    for row in fixture_gap_rows:
        if not isinstance(row, dict):
            continue
        request_id = str(row.get("request_id", "") or "")
        if not request_id:
            continue
        rows_by_request.setdefault(request_id, []).append(row)

    audit: Dict[str, Dict[str, Any]] = {}
    for target in fixture_targets:
        if not isinstance(target, dict):
            continue
        request_id = str(target.get("request_id", "") or "")
        if not request_id:
            continue
        request_rows = rows_by_request.get(request_id, [])
        candidate_source_domains = {
            str(domain or "local")
            for domain in (
                target.get("candidate_source_domains", [])
                if isinstance(target.get("candidate_source_domains", []), list)
                else []
            )
            if str(domain or "local")
        }
        row_count = len(request_rows)
        lineage_ready_count = sum(
            1
            for row in request_rows
            if str(row.get("source_url", "") or "").strip()
            or str(row.get("source_path", "") or "").strip()
        )
        collection_time_ready_count = sum(
            1 for row in request_rows if str(row.get("collection_time", "") or "").strip()
        )
        source_hash_ready_count = sum(
            1 for row in request_rows if str(row.get("source_hash", "") or "").strip()
        )
        source_revision_ready_count = sum(
            1 for row in request_rows if str(row.get("source_revision", "") or "").strip()
        )
        lineage_coverage = 1.0 if row_count <= 0 else lineage_ready_count / float(row_count)
        collection_time_coverage = (
            1.0 if row_count <= 0 else collection_time_ready_count / float(row_count)
        )
        source_hash_coverage = 1.0 if row_count <= 0 else source_hash_ready_count / float(row_count)
        source_revision_coverage = (
            1.0 if row_count <= 0 else source_revision_ready_count / float(row_count)
        )
        axis_status = {
            "source_domain": bool(candidate_source_domains) or bool(accepted_source_domains),
            "source_lineage": row_count <= 0 or lineage_coverage >= 1.0,
            "collection_time": row_count <= 0 or collection_time_coverage >= 1.0,
            "source_hash": row_count <= 0 or source_hash_coverage >= 1.0,
            "source_revision": row_count <= 0 or source_revision_coverage >= 1.0,
        }
        missing_axes = sorted(key for key, ready in axis_status.items() if not bool(ready))
        audit[request_id] = {
            "row_count": row_count,
            "candidate_source_domain_count": len(candidate_source_domains),
            "accepted_source_domain_count": len(accepted_source_domains),
            "lineage_coverage": lineage_coverage,
            "collection_time_coverage": collection_time_coverage,
            "source_hash_coverage": source_hash_coverage,
            "source_revision_coverage": source_revision_coverage,
            "axis_status": axis_status,
            "missing_axes": missing_axes,
        }
    return dict(sorted(audit.items()))


def build_report(
    *,
    loop_report: Optional[Dict[str, Any]],
    dataset_report: Optional[Dict[str, Any]],
    gap_report: Optional[Dict[str, Any]],
    enqueue_report: Optional[Dict[str, Any]],
    collection_targets: Optional[Dict[str, Any]],
    accepted_rows: Sequence[Dict[str, Any]],
    gap_rows: Sequence[Dict[str, Any]],
    min_accepted_count: int,
    min_gap_build_coverage: float,
    input_paths: Optional[Mapping[str, str]] = None,
    isolation_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_count = 0 if not isinstance(collection_targets, dict) else _safe_int(collection_targets.get("target_count"))
    requested_slot_count = _count_requested_slots(collection_targets)
    accepted_count = 0 if not isinstance(dataset_report, dict) else _safe_int(dataset_report.get("accepted_count"))
    built_count = 0 if not isinstance(gap_report, dict) else _safe_int(gap_report.get("built_count"))
    skipped_count = 0 if not isinstance(gap_report, dict) else _safe_int(gap_report.get("skipped_count"))
    enqueued_count = 0 if not isinstance(enqueue_report, dict) else _safe_int(enqueue_report.get("enqueued_count"))
    queue_pending = 0 if not isinstance(enqueue_report, dict) else _safe_int(enqueue_report.get("queue_pending"))
    build_coverage = 1.0 if requested_slot_count <= 0 else built_count / float(requested_slot_count)
    enqueue_coverage = 1.0 if built_count <= 0 else enqueued_count / float(built_count)
    skipped_ratio = 0.0 if requested_slot_count <= 0 else skipped_count / float(requested_slot_count)
    gap_curriculum_distribution = {}
    if isinstance(gap_report, dict) and isinstance(gap_report.get("curriculum_distribution"), dict):
        gap_curriculum_distribution = dict(gap_report.get("curriculum_distribution", {}))
    repair_count = _safe_int(gap_curriculum_distribution.get("repair"))
    replay_count = _safe_int(gap_curriculum_distribution.get("replay"))
    total_curriculum = sum(_safe_int(value) for value in gap_curriculum_distribution.values())
    repair_share = 0.0 if total_curriculum <= 0 else repair_count / float(total_curriculum)
    replay_share = 0.0 if total_curriculum <= 0 else replay_count / float(total_curriculum)
    fixture_targets = _fixture_targets(collection_targets)
    fixture_request_count = len(fixture_targets)
    fixture_requested_slot_count = _count_fixture_requested_slots(collection_targets)
    fixture_requested_slots_by_request = {
        str(item.get("request_id", "") or ""): len(
            [value for value in item.get("missing_material_types", []) if str(value)]
        )
        for item in fixture_targets
        if str(item.get("request_id", "") or "")
    }
    fixture_gap_rows = [
        row
        for row in gap_rows
        if str(row.get("request_id", "") or "").startswith("fixture_")
    ]
    fixture_gap_built_count = len(fixture_gap_rows)
    fixture_built_by_request = Counter(
        str(row.get("request_id", "") or "")
        for row in fixture_gap_rows
        if str(row.get("request_id", "") or "")
    )
    gap_report_skipped = (
        gap_report.get("skipped", [])
        if isinstance(gap_report, dict) and isinstance(gap_report.get("skipped"), list)
        else []
    )
    fixture_skipped_by_request = Counter(
        str(row.get("request_id", "") or "")
        for row in gap_report_skipped
        if isinstance(row, dict) and str(row.get("request_id", "") or "").startswith("fixture_")
    )
    fixture_build_coverage = (
        1.0
        if fixture_requested_slot_count <= 0
        else fixture_gap_built_count / float(fixture_requested_slot_count)
    )
    fixture_source_domain_counter = Counter(
        str(row.get("source_domain", "") or "local") for row in fixture_gap_rows
    )
    fixture_source_domain_count = len(fixture_source_domain_counter)
    fixture_lineage_ready_count = sum(
        1
        for row in fixture_gap_rows
        if str(row.get("request_id", "") or "").startswith("fixture_")
        and (
            str(row.get("source_url", "") or "").strip()
            or str(row.get("source_path", "") or "").strip()
        )
    )
    fixture_lineage_coverage = (
        1.0 if fixture_gap_built_count <= 0 else fixture_lineage_ready_count / float(fixture_gap_built_count)
    )
    accepted_source_domains = {
        str(row.get("source_domain", "") or "local")
        for row in accepted_rows
        if isinstance(row, dict)
    }
    fixture_candidate_source_domains = {
        str(domain or "local")
        for item in fixture_targets
        for domain in (
            item.get("candidate_source_domains", [])
            if isinstance(item.get("candidate_source_domains", []), list)
            else []
        )
        if str(domain or "local")
    }
    fixture_collection_time_ready_count = sum(
        1 for row in fixture_gap_rows if str(row.get("collection_time", "") or "").strip()
    )
    fixture_collection_time_coverage = (
        1.0
        if fixture_gap_built_count <= 0
        else fixture_collection_time_ready_count / float(fixture_gap_built_count)
    )
    fixture_isolation_axis_status = {
        "source_domain": bool(fixture_candidate_source_domains) or bool(accepted_source_domains),
        "source_lineage": fixture_gap_built_count <= 0 or fixture_lineage_coverage >= 1.0,
        "collection_time": fixture_gap_built_count <= 0 or fixture_collection_time_coverage >= 1.0,
    }
    missing_isolation_axes = sorted(
        key for key, ready in fixture_isolation_axis_status.items() if not bool(ready)
    )
    fixture_request_isolation_audit = _fixture_request_isolation_audit(
        fixture_targets=fixture_targets,
        fixture_gap_rows=fixture_gap_rows,
        accepted_rows=accepted_rows,
    )
    isolation_audit_available = isinstance(isolation_audit, dict) and bool(isolation_audit)
    isolation_checks = (
        isolation_audit.get("checks", {})
        if isolation_audit_available and isinstance(isolation_audit.get("checks"), dict)
        else {}
    )
    isolation_missing_axes = sorted(
        str(axis) for axis, passed in isolation_checks.items() if passed is False and str(axis)
    )
    isolation_audit_passed = (
        bool(isolation_audit.get("passed", False)) if isolation_audit_available else None
    )
    blocked_request_ids = []
    blocked_request_missing_axes = {}
    if isinstance(collection_targets, dict):
        blocked_request_ids = [
            str(item)
            for item in (
                collection_targets.get("blocked_request_ids", [])
                if isinstance(collection_targets.get("blocked_request_ids"), list)
                else []
            )
            if str(item)
        ]
        blocked_request_missing_axes = (
            collection_targets.get("blocked_request_missing_axes", {})
            if isinstance(collection_targets.get("blocked_request_missing_axes"), dict)
            else {}
        )
    clearable_blocked_request_ids = sorted(
        request_id
        for request_id in blocked_request_ids
        if (
            (
                not [str(axis) for axis in blocked_request_missing_axes.get(request_id, []) if str(axis)]
                and (not isolation_audit_available or isolation_audit_passed is True)
            )
            or (
            isinstance(fixture_request_isolation_audit.get(request_id, {}), dict)
            and int(fixture_request_isolation_audit.get(request_id, {}).get("row_count", 0) or 0) > 0
            and not fixture_request_isolation_audit.get(request_id, {}).get("missing_axes", [])
            and (not isolation_audit_available or isolation_audit_passed is True)
            )
        )
    )
    checks = {
        "loop_report_present": _check(isinstance(loop_report, dict), bool(loop_report)),
        "dataset_report_present": _check(isinstance(dataset_report, dict), bool(dataset_report)),
        "gap_report_present": _check(isinstance(gap_report, dict), bool(gap_report)),
        "enqueue_report_present": _check(isinstance(enqueue_report, dict), bool(enqueue_report)),
        "collection_targets_present": _check(isinstance(collection_targets, dict), bool(collection_targets)),
        "loop_passed": _check(bool(loop_report and loop_report.get("passed")), bool(loop_report and loop_report.get("passed"))),
        "accepted_materials_ready": _check(
            accepted_count >= int(min_accepted_count),
            accepted_count,
            f"min_accepted_count={int(min_accepted_count)}",
        ),
        "target_generation_ready": _check(target_count >= 0, target_count),
        "gap_material_coverage_ready": _check(
            requested_slot_count <= 0 or build_coverage >= float(min_gap_build_coverage),
            round(build_coverage, 6),
            f"min_gap_build_coverage={float(min_gap_build_coverage):.3f}",
        ),
        "gap_enqueue_ready": _check(
            built_count <= 0 or enqueued_count > 0,
            enqueued_count,
            "gap materials should reach the managed training queue",
        ),
        "repair_curriculum_present": _check(
            built_count <= 0 or (repair_count + replay_count) > 0,
            {"repair": repair_count, "replay": replay_count},
        ),
        "fixture_lane_coverage_ready": _check(
            fixture_requested_slot_count <= 0 or fixture_build_coverage >= float(min_gap_build_coverage),
            round(fixture_build_coverage, 6),
            f"fixture_requested_slot_count={fixture_requested_slot_count}",
        ),
        "fixture_source_lineage_ready": _check(
            fixture_gap_built_count <= 0 or fixture_lineage_coverage >= 1.0,
            round(fixture_lineage_coverage, 6),
            "fixture rows should preserve request_id plus source_url/source_path lineage",
        ),
        "fixture_source_isolation_ready": _check(
            fixture_request_count <= 0
            or bool(fixture_candidate_source_domains)
            or bool(accepted_source_domains),
            {
                "candidate_source_domain_count": len(fixture_candidate_source_domains),
                "accepted_source_domain_count": len(accepted_source_domains),
            },
            "fixture repair lane should retain source-aware domain candidates",
        ),
        "fixture_collection_time_ready": _check(
            fixture_gap_built_count <= 0 or fixture_collection_time_coverage >= 1.0,
            round(fixture_collection_time_coverage, 6),
            "fixture rows should preserve collection_time for train/evaluation split audits",
        ),
    }
    passed = all(bool(item.get("passed")) for item in checks.values())
    fixture_repair_actions = _build_fixture_repair_actions(
        fixture_targets=fixture_targets,
        requested_slots_by_request=fixture_requested_slots_by_request,
        built_by_request=fixture_built_by_request,
        skipped_by_request=fixture_skipped_by_request,
        input_paths=input_paths,
        blocked_request_ids=blocked_request_ids,
        blocked_request_missing_axes=blocked_request_missing_axes,
        clearable_blocked_request_ids=clearable_blocked_request_ids,
        request_isolation_audit=fixture_request_isolation_audit,
        global_isolation_audit={
            "available": isolation_audit_available,
            "passed": isolation_audit_passed,
            "missing_axes": isolation_missing_axes,
            "shared_source_hashes": (
                isolation_audit.get("metrics", {}).get("shared_source_hashes", [])
                if isolation_audit_available and isinstance(isolation_audit.get("metrics"), Mapping)
                else []
            ),
            "shared_source_revisions": (
                isolation_audit.get("metrics", {}).get("shared_source_revisions", [])
                if isolation_audit_available and isinstance(isolation_audit.get("metrics"), Mapping)
                else []
            ),
            "shared_source_domains": (
                isolation_audit.get("metrics", {}).get("shared_source_domains", [])
                if isolation_audit_available and isinstance(isolation_audit.get("metrics"), Mapping)
                else []
            ),
            "near_duplicate_pairs": (
                isolation_audit.get("metrics", {}).get("near_duplicate_pairs", [])
                if isolation_audit_available and isinstance(isolation_audit.get("metrics"), Mapping)
                else []
            ),
            "time_split_isolated": (
                isolation_checks.get("time_split_isolated") if isolation_audit_available else None
            ),
        },
    )
    return {
        "schema": "sara-autobot-gap-loop-readiness-v1",
        "passed": passed,
        "metrics": {
            "accepted_count": accepted_count,
            "collection_target_count": target_count,
            "requested_slot_count": requested_slot_count,
            "gap_material_built_count": built_count,
            "gap_material_skipped_count": skipped_count,
            "gap_curriculum_enqueued_count": enqueued_count,
            "queue_pending": queue_pending,
            "gap_build_coverage": build_coverage,
            "gap_enqueue_coverage": enqueue_coverage,
            "gap_skip_ratio": skipped_ratio,
            "repair_curriculum_share": repair_share,
            "replay_curriculum_share": replay_share,
            "fixture_request_count": fixture_request_count,
            "fixture_requested_slot_count": fixture_requested_slot_count,
            "fixture_gap_material_built_count": fixture_gap_built_count,
            "fixture_gap_build_coverage": fixture_build_coverage,
            "fixture_source_domain_count": fixture_source_domain_count,
            "fixture_source_lineage_coverage": fixture_lineage_coverage,
            "fixture_candidate_source_domain_count": len(fixture_candidate_source_domains),
            "fixture_accepted_source_domain_count": len(accepted_source_domains),
            "fixture_collection_time_coverage": fixture_collection_time_coverage,
        },
        "fixture_isolation_audit": {
            "axis_status": fixture_isolation_axis_status,
            "missing_axes": missing_isolation_axes,
        },
        "phase7_global_isolation_audit": {
            "available": isolation_audit_available,
            "passed": isolation_audit_passed,
            "missing_axes": isolation_missing_axes,
        },
        "fixture_request_isolation_audit": fixture_request_isolation_audit,
        "fixture_lane": {
            "requested_slots_by_request": dict(sorted(fixture_requested_slots_by_request.items())),
            "built_by_request": dict(sorted(fixture_built_by_request.items())),
            "skipped_by_request": dict(sorted(fixture_skipped_by_request.items())),
        },
        "fixture_execution_policy": {
            "blocked_request_count": len(blocked_request_ids),
            "blocked_request_ids": sorted(blocked_request_ids),
            "blocked_request_missing_axes": {
                str(request_id): [
                    str(axis)
                    for axis in (
                        axes if isinstance(axes, list) else []
                    )
                    if str(axis)
                ]
                for request_id, axes in blocked_request_missing_axes.items()
                if str(request_id)
            },
            "clearable_blocked_request_ids": clearable_blocked_request_ids,
        },
        "fixture_repair_actions": list(fixture_repair_actions),
        "checks": checks,
        "input_paths": {},
        "policy_notes": [
            "Readiness does not claim benchmark quality gains by itself; it only verifies that source-backed gap requests become managed repair or replay curriculum.",
            "Requested-slot coverage counts missing material slots, not abstract request objects, so counterexample and transcript needs remain separately visible.",
            "This report is Phase 7 evidence about autonomous data preparation, not Phase 6 physical energy evidence or Phase 8 ANN baseline evidence.",
        ],
    }


def summarize_report(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    fixture_lane = report.get("fixture_lane", {}) if isinstance(report.get("fixture_lane"), dict) else {}
    fixture_isolation_audit = (
        report.get("fixture_isolation_audit", {})
        if isinstance(report.get("fixture_isolation_audit"), dict)
        else {}
    )
    fixture_request_isolation_audit = (
        report.get("fixture_request_isolation_audit", {})
        if isinstance(report.get("fixture_request_isolation_audit"), dict)
        else {}
    )
    fixture_execution_policy = (
        report.get("fixture_execution_policy", {})
        if isinstance(report.get("fixture_execution_policy"), dict)
        else {}
    )
    global_isolation = (
        report.get("phase7_global_isolation_audit", {})
        if isinstance(report.get("phase7_global_isolation_audit"), dict)
        else {}
    )
    lines = [
        f"Autobot gap loop readiness: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Accepted materials: {metrics.get('accepted_count')}",
        f"Collection targets: {metrics.get('collection_target_count')}",
        f"Requested slots: {metrics.get('requested_slot_count')}",
        f"Gap materials built: {metrics.get('gap_material_built_count')}",
        f"Gap materials skipped: {metrics.get('gap_material_skipped_count')}",
        f"Gap curriculum enqueued: {metrics.get('gap_curriculum_enqueued_count')}",
        f"Gap build coverage: {float(metrics.get('gap_build_coverage', 0.0) or 0.0):.3f}",
        f"Gap enqueue coverage: {float(metrics.get('gap_enqueue_coverage', 0.0) or 0.0):.3f}",
        f"Fixture requests: {metrics.get('fixture_request_count')}",
        f"Fixture requested slots: {metrics.get('fixture_requested_slot_count')}",
        f"Fixture gap materials built: {metrics.get('fixture_gap_material_built_count')}",
        f"Fixture build coverage: {float(metrics.get('fixture_gap_build_coverage', 0.0) or 0.0):.3f}",
        f"Fixture source domain count: {metrics.get('fixture_source_domain_count')}",
        f"Fixture candidate source domain count: {metrics.get('fixture_candidate_source_domain_count')}",
        f"Fixture accepted source domain count: {metrics.get('fixture_accepted_source_domain_count')}",
        f"Fixture lineage coverage: {float(metrics.get('fixture_source_lineage_coverage', 0.0) or 0.0):.3f}",
        f"Fixture collection-time coverage: {float(metrics.get('fixture_collection_time_coverage', 0.0) or 0.0):.3f}",
        "Phase 7 global isolation audit: "
        + (
            "unavailable"
            if not bool(global_isolation.get("available", False))
            else "PASS" if bool(global_isolation.get("passed", False)) else "FAIL"
        ),
        "Checks:",
    ]
    requested_slots_by_request = (
        fixture_lane.get("requested_slots_by_request", {})
        if isinstance(fixture_lane.get("requested_slots_by_request"), dict)
        else {}
    )
    built_by_request = (
        fixture_lane.get("built_by_request", {})
        if isinstance(fixture_lane.get("built_by_request"), dict)
        else {}
    )
    skipped_by_request = (
        fixture_lane.get("skipped_by_request", {})
        if isinstance(fixture_lane.get("skipped_by_request"), dict)
        else {}
    )
    fixture_repair_actions = (
        report.get("fixture_repair_actions", [])
        if isinstance(report.get("fixture_repair_actions"), list)
        else []
    )
    if requested_slots_by_request:
        lines.append("Fixture lane by request:")
        for request_id in sorted(requested_slots_by_request):
            lines.append(
                "- "
                f"{request_id}: "
                f"requested_slots={int(requested_slots_by_request.get(request_id, 0) or 0)}, "
                f"built={int(built_by_request.get(request_id, 0) or 0)}, "
                f"skipped={int(skipped_by_request.get(request_id, 0) or 0)}"
            )
    if fixture_repair_actions:
        lines.append("Fixture repair actions:")
        for action in fixture_repair_actions:
            if not isinstance(action, dict):
                continue
            lines.append(
                "- "
                f"{str(action.get('request_id', '') or '')}: "
                f"missing_slots={int(action.get('missing_slots', 0) or 0)}, "
                f"skipped_slots={int(action.get('skipped_slots', 0) or 0)}, "
                f"missing_types={','.join(str(item) for item in action.get('missing_material_types', []) if str(item)) or 'none'}, "
                f"blocked={bool(action.get('blocked_by_isolation_review', False))}"
            )
    blocked_request_ids = (
        fixture_execution_policy.get("blocked_request_ids", [])
        if isinstance(fixture_execution_policy.get("blocked_request_ids"), list)
        else []
    )
    lines.append(
        "Fixture execution blocked requests: "
        + (",".join(str(item) for item in blocked_request_ids if str(item)) or "none")
    )
    clearable_blocked_request_ids = (
        fixture_execution_policy.get("clearable_blocked_request_ids", [])
        if isinstance(fixture_execution_policy.get("clearable_blocked_request_ids"), list)
        else []
    )
    lines.append(
        "Fixture execution clearable blocked requests: "
        + (",".join(str(item) for item in clearable_blocked_request_ids if str(item)) or "none")
    )
    axis_status = (
        fixture_isolation_audit.get("axis_status", {})
        if isinstance(fixture_isolation_audit.get("axis_status"), dict)
        else {}
    )
    missing_axes = (
        fixture_isolation_audit.get("missing_axes", [])
        if isinstance(fixture_isolation_audit.get("missing_axes"), list)
        else []
    )
    if axis_status:
        lines.append("Fixture isolation axes:")
        for axis_name, ready in sorted(axis_status.items()):
            lines.append(f"- {axis_name}: {bool(ready)}")
        lines.append(
            f"Fixture isolation missing axes: {','.join(str(item) for item in missing_axes if str(item)) or 'none'}"
        )
    if fixture_request_isolation_audit:
        lines.append("Fixture isolation by request:")
        for request_id, payload in sorted(fixture_request_isolation_audit.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                "- "
                f"{request_id}: "
                f"row_count={int(payload.get('row_count', 0) or 0)}, "
                f"lineage={float(payload.get('lineage_coverage', 0.0) or 0.0):.3f}, "
                f"collection_time={float(payload.get('collection_time_coverage', 0.0) or 0.0):.3f}, "
                f"missing_axes={','.join(str(item) for item in payload.get('missing_axes', []) if str(item)) or 'none'}"
            )
    for name, payload in sorted(checks.items()):
        if not isinstance(payload, dict):
            continue
        lines.append(f"- {name}: {bool(payload.get('passed'))} ({payload.get('value')})")
    return "\n".join(lines) + "\n"


def run_readiness(
    *,
    loop_report_path: str = DEFAULT_LOOP_REPORT_PATH,
    collection_targets_path: str = DEFAULT_COLLECTION_TARGETS_PATH,
    dataset_report_path: str = "",
    gap_report_path: str = "",
    enqueue_report_path: str = "",
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    min_accepted_count: int = 4,
    min_gap_build_coverage: float = 0.5,
    isolation_audit_path: str = DEFAULT_ISOLATION_AUDIT_PATH,
) -> Dict[str, Any]:
    loop_report = read_json(loop_report_path)
    if not dataset_report_path:
        dataset_report_path = _resolve_output_path(loop_report, "dataset_report")
    if not gap_report_path:
        gap_report_path = _resolve_output_path(loop_report, "gap_report")
    if not enqueue_report_path:
        enqueue_report_path = _resolve_output_path(loop_report, "enqueue_report")
    dataset_report = read_json(dataset_report_path)
    gap_report = read_json(gap_report_path)
    enqueue_report = read_json(enqueue_report_path)
    collection_targets = read_json(collection_targets_path)
    isolation_audit = read_json(isolation_audit_path)
    accepted_rows = read_jsonl(_resolve_output_path(loop_report, "accepted_materials"))
    gap_rows = read_jsonl(_resolve_output_path(loop_report, "gap_materials"))
    input_paths = {
        "loop_report": os.path.abspath(loop_report_path),
        "collection_targets": os.path.abspath(collection_targets_path),
        "dataset_report": os.path.abspath(dataset_report_path) if dataset_report_path else "",
        "gap_report": os.path.abspath(gap_report_path) if gap_report_path else "",
        "enqueue_report": os.path.abspath(enqueue_report_path) if enqueue_report_path else "",
    }
    report = build_report(
        loop_report=loop_report,
        dataset_report=dataset_report,
        gap_report=gap_report,
        enqueue_report=enqueue_report,
        collection_targets=collection_targets,
        accepted_rows=accepted_rows,
        gap_rows=gap_rows,
        min_accepted_count=min_accepted_count,
        min_gap_build_coverage=min_gap_build_coverage,
        input_paths=input_paths,
        isolation_audit=isolation_audit,
    )
    report["input_paths"] = input_paths
    report["report_path"] = write_json(report_path, report)
    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    report["summary_path"] = resolved_summary_path
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate managed autobot gap-loop readiness.")
    parser.add_argument("--loop-report-path", default=DEFAULT_LOOP_REPORT_PATH)
    parser.add_argument("--collection-targets-path", default=DEFAULT_COLLECTION_TARGETS_PATH)
    parser.add_argument("--dataset-report-path", default="")
    parser.add_argument("--gap-report-path", default="")
    parser.add_argument("--enqueue-report-path", default="")
    parser.add_argument("--isolation-audit-path", default=DEFAULT_ISOLATION_AUDIT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--min-accepted-count", type=int, default=4)
    parser.add_argument("--min-gap-build-coverage", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_readiness(
        loop_report_path=args.loop_report_path,
        collection_targets_path=args.collection_targets_path,
        dataset_report_path=args.dataset_report_path,
        gap_report_path=args.gap_report_path,
        enqueue_report_path=args.enqueue_report_path,
        isolation_audit_path=args.isolation_audit_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        min_accepted_count=args.min_accepted_count,
        min_gap_build_coverage=args.min_gap_build_coverage,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "accepted_count": report["metrics"]["accepted_count"],
                "gap_material_built_count": report["metrics"]["gap_material_built_count"],
                "gap_curriculum_enqueued_count": report["metrics"]["gap_curriculum_enqueued_count"],
                "gap_build_coverage": report["metrics"]["gap_build_coverage"],
                "report_path": report["report_path"],
                "summary_path": report["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
