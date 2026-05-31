# Directory Path: scripts/eval/release_soak.py
# English Title: Release Soak Runner
# Purpose/Content: Runs a lightweight wall-clock soak test for agent dialogue and inference memory loops, then saves a managed report for release validation.

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
UTILS_PATH = os.path.join(PROJECT_ROOT, "scripts", "utils")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

# Keep optional plotting/font caches inside the managed workspace to avoid
# noisy warnings on restricted environments during release validation.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.evaluation.phase3_tracking import (
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
    compact_neuromorphic_profile_trend,
    extract_cognitive_delta_memory_metrics,
    extract_cognitive_linear_snn_fusion_metrics,
    extract_cognitive_manifold_trace_metrics,
    extract_cognitive_plastic_submodel_metrics,
)
from sara_engine.evaluation.stage_d_contract import (
    STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT,
    STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS,
)
from sara_engine.evaluation.stage_e_contract import STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path
from memory_health import inspect_inference_memory
from phase3_accuracy_suite import run_phase3_accuracy_suite
from release_gate import (
    collect_release_gate_artifacts,
    validate_packaging_metadata,
    validate_release_report,
)
from research_automation_benchmark import (
    build_research_review_report,
    compact_research_review_report,
    summarize_completed_roadmap_patch_evidence_review,
)


SOAK_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "duration_seconds": 1.0,
        "max_agent_turns": 8,
        "min_agent_turns": 4,
        "max_inference_iterations": 12,
        "min_inference_iterations": 6,
        "shipping_ready": False,
    },
    "release": {
        "duration_seconds": 5.0,
        "max_agent_turns": 120,
        "min_agent_turns": 24,
        "max_inference_iterations": 256,
        "min_inference_iterations": 32,
        "shipping_ready": False,
    },
    "extended": {
        "duration_seconds": 30.0,
        "max_agent_turns": 360,
        "min_agent_turns": 60,
        "max_inference_iterations": 768,
        "min_inference_iterations": 96,
        "shipping_ready": True,
    },
}


STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES = (
    "synaptic_tag_integrity_observed",
    "synaptic_tag_importance_score_observed",
    "synaptic_tag_replay_priority_observed",
    "synaptic_tag_pruning_candidate_observed",
    "synaptic_tag_state_budget_observed",
    "memory_phase_transition_integrity_observed",
    "memory_phase_retention_protection_observed",
    "memory_phase_plasticity_guard_observed",
    "memory_phase_overfixation_guard_observed",
    "memory_phase_state_budget_observed",
    "metabolic_budget_integrity_observed",
    "plasticity_reserve_integrity_observed",
    "structural_growth_bounded_observed",
    "pruning_reason_trace_observed",
    "resource_pressure_observed",
    "sleep_consolidation_retention_observed",
    "latent_replay_noise_resilience_observed",
    "sleep_consolidation_memory_health_observed",
    "latent_replay_counterfactual_branch_observed",
    "sleep_consolidation_energy_budget_observed",
    "astro_structural_unlock_observed",
    "astro_structural_lock_observed",
    "astro_bounded_stdp_fallback_observed",
    "world_model_replay_policy_trace_observed",
    "astro_policy_state_budget_observed",
    "delta_memory_phase_retention_policy_observed",
    "delta_memory_crystal_retention_observed",
    "delta_memory_liquid_forget_observed",
    "delta_memory_astro_gate_alignment_observed",
    "delta_memory_policy_state_budget_observed",
    "delta_memory_multi_history_recall_observed",
    "delta_memory_multi_history_noise_resilience_observed",
    "delta_memory_multi_history_health_observed",
    "delta_memory_multi_history_manifold_guard_observed",
    "delta_memory_erase_write_decoupling_observed",
    "delta_memory_erase_preserves_stable_memory_observed",
    "delta_memory_write_commits_residual_observed",
)

DEFAULT_REPAIR_LOG_PATH = workspace_path("release", "release_repair_execution_log.json")


def _stage_d_candidate_failure_description(failure: Dict[str, Any]) -> str:
    description = str(failure.get("description", "") or "").strip()
    if description:
        return description
    check_name = str(failure.get("check", "") or "").strip()
    metric_name = str(failure.get("metric", "") or "").strip()
    if not check_name and metric_name:
        check_name = f"metric.{metric_name}"
    return str(
        STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS.get(
            check_name,
            STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS.get(check_name, ""),
        )
        or ""
    )


def _stage_e_observed_candidate_failure_description(failure: Dict[str, Any]) -> str:
    description = str(failure.get("description", "") or "").strip()
    if description:
        return description
    check_name = str(failure.get("check", "") or "").strip()
    metric_name = str(failure.get("metric", "") or "").strip()
    if not check_name and metric_name:
        check_name = f"metric.{metric_name}"
    return str(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS.get(check_name, "") or "")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_repair_execution_log(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, dict)]
    return []


def save_repair_execution_log(path: str, entries: List[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    return resolved


def parse_repair_checks_csv(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]


def append_repair_execution_entry(
    entries: List[Dict[str, Any]],
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    title: str = "",
    source: str = "manual",
) -> bool:
    cmd = str(command).strip()
    state = str(status).strip().lower()
    checks = sorted({str(item) for item in (covered_checks or []) if str(item).strip()})
    if not cmd or not state:
        return False
    if state in {"success", "failed", "skipped"}:
        finalized = finalize_pending_repair_entries(
            entries,
            command=cmd,
            status=state,
            covered_checks=checks,
            title=title,
            source=source,
        )
        if finalized > 0:
            return True
    entry = {
        "command": cmd,
        "status": state,
        "covered_checks": checks,
        "title": str(title).strip(),
        "source": str(source).strip() or "manual",
        "timestamp": time.time(),
    }
    entries.append(entry)
    return True


def finalize_pending_repair_entries(
    entries: List[Dict[str, Any]],
    *,
    command: str,
    status: str,
    covered_checks: Optional[List[str]] = None,
    title: str = "",
    source: str = "manual_completion",
) -> int:
    cmd = str(command).strip()
    state = str(status).strip().lower()
    if not cmd or state not in {"success", "failed", "skipped"}:
        return 0
    checks = sorted({str(item) for item in (covered_checks or []) if str(item).strip()})
    updated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("command", "")).strip() != cmd:
            continue
        if str(entry.get("status", "")).strip().lower() != "pending":
            continue
        previous_checks = (
            {str(item) for item in entry.get("covered_checks", []) if str(item).strip()}
            if isinstance(entry.get("covered_checks"), list)
            else set()
        )
        entry["status"] = state
        entry["covered_checks"] = sorted(previous_checks.union(set(checks)))
        if str(title).strip():
            entry["title"] = str(title).strip()
        entry["source"] = str(source).strip() or "manual_completion"
        entry["resolved_timestamp"] = time.time()
        updated += 1
    return updated


def expire_pending_repair_entries(
    entries: List[Dict[str, Any]],
    *,
    ttl_seconds: float,
    now_timestamp: Optional[float] = None,
    source: str = "pending_ttl_timeout",
) -> int:
    if ttl_seconds <= 0:
        return 0
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()
    expired = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "")).strip().lower() != "pending":
            continue
        created_raw = entry.get("timestamp", 0.0)
        try:
            created = float(created_raw)
        except (TypeError, ValueError):
            created = 0.0
        age = max(now - created, 0.0)
        if age < float(ttl_seconds):
            continue
        entry["status"] = "timeout"
        entry["source"] = str(source).strip() or "pending_ttl_timeout"
        entry["resolved_timestamp"] = now
        entry["timeout_after_seconds"] = float(ttl_seconds)
        expired += 1
    return expired


def _repair_entry_event_timestamp(entry: Dict[str, Any]) -> float:
    raw = entry.get("resolved_timestamp", entry.get("timestamp", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _group_repair_entries_by_command(
    entries: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_command: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command", "")).strip()
        if not command:
            continue
        by_command.setdefault(command, []).append(entry)
    return by_command


def build_retry_queue_from_repair_log(
    entries: List[Dict[str, Any]],
    *,
    max_attempts: int = 2,
    cooldown_seconds: float = 0.0,
    now_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if max_attempts < 1:
        max_attempts = 1
    if cooldown_seconds < 0:
        cooldown_seconds = 0.0
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()

    by_command = _group_repair_entries_by_command(entries)

    queue: List[Dict[str, Any]] = []
    for command, command_entries in by_command.items():
        sorted_entries = sorted(
            command_entries,
            key=_repair_entry_event_timestamp,
        )
        latest = sorted_entries[-1]
        latest_status = str(latest.get("status", "")).strip().lower()
        if latest_status not in {"failed", "timeout"}:
            continue
        attempts = sum(
            1
            for item in sorted_entries
            if str(item.get("status", "")).strip().lower() in {"failed", "timeout"}
        )
        if attempts >= max_attempts:
            continue
        latest_timestamp = _repair_entry_event_timestamp(latest)
        elapsed_since_latest = max(now - latest_timestamp, 0.0) if latest_timestamp > 0 else cooldown_seconds
        if cooldown_seconds > 0 and latest_timestamp > 0 and elapsed_since_latest < cooldown_seconds:
            continue
        covered_checks = (
            [str(item) for item in latest.get("covered_checks", []) if str(item).strip()]
            if isinstance(latest.get("covered_checks"), list)
            else []
        )
        queue.append(
            {
                "command": command,
                "title": str(latest.get("title", "")).strip(),
                "reason": latest_status,
                "covered_checks": sorted(set(covered_checks)),
                "attempts_used": int(attempts),
                "max_attempts": int(max_attempts),
                "next_attempt": int(attempts + 1),
                "last_attempt_timestamp": float(latest_timestamp),
            }
        )

    queue.sort(key=lambda item: (str(item.get("reason", "")), str(item.get("command", ""))))
    return queue


def build_retry_cooldown_blocked_from_repair_log(
    entries: List[Dict[str, Any]],
    *,
    max_attempts: int = 2,
    cooldown_seconds: float = 0.0,
    now_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if max_attempts < 1:
        max_attempts = 1
    if cooldown_seconds <= 0:
        return []
    now = float(now_timestamp) if isinstance(now_timestamp, (int, float)) else time.time()

    by_command = _group_repair_entries_by_command(entries)
    blocked: List[Dict[str, Any]] = []

    for command, command_entries in by_command.items():
        sorted_entries = sorted(
            command_entries,
            key=_repair_entry_event_timestamp,
        )
        latest = sorted_entries[-1]
        latest_status = str(latest.get("status", "")).strip().lower()
        if latest_status not in {"failed", "timeout"}:
            continue
        attempts = sum(
            1
            for item in sorted_entries
            if str(item.get("status", "")).strip().lower() in {"failed", "timeout"}
        )
        if attempts >= max_attempts:
            continue

        latest_timestamp = _repair_entry_event_timestamp(latest)
        if latest_timestamp <= 0:
            continue
        elapsed_since_latest = max(now - latest_timestamp, 0.0)
        if elapsed_since_latest >= cooldown_seconds:
            continue

        covered_checks = (
            [str(item) for item in latest.get("covered_checks", []) if str(item).strip()]
            if isinstance(latest.get("covered_checks"), list)
            else []
        )
        remaining = max(float(cooldown_seconds) - elapsed_since_latest, 0.0)
        blocked.append(
            {
                "command": command,
                "title": str(latest.get("title", "")).strip(),
                "reason": latest_status,
                "covered_checks": sorted(set(covered_checks)),
                "attempts_used": int(attempts),
                "max_attempts": int(max_attempts),
                "next_attempt": int(attempts + 1),
                "last_attempt_timestamp": float(latest_timestamp),
                "cooldown_remaining_seconds": float(remaining),
            }
        )

    blocked.sort(key=lambda item: (str(item.get("reason", "")), str(item.get("command", ""))))
    return blocked


def prioritize_retry_queue(
    retry_queue: List[Dict[str, Any]],
    *,
    iterative_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not retry_queue:
        return []

    remaining_checks = set()
    if isinstance(iterative_plan, dict):
        values = iterative_plan.get("remaining_checks", [])
        if isinstance(values, list):
            remaining_checks = {
                str(item) for item in values if str(item).strip()
            }

    scored: List[Dict[str, Any]] = []
    reason_base = {"timeout": 3.0, "failed": 2.0}
    for item in retry_queue:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        reason = str(payload.get("reason", "")).strip().lower()
        checks = payload.get("covered_checks", [])
        checks_set = (
            {str(value) for value in checks if str(value).strip()}
            if isinstance(checks, list)
            else set()
        )
        overlap = len(checks_set.intersection(remaining_checks))

        attempts_used = int(payload.get("attempts_used", 0) or 0)
        max_attempts = max(int(payload.get("max_attempts", 1) or 1), 1)
        pressure = float(attempts_used) / float(max_attempts)
        score = reason_base.get(reason, 1.0) + float(overlap) * 2.0 + pressure

        if score >= 5.0:
            tier = "high"
        elif score >= 3.5:
            tier = "medium"
        else:
            tier = "low"

        payload["priority_score"] = round(float(score), 3)
        payload["priority_tier"] = tier
        payload["priority_overlap_checks"] = int(overlap)
        scored.append(payload)

    scored.sort(
        key=lambda value: (
            -float(value.get("priority_score", 0.0) or 0.0),
            str(value.get("command", "")),
        )
    )
    return scored


def select_retry_dispatch_batch(
    retry_queue: List[Dict[str, Any]],
    *,
    max_dispatch: int,
    min_priority_tier: str = "low",
    diversify_checks: bool = False,
    max_per_check: int = 0,
) -> Dict[str, Any]:
    priority_rank = {"low": 0, "medium": 1, "high": 2}
    normalized_tier = str(min_priority_tier).strip().lower()
    if normalized_tier not in priority_rank:
        normalized_tier = "low"
    threshold = priority_rank[normalized_tier]
    per_check_limit = int(max_per_check) if isinstance(max_per_check, int) else 0
    if per_check_limit < 0:
        per_check_limit = 0

    allowed: List[Dict[str, Any]] = []
    skipped_low_priority_commands: List[str] = []
    for item in retry_queue:
        if not isinstance(item, dict):
            continue
        tier = str(item.get("priority_tier", "low")).strip().lower()
        if priority_rank.get(tier, 0) < threshold:
            command = str(item.get("command", "")).strip()
            if command:
                skipped_low_priority_commands.append(command)
            continue
        allowed.append(item)

    selection_budget = max(int(max_dispatch), 0)
    selected: List[Dict[str, Any]] = []
    skipped_check_quota_commands: List[str] = []
    check_counts: Dict[str, int] = {}

    def _covered_checks(payload: Dict[str, Any]) -> List[str]:
        checks = payload.get("covered_checks", [])
        if not isinstance(checks, list):
            return []
        return sorted({str(value) for value in checks if str(value).strip()})

    def _violates_quota(checks: List[str]) -> bool:
        if per_check_limit <= 0:
            return False
        for check in checks:
            if int(check_counts.get(check, 0)) >= per_check_limit:
                return True
        return False

    def _apply_quota(checks: List[str]) -> None:
        if per_check_limit <= 0:
            return
        for check in checks:
            check_counts[check] = int(check_counts.get(check, 0)) + 1

    if not diversify_checks:
        for item in allowed:
            if len(selected) >= selection_budget:
                break
            checks = _covered_checks(item)
            if _violates_quota(checks):
                command = str(item.get("command", "")).strip()
                if command:
                    skipped_check_quota_commands.append(command)
                continue
            selected.append(item)
            _apply_quota(checks)
    else:
        remaining = list(allowed)
        selected_checks: set[str] = set()
        while remaining and len(selected) < selection_budget:
            best_index = -1
            best_gain = -1
            for index, item in enumerate(remaining):
                checks_set = set(_covered_checks(item))
                if _violates_quota(sorted(checks_set)):
                    continue
                gain = len(checks_set.difference(selected_checks))
                if gain > best_gain:
                    best_gain = gain
                    best_index = index
            if best_index < 0:
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            chosen_checks = _covered_checks(chosen)
            selected_checks.update(set(chosen_checks))
            _apply_quota(chosen_checks)
        if per_check_limit > 0:
            selected_commands = {
                str(item.get("command", "")).strip()
                for item in selected
                if isinstance(item, dict) and str(item.get("command", "")).strip()
            }
            for item in allowed:
                command = str(item.get("command", "")).strip()
                if not command or command in selected_commands:
                    continue
                checks = _covered_checks(item)
                if _violates_quota(checks):
                    skipped_check_quota_commands.append(command)
    selected_unique_checks = set()
    for item in selected:
        selected_unique_checks.update(set(_covered_checks(item)))
    return {
        "min_priority_tier": normalized_tier,
        "selection_mode": "priority_diversified" if diversify_checks else "priority",
        "max_per_check": int(per_check_limit),
        "eligible_count": int(len(allowed)),
        "selected": selected,
        "selected_count": int(len(selected)),
        "selected_unique_check_count": int(len(selected_unique_checks)),
        "skipped_low_priority_commands": skipped_low_priority_commands,
        "skipped_low_priority_count": int(len(skipped_low_priority_commands)),
        "skipped_check_quota_commands": skipped_check_quota_commands,
        "skipped_check_quota_count": int(len(skipped_check_quota_commands)),
    }


def dispatch_retry_queue_to_pending(
    entries: List[Dict[str, Any]],
    retry_queue: List[Dict[str, Any]],
    *,
    max_dispatch: int = 1,
) -> int:
    report = dispatch_retry_queue_to_pending_with_report(
        entries,
        retry_queue,
        max_dispatch=max_dispatch,
    )
    return int(report.get("dispatched", 0) or 0)


def dispatch_retry_queue_to_pending_with_report(
    entries: List[Dict[str, Any]],
    retry_queue: List[Dict[str, Any]],
    *,
    max_dispatch: int = 1,
) -> Dict[str, Any]:
    if max_dispatch < 0:
        max_dispatch = 0
    dispatched_commands: List[str] = []
    skipped_pending_commands: List[str] = []
    skipped_limit_commands: List[str] = []
    candidate_count = 0
    if max_dispatch < 1:
        return {
            "requested": int(max_dispatch),
            "candidate_count": 0,
            "dispatched": 0,
            "dispatched_commands": [],
            "skipped_pending_commands": [],
            "skipped_limit_commands": [],
        }
    existing_pending = {
        str(item.get("command", "")).strip()
        for item in entries
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "pending"
        and str(item.get("command", "")).strip()
    }
    dispatched = 0
    for retry in retry_queue:
        if not isinstance(retry, dict):
            continue
        command = str(retry.get("command", "")).strip()
        if not command:
            continue
        candidate_count += 1
        if command in existing_pending:
            skipped_pending_commands.append(command)
            continue
        if dispatched >= max_dispatch:
            skipped_limit_commands.append(command)
            continue
        append_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=(
                [str(item) for item in retry.get("covered_checks", []) if str(item).strip()]
                if isinstance(retry.get("covered_checks"), list)
                else []
            ),
            title=str(retry.get("title", "")).strip() or "retry_queue_dispatch",
            source="retry_queue_dispatch",
        )
        existing_pending.add(command)
        dispatched_commands.append(command)
        dispatched += 1
    return {
        "requested": int(max_dispatch),
        "candidate_count": int(candidate_count),
        "dispatched": int(dispatched),
        "dispatched_commands": dispatched_commands,
        "skipped_pending_commands": skipped_pending_commands,
        "skipped_limit_commands": skipped_limit_commands,
    }


def append_iterative_next_actions_to_repair_log(
    entries: List[Dict[str, Any]],
    iterative_plan: Dict[str, Any],
) -> int:
    next_actions = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    if not next_actions:
        return 0
    existing_pending = {
        str(item.get("command", "")).strip()
        for item in entries
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "pending"
        and str(item.get("source", "")).strip() == "iterative_next_action"
        and str(item.get("command", "")).strip()
    }
    appended = 0
    for action in next_actions:
        if not isinstance(action, dict):
            continue
        command = str(action.get("command", "")).strip()
        if not command or command in existing_pending:
            continue
        append_repair_execution_entry(
            entries,
            command=command,
            status="pending",
            covered_checks=(
                [str(item) for item in action.get("affected_checks", []) if str(item).strip()]
                if isinstance(action.get("affected_checks"), list)
                else []
            ),
            title=str(action.get("title", "")).strip(),
            source="iterative_next_action",
        )
        existing_pending.add(command)
        appended += 1
    return appended


def collect_release_metadata(project_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    cargo_path = os.path.join(project_root, "Cargo.toml")
    notes_path = os.path.join(project_root, "doc", "RELEASE_NOTES.md")

    pyproject = _read_text(pyproject_path)
    cargo = _read_text(cargo_path)
    notes = _read_text(notes_path)

    pyproject_version_match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cargo_version_match = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE)
    current_release_heading = re.search(r"^##\s+(.+)$", notes, re.MULTILINE)
    note_sections = re.findall(r"^###\s+(.+)$", notes, re.MULTILINE)

    pyproject_version = pyproject_version_match.group(1) if pyproject_version_match else ""
    cargo_version = cargo_version_match.group(1) if cargo_version_match else ""
    console_scripts = []
    if 'sara-chat = "sara_engine.cli:chat"' in pyproject:
        console_scripts.append("sara-chat")
    if 'sara-train = "sara_engine.cli:train"' in pyproject:
        console_scripts.append("sara-train")

    return {
        "pyproject_version": pyproject_version,
        "cargo_version": cargo_version,
        "versions_match": bool(pyproject_version and pyproject_version == cargo_version),
        "console_scripts": console_scripts,
        "has_expected_console_scripts": set(console_scripts) == {"sara-chat", "sara-train"},
        "release_notes_heading": current_release_heading.group(1) if current_release_heading else "",
        "release_note_sections": note_sections,
        "release_notes_path": notes_path,
    }


def resolve_soak_profile(
    profile_name: str,
    duration_seconds: Optional[float],
    max_agent_turns: Optional[int],
    min_agent_turns: Optional[int],
    max_inference_iterations: Optional[int],
    min_inference_iterations: Optional[int],
) -> Dict[str, Any]:
    if profile_name not in SOAK_PROFILES:
        raise ValueError(f"Unknown soak profile: {profile_name}")

    baseline = dict(SOAK_PROFILES[profile_name])
    profile = dict(baseline)
    if duration_seconds is not None:
        profile["duration_seconds"] = duration_seconds
    if max_agent_turns is not None:
        profile["max_agent_turns"] = max_agent_turns
    if min_agent_turns is not None:
        profile["min_agent_turns"] = min_agent_turns
    if max_inference_iterations is not None:
        profile["max_inference_iterations"] = max_inference_iterations
    if min_inference_iterations is not None:
        profile["min_inference_iterations"] = min_inference_iterations
    profile["profile_name"] = profile_name
    profile["shipping_ready"] = bool(
        baseline["shipping_ready"]
        and profile["duration_seconds"] >= baseline["duration_seconds"]
        and profile["max_agent_turns"] >= baseline["max_agent_turns"]
        and profile["min_agent_turns"] >= baseline["min_agent_turns"]
        and profile["max_inference_iterations"] >= baseline["max_inference_iterations"]
        and profile["min_inference_iterations"] >= baseline["min_inference_iterations"]
    )
    return profile


def run_agent_soak(duration_seconds: float, max_turns: int, min_turns: int) -> Dict[str, Any]:
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    agent.register_tool("<CALC>", lambda _: "5")

    start = time.time()
    turns = 0
    required_turns = min(int(max_turns), int(min_turns))
    while turns < max_turns and ((time.time() - start) < duration_seconds or turns < required_turns):
        agent.chat(f"Python の補足知識 {turns} は 可読性 を高めます。", teaching_mode=True)
        agent.chat(f"この要点を教えて <CALC> {turns}", teaching_mode=False)
        turns += 1

    return {
        "turns": turns,
        "elapsed_seconds": time.time() - start,
        "history_size": len(agent.dialogue_history),
        "history_limit": agent.max_history_turns * 2,
        "issue_count": len(agent.get_recent_issues(limit=100)),
        "active_terms": agent.topic_tracker.active_terms(limit=5),
        "history_bounded": len(agent.dialogue_history) <= agent.max_history_turns * 2,
        "min_turns_required": min_turns,
        "meets_min_turns": turns >= min_turns,
    }


def run_inference_soak(duration_seconds: float, max_iterations: int, min_iterations: int) -> Dict[str, Any]:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = model_path("tests", "release_soak_runtime.msgpack")
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.lif_network = None

    start = time.time()
    iterations = 0
    required_iterations = min(int(max_iterations), int(min_iterations))
    while iterations < max_iterations and (
        (time.time() - start) < duration_seconds or iterations < required_iterations
    ):
        base = iterations % 256
        engine.learn_sequence([base, base + 1, base + 2, base + 3])
        iterations += 1

    engine.session_memory["name"] = "ReleaseSoakUser"
    engine.session_memory["location"] = "Tokyo"
    engine.session_memory["preference"] = "structured reports"
    engine.session_memory["goal"] = "finish the release soak"
    engine.session_memory["task"] = "the inference runtime"
    engine.generate("You: Do you remember me?\nSARA:")
    engine.generate("You: What should I do next?\nSARA:")
    engine.generate("You: What should I do next?\nSARA:")

    ensure_parent_directory(engine.model_path)
    engine.save_pretrained(engine.model_path)

    reloaded = SaraInference.__new__(SaraInference)
    reloaded.model_path = engine.model_path
    reloaded.direct_map = {}
    reloaded.refractory_buffer = []
    reloaded.lif_network = None
    reloaded._load_memory()
    memory_health = inspect_inference_memory(engine.model_path)

    return {
        "iterations": iterations,
        "elapsed_seconds": time.time() - start,
        "pattern_count": len(engine.direct_map),
        "roundtrip_ok": reloaded.direct_map == engine.direct_map,
        "tuple_keys_only": all(isinstance(key, tuple) for key in engine.direct_map.keys()),
        "min_iterations_required": min_iterations,
        "meets_min_iterations": iterations >= min_iterations,
        "memory_health": {
            "session_memory_keys": memory_health.get("session_memory_keys", []),
            "diagnostic_memory_hits": memory_health.get("diagnostic_memory_hits", []),
            "predictor_state_keys": memory_health.get("predictor_state_keys", []),
            "predictor_state_snapshot": memory_health.get("predictor_state_snapshot", {}),
            "adaptation_state_keys": memory_health.get("adaptation_state_keys", []),
            "adaptation_state_snapshot": memory_health.get("adaptation_state_snapshot", {}),
            "future_state_runtime_state": reloaded._get_future_state_runtime_snapshot(),
            "conversational_readiness": memory_health.get("conversational_readiness", {}),
        },
    }


def run_accuracy_soak(
    history_path: Optional[str] = None,
    history_limit: int = 50,
    stage_b_promotion_required_streak: int = 3,
) -> Dict[str, Any]:
    report = run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=bool(history_path),
        history_limit=history_limit,
        stage_b_promotion_required_streak=stage_b_promotion_required_streak,
    )
    return {
        "suite_name": report.get("suite_name", "Phase3AccuracySuite"),
        "overall_score": float(report.get("overall_score", 0.0)),
        "passed": bool(report.get("passed", False)),
        "trend": report.get("trend", {}),
        "component_reports": report.get("component_reports", {}),
        "focus_summary": report.get("focus_summary", {}),
        "focus_trend": report.get("focus_trend", {}),
        "stage_a_acceptance": report.get("stage_a_acceptance", {}),
        "stage_b_readiness": report.get("stage_b_readiness", {}),
        "stage_c_readiness": report.get("stage_c_readiness", {}),
        "stage_d_readiness": report.get("stage_d_readiness", {}),
        "stage_e_readiness": report.get("stage_e_readiness", {}),
        "phase3_completion": report.get("phase3_completion", {}),
        "history_length": int(report.get("history_length", 0)) if history_path else 0,
    }


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "WARN"


def _extract_metric_trend(
    trend: Dict[str, Any],
    metric_name: str,
) -> Dict[str, Any]:
    for bucket, status in [("improvements", "UP"), ("regressions", "DOWN")]:
        entries = trend.get(bucket, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("metric") == metric_name:
                return {
                    "status": status,
                    "delta": float(entry.get("delta", 0.0) or 0.0),
                }

    unchanged = trend.get("unchanged", [])
    if isinstance(unchanged, list) and metric_name in unchanged:
        return {"status": "FLAT", "delta": 0.0}

    new_metrics = trend.get("new_metrics", [])
    if isinstance(new_metrics, list) and metric_name in new_metrics:
        return {"status": "NEW", "delta": None}

    return {"status": "NEW", "delta": None}


def _build_stage_b_promotion_actions(stage_b_promotion_readiness: Dict[str, Any]) -> Dict[str, Any]:
    if bool(stage_b_promotion_readiness.get("promoted_to_minimum", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    recommended = bool(stage_b_promotion_readiness.get("recommended", False))
    if not recommended:
        return {
            "next_step_hint": "",
            "actions": [],
        }
    actions = [
        "review stage_b_contract minimum list and add the three promotion-candidate metrics",
        "run python scripts/eval/phase3_accuracy_suite.py and verify Stage B minimum remains green",
        "run python scripts/eval/release_gate.py --skip-accuracy to verify release gate compatibility",
    ]
    return {
        "next_step_hint": "promote_stage_b_reward_policy_metrics_to_minimum_gate",
        "actions": actions,
    }


def _build_stage_b_rlm_observation_promotion_actions(
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(readiness.get("promoted_to_minimum", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    if not bool(readiness.get("recommended", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    return {
        "next_step_hint": "promote_stage_b_rlm_observation_metrics_to_minimum_gate",
        "actions": [
            "review stage_b_contract minimum list and add the two RLM observation metrics",
            "run python scripts/eval/phase3_accuracy_suite.py and verify long-context Stage B remains green",
            "run python scripts/eval/release_gate.py --skip-accuracy to verify release gate compatibility",
        ],
    }


def _build_stage_d_delta_memory_promotion_actions(readiness: Dict[str, Any]) -> Dict[str, Any]:
    if bool(readiness.get("promoted_to_minimum", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    if not bool(readiness.get("recommended", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    return {
        "next_step_hint": "promote_stage_d_delta_memory_metrics_to_minimum_gate",
        "actions": [
            "review stage_d_contract minimum list and add the delta-memory promotion metrics",
            "run python scripts/eval/phase3_accuracy_suite.py and verify Stage D remains green",
            "run python scripts/eval/release_gate.py --skip-accuracy to verify release gate compatibility",
        ],
    }


def _build_stage_d_acceptance_candidate_stability_actions(
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(readiness.get("recommended", False)):
        return {
            "next_step_hint": "",
            "actions": [],
        }
    return {
        "next_step_hint": STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT,
        "actions": list(STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS),
    }


def _agent_status(agent: Dict[str, Any]) -> bool:
    return bool(
        agent.get("meets_min_turns", False)
        and agent.get("history_bounded", False)
        and int(agent.get("issue_count", 0)) == 0
    )


def _inference_status(inference: Dict[str, Any]) -> bool:
    return bool(
        inference.get("meets_min_iterations", False)
        and inference.get("roundtrip_ok", False)
        and inference.get("tuple_keys_only", False)
        and int(inference.get("pattern_count", 0)) >= 1
    )


def _metadata_status(metadata: Dict[str, Any]) -> bool:
    return bool(
        metadata.get("versions_match", False)
        and metadata.get("has_expected_console_scripts", False)
        and str(metadata.get("release_notes_heading", "")).strip()
    )


def _accuracy_status(accuracy: Dict[str, Any]) -> bool:
    trend = accuracy.get("trend", {}) if isinstance(accuracy.get("trend"), dict) else {}
    return bool(
        accuracy.get("passed", False)
        and int(trend.get("gate_regression_count", trend.get("regression_count", 0)) or 0) == 0
    )


def collect_release_gate_feedback(
    report: Dict[str, Any],
    project_root: str = PROJECT_ROOT,
    retry_max_attempts: int = 2,
    retry_cooldown_seconds: float = 0.0,
) -> Dict[str, Any]:
    criteria = report.get("criteria", {}) if isinstance(report.get("criteria"), dict) else {}
    accuracy = report.get("accuracy", {}) if isinstance(report.get("accuracy"), dict) else {}
    stage_a_acceptance = (
        accuracy.get("stage_a_acceptance", {})
        if isinstance(accuracy.get("stage_a_acceptance"), dict)
        else {}
    )
    stage_b_readiness = (
        accuracy.get("stage_b_readiness", {})
        if isinstance(accuracy.get("stage_b_readiness"), dict)
        else {}
    )
    stage_b_minimum_failures = (
        stage_b_readiness.get("minimum_failures", [])
        if isinstance(stage_b_readiness.get("minimum_failures"), list)
        else []
    )
    stage_b_promotion_readiness = (
        stage_b_readiness.get("promotion_readiness", {})
        if isinstance(stage_b_readiness.get("promotion_readiness"), dict)
        else {}
    )
    stage_b_promotion_actions = _build_stage_b_promotion_actions(stage_b_promotion_readiness)
    stage_b_rlm_observation_readiness = (
        stage_b_readiness.get("rlm_observation_promotion_readiness", {})
        if isinstance(stage_b_readiness.get("rlm_observation_promotion_readiness"), dict)
        else {}
    )
    stage_b_rlm_observation_actions = _build_stage_b_rlm_observation_promotion_actions(
        stage_b_rlm_observation_readiness
    )
    stage_c_readiness = (
        accuracy.get("stage_c_readiness", {})
        if isinstance(accuracy.get("stage_c_readiness"), dict)
        else {}
    )
    stage_c_minimum_failures = (
        stage_c_readiness.get("minimum_failures", [])
        if isinstance(stage_c_readiness.get("minimum_failures"), list)
        else []
    )
    stage_d_readiness = (
        accuracy.get("stage_d_readiness", {})
        if isinstance(accuracy.get("stage_d_readiness"), dict)
        else {}
    )
    stage_d_minimum_failures = (
        stage_d_readiness.get("minimum_failures", [])
        if isinstance(stage_d_readiness.get("minimum_failures"), list)
        else []
    )
    stage_d_delta_memory_readiness = (
        stage_d_readiness.get("delta_memory_promotion_readiness", {})
        if isinstance(stage_d_readiness.get("delta_memory_promotion_readiness"), dict)
        else {}
    )
    stage_d_acceptance_candidate_stability = (
        stage_d_readiness.get("acceptance_candidate_stability", {})
        if isinstance(stage_d_readiness.get("acceptance_candidate_stability"), dict)
        else {}
    )
    stage_d_delta_memory_actions = _build_stage_d_delta_memory_promotion_actions(
        stage_d_delta_memory_readiness
    )
    stage_d_acceptance_candidate_actions = (
        _build_stage_d_acceptance_candidate_stability_actions(
            stage_d_acceptance_candidate_stability
        )
    )
    stage_d_metrics = (
        stage_d_readiness.get("metrics", {})
        if isinstance(stage_d_readiness.get("metrics"), dict)
        else {}
    )
    stage_e_readiness = (
        accuracy.get("stage_e_readiness", {})
        if isinstance(accuracy.get("stage_e_readiness"), dict)
        else {}
    )
    stage_e_minimum_failures = (
        stage_e_readiness.get("minimum_failures", [])
        if isinstance(stage_e_readiness.get("minimum_failures"), list)
        else []
    )
    stage_e_metrics = (
        stage_e_readiness.get("metrics", {})
        if isinstance(stage_e_readiness.get("metrics"), dict)
        else {}
    )
    focus_summary = (
        accuracy.get("focus_summary", {})
        if isinstance(accuracy.get("focus_summary"), dict)
        else {}
    )
    phase5_entry = (
        focus_summary.get("phase5_entry_readiness", {})
        if isinstance(focus_summary.get("phase5_entry_readiness"), dict)
        else {}
    )
    phase5_metrics = (
        phase5_entry.get("metrics", {})
        if isinstance(phase5_entry.get("metrics"), dict)
        else {}
    )
    component_reports = (
        accuracy.get("component_reports", {})
        if isinstance(accuracy.get("component_reports"), dict)
        else {}
    )
    phase5_component = (
        component_reports.get("phase5_predictive_coding", {})
        if isinstance(component_reports.get("phase5_predictive_coding"), dict)
        else {}
    )
    phase5_component_metrics = (
        phase5_component.get("metrics", {})
        if isinstance(phase5_component.get("metrics"), dict)
        else {}
    )
    linear_snn_fusion_observed_trend = (
        accuracy.get("linear_snn_fusion_observed_trend", {})
        if isinstance(accuracy.get("linear_snn_fusion_observed_trend"), dict)
        else {}
    )
    stage_e_architecture_integration_observed_trend = (
        accuracy.get("stage_e_architecture_integration_observed_trend", {})
        if isinstance(accuracy.get("stage_e_architecture_integration_observed_trend"), dict)
        else {}
    )
    continual_component = (
        component_reports.get("continual_consolidation", {})
        if isinstance(component_reports.get("continual_consolidation"), dict)
        else {}
    )
    continual_component_metrics = (
        continual_component.get("metrics", {})
        if isinstance(continual_component.get("metrics"), dict)
        else {}
    )
    cognitive_manifold_trace_metrics = extract_cognitive_manifold_trace_metrics(accuracy)
    cognitive_delta_memory_metrics = extract_cognitive_delta_memory_metrics(accuracy)
    cognitive_linear_snn_fusion_metrics = extract_cognitive_linear_snn_fusion_metrics(accuracy)
    cognitive_plastic_submodel_metrics = extract_cognitive_plastic_submodel_metrics(accuracy)
    release_errors = validate_release_report(report)
    packaging_errors = validate_packaging_metadata(project_root)
    errors = [*release_errors, *packaging_errors]
    repair_execution_log = (
        [dict(item) for item in report.get("repair_execution_log", []) if isinstance(item, dict)]
        if isinstance(report.get("repair_execution_log"), list)
        else []
    )
    artifacts = collect_release_gate_artifacts(errors, execution_log=repair_execution_log)
    recovery_actions = (
        artifacts.get("recovery_actions", [])
        if isinstance(artifacts.get("recovery_actions"), list)
        else []
    )
    repair_plan = (
        artifacts.get("repair_plan", {})
        if isinstance(artifacts.get("repair_plan"), dict)
        else {}
    )
    iterative_repair_plan = (
        artifacts.get("iterative_repair_plan", {})
        if isinstance(artifacts.get("iterative_repair_plan"), dict)
        else {}
    )
    error_details = (
        artifacts.get("error_details", [])
        if isinstance(artifacts.get("error_details"), list)
        else []
    )
    error_details_summary = (
        artifacts.get("error_details_summary", {})
        if isinstance(artifacts.get("error_details_summary"), dict)
        else {}
    )
    failure_focus = (
        artifacts.get("failure_focus", {})
        if isinstance(artifacts.get("failure_focus"), dict)
        else {}
    )
    pending_count = sum(
        1
        for item in repair_execution_log
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
    )
    timeout_count = sum(
        1
        for item in repair_execution_log
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "timeout"
    )
    retry_queue = build_retry_queue_from_repair_log(
        repair_execution_log,
        max_attempts=retry_max_attempts,
        cooldown_seconds=retry_cooldown_seconds,
    )
    retry_cooldown_blocked = build_retry_cooldown_blocked_from_repair_log(
        repair_execution_log,
        max_attempts=retry_max_attempts,
        cooldown_seconds=retry_cooldown_seconds,
    )
    prioritized_retry_queue = prioritize_retry_queue(
        retry_queue,
        iterative_plan=iterative_repair_plan,
    )
    prioritized_cooldown_blocked = prioritize_retry_queue(
        retry_cooldown_blocked,
        iterative_plan=iterative_repair_plan,
    )
    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "recovery_actions": recovery_actions,
        "error_details": error_details,
        "error_details_summary": error_details_summary,
        "failure_focus": failure_focus,
        "repair_plan": repair_plan,
        "repair_execution_log": repair_execution_log,
        "repair_pending_count": int(pending_count),
        "repair_timeout_count": int(timeout_count),
        "repair_retry_queue": prioritized_retry_queue,
        "repair_retry_queue_count": int(len(prioritized_retry_queue)),
        "repair_retry_cooldown_seconds": float(max(retry_cooldown_seconds, 0.0)),
        "repair_retry_cooldown_blocked": prioritized_cooldown_blocked,
        "repair_retry_cooldown_blocked_count": int(len(prioritized_cooldown_blocked)),
        "iterative_repair_plan": iterative_repair_plan,
        "accuracy_required": bool(criteria.get("require_phase3_accuracy", False)),
        "embedded_accuracy_present": isinstance(report.get("accuracy"), dict),
        "stage_a_passed": bool(stage_a_acceptance.get("passed", False)),
        "stage_b_passed": bool(stage_b_readiness.get("passed", False)),
        "stage_b_minimum_requirements_passed": bool(
            stage_b_readiness.get("minimum_requirements_passed", False)
        ),
        "stage_b_minimum_failure_count": int(
            stage_b_readiness.get("minimum_failure_count", len(stage_b_minimum_failures)) or 0
        ),
        "stage_b_minimum_failures": [
            dict(item) for item in stage_b_minimum_failures if isinstance(item, dict)
        ],
        "stage_b_promotion_candidate_ready": bool(
            stage_b_readiness.get("promotion_candidate_ready", False)
        ),
        "stage_b_promotion_candidate_failure_count": int(
            stage_b_readiness.get("promotion_candidate_failure_count", 0) or 0
        ),
        "stage_b_promotion_candidate_promoted": bool(
            stage_b_readiness.get("promotion_candidate_promoted", False)
        ),
        "stage_b_promotion_consecutive_passes": int(
            stage_b_promotion_readiness.get("consecutive_passes", 0) or 0
        ),
        "stage_b_promotion_required_streak": int(
            stage_b_promotion_readiness.get("required_streak", 3) or 3
        ),
        "stage_b_promotion_recommended": bool(
            stage_b_promotion_readiness.get("recommended", False)
        ),
        "stage_b_promotion_next_step_hint": str(
            stage_b_promotion_actions.get("next_step_hint", "") or ""
        ),
        "stage_b_promotion_actions": [
            str(item)
            for item in stage_b_promotion_actions.get("actions", [])
            if str(item).strip()
        ] if isinstance(stage_b_promotion_actions.get("actions", []), list) else [],
        "stage_b_rlm_observation_candidate_ready": bool(
            stage_b_readiness.get("rlm_observation_candidate_ready", False)
        ),
        "stage_b_rlm_observation_candidate_failure_count": int(
            stage_b_readiness.get("rlm_observation_candidate_failure_count", 0) or 0
        ),
        "stage_b_rlm_observation_candidate_promoted": bool(
            stage_b_readiness.get("rlm_observation_candidate_promoted", False)
        ),
        "stage_b_rlm_observation_consecutive_passes": int(
            stage_b_rlm_observation_readiness.get("consecutive_passes", 0) or 0
        ),
        "stage_b_rlm_observation_required_streak": int(
            stage_b_rlm_observation_readiness.get("required_streak", 3) or 3
        ),
        "stage_b_rlm_observation_promotion_recommended": bool(
            stage_b_rlm_observation_readiness.get("recommended", False)
        ),
        "stage_b_rlm_observation_next_step_hint": str(
            stage_b_rlm_observation_actions.get("next_step_hint", "") or ""
        ),
        "stage_b_rlm_observation_actions": [
            str(item)
            for item in stage_b_rlm_observation_actions.get("actions", [])
            if str(item).strip()
        ] if isinstance(stage_b_rlm_observation_actions.get("actions", []), list) else [],
        "stage_c_passed": bool(stage_c_readiness.get("passed", False)),
        "stage_c_minimum_requirements_passed": bool(
            stage_c_readiness.get("minimum_requirements_passed", False)
        ),
        "stage_c_minimum_failure_count": int(
            stage_c_readiness.get("minimum_failure_count", len(stage_c_minimum_failures)) or 0
        ),
        "stage_c_minimum_failures": [
            dict(item) for item in stage_c_minimum_failures if isinstance(item, dict)
        ],
        "stage_d_passed": bool(stage_d_readiness.get("passed", False)),
        "stage_d_minimum_requirements_passed": bool(
            stage_d_readiness.get("minimum_requirements_passed", False)
        ),
        "stage_d_minimum_failure_count": int(
            stage_d_readiness.get("minimum_failure_count", len(stage_d_minimum_failures)) or 0
        ),
        "stage_d_minimum_failures": [
            dict(item) for item in stage_d_minimum_failures if isinstance(item, dict)
        ],
        "stage_d_readiness_score": float(stage_d_readiness.get("readiness_score", 0.0) or 0.0),
        "stage_d_acceptance_candidate_count": int(
            stage_d_readiness.get("acceptance_candidate_count", 0) or 0
        ),
        "stage_d_acceptance_candidate_ready_count": int(
            stage_d_readiness.get("acceptance_candidate_ready_count", 0) or 0
        ),
        "stage_d_acceptance_candidates_ready": bool(
            stage_d_readiness.get("acceptance_candidates_ready", False)
        ),
        "stage_d_acceptance_candidate_failure_count": int(
            stage_d_readiness.get("acceptance_candidate_failure_count", 0) or 0
        ),
        "stage_d_acceptance_candidate_failures": [
            dict(item)
            for item in stage_d_readiness.get("acceptance_candidate_failures", [])
            if isinstance(item, dict)
        ] if isinstance(stage_d_readiness.get("acceptance_candidate_failures", []), list) else [],
        "stage_d_acceptance_candidate_consecutive_passes": int(
            stage_d_acceptance_candidate_stability.get("consecutive_passes", 0) or 0
        ),
        "stage_d_acceptance_candidate_required_streak": int(
            stage_d_acceptance_candidate_stability.get("required_streak", 3) or 3
        ),
        "stage_d_acceptance_candidate_stability_recommended": bool(
            stage_d_acceptance_candidate_stability.get("recommended", False)
        ),
        "stage_d_acceptance_candidate_next_step_hint": str(
            stage_d_acceptance_candidate_actions.get("next_step_hint", "") or ""
        ),
        "stage_d_acceptance_candidate_actions": [
            str(item)
            for item in stage_d_acceptance_candidate_actions.get("actions", [])
            if str(item).strip()
        ] if isinstance(stage_d_acceptance_candidate_actions.get("actions", []), list) else [],
        "stage_d_acceptance_candidate_action_count": int(
            len(stage_d_acceptance_candidate_actions.get("actions", []))
            if isinstance(stage_d_acceptance_candidate_actions.get("actions", []), list)
            else 0
        ),
        "stage_d_delta_memory_candidate_ready": bool(
            stage_d_readiness.get("delta_memory_candidate_ready", False)
        ),
        "stage_d_delta_memory_candidate_failure_count": int(
            stage_d_readiness.get("delta_memory_candidate_failure_count", 0) or 0
        ),
        "stage_d_delta_memory_candidate_failures": [
            dict(item)
            for item in stage_d_readiness.get("delta_memory_candidate_failures", [])
            if isinstance(item, dict)
        ] if isinstance(stage_d_readiness.get("delta_memory_candidate_failures", []), list) else [],
        "stage_d_delta_memory_candidate_promoted": bool(
            stage_d_readiness.get("delta_memory_candidate_promoted", False)
        ),
        "stage_d_delta_memory_consecutive_passes": int(
            stage_d_delta_memory_readiness.get("consecutive_passes", 0) or 0
        ),
        "stage_d_delta_memory_required_streak": int(
            stage_d_delta_memory_readiness.get("required_streak", 3) or 3
        ),
        "stage_d_delta_memory_promotion_recommended": bool(
            stage_d_delta_memory_readiness.get("recommended", False)
        ),
        "stage_d_delta_memory_next_step_hint": str(
            stage_d_delta_memory_actions.get("next_step_hint", "") or ""
        ),
        "stage_d_delta_memory_actions": [
            str(item)
            for item in stage_d_delta_memory_actions.get("actions", [])
            if str(item).strip()
        ] if isinstance(stage_d_delta_memory_actions.get("actions", []), list) else [],
        "stage_d_replay_recovery_integrity": float(stage_d_metrics.get("replay_recovery_integrity", 0.0) or 0.0),
        "stage_d_replay_upgrade_reindex_integrity": float(
            stage_d_metrics.get("replay_upgrade_reindex_integrity", 0.0) or 0.0
        ),
        "stage_d_memory_health_index_integrity": float(
            stage_d_metrics.get("memory_health_index_integrity", 0.0) or 0.0
        ),
        "stage_d_replay_noise_resilience_integrity": float(
            stage_d_metrics.get("replay_noise_resilience_integrity", 0.0) or 0.0
        ),
        "stage_d_astro_modulation_stability": float(
            stage_d_metrics.get("astro_modulation_stability", 0.0) or 0.0
        ),
        "stage_d_manifold_continual_retention_observed": float(
            continual_component_metrics.get("manifold_continual_retention_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_trajectory_case_coverage_observed": float(
            continual_component_metrics.get("manifold_trajectory_case_coverage_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_average_case_recall_observed": float(
            continual_component_metrics.get("manifold_average_case_recall_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_scan_budget_integrity_observed": float(
            continual_component_metrics.get("manifold_scan_budget_integrity_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_indexed_candidate_integrity_observed": float(
            continual_component_metrics.get("manifold_indexed_candidate_integrity_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_index_scan_reduction_observed": float(
            continual_component_metrics.get("manifold_index_scan_reduction_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_capacity_pressure_recall_observed": float(
            continual_component_metrics.get("manifold_capacity_pressure_recall_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_capacity_pressure_scan_reduction_observed": float(
            continual_component_metrics.get("manifold_capacity_pressure_scan_reduction_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_replay_refresh_retention_observed": float(
            continual_component_metrics.get("manifold_replay_refresh_retention_observed", 0.0) or 0.0
        ),
        "stage_d_manifold_replay_refresh_eviction_integrity_observed": float(
            continual_component_metrics.get("manifold_replay_refresh_eviction_integrity_observed", 0.0) or 0.0
        ),
        **{
            f"stage_d_{metric_name}": float(continual_component_metrics.get(metric_name, 0.0) or 0.0)
            for metric_name in STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES
        },
        "stage_e_passed": bool(stage_e_readiness.get("passed", False)),
        "stage_e_minimum_requirements_passed": bool(
            stage_e_readiness.get("minimum_requirements_passed", False)
        ),
        "stage_e_minimum_failure_count": int(
            stage_e_readiness.get("minimum_failure_count", len(stage_e_minimum_failures)) or 0
        ),
        "stage_e_minimum_failures": [
            dict(item) for item in stage_e_minimum_failures if isinstance(item, dict)
        ],
        "stage_e_readiness_score": float(stage_e_readiness.get("readiness_score", 0.0) or 0.0),
        "stage_e_observed_acceptance_candidate_count": int(
            stage_e_readiness.get("observed_acceptance_candidate_count", 0) or 0
        ),
        "stage_e_observed_acceptance_candidate_ready_count": int(
            stage_e_readiness.get("observed_acceptance_candidate_ready_count", 0) or 0
        ),
        "stage_e_observed_acceptance_candidates_ready": bool(
            stage_e_readiness.get("observed_acceptance_candidates_ready", False)
        ),
        "stage_e_observed_acceptance_candidate_failure_count": int(
            stage_e_readiness.get("observed_acceptance_candidate_failure_count", 0) or 0
        ),
        "stage_e_observed_acceptance_candidate_failures": [
            dict(item)
            for item in stage_e_readiness.get("observed_acceptance_candidate_failures", [])
            if isinstance(item, dict)
        ] if isinstance(stage_e_readiness.get("observed_acceptance_candidate_failures", []), list) else [],
        "stage_e_observed_acceptance_candidate_consecutive_passes": int(
            stage_e_readiness.get("observed_acceptance_candidate_stability", {}).get("consecutive_passes", 0)
            if isinstance(stage_e_readiness.get("observed_acceptance_candidate_stability"), dict)
            else 0
        ),
        "stage_e_observed_acceptance_candidate_required_streak": int(
            stage_e_readiness.get("observed_acceptance_candidate_stability", {}).get("required_streak", 3)
            if isinstance(stage_e_readiness.get("observed_acceptance_candidate_stability"), dict)
            else 3
        ),
        "stage_e_observed_acceptance_candidate_stability_recommended": bool(
            stage_e_readiness.get("observed_acceptance_candidate_stability", {}).get("recommended", False)
            if isinstance(stage_e_readiness.get("observed_acceptance_candidate_stability"), dict)
            else False
        ),
        "stage_e_common_spike_space_integrity": float(
            stage_e_metrics.get("common_spike_space_integrity", 0.0) or 0.0
        ),
        "stage_e_temporal_compression_efficiency": float(
            stage_e_metrics.get("temporal_compression_efficiency", 0.0) or 0.0
        ),
        "stage_e_modality_temporal_budget_integrity": float(
            stage_e_metrics.get("modality_temporal_budget_integrity", 0.0) or 0.0
        ),
        "stage_e_dendritic_context_gate_stability": float(
            stage_e_metrics.get("dendritic_context_gate_stability", 0.0) or 0.0
        ),
        "stage_e_spiking_hjepa_latent_transition": float(
            stage_e_metrics.get("spiking_hjepa_latent_transition", 0.0) or 0.0
        ),
        "stage_e_reverse_reasoning_trace_integrity": float(
            stage_e_metrics.get("reverse_reasoning_trace_integrity", 0.0) or 0.0
        ),
        "stage_e_causal_candidate_trace_integrity": float(
            stage_e_metrics.get("causal_candidate_trace_integrity", 0.0) or 0.0
        ),
        "stage_e_module_orchestration_integrity": float(
            stage_e_metrics.get("module_orchestration_integrity", 0.0) or 0.0
        ),
        "stage_e_counterfactual_lane_integrity": float(
            stage_e_metrics.get("counterfactual_lane_integrity", 0.0) or 0.0
        ),
        "stage_e_action_trace_observability": float(
            stage_e_metrics.get("action_trace_observability", 0.0) or 0.0
        ),
        "stage_e_runtime_trace_replay_consistency": float(
            stage_e_metrics.get("runtime_trace_replay_consistency", 0.0) or 0.0
        ),
        **{
            f"stage_e_{metric_name}": float(cognitive_manifold_trace_metrics[metric_name])
            for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
        },
        **{
            f"stage_e_{metric_name}": float(cognitive_delta_memory_metrics[metric_name])
            for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
        },
        "stage_e_linear_snn_fusion_observed_policy": "excluded_from_score_and_release_gate",
        "stage_e_linear_snn_fusion_trend_has_previous": bool(
            linear_snn_fusion_observed_trend.get("has_previous", False)
        ),
        "stage_e_linear_snn_fusion_trend_regression_count": int(
            linear_snn_fusion_observed_trend.get("regression_count", 0) or 0
        ),
        "stage_e_linear_snn_fusion_trend_release_gate_blocking": bool(
            linear_snn_fusion_observed_trend.get("release_gate_blocking", False)
        ),
        "stage_e_architecture_integration_observed_policy": "excluded_from_score_and_release_gate",
        "stage_e_architecture_integration_trend_has_previous": bool(
            stage_e_architecture_integration_observed_trend.get("has_previous", False)
        ),
        "stage_e_architecture_integration_trend_regression_count": int(
            stage_e_architecture_integration_observed_trend.get("regression_count", 0) or 0
        ),
        "stage_e_architecture_integration_trend_release_gate_blocking": bool(
            stage_e_architecture_integration_observed_trend.get("release_gate_blocking", False)
        ),
        **{
            f"stage_e_{metric_name}": float(cognitive_linear_snn_fusion_metrics[metric_name])
            for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
        },
        **{
            f"stage_e_{metric_name}": float(cognitive_plastic_submodel_metrics[metric_name])
            for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
        },
        "phase5_entry_passed": bool(phase5_entry.get("passed", False)),
        "phase5_entry_readiness_score": float(phase5_entry.get("score", 0.0) or 0.0),
        "phase5_latent_transition_alignment": float(
            phase5_metrics.get("phase5_predictive_coding.latent_transition_alignment", 0.0) or 0.0
        ),
        "phase5_prediction_error_observability": float(
            phase5_metrics.get("phase5_predictive_coding.prediction_error_observability", 0.0) or 0.0
        ),
        "phase5_correction_event_coverage": float(
            phase5_metrics.get("phase5_predictive_coding.correction_event_coverage", 0.0) or 0.0
        ),
        "phase5_anti_collapse_event_diversity": float(
            phase5_metrics.get("phase5_predictive_coding.anti_collapse_event_diversity", 0.0) or 0.0
        ),
        "phase5_counterfactual_transition_separation": float(
            phase5_metrics.get("phase5_predictive_coding.counterfactual_transition_separation", 0.0) or 0.0
        ),
        "phase5_multi_step_latent_chain_integrity": float(
            phase5_metrics.get("phase5_predictive_coding.multi_step_latent_chain_integrity", 0.0) or 0.0
        ),
        "phase5_long_horizon_error_correction_convergence": float(
            phase5_metrics.get("phase5_predictive_coding.long_horizon_error_correction_convergence", 0.0) or 0.0
        ),
        "phase5_horizon_bucket_stability": float(
            phase5_metrics.get("phase5_predictive_coding.horizon_bucket_stability", 0.0) or 0.0
        ),
        "phase5_macro_action_effectiveness": float(
            phase5_metrics.get("phase5_predictive_coding.macro_action_effectiveness", 0.0) or 0.0
        ),
        "phase5_subgoal_decomposition_integrity": float(
            phase5_metrics.get("phase5_predictive_coding.subgoal_decomposition_integrity", 0.0) or 0.0
        ),
        "phase5_depth_selective_routing_integrity": float(
            phase5_metrics.get("phase5_predictive_coding.depth_selective_routing_integrity", 0.0) or 0.0
        ),
        "phase5_micro_es_policy_refinement_integrity": float(
            phase5_metrics.get("phase5_predictive_coding.micro_es_policy_refinement_integrity", 0.0) or 0.0
        ),
        "phase5_manifold_transition_locality_observed": float(
            phase5_component_metrics.get("manifold_transition_locality", 0.0) or 0.0
        ),
        "phase5_manifold_rollout_stability_observed": float(
            phase5_component_metrics.get("manifold_rollout_stability", 0.0) or 0.0
        ),
        "phase5_causal_route_sparsity_observed": float(
            phase5_component_metrics.get("causal_route_sparsity", 0.0) or 0.0
        ),
        "phase5_withheld_trajectory_recall_observed": float(
            phase5_component_metrics.get("withheld_trajectory_recall", 0.0) or 0.0
        ),
        "phase5_manifold_trajectory_case_coverage_observed": float(
            phase5_component_metrics.get("manifold_trajectory_case_coverage", 0.0) or 0.0
        ),
        "phase5_manifold_average_case_recall_observed": float(
            phase5_component_metrics.get("manifold_average_case_recall", 0.0) or 0.0
        ),
        "phase5_manifold_scan_budget_integrity_observed": float(
            phase5_component_metrics.get("manifold_scan_budget_integrity", 0.0) or 0.0
        ),
        "phase5_manifold_indexed_candidate_integrity_observed": float(
            phase5_component_metrics.get("manifold_indexed_candidate_integrity", 0.0) or 0.0
        ),
        "phase5_manifold_index_scan_reduction_observed": float(
            phase5_component_metrics.get("manifold_index_scan_reduction", 0.0) or 0.0
        ),
        "phase5_manifold_candidate_miss_guard_observed": float(
            phase5_component_metrics.get("manifold_candidate_miss_guard", 0.0) or 0.0
        ),
        "packaging_metadata_passed": len(packaging_errors) == 0,
    }


def _is_within_workspace(path: str) -> bool:
    workspace_root = os.path.abspath(workspace_path(""))
    abs_path = os.path.abspath(path)
    return os.path.commonpath([abs_path, workspace_root]) == workspace_root


def collect_release_checklist_status(
    report: Dict[str, Any],
    report_path: str,
    summary_path: str,
) -> Dict[str, Any]:
    criteria = report.get("criteria", {}) if isinstance(report.get("criteria"), dict) else {}
    metadata = report.get("release_metadata", {}) if isinstance(report.get("release_metadata"), dict) else {}

    report_path_resolved = os.path.abspath(report_path)
    summary_path_resolved = os.path.abspath(summary_path)
    managed_output_paths_ok = _is_within_workspace(report_path_resolved) and _is_within_workspace(summary_path_resolved)
    release_notes_reviewed = bool(str(metadata.get("release_notes_heading", "")).strip())
    report_summary_review_ready = managed_output_paths_ok
    extended_profile_ready = bool(criteria.get("profile_name") == "extended" and criteria.get("shipping_ready", False))
    # Keep checklist as a documentation/artifact hygiene gate.
    # Functional release eligibility is already represented by release_gate fields.
    checklist_passed = bool(
        managed_output_paths_ok
        and release_notes_reviewed
        and report_summary_review_ready
    )

    return {
        "passed": checklist_passed,
        "report_path": report_path_resolved,
        "summary_path": summary_path_resolved,
        "managed_output_paths_ok": managed_output_paths_ok,
        "release_notes_reviewed": release_notes_reviewed,
        "report_summary_review_ready": report_summary_review_ready,
        "extended_profile_ready": extended_profile_ready,
        "profile_name": criteria.get("profile_name", "unknown"),
        "shipping_ready_profile": bool(criteria.get("shipping_ready", False)),
    }


def build_release_research_review(report: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = report.get("accuracy", {}) if isinstance(report.get("accuracy"), dict) else None
    review = build_research_review_report(
        phase3_report=accuracy,
        release_soak_report=report,
        operational_report=None,
        input_snapshots=[
            {
                "path": "embedded:release_soak.accuracy",
                "exists": isinstance(accuracy, dict),
                "loaded": isinstance(accuracy, dict),
                "error": "" if isinstance(accuracy, dict) else "Embedded Phase 3 accuracy report is missing.",
            },
            {
                "path": "embedded:release_soak",
                "exists": True,
                "loaded": True,
                "error": "",
            },
        ],
        generated_at=float(report.get("generated_at", time.time()) or time.time()),
        require_operational_readiness=False,
    )
    compact = compact_research_review_report(review)
    return {
        "report": review,
        "compact": compact,
        "planner_task_status": compact_release_research_planner_task_status(
            report,
            research_review_compact=compact,
        ),
    }


def _first_mapping(*candidates: Any) -> Dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def compact_release_research_planner_task_status(
    report: Dict[str, Any],
    *,
    research_review_compact: Optional[Dict[str, Any]] = None,
    cleanup_threshold: int = 2,
) -> Dict[str, Any]:
    source_report = report if isinstance(report, dict) else {}
    operational = _first_mapping(
        source_report.get("operational_readiness"),
        source_report.get("operational_report"),
    )
    operational_research_review = _first_mapping(operational.get("research_review"))
    release_research_review = _first_mapping(source_report.get("research_review"))
    compact = _first_mapping(
        research_review_compact,
        release_research_review.get("compact"),
        operational_research_review.get("compact"),
        source_report.get("research_review_compact"),
    )
    journal = _first_mapping(
        source_report.get("research_journal_summary"),
        operational.get("research_journal_summary"),
    )
    existing_status = _first_mapping(
        release_research_review.get("planner_task_status"),
        source_report.get("research_planner_task_status"),
        operational.get("research_planner_task_status"),
    )
    if existing_status:
        return dict(existing_status)

    pending_cause_boundary = int(compact.get("cause_boundary_documentation_count", 0) or 0)
    pending_fixture_repair = int(compact.get("targeted_fixture_repair_count", 0) or 0)
    pending_count = int(pending_cause_boundary + pending_fixture_repair)
    completed_count = int(journal.get("completed_research_planner_task_count", 0) or 0)
    cleanup_pending_count = int(journal.get("research_planner_task_cleanup_pending_count", 0) or 0)
    cleanup_success_count = int(journal.get("research_planner_task_cleanup_success_count", 0) or 0)
    cleanup_skipped_count = int(journal.get("research_planner_task_cleanup_skipped_count", 0) or 0)
    cleanup_entries = (
        journal.get("research_planner_task_cleanup_entries", [])
        if isinstance(journal.get("research_planner_task_cleanup_entries", []), list)
        else []
    )
    total_count = int(pending_count + completed_count)
    completion_ratio = float(completed_count) / float(total_count) if total_count > 0 else 1.0
    threshold = int(max(cleanup_threshold, 1))
    pending_sources = {
        str(item.get("source", "") or "")
        for item in cleanup_entries
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
    }
    pending_commands = {
        str(item.get("command", "") or "")
        for item in cleanup_entries
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "pending"
    }
    stalled_reason = ""
    stalled_action_source = ""
    cleanup_stalled = bool(cleanup_pending_count > 0 and pending_count >= threshold)
    if cleanup_stalled:
        if pending_fixture_repair > 0:
            stalled_reason = "fixture_implementation_wait"
            stalled_action_source = "research_planner_fixture_repair_followup"
        elif pending_cause_boundary > 0 and (
            any("manual" in source for source in pending_sources)
            or any("review" in command for command in pending_commands)
        ):
            stalled_reason = "manual_review_wait"
            stalled_action_source = "research_planner_manual_review_followup"
        elif pending_cause_boundary > 0:
            stalled_reason = "documentation_not_reflected"
            stalled_action_source = "research_planner_documentation_followup"
        else:
            stalled_reason = "cleanup_pending"
            stalled_action_source = "research_planner_task_cleanup_stalled"
    return {
        "pending_count": pending_count,
        "pending_cause_boundary_documentation_count": pending_cause_boundary,
        "pending_targeted_fixture_repair_count": pending_fixture_repair,
        "completed_count": completed_count,
        "total_count": total_count,
        "completion_ratio": float(completion_ratio),
        "cleanup_threshold": threshold,
        "cleanup_pending_count": cleanup_pending_count,
        "cleanup_success_count": cleanup_success_count,
        "cleanup_skipped_count": cleanup_skipped_count,
        "cleanup_stalled": cleanup_stalled,
        "cleanup_stalled_reason": stalled_reason,
        "cleanup_stalled_action_source": stalled_action_source,
        "cleanup_needed": bool(pending_count >= threshold and cleanup_pending_count <= 0),
    }


def compact_release_stage_e_recovery_review_status(report: Dict[str, Any]) -> Dict[str, Any]:
    source_report = report if isinstance(report, dict) else {}
    operational = _first_mapping(
        source_report.get("operational_readiness"),
        source_report.get("operational_report"),
    )
    journal = _first_mapping(
        source_report.get("research_journal_summary"),
        operational.get("research_journal_summary"),
    )
    repair_loop = _first_mapping(
        journal.get("stage_e_observed_acceptance_candidate_repair_loop")
    )
    status_counts = (
        journal.get("stage_e_observed_acceptance_candidate_recovery_review_status_counts", {})
        if isinstance(
            journal.get("stage_e_observed_acceptance_candidate_recovery_review_status_counts"),
            dict,
        )
        else {}
    )
    latest_status = str(
        journal.get("stage_e_observed_acceptance_candidate_recovery_review_latest_status", "")
        or repair_loop.get("promotion_review_latest_status", "")
        or ""
    )
    completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_completed",
            repair_loop.get("promotion_review_completed", False),
        )
    )
    in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_in_progress",
            repair_loop.get("promotion_review_in_progress", False),
        )
    )
    stale = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_stale",
            repair_loop.get("promotion_review_stale", False),
        )
    )
    followup_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_in_progress",
            repair_loop.get("promotion_review_followup_in_progress", False),
        )
    )
    followup_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_completed",
            repair_loop.get("promotion_review_followup_completed", False),
        )
    )
    followup_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_failed",
            repair_loop.get("promotion_review_followup_failed", False),
        )
    )
    followup_retry_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_in_progress",
            repair_loop.get("promotion_review_followup_retry_in_progress", False),
        )
    )
    followup_retry_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_completed",
            repair_loop.get("promotion_review_followup_retry_completed", False),
        )
    )
    followup_retry_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_failed",
            repair_loop.get("promotion_review_followup_retry_failed", False),
        )
    )
    followup_retry_escalation_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_in_progress",
            repair_loop.get("promotion_review_followup_retry_escalation_in_progress", False),
        )
    )
    followup_retry_escalation_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_completed",
            repair_loop.get("promotion_review_followup_retry_escalation_completed", False),
        )
    )
    followup_retry_escalation_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_failed",
            repair_loop.get("promotion_review_followup_retry_escalation_failed", False),
        )
    )
    evidence_collection_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_in_progress",
            repair_loop.get("promotion_review_evidence_collection_in_progress", False),
        )
    )
    evidence_collection_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_completed",
            repair_loop.get("promotion_review_evidence_collection_completed", False),
        )
    )
    evidence_collection_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_failed",
            repair_loop.get("promotion_review_evidence_collection_failed", False),
        )
    )
    evidence_recheck_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_in_progress",
            repair_loop.get("promotion_review_evidence_recheck_in_progress", False),
        )
    )
    evidence_recheck_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_completed",
            repair_loop.get("promotion_review_evidence_recheck_completed", False),
        )
    )
    evidence_recheck_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_failed",
            repair_loop.get("promotion_review_evidence_recheck_failed", False),
        )
    )
    targeted_probe_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_in_progress",
            repair_loop.get("promotion_review_targeted_probe_in_progress", False),
        )
    )
    targeted_probe_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_completed",
            repair_loop.get("promotion_review_targeted_probe_completed", False),
        )
    )
    targeted_probe_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_failed",
            repair_loop.get("promotion_review_targeted_probe_failed", False),
        )
    )
    targeted_probe_recheck_in_progress = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_in_progress",
            repair_loop.get("promotion_review_targeted_probe_recheck_in_progress", False),
        )
    )
    targeted_probe_recheck_completed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_completed",
            repair_loop.get("promotion_review_targeted_probe_recheck_completed", False),
        )
    )
    targeted_probe_recheck_failed = bool(
        journal.get(
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_failed",
            repair_loop.get("promotion_review_targeted_probe_recheck_failed", False),
        )
    )
    recommended = bool(repair_loop.get("promotion_review_recommended", False))
    return {
        "available": bool(
            repair_loop
            or int(journal.get("stage_e_observed_acceptance_candidate_recovery_review_count", 0) or 0) > 0
        ),
        "recovery_confirmed": bool(repair_loop.get("recovery_confirmed", False)),
        "promotion_review_recommended": bool(recommended and not completed and not in_progress),
        "promotion_review_completed": completed,
        "promotion_review_in_progress": in_progress,
        "promotion_review_stale": stale,
        "promotion_review_followup_in_progress": followup_in_progress,
        "promotion_review_followup_completed": followup_completed,
        "promotion_review_followup_failed": followup_failed,
        "promotion_review_followup_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_latest_status",
                repair_loop.get("promotion_review_followup_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_followup_retry_in_progress": followup_retry_in_progress,
        "promotion_review_followup_retry_completed": followup_retry_completed,
        "promotion_review_followup_retry_failed": followup_retry_failed,
        "promotion_review_followup_retry_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_latest_status",
                repair_loop.get("promotion_review_followup_retry_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_followup_retry_escalation_in_progress": (
            followup_retry_escalation_in_progress
        ),
        "promotion_review_followup_retry_escalation_completed": (
            followup_retry_escalation_completed
        ),
        "promotion_review_followup_retry_escalation_failed": (
            followup_retry_escalation_failed
        ),
        "promotion_review_followup_retry_escalation_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_latest_status",
                repair_loop.get("promotion_review_followup_retry_escalation_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_evidence_collection_in_progress": evidence_collection_in_progress,
        "promotion_review_evidence_collection_completed": evidence_collection_completed,
        "promotion_review_evidence_collection_failed": evidence_collection_failed,
        "promotion_review_evidence_collection_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_latest_status",
                repair_loop.get("promotion_review_evidence_collection_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_evidence_recheck_in_progress": evidence_recheck_in_progress,
        "promotion_review_evidence_recheck_completed": evidence_recheck_completed,
        "promotion_review_evidence_recheck_failed": evidence_recheck_failed,
        "promotion_review_evidence_recheck_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_latest_status",
                repair_loop.get("promotion_review_evidence_recheck_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_targeted_probe_in_progress": targeted_probe_in_progress,
        "promotion_review_targeted_probe_completed": targeted_probe_completed,
        "promotion_review_targeted_probe_failed": targeted_probe_failed,
        "promotion_review_targeted_probe_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_latest_status",
                repair_loop.get("promotion_review_targeted_probe_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_targeted_probe_recheck_in_progress": targeted_probe_recheck_in_progress,
        "promotion_review_targeted_probe_recheck_completed": targeted_probe_recheck_completed,
        "promotion_review_targeted_probe_recheck_failed": targeted_probe_recheck_failed,
        "promotion_review_targeted_probe_recheck_latest_status": str(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_latest_status",
                repair_loop.get("promotion_review_targeted_probe_recheck_latest_status", ""),
            )
            or ""
        ),
        "promotion_review_latest_status": latest_status,
        "promotion_review_latest_age_seconds": float(
            journal.get(
                "stage_e_observed_acceptance_candidate_recovery_review_latest_age_seconds",
                repair_loop.get("promotion_review_latest_age_seconds", 0.0),
            )
            or 0.0
        ),
        "promotion_review_count": int(
            journal.get("stage_e_observed_acceptance_candidate_recovery_review_count", 0) or 0
        ),
        "promotion_review_success_count": int(status_counts.get("success", 0) or 0),
        "promotion_review_pending_count": int(status_counts.get("pending", 0) or 0),
        "promotion_review_failed_count": int(
            status_counts.get("failed", 0)
            + status_counts.get("timeout", 0)
            + status_counts.get("error", 0)
        ),
        "recovery_source": str(repair_loop.get("recovery_source", "") or ""),
        "next_review_action": str(repair_loop.get("next_review_action", "") or ""),
    }


def format_release_summary(report: Dict[str, Any]) -> str:
    criteria = report.get("criteria", {}) if isinstance(report.get("criteria"), dict) else {}
    agent = report.get("agent", {}) if isinstance(report.get("agent"), dict) else {}
    inference = report.get("inference", {}) if isinstance(report.get("inference"), dict) else {}
    memory_health = inference.get("memory_health", {}) if isinstance(inference.get("memory_health"), dict) else {}
    conversational_readiness = (
        memory_health.get("conversational_readiness", {})
        if isinstance(memory_health.get("conversational_readiness"), dict)
        else {}
    )
    predictor_state_snapshot = (
        memory_health.get("predictor_state_snapshot", {})
        if isinstance(memory_health.get("predictor_state_snapshot"), dict)
        else {}
    )
    adaptation_state_snapshot = (
        memory_health.get("adaptation_state_snapshot", {})
        if isinstance(memory_health.get("adaptation_state_snapshot"), dict)
        else {}
    )
    future_state_runtime_state = (
        memory_health.get("future_state_runtime_state", {})
        if isinstance(memory_health.get("future_state_runtime_state"), dict)
        else {}
    )
    accuracy = report.get("accuracy", {}) if isinstance(report.get("accuracy"), dict) else {}
    stage_a_acceptance = (
        accuracy.get("stage_a_acceptance", {})
        if isinstance(accuracy.get("stage_a_acceptance"), dict)
        else {}
    )
    stage_b_readiness = (
        accuracy.get("stage_b_readiness", {})
        if isinstance(accuracy.get("stage_b_readiness"), dict)
        else {}
    )
    metadata = report.get("release_metadata", {}) if isinstance(report.get("release_metadata"), dict) else {}
    gate = report.get("release_gate", {}) if isinstance(report.get("release_gate"), dict) else {}
    checklist = report.get("release_checklist", {}) if isinstance(report.get("release_checklist"), dict) else {}
    auto_dispatch = (
        report.get("repair_auto_dispatch", {})
        if isinstance(report.get("repair_auto_dispatch"), dict)
        else {}
    )
    research_review = (
        report.get("research_review", {})
        if isinstance(report.get("research_review"), dict)
        else {}
    )
    research_review_compact = (
        research_review.get("compact", {})
        if isinstance(research_review.get("compact"), dict)
        else {}
    )
    research_journal_summary = (
        report.get("research_journal_summary", {})
        if isinstance(report.get("research_journal_summary"), dict)
        else {}
    )
    completed_evidence_review = summarize_completed_roadmap_patch_evidence_review(
        research_journal_summary
    )
    research_planner_task_status = compact_release_research_planner_task_status(
        report,
        research_review_compact=research_review_compact,
    )
    stage_e_recovery_review_status = compact_release_stage_e_recovery_review_status(report)
    accuracy_required = bool(criteria.get("require_phase3_accuracy", False))

    agent_ok = _agent_status(agent)
    inference_ok = _inference_status(inference)
    metadata_ok = _metadata_status(metadata)
    accuracy_ok = _accuracy_status(accuracy) if accuracy else (not accuracy_required)
    gate_ok = bool(gate.get("passed", False)) if gate else False
    checklist_ok = bool(checklist.get("passed", False)) if checklist else False
    overall_ok = agent_ok and inference_ok and metadata_ok and accuracy_ok and gate_ok and checklist_ok

    lines = [
        "SARA Engine Release Soak Summary",
        f"overall_status: {_status_label(overall_ok)}",
        f"profile: {criteria.get('profile_name', 'unknown')}",
        f"duration_seconds: {report.get('duration_seconds', 0.0)}",
        f"shipping_ready_profile: {criteria.get('shipping_ready', False)}",
        "",
        "Agent",
        f"- status: {_status_label(agent_ok)}",
        f"- turns: {agent.get('turns', 0)} / min {agent.get('min_turns_required', 0)}",
        f"- history_bounded: {agent.get('history_bounded', False)}",
        f"- issue_count: {agent.get('issue_count', 0)}",
        "",
        "Inference",
        f"- status: {_status_label(inference_ok)}",
        f"- iterations: {inference.get('iterations', 0)} / min {inference.get('min_iterations_required', 0)}",
        f"- roundtrip_ok: {inference.get('roundtrip_ok', False)}",
        f"- tuple_keys_only: {inference.get('tuple_keys_only', False)}",
        f"- pattern_count: {inference.get('pattern_count', 0)}",
        f"- session_memory_keys: {', '.join(memory_health.get('session_memory_keys', []))}",
        f"- diagnostic_memory_hits: {', '.join(memory_health.get('diagnostic_memory_hits', []))}",
        f"- predictor_state_keys: {', '.join(memory_health.get('predictor_state_keys', []))}",
        f"- adaptation_state_keys: {', '.join(memory_health.get('adaptation_state_keys', []))}",
        f"- profile_memory_ready: {conversational_readiness.get('profile_memory_ready', False)}",
        f"- next_step_ready: {conversational_readiness.get('next_step_ready', False)}",
        f"- predictor_state_ready: {conversational_readiness.get('predictor_state_ready', False)}",
        f"- predictive_simulation_ready: {conversational_readiness.get('predictive_simulation_ready', False)}",
        f"- meta_adaptation_ready: {conversational_readiness.get('meta_adaptation_ready', False)}",
        f"- session_memory_observable: {conversational_readiness.get('session_memory_observable', False)}",
        f"- operator_trace_ready: {conversational_readiness.get('operator_trace_ready', False)}",
        f"- speculative_trace_ready: {conversational_readiness.get('speculative_trace_ready', False)}",
        f"- fluid_trace_ready: {conversational_readiness.get('fluid_trace_ready', False)}",
        f"- runtime_transition_count: {int(future_state_runtime_state.get('transition_count', 0) or 0)}",
        f"- runtime_shift_count: {int(future_state_runtime_state.get('shift_count', 0) or 0)}",
        f"- runtime_simulated_branch_count: {int(future_state_runtime_state.get('last_simulated_branch_count', 0) or 0)}",
        f"- runtime_best_simulated_branch: {future_state_runtime_state.get('last_best_simulated_branch', '')}",
        f"- runtime_transition_operator: {future_state_runtime_state.get('last_transition_operator', '')}",
        f"- runtime_verified_operator: {future_state_runtime_state.get('last_verified_operator', '')}",
        f"- runtime_operator_consistency_ratio: {float(future_state_runtime_state.get('operator_consistency_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_speculative_acceptance_ratio: {float(future_state_runtime_state.get('speculative_acceptance_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_speculative_rollback_ratio: {float(future_state_runtime_state.get('speculative_rollback_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_counterfactual_viability_ratio: {float(future_state_runtime_state.get('counterfactual_viability_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_rewarded_selection_ratio: {float(future_state_runtime_state.get('rewarded_selection_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_policy_stability_ratio: {float(future_state_runtime_state.get('policy_stability_ratio', 0.0) or 0.0):.3f}",
        f"- runtime_energy_aware_preference_ratio: {float(future_state_runtime_state.get('energy_aware_preference_ratio', 0.0) or 0.0):.3f}",
        "",
        "Release Metadata",
        f"- status: {_status_label(metadata_ok)}",
        f"- version: {metadata.get('pyproject_version', '')}",
        f"- versions_match: {metadata.get('versions_match', False)}",
        f"- console_scripts: {', '.join(metadata.get('console_scripts', []))}",
        f"- release_notes_heading: {metadata.get('release_notes_heading', '')}",
    ]

    if accuracy:
        trend = accuracy.get("trend", {}) if isinstance(accuracy.get("trend"), dict) else {}
        focus_summary = accuracy.get("focus_summary", {}) if isinstance(accuracy.get("focus_summary"), dict) else {}
        focus_trend = accuracy.get("focus_trend", {}) if isinstance(accuracy.get("focus_trend"), dict) else {}
        component_reports = (
            accuracy.get("component_reports", {})
            if isinstance(accuracy.get("component_reports"), dict)
            else {}
        )
        lines.extend(
            [
                "",
                "Accuracy",
                f"- status: {_status_label(accuracy_ok)}",
                f"- suite_name: {accuracy.get('suite_name', '')}",
                f"- passed: {accuracy.get('passed', False)}",
                f"- overall_score: {accuracy.get('overall_score', 0.0):.3f}",
                f"- regression_count: {trend.get('regression_count', 0)}",
                f"- gate_regression_count: {trend.get('gate_regression_count', trend.get('regression_count', 0))}",
                f"- stage_a_status: {_status_label(bool(stage_a_acceptance.get('passed', False)))}",
                f"- stage_a_acc_target_met: {bool(stage_a_acceptance.get('checks', {}).get('overall.acc_target_0_95', False))}",
                f"- stage_a_zero_regressions: {bool(stage_a_acceptance.get('checks', {}).get('trend.zero_regressions', False))}",
                f"- stage_b_status: {_status_label(bool(stage_b_readiness.get('passed', False)))}",
                f"- stage_b_readiness_score: {float(stage_b_readiness.get('readiness_score', 0.0)):.3f}",
                f"- stage_b_minimum_requirements_passed: {bool(stage_b_readiness.get('minimum_requirements_passed', False))}",
                f"- stage_b_transition_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_integrity', False))}",
                f"- stage_b_command_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_command_integrity', False))}",
                f"- stage_b_predictor_snapshot_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_predictor_snapshot_integrity', False))}",
                f"- stage_b_runtime_tracking_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_runtime_tracking_integrity', False))}",
                f"- stage_b_shift_tracking_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_shift_tracking_integrity', False))}",
                f"- stage_b_operator_coverage_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_operator_coverage', False))}",
                f"- stage_b_operator_consistency_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_operator_consistency', False))}",
                f"- stage_b_counterfactual_viability_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_counterfactual_branch_viability', False))}",
                f"- stage_b_fluid_trace_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_fluid_trace_integrity', False))}",
                f"- stage_b_fluid_support_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_fluid_support_integrity', False))}",
                f"- stage_b_refinement_loop_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_refinement_loop_integrity', False))}",
                f"- stage_b_adaptive_refinement_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_adaptive_refinement', False))}",
                f"- stage_b_rewarded_action_selection_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_rewarded_action_selection_integrity', False))}",
                f"- stage_b_policy_update_stability_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_policy_update_stability', False))}",
                f"- stage_b_energy_aware_preference_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_energy_aware_action_preference', False))}",
                f"- stage_b_focused_retrieval_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_focused_retrieval_hit_ratio', False))}",
                f"- stage_b_branch_decision_consistency_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_branch_level_decision_consistency', False))}",
                f"- stage_b_rlm_observation_candidate_ready: {bool(stage_b_readiness.get('rlm_observation_candidate_ready', False))}",
                f"- stage_b_rlm_observation_candidate_failure_count: {int(stage_b_readiness.get('rlm_observation_candidate_failure_count', 0) or 0)}",
                f"- stage_b_rlm_observation_candidate_promoted: {bool(stage_b_readiness.get('rlm_observation_candidate_promoted', False))}",
                f"- stage_b_rlm_observation_consecutive_passes: {int(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('consecutive_passes', 0) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else 0)}",
                f"- stage_b_rlm_observation_required_streak: {int(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('required_streak', 3) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else 3)}",
                f"- stage_b_rlm_observation_promotion_recommended: {bool(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('recommended', False) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else False)}",
                f"- stage_b_branching_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_branching_integrity', False))}",
                f"- stage_b_simulation_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_simulation_integrity', False))}",
                f"- stage_b_speculative_acceptance_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_speculative_acceptance_ratio', False))}",
                f"- stage_b_speculative_rollback_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_speculative_rollback_observability', False))}",
                f"- stage_b_promotion_candidate_ready: {bool(stage_b_readiness.get('promotion_candidate_ready', False))}",
                f"- stage_b_promotion_candidate_failure_count: {int(stage_b_readiness.get('promotion_candidate_failure_count', 0) or 0)}",
                f"- stage_b_promotion_candidate_promoted: {bool(stage_b_readiness.get('promotion_candidate_promoted', False))}",
                f"- stage_b_promotion_consecutive_passes: {int(stage_b_readiness.get('promotion_readiness', {}).get('consecutive_passes', 0) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else 0)}",
                f"- stage_b_promotion_required_streak: {int(stage_b_readiness.get('promotion_readiness', {}).get('required_streak', 3) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else 3)}",
                f"- stage_b_promotion_recommended: {bool(stage_b_readiness.get('promotion_readiness', {}).get('recommended', False) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else False)}",
            ]
        )
        if research_review_compact:
            lines.extend(
                [
                    "",
                    "Research Review",
                    f"- status: {_status_label(bool(research_review_compact.get('passed', False)))}",
                    f"- review_score: {float(research_review_compact.get('review_score', 0.0) or 0.0):.3f}",
                    f"- release_gate_blocking: {bool(research_review_compact.get('release_gate_blocking', False))}",
                    f"- requires_human_approval: {bool(research_review_compact.get('requires_human_approval', True))}",
                    f"- next_hypothesis_count: {int(research_review_compact.get('next_hypothesis_count', 0) or 0)}",
                    f"- stable_hypothesis_count: {int(research_review_compact.get('stable_hypothesis_count', 0) or 0)}",
                    f"- regression_watchlist_count: {int(research_review_compact.get('regression_watchlist_count', 0) or 0)}",
                    f"- negative_result_count: {int(research_review_compact.get('negative_result_count', 0) or 0)}",
                    f"- bounded_experiment_graph_node_count: {int(research_review_compact.get('bounded_experiment_graph_node_count', 0) or 0)}",
                    f"- bounded_experiment_graph_edge_count: {int(research_review_compact.get('bounded_experiment_graph_edge_count', 0) or 0)}",
                    f"- sara_policy_dimension_count: {int(research_review_compact.get('sara_policy_dimension_count', 0) or 0)}",
                    f"- sara_policy_needs_review_count: {int(research_review_compact.get('sara_policy_needs_review_count', 0) or 0)}",
                    f"- experiment_adoption_candidate_count: {int(research_review_compact.get('experiment_adoption_candidate_count', 0) or 0)}",
                    f"- experiment_regressing_item_count: {int(research_review_compact.get('experiment_regressing_item_count', 0) or 0)}",
                    f"- experiment_falsified_item_count: {int(research_review_compact.get('experiment_falsified_item_count', 0) or 0)}",
                    f"- experiment_human_review_pending_count: {int(research_review_compact.get('experiment_human_review_pending_count', 0) or 0)}",
                    f"- experiment_priority_action_count: {int(research_review_compact.get('experiment_priority_action_count', 0) or 0)}",
                    f"- experiment_top_priority_source: {str(research_review_compact.get('experiment_top_priority_source', '') or '')}",
                    f"- experiment_top_priority_category: {str(research_review_compact.get('experiment_top_priority_category', '') or '')}",
                    f"- experiment_promotion_target_candidate_count: {int(research_review_compact.get('experiment_promotion_target_candidate_count', 0) or 0)}",
                    f"- experiment_promotion_target_review_action_count: {int(research_review_compact.get('experiment_promotion_target_review_action_count', 0) or 0)}",
                    f"- roadmap_patch_rejection_suppressed_count: {int(research_review_compact.get('roadmap_patch_rejection_suppressed_count', 0) or 0)}",
                    f"- roadmap_patch_rejection_refreshed_count: {int(research_review_compact.get('roadmap_patch_rejection_refreshed_count', 0) or 0)}",
                    f"- completed_evidence_pending_review_count: {int(completed_evidence_review.get('pending_review_count', 0) or 0)}",
                    f"- completed_evidence_pending_review_keys: {', '.join(completed_evidence_review.get('pending_review_keys', [])[:5]) if isinstance(completed_evidence_review.get('pending_review_keys', []), list) else ''}",
                    f"- planner_task_pending_count: {int(research_planner_task_status.get('pending_count', 0) or 0)}",
                    f"- planner_task_completed_count: {int(research_planner_task_status.get('completed_count', 0) or 0)}",
                    f"- planner_task_completion_ratio: {float(research_planner_task_status.get('completion_ratio', 1.0) or 0.0):.3f}",
                    f"- planner_task_cleanup_needed: {bool(research_planner_task_status.get('cleanup_needed', False))}",
                    f"- planner_task_cleanup_pending_count: {int(research_planner_task_status.get('cleanup_pending_count', 0) or 0)}",
                    f"- planner_task_cleanup_success_count: {int(research_planner_task_status.get('cleanup_success_count', 0) or 0)}",
                    f"- planner_task_cleanup_skipped_count: {int(research_planner_task_status.get('cleanup_skipped_count', 0) or 0)}",
                    f"- planner_task_cleanup_stalled: {bool(research_planner_task_status.get('cleanup_stalled', False))}",
                    f"- planner_task_cleanup_stalled_reason: {str(research_planner_task_status.get('cleanup_stalled_reason', '') or '')}",
                    f"- planner_task_cleanup_stalled_action_source: {str(research_planner_task_status.get('cleanup_stalled_action_source', '') or '')}",
                    f"- stage_e_recovery_review_available: {bool(stage_e_recovery_review_status.get('available', False))}",
                    f"- stage_e_recovery_confirmed: {bool(stage_e_recovery_review_status.get('recovery_confirmed', False))}",
                    f"- stage_e_recovery_review_recommended: {bool(stage_e_recovery_review_status.get('promotion_review_recommended', False))}",
                    f"- stage_e_recovery_review_completed: {bool(stage_e_recovery_review_status.get('promotion_review_completed', False))}",
                    f"- stage_e_recovery_review_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_in_progress', False))}",
                    f"- stage_e_recovery_review_stale: {bool(stage_e_recovery_review_status.get('promotion_review_stale', False))}",
                    f"- stage_e_recovery_review_followup_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_followup_in_progress', False))}",
                    f"- stage_e_recovery_review_followup_completed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_completed', False))}",
                    f"- stage_e_recovery_review_followup_failed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_failed', False))}",
                    f"- stage_e_recovery_review_followup_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_followup_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_followup_retry_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_in_progress', False))}",
                    f"- stage_e_recovery_review_followup_retry_completed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_completed', False))}",
                    f"- stage_e_recovery_review_followup_retry_failed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_failed', False))}",
                    f"- stage_e_recovery_review_followup_retry_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_followup_retry_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_followup_retry_escalation_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_escalation_in_progress', False))}",
                    f"- stage_e_recovery_review_followup_retry_escalation_completed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_escalation_completed', False))}",
                    f"- stage_e_recovery_review_followup_retry_escalation_failed: {bool(stage_e_recovery_review_status.get('promotion_review_followup_retry_escalation_failed', False))}",
                    f"- stage_e_recovery_review_followup_retry_escalation_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_followup_retry_escalation_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_evidence_collection_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_collection_in_progress', False))}",
                    f"- stage_e_recovery_review_evidence_collection_completed: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_collection_completed', False))}",
                    f"- stage_e_recovery_review_evidence_collection_failed: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_collection_failed', False))}",
                    f"- stage_e_recovery_review_evidence_collection_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_evidence_collection_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_evidence_recheck_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_recheck_in_progress', False))}",
                    f"- stage_e_recovery_review_evidence_recheck_completed: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_recheck_completed', False))}",
                    f"- stage_e_recovery_review_evidence_recheck_failed: {bool(stage_e_recovery_review_status.get('promotion_review_evidence_recheck_failed', False))}",
                    f"- stage_e_recovery_review_evidence_recheck_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_evidence_recheck_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_targeted_probe_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_in_progress', False))}",
                    f"- stage_e_recovery_review_targeted_probe_completed: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_completed', False))}",
                    f"- stage_e_recovery_review_targeted_probe_failed: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_failed', False))}",
                    f"- stage_e_recovery_review_targeted_probe_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_targeted_probe_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_targeted_probe_recheck_in_progress: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_recheck_in_progress', False))}",
                    f"- stage_e_recovery_review_targeted_probe_recheck_completed: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_recheck_completed', False))}",
                    f"- stage_e_recovery_review_targeted_probe_recheck_failed: {bool(stage_e_recovery_review_status.get('promotion_review_targeted_probe_recheck_failed', False))}",
                    f"- stage_e_recovery_review_targeted_probe_recheck_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_targeted_probe_recheck_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_latest_status: {str(stage_e_recovery_review_status.get('promotion_review_latest_status', '') or '')}",
                    f"- stage_e_recovery_review_latest_age_seconds: {float(stage_e_recovery_review_status.get('promotion_review_latest_age_seconds', 0.0) or 0.0):.1f}",
                    f"- stage_e_recovery_review_count: {int(stage_e_recovery_review_status.get('promotion_review_count', 0) or 0)}",
                    f"- stage_e_recovery_review_success_count: {int(stage_e_recovery_review_status.get('promotion_review_success_count', 0) or 0)}",
                    f"- stage_e_recovery_review_pending_count: {int(stage_e_recovery_review_status.get('promotion_review_pending_count', 0) or 0)}",
                    f"- stage_e_recovery_review_failed_count: {int(stage_e_recovery_review_status.get('promotion_review_failed_count', 0) or 0)}",
                    f"- stage_e_recovery_source: {str(stage_e_recovery_review_status.get('recovery_source', '') or '')}",
                    f"- stage_e_recovery_next_review_action: {str(stage_e_recovery_review_status.get('next_review_action', '') or '')}",
                    f"- next_hypothesis_ids: {', '.join(research_review_compact.get('next_hypothesis_ids', [])) if isinstance(research_review_compact.get('next_hypothesis_ids', []), list) else ''}",
                    f"- regression_watchlist_ids: {', '.join(research_review_compact.get('regression_watchlist_ids', [])) if isinstance(research_review_compact.get('regression_watchlist_ids', []), list) else ''}",
                ]
            )
        if focus_summary:
            few_shot = focus_summary.get("few_shot", {}) if isinstance(focus_summary.get("few_shot"), dict) else {}
            continual = focus_summary.get("continual", {}) if isinstance(focus_summary.get("continual"), dict) else {}
            retrieval_hygiene = (
                focus_summary.get("retrieval_hygiene", {})
                if isinstance(focus_summary.get("retrieval_hygiene"), dict)
                else {}
            )
            adaptive_readiness = (
                focus_summary.get("adaptive_readiness", {})
                if isinstance(focus_summary.get("adaptive_readiness"), dict)
                else {}
            )
            adaptive_metrics_detail = (
                adaptive_readiness.get("metrics", {})
                if isinstance(adaptive_readiness.get("metrics"), dict)
                else {}
            )
            predictive_readiness = (
                focus_summary.get("predictive_readiness", {})
                if isinstance(focus_summary.get("predictive_readiness"), dict)
                else {}
            )
            efficiency_readiness = (
                focus_summary.get("efficiency_readiness", {})
                if isinstance(focus_summary.get("efficiency_readiness"), dict)
                else {}
            )
            consolidation_readiness = (
                focus_summary.get("consolidation_readiness", {})
                if isinstance(focus_summary.get("consolidation_readiness"), dict)
                else {}
            )
            efficiency_metrics_detail = (
                efficiency_readiness.get("metrics", {})
                if isinstance(efficiency_readiness.get("metrics"), dict)
                else {}
            )
            efficiency_component = (
                component_reports.get("energy_efficiency", {})
                if isinstance(component_reports.get("energy_efficiency"), dict)
                else {}
            )
            efficiency_component_details = (
                efficiency_component.get("details", {})
                if isinstance(efficiency_component.get("details"), dict)
                else {}
            )
            efficiency_component_metrics = (
                efficiency_component.get("metrics", {})
                if isinstance(efficiency_component.get("metrics"), dict)
                else {}
            )
            neuromorphic_profile_trend = (
                efficiency_component.get("neuromorphic_profile_trend", {})
                if isinstance(efficiency_component.get("neuromorphic_profile_trend"), dict)
                else {}
            )
            neuromorphic_trend_compact = compact_neuromorphic_profile_trend(
                neuromorphic_profile_trend
            )
            neuromorphic_regression_detail_line = str(
                neuromorphic_trend_compact.get("regression_detail_line", "none") or "none"
            )
            neuromorphic_policy_change_detail_line = str(
                neuromorphic_trend_compact.get("policy_change_detail_line", "none") or "none"
            )
            retrieval_hygiene_trend = (
                focus_trend.get("retrieval_hygiene", {})
                if isinstance(focus_trend.get("retrieval_hygiene"), dict)
                else {}
            )
            adaptive_readiness_trend = (
                focus_trend.get("adaptive_readiness", {})
                if isinstance(focus_trend.get("adaptive_readiness"), dict)
                else {}
            )
            predictive_readiness_trend = (
                focus_trend.get("predictive_readiness", {})
                if isinstance(focus_trend.get("predictive_readiness"), dict)
                else {}
            )
            efficiency_readiness_trend = (
                focus_trend.get("efficiency_readiness", {})
                if isinstance(focus_trend.get("efficiency_readiness"), dict)
                else {}
            )
            consolidation_readiness_trend = (
                focus_trend.get("consolidation_readiness", {})
                if isinstance(focus_trend.get("consolidation_readiness"), dict)
                else {}
            )
            direction_shift_trend = _extract_metric_trend(
                trend,
                "agent_dialogue.direction_shift_following",
            )
            adaptation_parameter_integrity_trend = _extract_metric_trend(
                trend,
                "task_switch_adaptation.meta_adaptation_parameter_integrity",
            )
            predictive_command_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_command_integrity",
            )
            predictive_counterfactual_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_counterfactual_integrity",
            )
            predictive_counterfactual_usefulness_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_counterfactual_usefulness",
            )
            predictive_branching_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_branching_integrity",
            )
            predictive_options_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_options_integrity",
            )
            predictive_ranking_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_ranking_integrity",
            )
            predictive_decision_brief_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_decision_brief_integrity",
            )
            predictive_choice_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_choice_integrity",
            )
            predictive_choice_reason_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_choice_reason_integrity",
            )
            hierarchical_context_trend = _extract_metric_trend(
                trend,
                "spiking_llm.hierarchical_context_integrity",
            )
            memory_per_success_trend = _extract_metric_trend(
                trend,
                "energy_efficiency.memory_per_success_proxy",
            )
            stochastic_readout_trend = _extract_metric_trend(
                trend,
                "energy_efficiency.stochastic_readout_integrity",
            )
            predictive_shift_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_shift_tracking_integrity",
            )
            predictive_simulation_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_simulation_integrity",
            )
            predictive_fluid_trace_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_fluid_trace_integrity",
            )
            predictive_fluid_support_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_fluid_support_integrity",
            )
            predictive_refinement_loop_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_refinement_loop_integrity",
            )
            predictive_adaptive_refinement_trend = _extract_metric_trend(
                trend,
                "future_state_consistency.future_state_adaptive_refinement",
            )
            consolidation_replay_recovery_trend = _extract_metric_trend(
                trend,
                "continual_consolidation.replay_recovery_integrity",
            )
            consolidation_reindex_trend = _extract_metric_trend(
                trend,
                "continual_consolidation.replay_upgrade_reindex_integrity",
            )
            consolidation_health_index_trend = _extract_metric_trend(
                trend,
                "continual_consolidation.memory_health_index_integrity",
            )
            consolidation_replay_noise_resilience_trend = _extract_metric_trend(
                trend,
                "continual_consolidation.replay_noise_resilience_integrity",
            )
            consolidation_astro_modulation_trend = _extract_metric_trend(
                trend,
                "continual_consolidation.astro_modulation_stability",
            )
            predictive_component = (
                component_reports.get("future_state_consistency", {})
                if isinstance(component_reports.get("future_state_consistency"), dict)
                else {}
            )
            agent_dialogue_component = (
                component_reports.get("agent_dialogue", {})
                if isinstance(component_reports.get("agent_dialogue"), dict)
                else {}
            )
            agent_dialogue_metrics = (
                agent_dialogue_component.get("metrics", {})
                if isinstance(agent_dialogue_component.get("metrics"), dict)
                else {}
            )
            agent_dialogue_details = (
                agent_dialogue_component.get("details", {})
                if isinstance(agent_dialogue_component.get("details"), dict)
                else {}
            )
            agent_dialogue_results = (
                agent_dialogue_details.get("test_results", [])
                if isinstance(agent_dialogue_details.get("test_results"), list)
                else []
            )
            predictive_details = (
                predictive_component.get("details", {})
                if isinstance(predictive_component.get("details"), dict)
                else {}
            )
            predictive_results = (
                predictive_details.get("test_results", [])
                if isinstance(predictive_details.get("test_results"), list)
                else []
            )
            lines.extend(
                [
                    "",
                    "Phase 3 Focus",
                    f"- few_shot_status: {_status_label(bool(few_shot.get('passed', False)))}",
                    f"- few_shot_score: {float(few_shot.get('score', 0.0)):.3f}",
                    f"- hierarchical_context_trend: {hierarchical_context_trend.get('status', 'NEW')}",
                    f"- hierarchical_context_delta: {float(hierarchical_context_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- continual_status: {_status_label(bool(continual.get('passed', False)))}",
                    f"- continual_score: {float(continual.get('score', 0.0)):.3f}",
                    f"- retrieval_hygiene_status: {_status_label(bool(retrieval_hygiene.get('passed', False)))}",
                    f"- retrieval_hygiene_score: {float(retrieval_hygiene.get('score', 0.0)):.3f}",
                    f"- retrieval_hygiene_trend: {retrieval_hygiene_trend.get('status', 'NEW')}",
                    f"- retrieval_hygiene_delta: {float(retrieval_hygiene_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- adaptive_readiness_status: {_status_label(bool(adaptive_readiness.get('passed', False)))}",
                    f"- adaptive_readiness_score: {float(adaptive_readiness.get('score', 0.0)):.3f}",
                    f"- adaptive_readiness_trend: {adaptive_readiness_trend.get('status', 'NEW')}",
                    f"- adaptive_readiness_delta: {float(adaptive_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- adaptation_parameter_integrity: {float(adaptive_metrics_detail.get('task_switch_adaptation.meta_adaptation_parameter_integrity', 0.0)):.3f}",
                    f"- adaptation_parameter_integrity_trend: {adaptation_parameter_integrity_trend.get('status', 'NEW')}",
                    f"- adaptation_parameter_integrity_delta: {float(adaptation_parameter_integrity_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- direction_shift_following: {float(agent_dialogue_metrics.get('direction_shift_following', 0.0)):.3f}",
                    f"- direction_shift_trend: {direction_shift_trend.get('status', 'NEW')}",
                    f"- direction_shift_delta: {float(direction_shift_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_readiness_status: {_status_label(bool(predictive_readiness.get('passed', False)))}",
                    f"- predictive_readiness_score: {float(predictive_readiness.get('score', 0.0)):.3f}",
                    f"- predictive_readiness_trend: {predictive_readiness_trend.get('status', 'NEW')}",
                    f"- predictive_readiness_delta: {float(predictive_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_command_trend: {predictive_command_trend.get('status', 'NEW')}",
                    f"- predictive_command_delta: {float(predictive_command_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_counterfactual_trend: {predictive_counterfactual_trend.get('status', 'NEW')}",
                    f"- predictive_counterfactual_delta: {float(predictive_counterfactual_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_counterfactual_usefulness_trend: {predictive_counterfactual_usefulness_trend.get('status', 'NEW')}",
                    f"- predictive_counterfactual_usefulness_delta: {float(predictive_counterfactual_usefulness_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_branching_trend: {predictive_branching_trend.get('status', 'NEW')}",
                    f"- predictive_branching_delta: {float(predictive_branching_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_options_trend: {predictive_options_trend.get('status', 'NEW')}",
                    f"- predictive_options_delta: {float(predictive_options_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_ranking_trend: {predictive_ranking_trend.get('status', 'NEW')}",
                    f"- predictive_ranking_delta: {float(predictive_ranking_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_decision_brief_trend: {predictive_decision_brief_trend.get('status', 'NEW')}",
                    f"- predictive_decision_brief_delta: {float(predictive_decision_brief_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_choice_trend: {predictive_choice_trend.get('status', 'NEW')}",
                    f"- predictive_choice_delta: {float(predictive_choice_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_choice_reason_trend: {predictive_choice_reason_trend.get('status', 'NEW')}",
                    f"- predictive_choice_reason_delta: {float(predictive_choice_reason_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_shift_trend: {predictive_shift_trend.get('status', 'NEW')}",
                    f"- predictive_shift_delta: {float(predictive_shift_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_simulation_trend: {predictive_simulation_trend.get('status', 'NEW')}",
                    f"- predictive_simulation_delta: {float(predictive_simulation_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_fluid_trace_trend: {predictive_fluid_trace_trend.get('status', 'NEW')}",
                    f"- predictive_fluid_trace_delta: {float(predictive_fluid_trace_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_fluid_support_trend: {predictive_fluid_support_trend.get('status', 'NEW')}",
                    f"- predictive_fluid_support_delta: {float(predictive_fluid_support_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_refinement_loop_trend: {predictive_refinement_loop_trend.get('status', 'NEW')}",
                    f"- predictive_refinement_loop_delta: {float(predictive_refinement_loop_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- predictive_adaptive_refinement_trend: {predictive_adaptive_refinement_trend.get('status', 'NEW')}",
                    f"- predictive_adaptive_refinement_delta: {float(predictive_adaptive_refinement_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- efficiency_readiness_status: {_status_label(bool(efficiency_readiness.get('passed', False)))}",
                    f"- efficiency_readiness_score: {float(efficiency_readiness.get('score', 0.0)):.3f}",
                    f"- energy_per_success_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.energy_per_success_proxy', 0.0)):.3f}",
                    f"- performance_energy_ratio_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.performance_energy_ratio_proxy', 0.0)):.3f}",
                    f"- ann_cost_advantage_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.ann_cost_advantage_proxy', 0.0)):.3f}",
                    f"- sparse_event_cost_score: {float(efficiency_metrics_detail.get('energy_efficiency.sparse_event_cost_score', 0.0)):.3f}",
                    f"- brain_efficiency_alignment_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.brain_efficiency_alignment_proxy', 0.0)):.3f}",
                    f"- memory_per_success_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.memory_per_success_proxy', 0.0)):.3f}",
                    f"- low_overhead_route_score: {float(efficiency_metrics_detail.get('energy_efficiency.low_overhead_route_score', 0.0)):.3f}",
                    f"- bounded_latency_score: {float(efficiency_metrics_detail.get('energy_efficiency.bounded_latency_score', 0.0)):.3f}",
                    f"- stochastic_readout_integrity: {float(efficiency_metrics_detail.get('energy_efficiency.stochastic_readout_integrity', 0.0)):.3f}",
                    f"- neuromorphic_stage_e_state_trace_ir_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_state_trace_ir_observed', 0.0)):.3f}",
                    f"- neuromorphic_stage_e_routing_hint_coverage_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_routing_hint_coverage_observed', 0.0)):.3f}",
                    f"- neuromorphic_stage_e_online_update_policy_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_online_update_policy_observed', 0.0)):.3f}",
                    f"- neuromorphic_stage_e_event_budget_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_event_budget_observed', 0.0)):.3f}",
                    f"- neuromorphic_profile_history_regression_observed: {float(efficiency_component_metrics.get('neuromorphic_profile_history_regression_observed', 0.0)):.3f}",
                    f"- neuromorphic_profile_trend_regression_count: {int(neuromorphic_profile_trend.get('regression_count', 0) or 0)}",
                    f"- neuromorphic_profile_trend_policy_change_count: {int(neuromorphic_profile_trend.get('policy_change_count', 0) or 0)}",
                    f"- neuromorphic_profile_trend_regression_details: {neuromorphic_regression_detail_line}",
                    f"- neuromorphic_profile_trend_policy_change_details: {neuromorphic_policy_change_detail_line}",
                    f"- average_state_units: {float(efficiency_component_details.get('average_state_units', 0.0) or 0.0):.3f}",
                    f"- memory_per_success_trend: {memory_per_success_trend.get('status', 'NEW')}",
                    f"- memory_per_success_delta: {float(memory_per_success_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- stochastic_readout_trend: {stochastic_readout_trend.get('status', 'NEW')}",
                    f"- stochastic_readout_delta: {float(stochastic_readout_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- efficiency_readiness_trend: {efficiency_readiness_trend.get('status', 'NEW')}",
                    f"- efficiency_readiness_delta: {float(efficiency_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_readiness_status: {_status_label(bool(consolidation_readiness.get('passed', False)))}",
                    f"- consolidation_readiness_score: {float(consolidation_readiness.get('score', 0.0)):.3f}",
                    f"- consolidation_replay_recovery_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_recovery_integrity', 0.0)):.3f}",
                    f"- consolidation_replay_upgrade_reindex_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_upgrade_reindex_integrity', 0.0)):.3f}",
                    f"- consolidation_memory_health_index_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.memory_health_index_integrity', 0.0)):.3f}",
                    f"- consolidation_replay_noise_resilience_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_noise_resilience_integrity', 0.0)):.3f}",
                    f"- consolidation_astro_modulation_stability: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.astro_modulation_stability', 0.0)):.3f}",
                    f"- consolidation_readiness_trend: {consolidation_readiness_trend.get('status', 'NEW')}",
                    f"- consolidation_readiness_delta: {float(consolidation_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_replay_recovery_trend: {consolidation_replay_recovery_trend.get('status', 'NEW')}",
                    f"- consolidation_replay_recovery_delta: {float(consolidation_replay_recovery_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_reindex_trend: {consolidation_reindex_trend.get('status', 'NEW')}",
                    f"- consolidation_reindex_delta: {float(consolidation_reindex_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_health_index_trend: {consolidation_health_index_trend.get('status', 'NEW')}",
                    f"- consolidation_health_index_delta: {float(consolidation_health_index_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_replay_noise_resilience_trend: {consolidation_replay_noise_resilience_trend.get('status', 'NEW')}",
                    f"- consolidation_replay_noise_resilience_delta: {float(consolidation_replay_noise_resilience_trend.get('delta', 0.0) or 0.0):+.3f}",
                    f"- consolidation_astro_modulation_trend: {consolidation_astro_modulation_trend.get('status', 'NEW')}",
                    f"- consolidation_astro_modulation_delta: {float(consolidation_astro_modulation_trend.get('delta', 0.0) or 0.0):+.3f}",
                ]
            )
            if agent_dialogue_results and isinstance(agent_dialogue_results[-1], dict):
                shift_detail = agent_dialogue_results[-1]
                lines.extend(
                    [
                        "",
                        "Dialogue Shift Detail",
                        f"- shift_from: {shift_detail.get('shift_from', '')}",
                        f"- shift_query: {shift_detail.get('user_input', '')}",
                        f"- shift_following_score: {float(shift_detail.get('shift_following_score', 0.0) or 0.0):.3f}",
                    ]
                )
            if predictive_results and isinstance(predictive_results[0], dict):
                representative = predictive_results[0]
                representative_predictor_state = (
                    representative.get("predictor_state", {})
                    if isinstance(representative.get("predictor_state"), dict)
                    else {}
                )
                fluid_trace_snapshot = (
                    predictor_state_snapshot.get("fluid_trace", {})
                    if isinstance(predictor_state_snapshot.get("fluid_trace"), dict)
                    else {}
                )
                representative_fluid_trace = (
                    representative.get("fluid_trace", {})
                    if isinstance(representative.get("fluid_trace"), dict)
                    else {}
                )
                speculative_trace_snapshot = (
                    predictor_state_snapshot.get("speculative_trace", {})
                    if isinstance(predictor_state_snapshot.get("speculative_trace"), dict)
                    else {}
                )
                representative_speculative_trace = (
                    representative.get("speculative_trace", {})
                    if isinstance(representative.get("speculative_trace"), dict)
                    else {}
                )
                refinement_trace_snapshot = (
                    predictor_state_snapshot.get("refinement_trace", {})
                    if isinstance(predictor_state_snapshot.get("refinement_trace"), dict)
                    else {}
                )
                representative_refinement_trace = (
                    representative.get("refinement_trace", {})
                    if isinstance(representative.get("refinement_trace"), dict)
                    else {}
                )
                lines.extend(
                    [
                        "",
                        "Predictive Detail",
                        f"- predicted_action: {representative.get('predicted_action', '')}",
                        f"- predicted_target_state: {representative.get('predicted_target_state', '')}",
                        f"- predicted_command: {representative.get('predicted_command', '')}",
                        f"- alternative_action: {representative.get('alternative_action', '')}",
                        f"- alternative_target_state: {representative.get('alternative_target_state', '')}",
                        f"- alternative_command: {representative.get('alternative_command', '')}",
                        f"- secondary_alternative_action: {representative.get('secondary_alternative_action', '')}",
                        f"- secondary_alternative_target_state: {representative.get('secondary_alternative_target_state', '')}",
                        f"- secondary_alternative_command: {representative.get('secondary_alternative_command', '')}",
                        f"- chosen_plan: {representative.get('chosen_plan', '')}",
                        f"- choice_reason: {representative.get('choice_reason', '')}",
                        f"- choice_response: {representative.get('choice_response', '')}",
                        f"- options_response: {representative.get('options_response', '')}",
                        f"- ranked_options_response: {representative.get('ranked_options_response', '')}",
                        f"- decision_brief_response: {representative.get('decision_brief_response', '')}",
                        f"- simulation_response: {representative.get('simulation_response', '')}",
                        f"- best_simulated_branch: {representative_predictor_state.get('best_simulated_branch', '')}",
                        f"- predictor_category: {predictor_state_snapshot.get('category', representative_predictor_state.get('category', ''))}",
                        f"- predictor_confidence: {float(predictor_state_snapshot.get('confidence', representative_predictor_state.get('confidence', 0.0) or 0.0)):.3f}",
                        f"- transition_operator: {predictor_state_snapshot.get('transition_operator', representative_predictor_state.get('transition_operator', ''))}",
                        f"- alternative_transition_operator: {predictor_state_snapshot.get('alternative_transition_operator', representative_predictor_state.get('alternative_transition_operator', ''))}",
                        f"- secondary_alternative_transition_operator: {predictor_state_snapshot.get('secondary_alternative_transition_operator', representative_predictor_state.get('secondary_alternative_transition_operator', ''))}",
                        f"- speculative_predicted_operator: {speculative_trace_snapshot.get('predicted_operator', representative_speculative_trace.get('predicted_operator', ''))}",
                        f"- speculative_verified_operator: {speculative_trace_snapshot.get('verified_operator', representative_speculative_trace.get('verified_operator', ''))}",
                        f"- speculative_operator_match: {bool(speculative_trace_snapshot.get('operator_match', representative_speculative_trace.get('operator_match', False)))}",
                        f"- speculative_acceptance: {bool(speculative_trace_snapshot.get('draft_verify_accepted', representative_speculative_trace.get('draft_verify_accepted', False)))}",
                        f"- speculative_rollback_observable: {bool(speculative_trace_snapshot.get('rollback_observable', representative_speculative_trace.get('rollback_observable', False)))}",
                        f"- speculative_counterfactual_viable: {bool(speculative_trace_snapshot.get('counterfactual_branch_viable', representative_speculative_trace.get('counterfactual_branch_viable', False)))}",
                        f"- refinement_triggered: {bool(refinement_trace_snapshot.get('triggered', representative_refinement_trace.get('triggered', False)))}",
                        f"- refinement_loop_count: {int(refinement_trace_snapshot.get('loop_count', representative_refinement_trace.get('loop_count', 0) or 0))}",
                        f"- refinement_selected_before: {refinement_trace_snapshot.get('selected_branch_before', representative_refinement_trace.get('selected_branch_before', ''))}",
                        f"- refinement_selected_after: {refinement_trace_snapshot.get('selected_branch_after', representative_refinement_trace.get('selected_branch_after', ''))}",
                        f"- refinement_score_gap_before: {float(refinement_trace_snapshot.get('score_gap_before', representative_refinement_trace.get('score_gap_before', 0.0) or 0.0)):.3f}",
                        f"- refinement_score_gap_after: {float(refinement_trace_snapshot.get('score_gap_after', representative_refinement_trace.get('score_gap_after', 0.0) or 0.0)):.3f}",
                        f"- fluid_bounded: {bool(fluid_trace_snapshot.get('bounded', representative_fluid_trace.get('bounded', False)))}",
                        f"- fluid_support_score: {float(fluid_trace_snapshot.get('support_score', representative_fluid_trace.get('support_score', 0.0) or 0.0)):.3f}",
                        f"- fluid_active_columns: {int(fluid_trace_snapshot.get('active_columns', representative_fluid_trace.get('active_columns', 0) or 0))}",
                        f"- fluid_total_spikes: {int(fluid_trace_snapshot.get('total_spikes', representative_fluid_trace.get('total_spikes', 0) or 0))}",
                        f"- adaptation_response_mode: {adaptation_state_snapshot.get('response_mode', '')}",
                        f"- adaptation_planning_confidence: {float(adaptation_state_snapshot.get('planning_confidence', 0.0) or 0.0):.3f}",
                        f"- adaptation_memory_weight: {float(adaptation_state_snapshot.get('memory_weight', 0.0) or 0.0):.3f}",
                        f"- adaptation_fallback_relaxation: {float(adaptation_state_snapshot.get('fallback_relaxation', 0.0) or 0.0):.3f}",
                        f"- runtime_stability_ratio: {float(future_state_runtime_state.get('stability_ratio', representative.get('runtime_state', {}).get('stability_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_operator_consistency_ratio: {float(future_state_runtime_state.get('operator_consistency_ratio', representative.get('runtime_state', {}).get('operator_consistency_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_speculative_acceptance_ratio: {float(future_state_runtime_state.get('speculative_acceptance_ratio', representative.get('runtime_state', {}).get('speculative_acceptance_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_speculative_rollback_ratio: {float(future_state_runtime_state.get('speculative_rollback_ratio', representative.get('runtime_state', {}).get('speculative_rollback_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_counterfactual_viability_ratio: {float(future_state_runtime_state.get('counterfactual_viability_ratio', representative.get('runtime_state', {}).get('counterfactual_viability_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_rewarded_selection_ratio: {float(future_state_runtime_state.get('rewarded_selection_ratio', representative.get('runtime_state', {}).get('rewarded_selection_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_policy_stability_ratio: {float(future_state_runtime_state.get('policy_stability_ratio', representative.get('runtime_state', {}).get('policy_stability_ratio', 0.0) or 0.0)):.3f}",
                        f"- runtime_energy_aware_preference_ratio: {float(future_state_runtime_state.get('energy_aware_preference_ratio', representative.get('runtime_state', {}).get('energy_aware_preference_ratio', 0.0) or 0.0)):.3f}",
                        f"- previous_target_state: {future_state_runtime_state.get('previous_target_state', representative.get('runtime_state', {}).get('previous_target_state', ''))}",
                    ]
                )
    elif accuracy_required:
        lines.extend(
            [
                "",
                "Accuracy",
                f"- status: {_status_label(False)}",
                "- suite_name: missing",
                "- passed: False",
                "- overall_score: 0.000",
                "- regression_count: 0",
                f"- stage_a_status: {_status_label(False)}",
                "- stage_a_acc_target_met: False",
                "- stage_a_zero_regressions: False",
                f"- stage_b_status: {_status_label(False)}",
                "- stage_b_readiness_score: 0.000",
                "- stage_b_minimum_requirements_passed: False",
                "- stage_b_transition_ready: False",
                "- stage_b_command_ready: False",
                "- stage_b_predictor_snapshot_ready: False",
                "- stage_b_runtime_tracking_ready: False",
                "- stage_b_shift_tracking_ready: False",
                "- stage_b_operator_coverage_ready: False",
                "- stage_b_operator_consistency_ready: False",
                "- stage_b_counterfactual_viability_ready: False",
                "- stage_b_fluid_trace_ready: False",
                "- stage_b_fluid_support_ready: False",
                "- stage_b_refinement_loop_ready: False",
                "- stage_b_adaptive_refinement_ready: False",
                "- stage_b_focused_retrieval_observed: False",
                "- stage_b_branch_decision_consistency_observed: False",
                "- stage_b_rlm_observation_candidate_ready: False",
                "- stage_b_rlm_observation_candidate_failure_count: 2",
                "- stage_b_rlm_observation_candidate_promoted: False",
                "- stage_b_rlm_observation_consecutive_passes: 0",
                "- stage_b_rlm_observation_required_streak: 3",
                "- stage_b_rlm_observation_promotion_recommended: False",
                "- stage_b_branching_ready: False",
                "- stage_b_simulation_ready: False",
                "- stage_b_speculative_acceptance_ready: False",
                "- stage_b_speculative_rollback_ready: False",
                "- stage_b_promotion_candidate_ready: False",
                "- stage_b_promotion_candidate_failure_count: 3",
                "- stage_b_promotion_consecutive_passes: 0",
                "- stage_b_promotion_required_streak: 3",
                "- stage_b_promotion_recommended: False",
            ]
        )
        lines.extend(
            [
                "",
                "Phase 3 Focus",
                f"- few_shot_status: {_status_label(False)}",
                "- few_shot_score: 0.000",
                f"- continual_status: {_status_label(False)}",
                "- continual_score: 0.000",
                f"- retrieval_hygiene_status: {_status_label(False)}",
                "- retrieval_hygiene_score: 0.000",
                "- retrieval_hygiene_trend: NEW",
                "- retrieval_hygiene_delta: +0.000",
                f"- adaptive_readiness_status: {_status_label(False)}",
                "- adaptive_readiness_score: 0.000",
                "- adaptive_readiness_trend: NEW",
                "- adaptive_readiness_delta: +0.000",
                "- adaptation_parameter_integrity: 0.000",
                "- adaptation_parameter_integrity_trend: NEW",
                "- adaptation_parameter_integrity_delta: +0.000",
                "- direction_shift_following: 0.000",
                "- direction_shift_trend: NEW",
                "- direction_shift_delta: +0.000",
                f"- predictive_readiness_status: {_status_label(False)}",
                "- predictive_readiness_score: 0.000",
                "- predictive_readiness_trend: NEW",
                "- predictive_readiness_delta: +0.000",
                "- predictive_command_trend: NEW",
                "- predictive_command_delta: +0.000",
                "- predictive_shift_trend: NEW",
                "- predictive_shift_delta: +0.000",
                "- predictive_simulation_trend: NEW",
                "- predictive_simulation_delta: +0.000",
                "- predictive_fluid_trace_trend: NEW",
                "- predictive_fluid_trace_delta: +0.000",
                "- predictive_fluid_support_trend: NEW",
                "- predictive_fluid_support_delta: +0.000",
                "- predictive_refinement_loop_trend: NEW",
                "- predictive_refinement_loop_delta: +0.000",
                "- predictive_adaptive_refinement_trend: NEW",
                "- predictive_adaptive_refinement_delta: +0.000",
                f"- efficiency_readiness_status: {_status_label(False)}",
                "- efficiency_readiness_score: 0.000",
                "- energy_per_success_proxy: 0.000",
                "- performance_energy_ratio_proxy: 0.000",
                "- ann_cost_advantage_proxy: 0.000",
                "- sparse_event_cost_score: 0.000",
                "- brain_efficiency_alignment_proxy: 0.000",
                "- memory_per_success_proxy: 0.000",
                "- low_overhead_route_score: 0.000",
                "- bounded_latency_score: 0.000",
                "- stochastic_readout_integrity: 0.000",
                "- average_state_units: 0.000",
                "- memory_per_success_trend: NEW",
                "- memory_per_success_delta: +0.000",
                "- stochastic_readout_trend: NEW",
                "- stochastic_readout_delta: +0.000",
                "- efficiency_readiness_trend: NEW",
                "- efficiency_readiness_delta: +0.000",
                f"- consolidation_readiness_status: {_status_label(False)}",
                "- consolidation_readiness_score: 0.000",
                "- consolidation_replay_recovery_integrity: 0.000",
                "- consolidation_replay_upgrade_reindex_integrity: 0.000",
                "- consolidation_memory_health_index_integrity: 0.000",
                "- consolidation_replay_noise_resilience_integrity: 0.000",
                "- consolidation_astro_modulation_stability: 0.000",
                "- consolidation_readiness_trend: NEW",
                "- consolidation_readiness_delta: +0.000",
                "- consolidation_replay_recovery_trend: NEW",
                "- consolidation_replay_recovery_delta: +0.000",
                "- consolidation_reindex_trend: NEW",
                "- consolidation_reindex_delta: +0.000",
                "- consolidation_health_index_trend: NEW",
                "- consolidation_health_index_delta: +0.000",
                "- consolidation_replay_noise_resilience_trend: NEW",
                "- consolidation_replay_noise_resilience_delta: +0.000",
                "- consolidation_astro_modulation_trend: NEW",
                "- consolidation_astro_modulation_delta: +0.000",
            ]
        )

    lines.extend(
        [
            "",
            "Gate",
            f"- status: {_status_label(gate_ok)}",
            f"- error_count: {gate.get('error_count', 0) if gate else 0}",
            f"- accuracy_required: {gate.get('accuracy_required', False) if gate else False}",
            f"- embedded_accuracy_present: {gate.get('embedded_accuracy_present', False) if gate else False}",
            f"- stage_a_passed: {gate.get('stage_a_passed', False) if gate else False}",
            f"- stage_b_passed: {gate.get('stage_b_passed', False) if gate else False}",
            f"- stage_b_minimum_requirements_passed: {gate.get('stage_b_minimum_requirements_passed', False) if gate else False}",
            f"- stage_b_minimum_failure_count: {gate.get('stage_b_minimum_failure_count', 0) if gate else 0}",
            f"- stage_b_promotion_candidate_ready: {gate.get('stage_b_promotion_candidate_ready', False) if gate else False}",
            f"- stage_b_promotion_candidate_failure_count: {gate.get('stage_b_promotion_candidate_failure_count', 0) if gate else 0}",
            f"- stage_b_promotion_candidate_promoted: {gate.get('stage_b_promotion_candidate_promoted', False) if gate else False}",
            f"- stage_b_promotion_consecutive_passes: {gate.get('stage_b_promotion_consecutive_passes', 0) if gate else 0}",
            f"- stage_b_promotion_required_streak: {gate.get('stage_b_promotion_required_streak', 3) if gate else 3}",
            f"- stage_b_promotion_recommended: {gate.get('stage_b_promotion_recommended', False) if gate else False}",
            f"- stage_b_promotion_next_step_hint: {gate.get('stage_b_promotion_next_step_hint', '') if gate else ''}",
            f"- stage_b_rlm_observation_candidate_ready: {gate.get('stage_b_rlm_observation_candidate_ready', False) if gate else False}",
            f"- stage_b_rlm_observation_candidate_failure_count: {gate.get('stage_b_rlm_observation_candidate_failure_count', 0) if gate else 0}",
            f"- stage_b_rlm_observation_candidate_promoted: {gate.get('stage_b_rlm_observation_candidate_promoted', False) if gate else False}",
            f"- stage_b_rlm_observation_consecutive_passes: {gate.get('stage_b_rlm_observation_consecutive_passes', 0) if gate else 0}",
            f"- stage_b_rlm_observation_required_streak: {gate.get('stage_b_rlm_observation_required_streak', 3) if gate else 3}",
            f"- stage_b_rlm_observation_promotion_recommended: {gate.get('stage_b_rlm_observation_promotion_recommended', False) if gate else False}",
            f"- stage_b_rlm_observation_next_step_hint: {gate.get('stage_b_rlm_observation_next_step_hint', '') if gate else ''}",
            f"- stage_c_passed: {gate.get('stage_c_passed', False) if gate else False}",
            f"- stage_c_minimum_requirements_passed: {gate.get('stage_c_minimum_requirements_passed', False) if gate else False}",
            f"- stage_c_minimum_failure_count: {gate.get('stage_c_minimum_failure_count', 0) if gate else 0}",
            f"- stage_d_passed: {gate.get('stage_d_passed', False) if gate else False}",
            f"- stage_d_minimum_requirements_passed: {gate.get('stage_d_minimum_requirements_passed', False) if gate else False}",
            f"- stage_d_minimum_failure_count: {gate.get('stage_d_minimum_failure_count', 0) if gate else 0}",
            f"- stage_d_readiness_score: {float(gate.get('stage_d_readiness_score', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_acceptance_candidate_count: {gate.get('stage_d_acceptance_candidate_count', 0) if gate else 0}",
            f"- stage_d_acceptance_candidate_ready_count: {gate.get('stage_d_acceptance_candidate_ready_count', 0) if gate else 0}",
            f"- stage_d_acceptance_candidates_ready: {gate.get('stage_d_acceptance_candidates_ready', False) if gate else False}",
            f"- stage_d_acceptance_candidate_failure_count: {gate.get('stage_d_acceptance_candidate_failure_count', 0) if gate else 0}",
            f"- stage_d_acceptance_candidate_consecutive_passes: {gate.get('stage_d_acceptance_candidate_consecutive_passes', 0) if gate else 0}",
            f"- stage_d_acceptance_candidate_required_streak: {gate.get('stage_d_acceptance_candidate_required_streak', 3) if gate else 3}",
            f"- stage_d_acceptance_candidate_stability_recommended: {gate.get('stage_d_acceptance_candidate_stability_recommended', False) if gate else False}",
            f"- stage_d_acceptance_candidate_next_step_hint: {gate.get('stage_d_acceptance_candidate_next_step_hint', '') if gate else ''}",
            f"- stage_d_acceptance_candidate_action_count: {gate.get('stage_d_acceptance_candidate_action_count', 0) if gate else 0}",
            f"- stage_d_delta_memory_candidate_ready: {gate.get('stage_d_delta_memory_candidate_ready', False) if gate else False}",
            f"- stage_d_delta_memory_candidate_failure_count: {gate.get('stage_d_delta_memory_candidate_failure_count', 0) if gate else 0}",
            f"- stage_d_delta_memory_candidate_promoted: {gate.get('stage_d_delta_memory_candidate_promoted', False) if gate else False}",
            f"- stage_d_delta_memory_consecutive_passes: {gate.get('stage_d_delta_memory_consecutive_passes', 0) if gate else 0}",
            f"- stage_d_delta_memory_required_streak: {gate.get('stage_d_delta_memory_required_streak', 3) if gate else 3}",
            f"- stage_d_delta_memory_promotion_recommended: {gate.get('stage_d_delta_memory_promotion_recommended', False) if gate else False}",
            f"- stage_d_delta_memory_next_step_hint: {gate.get('stage_d_delta_memory_next_step_hint', '') if gate else ''}",
            f"- stage_d_replay_recovery_integrity: {float(gate.get('stage_d_replay_recovery_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_replay_upgrade_reindex_integrity: {float(gate.get('stage_d_replay_upgrade_reindex_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_memory_health_index_integrity: {float(gate.get('stage_d_memory_health_index_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_replay_noise_resilience_integrity: {float(gate.get('stage_d_replay_noise_resilience_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_astro_modulation_stability: {float(gate.get('stage_d_astro_modulation_stability', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_continual_retention_observed: {float(gate.get('stage_d_manifold_continual_retention_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_trajectory_case_coverage_observed: {float(gate.get('stage_d_manifold_trajectory_case_coverage_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_average_case_recall_observed: {float(gate.get('stage_d_manifold_average_case_recall_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_scan_budget_integrity_observed: {float(gate.get('stage_d_manifold_scan_budget_integrity_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_indexed_candidate_integrity_observed: {float(gate.get('stage_d_manifold_indexed_candidate_integrity_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_index_scan_reduction_observed: {float(gate.get('stage_d_manifold_index_scan_reduction_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_capacity_pressure_recall_observed: {float(gate.get('stage_d_manifold_capacity_pressure_recall_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_capacity_pressure_scan_reduction_observed: {float(gate.get('stage_d_manifold_capacity_pressure_scan_reduction_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_replay_refresh_retention_observed: {float(gate.get('stage_d_manifold_replay_refresh_retention_observed', 0.0) if gate else 0.0):.3f}",
            f"- stage_d_manifold_replay_refresh_eviction_integrity_observed: {float(gate.get('stage_d_manifold_replay_refresh_eviction_integrity_observed', 0.0) if gate else 0.0):.3f}",
            *[
                f"- stage_d_{metric_name}: {float(gate.get(f'stage_d_{metric_name}', 0.0) if gate else 0.0):.3f}"
                for metric_name in STAGE_D_STRUCTURAL_OBSERVED_METRIC_NAMES
            ],
            f"- stage_e_passed: {gate.get('stage_e_passed', False) if gate else False}",
            f"- stage_e_minimum_requirements_passed: {gate.get('stage_e_minimum_requirements_passed', False) if gate else False}",
            f"- stage_e_minimum_failure_count: {gate.get('stage_e_minimum_failure_count', 0) if gate else 0}",
            f"- stage_e_readiness_score: {float(gate.get('stage_e_readiness_score', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_observed_acceptance_candidate_count: {gate.get('stage_e_observed_acceptance_candidate_count', 0) if gate else 0}",
            f"- stage_e_observed_acceptance_candidate_ready_count: {gate.get('stage_e_observed_acceptance_candidate_ready_count', 0) if gate else 0}",
            f"- stage_e_observed_acceptance_candidates_ready: {gate.get('stage_e_observed_acceptance_candidates_ready', False) if gate else False}",
            f"- stage_e_observed_acceptance_candidate_failure_count: {gate.get('stage_e_observed_acceptance_candidate_failure_count', 0) if gate else 0}",
            f"- stage_e_observed_acceptance_candidate_consecutive_passes: {gate.get('stage_e_observed_acceptance_candidate_consecutive_passes', 0) if gate else 0}",
            f"- stage_e_observed_acceptance_candidate_required_streak: {gate.get('stage_e_observed_acceptance_candidate_required_streak', 3) if gate else 3}",
            f"- stage_e_observed_acceptance_candidate_stability_recommended: {gate.get('stage_e_observed_acceptance_candidate_stability_recommended', False) if gate else False}",
            f"- stage_e_common_spike_space_integrity: {float(gate.get('stage_e_common_spike_space_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_temporal_compression_efficiency: {float(gate.get('stage_e_temporal_compression_efficiency', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_modality_temporal_budget_integrity: {float(gate.get('stage_e_modality_temporal_budget_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_dendritic_context_gate_stability: {float(gate.get('stage_e_dendritic_context_gate_stability', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_spiking_hjepa_latent_transition: {float(gate.get('stage_e_spiking_hjepa_latent_transition', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_reverse_reasoning_trace_integrity: {float(gate.get('stage_e_reverse_reasoning_trace_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_causal_candidate_trace_integrity: {float(gate.get('stage_e_causal_candidate_trace_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_module_orchestration_integrity: {float(gate.get('stage_e_module_orchestration_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_counterfactual_lane_integrity: {float(gate.get('stage_e_counterfactual_lane_integrity', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_action_trace_observability: {float(gate.get('stage_e_action_trace_observability', 0.0) if gate else 0.0):.3f}",
            f"- stage_e_runtime_trace_replay_consistency: {float(gate.get('stage_e_runtime_trace_replay_consistency', 0.0) if gate else 0.0):.3f}",
        *[
            f"- stage_e_{metric_name}: {float(gate.get(f'stage_e_{metric_name}', 0.0) if gate else 0.0):.3f}"
            for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
        ],
        *[
            f"- stage_e_{metric_name}: {float(gate.get(f'stage_e_{metric_name}', 0.0) if gate else 0.0):.3f}"
            for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
        ],
            f"- stage_e_linear_snn_fusion_observed_policy: {gate.get('stage_e_linear_snn_fusion_observed_policy', 'excluded_from_score_and_release_gate') if gate else 'excluded_from_score_and_release_gate'}",
            f"- stage_e_linear_snn_fusion_trend_has_previous: {gate.get('stage_e_linear_snn_fusion_trend_has_previous', False) if gate else False}",
            f"- stage_e_linear_snn_fusion_trend_regression_count: {gate.get('stage_e_linear_snn_fusion_trend_regression_count', 0) if gate else 0}",
            f"- stage_e_linear_snn_fusion_trend_release_gate_blocking: {gate.get('stage_e_linear_snn_fusion_trend_release_gate_blocking', False) if gate else False}",
            f"- stage_e_architecture_integration_observed_policy: {gate.get('stage_e_architecture_integration_observed_policy', 'excluded_from_score_and_release_gate') if gate else 'excluded_from_score_and_release_gate'}",
            f"- stage_e_architecture_integration_trend_has_previous: {gate.get('stage_e_architecture_integration_trend_has_previous', False) if gate else False}",
            f"- stage_e_architecture_integration_trend_regression_count: {gate.get('stage_e_architecture_integration_trend_regression_count', 0) if gate else 0}",
            f"- stage_e_architecture_integration_trend_release_gate_blocking: {gate.get('stage_e_architecture_integration_trend_release_gate_blocking', False) if gate else False}",
        *[
            f"- stage_e_{metric_name}: {float(gate.get(f'stage_e_{metric_name}', 0.0) if gate else 0.0):.3f}"
            for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
        ],
        *[
            f"- stage_e_{metric_name}: {float(gate.get(f'stage_e_{metric_name}', 0.0) if gate else 0.0):.3f}"
            for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
        ],
            f"- phase5_entry_passed: {gate.get('phase5_entry_passed', False) if gate else False}",
            f"- phase5_entry_readiness_score: {float(gate.get('phase5_entry_readiness_score', 0.0) if gate else 0.0):.3f}",
            f"- phase5_latent_transition_alignment: {float(gate.get('phase5_latent_transition_alignment', 0.0) if gate else 0.0):.3f}",
            f"- phase5_prediction_error_observability: {float(gate.get('phase5_prediction_error_observability', 0.0) if gate else 0.0):.3f}",
            f"- phase5_correction_event_coverage: {float(gate.get('phase5_correction_event_coverage', 0.0) if gate else 0.0):.3f}",
            f"- phase5_anti_collapse_event_diversity: {float(gate.get('phase5_anti_collapse_event_diversity', 0.0) if gate else 0.0):.3f}",
            f"- phase5_counterfactual_transition_separation: {float(gate.get('phase5_counterfactual_transition_separation', 0.0) if gate else 0.0):.3f}",
            f"- phase5_multi_step_latent_chain_integrity: {float(gate.get('phase5_multi_step_latent_chain_integrity', 0.0) if gate else 0.0):.3f}",
            f"- phase5_long_horizon_error_correction_convergence: {float(gate.get('phase5_long_horizon_error_correction_convergence', 0.0) if gate else 0.0):.3f}",
            f"- phase5_horizon_bucket_stability: {float(gate.get('phase5_horizon_bucket_stability', 0.0) if gate else 0.0):.3f}",
            f"- phase5_macro_action_effectiveness: {float(gate.get('phase5_macro_action_effectiveness', 0.0) if gate else 0.0):.3f}",
            f"- phase5_subgoal_decomposition_integrity: {float(gate.get('phase5_subgoal_decomposition_integrity', 0.0) if gate else 0.0):.3f}",
            f"- phase5_depth_selective_routing_integrity: {float(gate.get('phase5_depth_selective_routing_integrity', 0.0) if gate else 0.0):.3f}",
            f"- phase5_micro_es_policy_refinement_integrity: {float(gate.get('phase5_micro_es_policy_refinement_integrity', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_transition_locality_observed: {float(gate.get('phase5_manifold_transition_locality_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_rollout_stability_observed: {float(gate.get('phase5_manifold_rollout_stability_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_causal_route_sparsity_observed: {float(gate.get('phase5_causal_route_sparsity_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_withheld_trajectory_recall_observed: {float(gate.get('phase5_withheld_trajectory_recall_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_trajectory_case_coverage_observed: {float(gate.get('phase5_manifold_trajectory_case_coverage_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_average_case_recall_observed: {float(gate.get('phase5_manifold_average_case_recall_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_scan_budget_integrity_observed: {float(gate.get('phase5_manifold_scan_budget_integrity_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_indexed_candidate_integrity_observed: {float(gate.get('phase5_manifold_indexed_candidate_integrity_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_index_scan_reduction_observed: {float(gate.get('phase5_manifold_index_scan_reduction_observed', 0.0) if gate else 0.0):.3f}",
            f"- phase5_manifold_candidate_miss_guard_observed: {float(gate.get('phase5_manifold_candidate_miss_guard_observed', 0.0) if gate else 0.0):.3f}",
            f"- packaging_metadata_passed: {gate.get('packaging_metadata_passed', False) if gate else False}",
            f"- repair_pending_count: {gate.get('repair_pending_count', 0) if gate else 0}",
            f"- repair_timeout_count: {gate.get('repair_timeout_count', 0) if gate else 0}",
            f"- repair_retry_queue_count: {gate.get('repair_retry_queue_count', 0) if gate else 0}",
            f"- repair_retry_cooldown_seconds: {float(gate.get('repair_retry_cooldown_seconds', 0.0) if gate else 0.0):.1f}",
            f"- repair_retry_cooldown_blocked_count: {gate.get('repair_retry_cooldown_blocked_count', 0) if gate else 0}",
        ]
    )
    for action in (
        gate.get("stage_b_promotion_actions", [])
        if gate and isinstance(gate.get("stage_b_promotion_actions"), list)
        else []
    ):
        lines.append(f"- stage_b_promotion_action: {action}")
    for action in (
        gate.get("stage_b_rlm_observation_actions", [])
        if gate and isinstance(gate.get("stage_b_rlm_observation_actions"), list)
        else []
    ):
        lines.append(f"- stage_b_rlm_observation_action: {action}")
    for failure in (
        gate.get("stage_d_delta_memory_candidate_failures", [])
        if gate and isinstance(gate.get("stage_d_delta_memory_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_delta_memory_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_d_candidate_failure_description(failure)}"
        )
    for failure in (
        gate.get("stage_d_acceptance_candidate_failures", [])
        if gate and isinstance(gate.get("stage_d_acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_d_candidate_failure_description(failure)}"
        )
    for failure in (
        gate.get("stage_e_observed_acceptance_candidate_failures", [])
        if gate and isinstance(gate.get("stage_e_observed_acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_e_observed_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={_stage_e_observed_candidate_failure_description(failure)}"
        )
    for action in (
        gate.get("stage_d_acceptance_candidate_actions", [])
        if gate and isinstance(gate.get("stage_d_acceptance_candidate_actions"), list)
        else []
    ):
        lines.append(f"- stage_d_acceptance_candidate_action: {action}")
    for action in (
        gate.get("stage_d_delta_memory_actions", [])
        if gate and isinstance(gate.get("stage_d_delta_memory_actions"), list)
        else []
    ):
        lines.append(f"- stage_d_delta_memory_action: {action}")
    if auto_dispatch:
        lines.extend(
            [
                f"- auto_dispatch_requested: {int(auto_dispatch.get('requested', 0) or 0)}",
                f"- auto_dispatch_candidates: {int(auto_dispatch.get('candidate_count', 0) or 0)}",
                f"- auto_dispatch_eligible: {int(auto_dispatch.get('eligible_count', 0) or 0)}",
                f"- auto_dispatch_selected: {int(auto_dispatch.get('selected_count', 0) or 0)}",
                f"- auto_dispatch_selected_unique_checks: {int(auto_dispatch.get('selected_unique_check_count', 0) or 0)}",
                f"- auto_dispatch_min_priority_tier: {auto_dispatch.get('min_priority_tier', 'low')}",
                f"- auto_dispatch_selection_mode: {auto_dispatch.get('selection_mode', 'priority')}",
                f"- auto_dispatch_max_per_check: {int(auto_dispatch.get('max_per_check', 0) or 0)}",
                f"- auto_dispatch_dispatched: {int(auto_dispatch.get('dispatched', 0) or 0)}",
                f"- auto_dispatch_skipped_pending: {len(auto_dispatch.get('skipped_pending_commands', [])) if isinstance(auto_dispatch.get('skipped_pending_commands', []), list) else 0}",
                f"- auto_dispatch_skipped_limit: {len(auto_dispatch.get('skipped_limit_commands', [])) if isinstance(auto_dispatch.get('skipped_limit_commands', []), list) else 0}",
                f"- auto_dispatch_skipped_low_priority: {int(auto_dispatch.get('skipped_low_priority_count', 0) or 0)}",
                f"- auto_dispatch_skipped_check_quota: {int(auto_dispatch.get('skipped_check_quota_count', 0) or 0)}",
            ]
        )
        for command in (
            auto_dispatch.get("dispatched_commands", [])
            if isinstance(auto_dispatch.get("dispatched_commands"), list)
            else []
        ):
            lines.append(f"- auto_dispatch_command: {command}")
        for command in (
            auto_dispatch.get("skipped_pending_commands", [])
            if isinstance(auto_dispatch.get("skipped_pending_commands"), list)
            else []
        ):
            lines.append(f"- auto_dispatch_skipped_pending_command: {command}")
        for command in (
            auto_dispatch.get("skipped_limit_commands", [])
            if isinstance(auto_dispatch.get("skipped_limit_commands"), list)
            else []
        ):
            lines.append(f"- auto_dispatch_skipped_limit_command: {command}")
        for command in (
            auto_dispatch.get("skipped_low_priority_commands", [])
            if isinstance(auto_dispatch.get("skipped_low_priority_commands"), list)
            else []
        ):
            lines.append(f"- auto_dispatch_skipped_low_priority_command: {command}")
        for command in (
            auto_dispatch.get("skipped_check_quota_commands", [])
            if isinstance(auto_dispatch.get("skipped_check_quota_commands"), list)
            else []
        ):
            lines.append(f"- auto_dispatch_skipped_check_quota_command: {command}")
    retry_queue = (
        gate.get("repair_retry_queue", [])
        if gate and isinstance(gate.get("repair_retry_queue"), list)
        else []
    )
    for retry in retry_queue:
        if not isinstance(retry, dict):
            continue
        checks = retry.get("covered_checks", [])
        checks_text = ", ".join(checks) if isinstance(checks, list) else ""
        lines.append(
            "- retry_queue_entry: "
            f"{retry.get('command', '')} "
            f"(reason={retry.get('reason', '')}, attempt={int(retry.get('next_attempt', 0) or 0)}/{int(retry.get('max_attempts', 0) or 0)}, "
            f"priority={retry.get('priority_tier', '')}, score={float(retry.get('priority_score', 0.0) or 0.0):.3f}, "
            f"checks={checks_text})"
        )
    cooldown_blocked = (
        gate.get("repair_retry_cooldown_blocked", [])
        if gate and isinstance(gate.get("repair_retry_cooldown_blocked"), list)
        else []
    )
    for blocked in cooldown_blocked:
        if not isinstance(blocked, dict):
            continue
        checks = blocked.get("covered_checks", [])
        checks_text = ", ".join(checks) if isinstance(checks, list) else ""
        lines.append(
            "- retry_cooldown_blocked_entry: "
            f"{blocked.get('command', '')} "
            f"(reason={blocked.get('reason', '')}, attempt={int(blocked.get('next_attempt', 0) or 0)}/{int(blocked.get('max_attempts', 0) or 0)}, "
            f"priority={blocked.get('priority_tier', '')}, score={float(blocked.get('priority_score', 0.0) or 0.0):.3f}, "
            f"cooldown_remaining_seconds={float(blocked.get('cooldown_remaining_seconds', 0.0) or 0.0):.1f}, checks={checks_text})"
        )
    repair_plan = gate.get("repair_plan", {}) if gate and isinstance(gate.get("repair_plan"), dict) else {}
    selected_actions = (
        repair_plan.get("selected_actions", [])
        if isinstance(repair_plan.get("selected_actions"), list)
        else []
    )
    fallback_actions = (
        repair_plan.get("fallback_actions", [])
        if isinstance(repair_plan.get("fallback_actions"), list)
        else []
    )
    lines.append(
        f"- repair_plan_steps: {int(repair_plan.get('estimated_steps', len(selected_actions)) or 0)}"
    )
    covered_checks = repair_plan.get("covered_checks", [])
    if not isinstance(covered_checks, list):
        covered_checks = []
    uncovered_checks = repair_plan.get("uncovered_checks", [])
    if not isinstance(uncovered_checks, list):
        uncovered_checks = []
    lines.append(
        f"- repair_plan_coverage: {len(covered_checks)}/{len(covered_checks) + len(uncovered_checks)}"
    )
    for action in selected_actions:
        if not isinstance(action, dict):
            continue
        checks = action.get("affected_checks", [])
        checks_text = ", ".join(checks) if isinstance(checks, list) else ""
        lines.append(
            f"- repair_step: {int(action.get('step', 0) or 0)} {action.get('title', '')} -> "
            f"{action.get('command', '')} (covers={checks_text})"
        )
    lines.append(f"- fallback_plan_steps: {len(fallback_actions)}")
    for action in fallback_actions:
        if not isinstance(action, dict):
            continue
        checks = action.get("affected_checks", [])
        checks_text = ", ".join(checks) if isinstance(checks, list) else ""
        lines.append(
            f"- fallback_step: {int(action.get('step', 0) or 0)} {action.get('title', '')} -> "
            f"{action.get('command', '')} (covers={checks_text})"
        )
    iterative_plan = (
        gate.get("iterative_repair_plan", {})
        if gate and isinstance(gate.get("iterative_repair_plan"), dict)
        else {}
    )
    iterative_remaining = (
        iterative_plan.get("remaining_checks", [])
        if isinstance(iterative_plan.get("remaining_checks"), list)
        else []
    )
    iterative_next = (
        iterative_plan.get("next_actions", [])
        if isinstance(iterative_plan.get("next_actions"), list)
        else []
    )
    lines.append(
        f"- iterative_remaining_checks: {len(iterative_remaining)}"
    )
    lines.append(
        f"- iterative_next_steps: {len(iterative_next)}"
    )
    lines.append(
        f"- iterative_completed: {bool(iterative_plan.get('completed', False))}"
    )
    lines.append(
        f"- iterative_stop_reason: {iterative_plan.get('stop_reason', '')}"
    )
    lines.append(
        f"- iterative_next_step_hint: {iterative_plan.get('next_step_hint', '')}"
    )
    for action in iterative_next:
        if not isinstance(action, dict):
            continue
        checks = action.get("affected_checks", [])
        checks_text = ", ".join(checks) if isinstance(checks, list) else ""
        lines.append(
            f"- iterative_next_step: {int(action.get('step', 0) or 0)} {action.get('title', '')} -> "
            f"{action.get('command', '')} (covers={checks_text})"
        )
    if gate and isinstance(gate.get("stage_b_minimum_failures"), list):
        for failure in gate.get("stage_b_minimum_failures", []):
            if not isinstance(failure, dict):
                continue
            lines.append(
                "- stage_b_minimum_failure: "
                f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
                f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f}"
            )
    if gate and isinstance(gate.get("stage_c_minimum_failures"), list):
        for failure in gate.get("stage_c_minimum_failures", []):
            if not isinstance(failure, dict):
                continue
            lines.append(
                "- stage_c_minimum_failure: "
                f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
                f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f}"
            )
    if gate and isinstance(gate.get("stage_d_minimum_failures"), list):
        for failure in gate.get("stage_d_minimum_failures", []):
            if not isinstance(failure, dict):
                continue
            lines.append(
                "- stage_d_minimum_failure: "
                f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
                f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f}"
            )
    if gate and isinstance(gate.get("stage_e_minimum_failures"), list):
        for failure in gate.get("stage_e_minimum_failures", []):
            if not isinstance(failure, dict):
                continue
            lines.append(
                "- stage_e_minimum_failure: "
                f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
                f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f}"
            )
    if gate and isinstance(gate.get("errors"), list) and gate["errors"]:
        for error in gate["errors"]:
            lines.append(f"- error: {error}")
    error_details = (
        gate.get("error_details", [])
        if gate and isinstance(gate.get("error_details"), list)
        else []
    )
    if error_details:
        lines.append(f"- error_detail_count: {len(error_details)}")
        for detail in error_details[:5]:
            if not isinstance(detail, dict):
                continue
            detail_type = str(detail.get("type", "general_error"))
            category = str(detail.get("category", "release_gate.errors"))
            metric_name = str(detail.get("metric_name", ""))
            if detail_type == "minimum_threshold_failure":
                lines.append(
                    "- error_detail: "
                    f"type={detail_type}, category={category}, metric={metric_name}, "
                    f"value={float(detail.get('actual_value', 0.0) or 0.0):.3f}, "
                    f"required>={float(detail.get('required_value', 0.0) or 0.0):.3f}"
                )
            elif detail_type == "metric_threshold_drop":
                lines.append(
                    "- error_detail: "
                    f"type={detail_type}, category={category}, metric={metric_name}"
                )
            else:
                lines.append(
                    "- error_detail: "
                    f"type={detail_type}, category={category}"
                )
    error_details_summary = (
        gate.get("error_details_summary", {})
        if gate and isinstance(gate.get("error_details_summary"), dict)
        else {}
    )
    if error_details_summary:
        lines.append(
            f"- error_detail_total: {int(error_details_summary.get('total', len(error_details)) or 0)}"
        )
        top_types = (
            error_details_summary.get("top_types", [])
            if isinstance(error_details_summary.get("top_types"), list)
            else []
        )
        if top_types:
            for item in top_types[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- error_detail_type_count: {item.get('name', '')}={int(item.get('count', 0) or 0)}"
                )
        else:
            by_type = (
                error_details_summary.get("by_type", {})
                if isinstance(error_details_summary.get("by_type"), dict)
                else {}
            )
            for key, value in list(sorted(by_type.items()))[:5]:
                lines.append(f"- error_detail_type_count: {key}={int(value)}")

        top_categories = (
            error_details_summary.get("top_categories", [])
            if isinstance(error_details_summary.get("top_categories"), list)
            else []
        )
        if top_categories:
            for item in top_categories[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- error_detail_category_count: {item.get('name', '')}={int(item.get('count', 0) or 0)}"
                )
        else:
            by_category = (
                error_details_summary.get("by_category", {})
                if isinstance(error_details_summary.get("by_category"), dict)
                else {}
            )
            for key, value in list(sorted(by_category.items()))[:5]:
                lines.append(f"- error_detail_category_count: {key}={int(value)}")

        top_metrics = (
            error_details_summary.get("top_metrics", [])
            if isinstance(error_details_summary.get("top_metrics"), list)
            else []
        )
        if top_metrics:
            for item in top_metrics[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- error_detail_metric_count: {item.get('name', '')}={int(item.get('count', 0) or 0)}"
                )
        else:
            by_metric = (
                error_details_summary.get("by_metric", {})
                if isinstance(error_details_summary.get("by_metric"), dict)
                else {}
            )
            for key, value in list(sorted(by_metric.items()))[:5]:
                lines.append(f"- error_detail_metric_count: {key}={int(value)}")
    failure_focus = (
        gate.get("failure_focus", {})
        if gate and isinstance(gate.get("failure_focus"), dict)
        else {}
    )
    if failure_focus:
        lines.append(f"- failure_focus_primary_category: {failure_focus.get('primary_category', '')}")
        lines.append(f"- failure_focus_secondary_category: {failure_focus.get('secondary_category', '')}")
        lines.append(f"- failure_focus_primary_metric: {failure_focus.get('primary_metric', '')}")
        lines.append(f"- failure_focus_confidence: {float(failure_focus.get('confidence', 0.0) or 0.0):.3f}")
        primary_action = (
            failure_focus.get("primary_action", {})
            if isinstance(failure_focus.get("primary_action"), dict)
            else {}
        )
        if primary_action:
            lines.append(f"- failure_focus_primary_action_title: {primary_action.get('title', '')}")
            lines.append(f"- failure_focus_primary_action_command: {primary_action.get('command', '')}")
    if gate and isinstance(gate.get("recovery_actions"), list):
        for action in gate.get("recovery_actions", []):
            if not isinstance(action, dict):
                continue
            affected_checks = action.get("affected_checks", [])
            affected_text = ", ".join(affected_checks) if isinstance(affected_checks, list) else ""
            lines.append(
                "- recovery_action: "
                f"{action.get('title', '')} -> {action.get('command', '')} "
                f"(priority={action.get('priority', '')}, effect={action.get('expected_effect', '')}, "
                f"affected_checks={affected_text}, reason={action.get('reason', '')})"
            )

    lines.extend(
        [
            "",
            "Checklist",
            f"- status: {_status_label(checklist_ok)}",
            f"- profile_name: {checklist.get('profile_name', criteria.get('profile_name', 'unknown'))}",
            f"- managed_output_paths_ok: {checklist.get('managed_output_paths_ok', False)}",
            f"- report_summary_review_ready: {checklist.get('report_summary_review_ready', False)}",
            f"- release_notes_reviewed: {checklist.get('release_notes_reviewed', False)}",
            f"- extended_profile_ready: {checklist.get('extended_profile_ready', False)}",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight release soak checks.")
    parser.add_argument(
        "--profile",
        choices=sorted(SOAK_PROFILES.keys()),
        default="release",
        help="Named soak profile with SNN-friendly default thresholds.",
    )
    parser.add_argument("--duration-seconds", type=float, default=None, help="Wall-clock budget per soak section.")
    parser.add_argument("--max-agent-turns", type=int, default=None, help="Maximum agent turns to execute.")
    parser.add_argument(
        "--min-agent-turns",
        type=int,
        default=None,
        help="Minimum agent turns required for the soak report to satisfy the release gate.",
    )
    parser.add_argument(
        "--max-inference-iterations",
        type=int,
        default=None,
        help="Maximum inference learning iterations.",
    )
    parser.add_argument(
        "--min-inference-iterations",
        type=int,
        default=None,
        help="Minimum inference iterations required for the soak report to satisfy the release gate.",
    )
    parser.add_argument(
        "--report-path",
        default=workspace_path("release", "release_soak_report.json"),
        help="Managed output path for the soak report.",
    )
    parser.add_argument(
        "--summary-path",
        default=workspace_path("release", "release_soak_summary.txt"),
        help="Managed output path for the human-readable release summary.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Run the Phase 3 accuracy suite and embed its summary into the release soak report.",
    )
    parser.add_argument(
        "--accuracy-history-path",
        default=workspace_path("evaluation", "phase3_accuracy_history.json"),
        help="Managed output path for Phase 3 accuracy history snapshots.",
    )
    parser.add_argument(
        "--accuracy-history-limit",
        type=int,
        default=50,
        help="Maximum number of embedded Phase 3 accuracy history snapshots to keep.",
    )
    parser.add_argument(
        "--stage-b-promotion-required-streak",
        type=int,
        default=3,
        help="Consecutive pass count required before Stage B promotion is recommended.",
    )
    parser.add_argument(
        "--repair-log-path",
        default=DEFAULT_REPAIR_LOG_PATH,
        help="Managed path for the repair execution log JSON.",
    )
    parser.add_argument(
        "--repair-log-command",
        default="",
        help="Optional command to append into repair execution log.",
    )
    parser.add_argument(
        "--repair-log-status",
        choices=["success", "failed", "pending", "skipped", "timeout"],
        default="success",
        help="Status for --repair-log-command entry.",
    )
    parser.add_argument(
        "--repair-log-covered-checks",
        default="",
        help="Comma-separated covered checks for --repair-log-command.",
    )
    parser.add_argument(
        "--append-iterative-next-actions",
        action="store_true",
        help="Append iterative next-step commands as pending entries to the repair log.",
    )
    parser.add_argument(
        "--pending-ttl-seconds",
        type=float,
        default=0.0,
        help="Automatically expire pending repair entries older than this TTL (0 disables).",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=2,
        help="Maximum failed/timeout attempts before a command is excluded from retry queue.",
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=float,
        default=0.0,
        help="Cooldown window before failed/timeout commands re-enter retry queue (0 disables).",
    )
    parser.add_argument(
        "--auto-dispatch-retry",
        type=int,
        default=0,
        help="Automatically dispatch this many retry-queue commands to pending entries.",
    )
    parser.add_argument(
        "--auto-dispatch-min-priority",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum retry priority tier eligible for auto-dispatch.",
    )
    parser.add_argument(
        "--auto-dispatch-diversify-checks",
        action="store_true",
        help="Greedy-select retry commands to diversify covered checks within dispatch budget.",
    )
    parser.add_argument(
        "--auto-dispatch-max-per-check",
        type=int,
        default=0,
        help="Maximum auto-dispatch entries allowed per covered check (0 disables).",
    )
    parser.add_argument(
        "--repair-complete-command",
        default="",
        help="Optional command to complete from pending repair entries.",
    )
    parser.add_argument(
        "--repair-complete-status",
        choices=["success", "failed", "skipped"],
        default="success",
        help="Completion status for --repair-complete-command.",
    )
    parser.add_argument(
        "--repair-complete-covered-checks",
        default="",
        help="Comma-separated covered checks for --repair-complete-command.",
    )
    args = parser.parse_args()

    settings = resolve_soak_profile(
        profile_name=args.profile,
        duration_seconds=args.duration_seconds,
        max_agent_turns=args.max_agent_turns,
        min_agent_turns=args.min_agent_turns,
        max_inference_iterations=args.max_inference_iterations,
        min_inference_iterations=args.min_inference_iterations,
    )

    if settings["duration_seconds"] <= 0:
        raise ValueError("--duration-seconds must be greater than 0.")
    if settings["min_agent_turns"] < 1:
        raise ValueError("--min-agent-turns must be at least 1.")
    if settings["min_inference_iterations"] < 1:
        raise ValueError("--min-inference-iterations must be at least 1.")
    if settings["min_agent_turns"] > settings["max_agent_turns"]:
        raise ValueError("--min-agent-turns cannot exceed --max-agent-turns.")
    if settings["min_inference_iterations"] > settings["max_inference_iterations"]:
        raise ValueError("--min-inference-iterations cannot exceed --max-inference-iterations.")

    report = {
        "agent": run_agent_soak(
            settings["duration_seconds"],
            settings["max_agent_turns"],
            settings["min_agent_turns"],
        ),
        "inference": run_inference_soak(
            settings["duration_seconds"],
            settings["max_inference_iterations"],
            settings["min_inference_iterations"],
        ),
        "duration_seconds": settings["duration_seconds"],
        "criteria": {
            "profile_name": settings["profile_name"],
            "min_duration_seconds": settings["duration_seconds"],
            "max_agent_turns": settings["max_agent_turns"],
            "min_agent_turns": settings["min_agent_turns"],
            "max_inference_iterations": settings["max_inference_iterations"],
            "min_inference_iterations": settings["min_inference_iterations"],
            "require_zero_agent_issues": True,
            "require_bounded_history": True,
            "require_roundtrip_ok": True,
            "require_tuple_keys_only": True,
            "min_pattern_count": 1,
            "shipping_ready": settings["shipping_ready"],
        },
        "release_metadata": collect_release_metadata(),
    }
    repair_execution_log = load_repair_execution_log(args.repair_log_path)
    if args.pending_ttl_seconds > 0:
        expire_pending_repair_entries(
            repair_execution_log,
            ttl_seconds=float(args.pending_ttl_seconds),
        )
    if str(args.repair_log_command).strip():
        append_repair_execution_entry(
            repair_execution_log,
            command=args.repair_log_command,
            status=args.repair_log_status,
            covered_checks=parse_repair_checks_csv(args.repair_log_covered_checks),
            title="manual_repair_log_entry",
            source="manual_cli",
        )
    if str(args.repair_complete_command).strip():
        completed = finalize_pending_repair_entries(
            repair_execution_log,
            command=args.repair_complete_command,
            status=args.repair_complete_status,
            covered_checks=parse_repair_checks_csv(args.repair_complete_covered_checks),
            title="manual_repair_completion",
            source="manual_cli_completion",
        )
        if completed == 0:
            append_repair_execution_entry(
                repair_execution_log,
                command=args.repair_complete_command,
                status=args.repair_complete_status,
                covered_checks=parse_repair_checks_csv(args.repair_complete_covered_checks),
                title="manual_repair_completion",
                source="manual_cli_completion",
            )
    report["repair_execution_log"] = repair_execution_log
    if args.include_accuracy:
        report["accuracy"] = run_accuracy_soak(
            history_path=args.accuracy_history_path,
            history_limit=args.accuracy_history_limit,
            stage_b_promotion_required_streak=args.stage_b_promotion_required_streak,
        )
        report["criteria"]["require_phase3_accuracy"] = True
    else:
        report["criteria"]["require_phase3_accuracy"] = False
    report["release_gate"] = collect_release_gate_feedback(
        report,
        retry_max_attempts=args.retry_max_attempts,
        retry_cooldown_seconds=args.retry_cooldown_seconds,
    )
    report["repair_auto_dispatch"] = {
        "requested": int(args.auto_dispatch_retry),
        "dispatched": 0,
        "candidate_count": 0,
        "eligible_count": 0,
        "selected_count": 0,
        "selected_unique_check_count": 0,
        "min_priority_tier": str(args.auto_dispatch_min_priority).strip().lower(),
        "selection_mode": "priority_diversified" if bool(args.auto_dispatch_diversify_checks) else "priority",
        "max_per_check": int(args.auto_dispatch_max_per_check),
        "dispatched_commands": [],
        "skipped_pending_commands": [],
        "skipped_limit_commands": [],
        "skipped_low_priority_commands": [],
        "skipped_low_priority_count": 0,
        "skipped_check_quota_commands": [],
        "skipped_check_quota_count": 0,
    }
    if args.auto_dispatch_retry > 0:
        retry_queue = (
            report.get("release_gate", {}).get("repair_retry_queue", [])
            if isinstance(report.get("release_gate"), dict)
            and isinstance(report.get("release_gate", {}).get("repair_retry_queue"), list)
            else []
        )
        report["repair_auto_dispatch"]["candidate_count"] = int(len(retry_queue))
        batch = select_retry_dispatch_batch(
            retry_queue,
            max_dispatch=int(args.auto_dispatch_retry),
            min_priority_tier=str(args.auto_dispatch_min_priority).strip().lower(),
            diversify_checks=bool(args.auto_dispatch_diversify_checks),
            max_per_check=int(args.auto_dispatch_max_per_check),
        )
        dispatch_report = dispatch_retry_queue_to_pending_with_report(
            repair_execution_log,
            batch.get("selected", []) if isinstance(batch.get("selected"), list) else [],
            max_dispatch=int(args.auto_dispatch_retry),
        )
        report["repair_auto_dispatch"] = {
            **report["repair_auto_dispatch"],
            "min_priority_tier": batch.get("min_priority_tier", "low"),
            "candidate_count": int(len(retry_queue)),
            "eligible_count": int(batch.get("eligible_count", 0) or 0),
            "selected_count": int(batch.get("selected_count", 0) or 0),
            "selected_unique_check_count": int(batch.get("selected_unique_check_count", 0) or 0),
            "selection_mode": str(batch.get("selection_mode", "priority")).strip() or "priority",
            "max_per_check": int(batch.get("max_per_check", 0) or 0),
            "skipped_low_priority_commands": (
                batch.get("skipped_low_priority_commands", [])
                if isinstance(batch.get("skipped_low_priority_commands"), list)
                else []
            ),
            "skipped_low_priority_count": int(batch.get("skipped_low_priority_count", 0) or 0),
            "skipped_check_quota_commands": (
                batch.get("skipped_check_quota_commands", [])
                if isinstance(batch.get("skipped_check_quota_commands"), list)
                else []
            ),
            "skipped_check_quota_count": int(batch.get("skipped_check_quota_count", 0) or 0),
            **dispatch_report,
        }
        if int(dispatch_report.get("dispatched", 0) or 0) > 0:
            report["repair_execution_log"] = repair_execution_log
            report["release_gate"] = collect_release_gate_feedback(
                report,
                retry_max_attempts=args.retry_max_attempts,
                retry_cooldown_seconds=args.retry_cooldown_seconds,
            )
            report["release_checklist"] = collect_release_checklist_status(
                report,
                report_path=args.report_path,
                summary_path=args.summary_path,
            )
    report["release_checklist"] = collect_release_checklist_status(
        report,
        report_path=args.report_path,
        summary_path=args.summary_path,
    )
    if args.append_iterative_next_actions:
        appended = append_iterative_next_actions_to_repair_log(
            repair_execution_log,
            report.get("release_gate", {}).get("iterative_repair_plan", {})
            if isinstance(report.get("release_gate"), dict)
            else {},
        )
        if appended > 0:
            report["repair_execution_log"] = repair_execution_log
            report["release_gate"] = collect_release_gate_feedback(
                report,
                retry_max_attempts=args.retry_max_attempts,
                retry_cooldown_seconds=args.retry_cooldown_seconds,
            )
            report["release_checklist"] = collect_release_checklist_status(
                report,
                report_path=args.report_path,
                summary_path=args.summary_path,
            )
    report["release_gate"] = collect_release_gate_feedback(
        report,
        retry_max_attempts=args.retry_max_attempts,
        retry_cooldown_seconds=args.retry_cooldown_seconds,
    )
    report["release_checklist"] = collect_release_checklist_status(
        report,
        report_path=args.report_path,
        summary_path=args.summary_path,
    )
    report["research_review"] = build_release_research_review(report)

    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    repair_log_path = save_repair_execution_log(args.repair_log_path, repair_execution_log)
    summary_path = ensure_parent_directory(args.summary_path)
    summary_text = format_release_summary(report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    print("Release soak completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved repair log: {repair_log_path}")


if __name__ == "__main__":
    main()
