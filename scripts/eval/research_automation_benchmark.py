# Directory Path: scripts/eval/research_automation_benchmark.py
# English Title: Research Automation Benchmark
# Purpose/Content: Builds a lightweight research review report from existing SARA evaluation artifacts and writes managed research planning artifacts under workspace/.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase3_tracking import (  # noqa: E402
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
    COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES,
    compact_neuromorphic_profile_trend,
    extract_cognitive_linear_snn_fusion_metrics,
    extract_cognitive_plastic_submodel_metrics,
    extract_cognitive_stage_e_architecture_integration_metrics,
)
from sara_engine.evaluation.stage_b_contract import (  # noqa: E402
    STAGE_B_MINIMUM_METRIC_NAMES,
    STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES,
    STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES,
)
from sara_engine.evaluation.stage_d_contract import (  # noqa: E402
    STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES,
    STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES,
    STAGE_D_MINIMUM_METRIC_NAMES,
)
from sara_engine.evaluation.stage_e_contract import (  # noqa: E402
    STAGE_E_MINIMUM_METRIC_NAMES,
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_PHASE3_REPORT_PATH = workspace_path("evaluation", "phase3_accuracy_suite.json")
DEFAULT_RELEASE_SOAK_REPORT_PATH = workspace_path("release", "release_soak_report.json")
DEFAULT_OPERATIONAL_REPORT_PATH = workspace_path("release", "operational_readiness_report.json")
DEFAULT_RESEARCH_REVIEW_REPORT_PATH = workspace_path("evaluation", "research_review_report.json")
DEFAULT_ROADMAP_PATCH_SUGGESTION_PATH = workspace_path("evaluation", "roadmap_patch_suggestion.json")
DEFAULT_RESEARCH_JOURNAL_PATH = workspace_path("evaluation", "research_journal.jsonl")

LINEAR_SNN_READY_THRESHOLD = 0.95
STAGE_E_ARCHITECTURE_READY_THRESHOLD = 0.95
SARA_POLICY_ALIGNMENT_THRESHOLD = 0.95
DEFAULT_JOURNAL_DEDUPE_WINDOW_SECONDS = 24.0 * 60.0 * 60.0
DEFAULT_JOURNAL_MAX_ENTRIES = 512
DEFAULT_JOURNAL_MAX_AGE_SECONDS = 90.0 * 24.0 * 60.0 * 60.0
RECOVERED_REMEASURE_INTERVAL_SECONDS = 7.0 * 24.0 * 60.0 * 60.0
FAILED_REMEASURE_INTERVAL_SECONDS = 24.0 * 60.0 * 60.0
SKIPPED_REMEASURE_INTERVAL_SECONDS = 3.0 * 24.0 * 60.0 * 60.0
STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID = "stage_e_observed_acceptance_candidate_repair"
RESEARCH_JOURNAL_BENCHMARK_COMMANDS = {
    "predictive_spike_entropy_reduction_observed": "python scripts/eval/cognitive_runtime_benchmark.py",
    "phase_binding_coincidence_integrity_observed": "python scripts/eval/cognitive_runtime_benchmark.py",
    "forward_only_local_update_stability_observed": "python scripts/eval/cognitive_runtime_benchmark.py",
    "linear_snn_fusion_metric_recovery": "python scripts/eval/cognitive_runtime_benchmark.py",
    "stage_e_architecture_integration_metric_recovery": "python scripts/eval/cognitive_runtime_benchmark.py",
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID: "python scripts/eval/cognitive_runtime_benchmark.py",
    "sara_policy_alignment_recovery": "python scripts/eval/cognitive_runtime_benchmark.py",
    "neuromorphic_profile_regression_review": "python scripts/eval/energy_efficiency_benchmark.py --no-history-update",
    "release_gate_safety_review": "python scripts/eval/release_gate.py",
}
RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS = {
    "predictive_spike_entropy_reduction_observed": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics",
        "reason": "target predictive-error-gated spike fixture instead of rerunning the full cognitive runtime benchmark.",
    },
    "phase_binding_coincidence_integrity_observed": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase_synchronized_binding_trace.py",
        "reason": "target phase synchronization binding fixtures instead of rerunning the full cognitive runtime benchmark.",
    },
    "forward_only_local_update_stability_observed": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_forward_only_local_update.py",
        "reason": "target forward-only local update fixtures instead of rerunning the full cognitive runtime benchmark.",
    },
    "linear_snn_fusion_metric_recovery": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_phase3_tracking_detects_linear_snn_fusion_observed_regression_without_gate_block",
        "reason": "target linear SNN fusion regression tracking before rerunning the full cognitive runtime benchmark.",
    },
    "stage_e_architecture_integration_metric_recovery": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_phase3_tracking_extracts_stage_e_architecture_integration_observations",
        "reason": "target Stage E architecture-integration observed metrics before rerunning the full cognitive runtime benchmark.",
    },
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID: {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_stage_e_observed_acceptance_candidate_failures_are_structured",
        "reason": "target Stage E observed acceptance candidate failures before rerunning the full cognitive runtime benchmark.",
    },
    "sara_policy_alignment_recovery": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics tests/test_common_spike_space.py",
        "reason": "target SARA policy-alignment fixtures for no-backprop, sparse event, local learning, interpretability, and submodel integration.",
    },
    "neuromorphic_profile_regression_review": {
        "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_energy_efficiency_neuromorphic_profile_trend_detects_regression",
        "reason": "target neuromorphic profile regression fixture before rerunning energy history updates.",
    },
    "release_gate_safety_review": {
        "command": "python scripts/eval/release_gate.py --skip-accuracy",
        "reason": "validate release-gate safety without rerunning the heavier accuracy path.",
    },
}


def _normalize_experiment_node_id(source: str, item: Dict[str, Any]) -> str:
    item_id = _journal_item_id(item)
    if item_id:
        return item_id
    encoded = json.dumps(item, sort_keys=True, default=str)
    stable_value = sum((index + 1) * ord(char) for index, char in enumerate(encoded))
    return f"{source}_{stable_value % 100000}"


def build_bounded_experiment_graph(
    planner: Dict[str, Any],
    research_journal_summary: Optional[Dict[str, Any]] = None,
    *,
    max_nodes: int = 32,
) -> Dict[str, Any]:
    """Builds a compact experiment graph from planner items without opening unbounded search."""

    source_planner = planner if isinstance(planner, dict) else {}
    journal = research_journal_summary if isinstance(research_journal_summary, dict) else {}
    node_limit = max(1, int(max_nodes))
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def _journal_map(key: str) -> Dict[str, Dict[str, Any]]:
        items = journal.get(key, []) if isinstance(journal.get(key, []), list) else []
        return {
            str(item.get("id", "") or ""): dict(item)
            for item in items
            if isinstance(item, dict) and str(item.get("id", "") or "")
        }

    remeasure_by_id = _journal_map("remeasure_trends")
    alternative_by_id = _journal_map("alternative_probe_trends")
    rejected_by_id = _journal_map("roadmap_patch_rejected_items")
    refreshed_by_id = _journal_map("roadmap_patch_refreshed_items")

    def _merge_node(source: str, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        node_id = _normalize_experiment_node_id(source, item)
        if not node_id:
            return
        node = nodes_by_id.setdefault(
            node_id,
            {
                "id": node_id,
                "sources": [],
                "priority": str(item.get("priority", "medium") or "medium"),
                "experiment_modes": [],
                "benchmark_command": RESEARCH_JOURNAL_BENCHMARK_COMMANDS.get(node_id, ""),
                "alternative_probe_command": (
                    RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS.get(node_id, {}).get("command", "")
                    if isinstance(RESEARCH_JOURNAL_ALTERNATIVE_BENCHMARK_ACTIONS.get(node_id, {}), dict)
                    else ""
                ),
                "observed_metric": str(item.get("metric", "") or ""),
                "promotion_blockers": [],
                "evidence": {},
            },
        )
        if source not in node["sources"]:
            node["sources"].append(source)
        source_modes = {
            "next_hypotheses": ["template_based_probe", "template_free_probe"],
            "regression_watchlist": ["ablation", "regression_remeasure"],
            "negative_results": ["targeted_fixture", "negative_result_review"],
            "stable_hypotheses": ["promotion_candidate"],
            "cause_boundary_documentation_tasks": ["cause_boundary_documentation"],
            "targeted_fixture_repair_tasks": ["targeted_fixture_repair"],
            "roadmap_patch_evidence_collection_tasks": ["evidence_collection"],
        }.get(source, ["review"])
        for mode in source_modes:
            if mode not in node["experiment_modes"]:
                node["experiment_modes"].append(mode)
        if str(item.get("priority", "") or "").lower() == "high":
            node["priority"] = "high"
        if bool(item.get("requires_additional_evidence", False)):
            node["promotion_blockers"].append("requires_additional_evidence")
        if bool(item.get("roadmap_patch_review_suppressed", False)):
            node["promotion_blockers"].append("roadmap_patch_review_suppressed")
        if source in {"negative_results", "regression_watchlist"}:
            node["promotion_blockers"].append(source.rstrip("s"))
        if node_id in rejected_by_id:
            node["promotion_blockers"].append("roadmap_patch_rejected")
        if node_id in refreshed_by_id:
            node["evidence"]["roadmap_patch_refresh"] = refreshed_by_id[node_id]
        if node_id in remeasure_by_id:
            node["evidence"]["remeasure_trend"] = remeasure_by_id[node_id]
        if node_id in alternative_by_id:
            node["evidence"]["alternative_probe_trend"] = alternative_by_id[node_id]
        node["promotion_blockers"] = sorted(set(str(item) for item in node["promotion_blockers"] if str(item)))

    for source in (
        "next_hypotheses",
        "regression_watchlist",
        "negative_results",
        "stable_hypotheses",
        "cause_boundary_documentation_tasks",
        "targeted_fixture_repair_tasks",
        "roadmap_patch_evidence_collection_tasks",
    ):
        for item in source_planner.get(source, []) if isinstance(source_planner.get(source, []), list) else []:
            if len(nodes_by_id) >= node_limit and _normalize_experiment_node_id(source, item) not in nodes_by_id:
                continue
            _merge_node(source, item)

    for node_id, node in sorted(nodes_by_id.items()):
        if "negative_results" in node.get("sources", []) or "regression_watchlist" in node.get("sources", []):
            if node.get("benchmark_command"):
                edges.append(
                    {
                        "source": node_id,
                        "target": f"{node_id}:remeasure",
                        "kind": "proposes_remeasure",
                    }
                )
            if node.get("alternative_probe_command"):
                edges.append(
                    {
                        "source": node_id,
                        "target": f"{node_id}:alternative_probe",
                        "kind": "proposes_targeted_probe",
                    }
                )
        if "next_hypotheses" in node.get("sources", []) and node.get("promotion_blockers"):
            edges.append(
                {
                    "source": node_id,
                    "target": f"{node_id}:promotion_blocker",
                    "kind": "blocked_until_evidence",
                }
            )

    stage_counts: Dict[str, int] = {}
    for node in nodes_by_id.values():
        for mode in node.get("experiment_modes", []):
            stage_counts[str(mode)] = int(stage_counts.get(str(mode), 0)) + 1

    nodes = [
        {
            **node,
            "sources": sorted(node.get("sources", [])),
            "experiment_modes": sorted(node.get("experiment_modes", [])),
        }
        for node in sorted(
            nodes_by_id.values(),
            key=lambda item: (
                0 if str(item.get("priority", "")) == "high" else 1,
                str(item.get("id", "")),
            ),
        )
    ]
    return {
        "schema": "sara-bounded-experiment-graph-v1",
        "bounded": True,
        "max_nodes": int(node_limit),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "stage_counts": dict(sorted(stage_counts.items())),
        "nodes": nodes,
        "edges": edges[: node_limit * 2],
        "policy": {
            "uses_gpu_heavy_test_time_compute": False,
            "requires_human_approval_for_roadmap_patch": True,
            "release_gate_blocking": False,
        },
    }


def classify_experiment_graph_status(
    planner: Dict[str, Any],
    research_journal_summary: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 5,
) -> Dict[str, Any]:
    source_planner = planner if isinstance(planner, dict) else {}
    journal = research_journal_summary if isinstance(research_journal_summary, dict) else {}
    top_limit = max(1, int(limit))

    def _ids(items: Any) -> List[str]:
        return _journal_item_ids(items)

    adoption_candidates = sorted(
        set(_ids(source_planner.get("stable_hypotheses", [])))
        .union(
            str(item.get("id", "") or "")
            for item in journal.get("roadmap_patch_refreshed_items", [])
            if isinstance(item, dict) and str(item.get("id", "") or "")
        )
    )
    regressing_items = sorted(
        set(_ids(source_planner.get("regression_watchlist", [])))
        .union(
            str(item.get("id", "") or "")
            for item in journal.get("remeasure_trends", [])
            if isinstance(item, dict)
            and str(item.get("trend", "") or "") in {"still_failing", "regressed_after_success"}
            and str(item.get("id", "") or "")
        )
    )
    falsified_items = sorted(
        set(_ids(source_planner.get("negative_results", [])))
        .union(
            str(item.get("id", "") or "")
            for item in journal.get("roadmap_patch_rejected_items", [])
            if isinstance(item, dict) and str(item.get("id", "") or "")
        )
    )
    human_review_pending_items = sorted(
        set(
            str(item.get("id", "") or "")
            for item in source_planner.get("roadmap_patch_evidence_collection_tasks", [])
            if isinstance(item, dict) and str(item.get("id", "") or "")
        )
        .union(
            str(item).split(":", 1)[0]
            for item in journal.get("completed_roadmap_patch_evidence_collection_keys", [])
            if str(item).strip()
        )
        .union(
            str(item.get("id", "") or "")
            for key in ("next_hypotheses", "regression_watchlist", "negative_results")
            for item in (source_planner.get(key, []) if isinstance(source_planner.get(key, []), list) else [])
            if isinstance(item, dict)
            and bool(item.get("roadmap_patch_review_suppressed", False))
            and str(item.get("id", item.get("metric", "")) or "")
        )
    )

    return {
        "schema": "sara-experiment-status-summary-v1",
        "adoption_candidate_count": len(adoption_candidates),
        "regressing_item_count": len(regressing_items),
        "falsified_item_count": len(falsified_items),
        "human_review_pending_count": len(human_review_pending_items),
        "adoption_candidate_ids": adoption_candidates[:top_limit],
        "regressing_item_ids": regressing_items[:top_limit],
        "falsified_item_ids": falsified_items[:top_limit],
        "human_review_pending_ids": human_review_pending_items[:top_limit],
    }


def build_experiment_status_priority_plan(
    experiment_status_summary: Dict[str, Any],
    *,
    limit: int = 5,
) -> Dict[str, Any]:
    """Turns status buckets into a bounded operational priority plan."""

    status = experiment_status_summary if isinstance(experiment_status_summary, dict) else {}
    top_limit = max(1, int(limit))

    def _bucket_ids(key: str) -> List[str]:
        return [
            str(item).strip()
            for item in (status.get(key, []) if isinstance(status.get(key, []), list) else [])
            if str(item).strip()
        ][:top_limit]

    buckets = [
        {
            "category": "regressing",
            "source": "experiment_regression_remeasure",
            "priority": "high",
            "count": int(status.get("regressing_item_count", 0) or 0),
            "ids": _bucket_ids("regressing_item_ids"),
            "command_label": "experiment_regression_remeasure",
            "policy": "remeasure before promotion or roadmap refresh",
        },
        {
            "category": "human_review_pending",
            "source": "experiment_human_review_followup",
            "priority": "medium",
            "count": int(status.get("human_review_pending_count", 0) or 0),
            "ids": _bucket_ids("human_review_pending_ids"),
            "command_label": "experiment_human_review_followup",
            "policy": "collect or review missing evidence before ROADMAP changes",
        },
        {
            "category": "adoption_candidate",
            "source": "experiment_adoption_candidate_review",
            "priority": "medium",
            "count": int(status.get("adoption_candidate_count", 0) or 0),
            "ids": _bucket_ids("adoption_candidate_ids"),
            "command_label": "experiment_adoption_candidate_review",
            "policy": "review for bounded promotion after stable evidence",
        },
        {
            "category": "falsified",
            "source": "experiment_falsified_suppression_review",
            "priority": "low",
            "count": int(status.get("falsified_item_count", 0) or 0),
            "ids": _bucket_ids("falsified_item_ids"),
            "command_label": "experiment_falsified_suppression_review",
            "policy": "keep suppressed unless new evidence refreshes the item",
        },
    ]
    actions = [dict(item) for item in buckets if int(item.get("count", 0) or 0) > 0]
    return {
        "schema": "sara-experiment-priority-plan-v1",
        "bounded": True,
        "action_count": int(len(actions)),
        "top_priority_source": str(actions[0].get("source", "")) if actions else "",
        "top_priority_category": str(actions[0].get("category", "")) if actions else "",
        "actions": actions,
        "policy": {
            "direct_roadmap_write_allowed": False,
            "requires_human_approval_for_promotion": True,
            "release_gate_blocking": False,
        },
    }


def classify_experiment_promotion_target(candidate_id: str) -> Dict[str, Any]:
    """Classifies where an adoption candidate should be reviewed before promotion."""

    item_id = str(candidate_id).strip()
    stage_b_minimum = set(STAGE_B_MINIMUM_METRIC_NAMES)
    stage_b_reward_policy = set(STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES)
    stage_b_rlm_observation = set(STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES)
    stage_d_minimum = set(STAGE_D_MINIMUM_METRIC_NAMES)
    stage_d_delta = set(STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES)
    stage_d_acceptance = set(STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES)
    stage_e_minimum = set(STAGE_E_MINIMUM_METRIC_NAMES)
    stage_e_observed = set(COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES).union(
        COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
        COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES,
        COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
        COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    )
    aggregate_targets = {
        "linear_snn_fusion_observed_metrics": {
            "target_stage": "stage_e",
            "target_surface": "observed_acceptance_candidate",
            "promotion_path": "stage_e_sparse_runtime_acceptance_review",
            "policy": "keep observed-only until persisted history and release soak validate sparse runtime stability",
        },
        "stage_e_architecture_integration_observed_metrics": {
            "target_stage": "stage_e",
            "target_surface": "observed_acceptance_candidate",
            "promotion_path": "stage_e_architecture_acceptance_review",
            "policy": "review micro-turn and phase-block traces before minimum-gate expansion",
        },
        "neuromorphic_profile_compatibility": {
            "target_stage": "neuromorphic_profile",
            "target_surface": "hardware_portability_candidate",
            "promotion_path": "neuromorphic_adapter_policy_review",
            "policy": "review Lava, SpiNNaker, and Akida profile stability before hardware-oriented promotion",
        },
        "sara_policy_alignment_observed_metrics": {
            "target_stage": "research_policy",
            "target_surface": "policy_guard_candidate",
            "promotion_path": "sara_policy_guard_review",
            "policy": "keep as reviewer guardrail unless repeated failures require gate hardening",
        },
    }
    if item_id in aggregate_targets:
        payload = dict(aggregate_targets[item_id])
    elif item_id in stage_b_reward_policy:
        payload = {
            "target_stage": "stage_b",
            "target_surface": "reward_policy_minimum_candidate",
            "promotion_path": "stage_b_promotion",
            "policy": "use Stage B promotion streak before minimum gate changes",
        }
    elif item_id in stage_b_rlm_observation:
        payload = {
            "target_stage": "stage_b",
            "target_surface": "rlm_observation_minimum_candidate",
            "promotion_path": "stage_b_rlm_observation_promotion",
            "policy": "use Stage B RLM observation streak before minimum gate changes",
        }
    elif item_id in stage_b_minimum:
        payload = {
            "target_stage": "stage_b",
            "target_surface": "minimum_gate",
            "promotion_path": "already_minimum",
            "policy": "already belongs to the Stage B minimum contract",
        }
    elif item_id in stage_d_delta:
        payload = {
            "target_stage": "stage_d",
            "target_surface": "delta_memory_promotion_candidate",
            "promotion_path": "stage_d_delta_memory_promotion",
            "policy": "use delta-memory promotion streak before minimum gate changes",
        }
    elif item_id in stage_d_acceptance:
        payload = {
            "target_stage": "stage_d",
            "target_surface": "acceptance_candidate",
            "promotion_path": "stage_d_acceptance_candidate_stability",
            "policy": "review Stage D acceptance candidates as a bounded group before minimum gate changes",
        }
    elif item_id in stage_d_minimum:
        payload = {
            "target_stage": "stage_d",
            "target_surface": "minimum_gate",
            "promotion_path": "already_minimum",
            "policy": "already belongs to the Stage D minimum contract",
        }
    elif item_id in stage_e_minimum:
        payload = {
            "target_stage": "stage_e",
            "target_surface": "minimum_gate",
            "promotion_path": "already_minimum",
            "policy": "already belongs to the Stage E minimum contract",
        }
    elif item_id in stage_e_observed:
        payload = {
            "target_stage": "stage_e",
            "target_surface": "observed_acceptance_candidate",
            "promotion_path": "stage_e_observed_metric_acceptance_review",
            "policy": "keep observed-only until repeated history shows no sparse-runtime regression",
        }
    else:
        payload = {
            "target_stage": "research_review",
            "target_surface": "unmapped_observed_candidate",
            "promotion_path": "manual_mapping_review",
            "policy": "requires human mapping before any gate or acceptance-candidate change",
        }
    return {
        "id": item_id,
        **payload,
        "direct_minimum_gate_write_allowed": False,
        "requires_human_approval": True,
    }


def build_experiment_promotion_target_plan(
    experiment_status_summary: Dict[str, Any],
    *,
    limit: int = 5,
) -> Dict[str, Any]:
    status = experiment_status_summary if isinstance(experiment_status_summary, dict) else {}
    top_limit = max(1, int(limit))
    candidate_ids = [
        str(item).strip()
        for item in (
            status.get("adoption_candidate_ids", [])
            if isinstance(status.get("adoption_candidate_ids", []), list)
            else []
        )
        if str(item).strip()
    ][:top_limit]
    targets = [classify_experiment_promotion_target(item_id) for item_id in candidate_ids]
    stage_counts: Dict[str, int] = {}
    surface_counts: Dict[str, int] = {}
    review_actions: List[Dict[str, Any]] = []
    for target in targets:
        stage = str(target.get("target_stage", "") or "")
        surface = str(target.get("target_surface", "") or "")
        if stage:
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
        if surface:
            surface_counts[surface] = int(surface_counts.get(surface, 0)) + 1
        if str(target.get("promotion_path", "")) != "already_minimum":
            review_actions.append(
                {
                    "id": str(target.get("id", "")),
                    "source": "experiment_promotion_target_review",
                    "priority": "medium",
                    "target_stage": stage,
                    "target_surface": surface,
                    "promotion_path": str(target.get("promotion_path", "")),
                    "policy": str(target.get("policy", "")),
                }
            )
    return {
        "schema": "sara-experiment-promotion-target-plan-v1",
        "candidate_count": int(len(candidate_ids)),
        "mapped_candidate_count": int(len(targets)),
        "review_action_count": int(len(review_actions)),
        "target_stage_counts": dict(sorted(stage_counts.items())),
        "target_surface_counts": dict(sorted(surface_counts.items())),
        "targets": targets,
        "review_actions": review_actions,
        "policy": {
            "direct_minimum_gate_write_allowed": False,
            "requires_human_approval": True,
            "release_gate_blocking": False,
        },
    }


def _load_json_object_if_present(path: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    snapshot = {
        "path": os.path.abspath(path),
        "exists": bool(os.path.exists(path)),
        "loaded": False,
        "error": "",
    }
    if not snapshot["exists"]:
        return None, snapshot
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        snapshot["error"] = str(exc)
        return None, snapshot
    if not isinstance(payload, dict):
        snapshot["error"] = "JSON object expected."
        return None, snapshot
    snapshot["loaded"] = True
    return payload, snapshot


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metric_failures(metrics: Dict[str, float], threshold: float) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for metric_name, value in sorted(metrics.items()):
        if value < threshold:
            failures.append(
                {
                    "metric": metric_name,
                    "value": float(value),
                    "threshold": float(threshold),
                    "hypothesis": "Improve linear RNN + SNN fusion until observed metric clears the research threshold.",
                }
            )
    return failures


def _journal_item_id(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("id", "") or item.get("metric", "") or item.get("check", "") or "").strip()


def _lower_priority(priority: str) -> str:
    current = str(priority or "medium").strip().lower()
    if current == "critical":
        return "high"
    if current == "high":
        return "medium"
    return "low"


def summarize_completed_roadmap_patch_evidence_review(
    research_journal_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarizes completed evidence keys that still need roadmap review."""

    summary = research_journal_summary if isinstance(research_journal_summary, dict) else {}
    completed_keys = sorted(
        {
            str(item).strip()
            for item in (
                summary.get("completed_roadmap_patch_evidence_collection_keys", [])
                if isinstance(summary.get("completed_roadmap_patch_evidence_collection_keys", []), list)
                else []
            )
            if str(item).strip()
        }
    )
    refreshed_ids = {
        str(item.get("id", "") or "").strip()
        for item in (
            summary.get("roadmap_patch_refreshed_items", [])
            if isinstance(summary.get("roadmap_patch_refreshed_items", []), list)
            else []
        )
        if isinstance(item, dict) and str(item.get("id", "") or "").strip()
    }
    pending_keys = []
    for key in completed_keys:
        target_id = key.split(":", 1)[0].strip()
        if target_id and target_id not in refreshed_ids:
            pending_keys.append(key)
    return {
        "completed_count": int(len(completed_keys)),
        "completed_keys": completed_keys,
        "refreshed_id_count": int(len(refreshed_ids)),
        "refreshed_ids": sorted(refreshed_ids),
        "pending_review_count": int(len(pending_keys)),
        "pending_review_keys": pending_keys,
        "needs_review": bool(pending_keys),
    }


def _apply_remeasure_trends_to_experiment_planner(
    planner: Dict[str, Any],
    research_journal_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(planner, dict):
        return {}
    if not isinstance(research_journal_summary, dict):
        return planner
    trends = research_journal_summary.get("remeasure_trends", [])
    alternative_probe_trends = research_journal_summary.get("alternative_probe_trends", [])
    roadmap_patch_rejected_items = research_journal_summary.get("roadmap_patch_rejected_items", [])
    roadmap_patch_refreshed_items = research_journal_summary.get("roadmap_patch_refreshed_items", [])
    evidence_kind_counts = (
        research_journal_summary.get("roadmap_patch_evidence_collection_kind_counts", {})
        if isinstance(research_journal_summary.get("roadmap_patch_evidence_collection_kind_counts"), dict)
        else {}
    )
    evidence_next_required_kind = str(
        research_journal_summary.get("roadmap_patch_evidence_collection_next_required_kind", "")
        or ""
    )
    completed_evidence_tasks = (
        research_journal_summary.get("completed_roadmap_patch_evidence_collection_tasks", [])
        if isinstance(research_journal_summary.get("completed_roadmap_patch_evidence_collection_tasks", []), list)
        else []
    )
    completed_evidence_keys = {
        (
            str(target_id).strip(),
            str(task.get("evidence_kind", "") or "").strip(),
        )
        for task in completed_evidence_tasks
        if isinstance(task, dict)
        for target_id in (
            task.get("target_ids", [])
            if isinstance(task.get("target_ids", []), list)
            else []
        )
        if str(target_id).strip()
    }
    for item in (
        research_journal_summary.get("completed_roadmap_patch_evidence_collection_keys", [])
        if isinstance(research_journal_summary.get("completed_roadmap_patch_evidence_collection_keys", []), list)
        else []
    ):
        text = str(item).strip()
        if ":" not in text:
            continue
        target_id, evidence_kind = text.split(":", 1)
        if target_id.strip() and evidence_kind.strip():
            completed_evidence_keys.add((target_id.strip(), evidence_kind.strip()))
    if not isinstance(trends, list):
        trends = []
    if not isinstance(alternative_probe_trends, list):
        alternative_probe_trends = []
    if not isinstance(roadmap_patch_rejected_items, list):
        roadmap_patch_rejected_items = []
    if not isinstance(roadmap_patch_refreshed_items, list):
        roadmap_patch_refreshed_items = []
    trend_by_id = {
        str(item.get("id", "") or ""): dict(item)
        for item in trends
        if isinstance(item, dict) and str(item.get("id", "") or "")
    }
    alternative_probe_trend_by_id = {
        str(item.get("id", "") or ""): dict(item)
        for item in alternative_probe_trends
        if isinstance(item, dict) and str(item.get("id", "") or "")
    }
    roadmap_patch_rejection_by_id = {
        str(item.get("id", "") or ""): dict(item)
        for item in roadmap_patch_rejected_items
        if isinstance(item, dict) and str(item.get("id", "") or "")
    }
    roadmap_patch_refresh_by_id = {
        str(item.get("id", "") or ""): dict(item)
        for item in roadmap_patch_refreshed_items
        if isinstance(item, dict) and str(item.get("id", "") or "")
    }
    completed_cause_boundary_ids = {
        str(item).strip()
        for item in research_journal_summary.get("completed_cause_boundary_documentation_ids", [])
        if str(item).strip()
    } if isinstance(research_journal_summary.get("completed_cause_boundary_documentation_ids", []), list) else set()
    completed_targeted_fixture_ids = {
        str(item).strip()
        for item in research_journal_summary.get("completed_targeted_fixture_repair_ids", [])
        if str(item).strip()
    } if isinstance(research_journal_summary.get("completed_targeted_fixture_repair_ids", []), list) else set()
    completed_tasks = (
        research_journal_summary.get("completed_research_planner_tasks", [])
        if isinstance(research_journal_summary.get("completed_research_planner_tasks", []), list)
        else []
    )
    completed_task_refresh_by_id: Dict[str, Dict[str, Any]] = {}
    for task in completed_tasks:
        if not isinstance(task, dict):
            continue
        timestamp = _safe_float(task.get("resolved_timestamp", task.get("timestamp", 0.0)), 0.0)
        task_type = str(task.get("task_type", "") or "")
        target_ids = (
            [str(item).strip() for item in task.get("target_ids", []) if str(item).strip()]
            if isinstance(task.get("target_ids"), list)
            else []
        )
        for target_id in target_ids:
            previous = completed_task_refresh_by_id.get(target_id, {})
            if timestamp >= float(previous.get("latest_timestamp", 0.0) or 0.0):
                completed_task_refresh_by_id[target_id] = {
                    "id": target_id,
                    "latest_timestamp": float(timestamp),
                    "task_type": task_type,
                    "command": str(task.get("command", "") or ""),
                }
    if (
        not trend_by_id
        and not alternative_probe_trend_by_id
        and not roadmap_patch_rejection_by_id
        and not roadmap_patch_refresh_by_id
    ):
        return planner

    policy_summary = {
        "applied": True,
        "deprioritized_recovered_count": 0,
        "escalated_still_failing_count": 0,
        "cause_boundary_documentation_count": 0,
        "targeted_fixture_repair_count": 0,
        "roadmap_patch_evidence_collection_count": 0,
        "completed_roadmap_patch_evidence_collection_count": int(len(completed_evidence_keys)),
        "completed_cause_boundary_documentation_count": int(len(completed_cause_boundary_ids)),
        "completed_targeted_fixture_repair_count": int(len(completed_targeted_fixture_ids)),
        "roadmap_patch_rejection_suppressed_count": 0,
        "roadmap_patch_rejection_refreshed_count": 0,
        "annotated_item_count": 0,
    }

    def _evidence_refresh_for_rejection(item_id: str, rejection: Dict[str, Any]) -> Dict[str, Any]:
        rejected_at = _safe_float(rejection.get("latest_timestamp", 0.0), 0.0)
        previous_refresh = roadmap_patch_refresh_by_id.get(item_id, {})
        previous_refresh_at = _safe_float(previous_refresh.get("latest_refresh_timestamp", 0.0), 0.0)
        remeasure = trend_by_id.get(item_id, {})
        remeasure_label = str(remeasure.get("trend", "") or "")
        remeasure_at = _safe_float(remeasure.get("latest_timestamp", 0.0), 0.0)
        if remeasure_label in {"recovered", "confirmed"} and remeasure_at > rejected_at:
            if previous_refresh_at >= remeasure_at:
                return {
                    "refreshed": False,
                    "reason": "remeasure_refresh_already_surfaced",
                    "latest_timestamp": float(remeasure_at),
                }
            return {
                "refreshed": True,
                "reason": f"remeasure_{remeasure_label}",
                "latest_timestamp": float(remeasure_at),
                "latest_command": str(remeasure.get("latest_command", "") or ""),
            }
        alternative = alternative_probe_trend_by_id.get(item_id, {})
        alternative_label = str(alternative.get("trend", "") or "")
        alternative_at = _safe_float(alternative.get("latest_timestamp", 0.0), 0.0)
        if alternative_label == "targeted_probe_passed" and alternative_at > rejected_at:
            if previous_refresh_at >= alternative_at:
                return {
                    "refreshed": False,
                    "reason": "targeted_probe_refresh_already_surfaced",
                    "latest_timestamp": float(alternative_at),
                }
            return {
                "refreshed": True,
                "reason": "targeted_probe_passed",
                "latest_timestamp": float(alternative_at),
                "latest_command": str(alternative.get("latest_command", "") or ""),
            }
        completed = completed_task_refresh_by_id.get(item_id, {})
        completed_at = _safe_float(completed.get("latest_timestamp", 0.0), 0.0)
        if completed_at > rejected_at and str(completed.get("task_type", "") or "") in {
            "targeted_fixture_repair",
            "cause_boundary_documentation",
        }:
            if previous_refresh_at >= completed_at:
                return {
                    "refreshed": False,
                    "reason": "planner_task_refresh_already_surfaced",
                    "latest_timestamp": float(completed_at),
                }
            return {
                "refreshed": True,
                "reason": f"completed_{str(completed.get('task_type', '') or '')}",
                "latest_timestamp": float(completed_at),
                "latest_command": str(completed.get("command", "") or ""),
            }
        return {"refreshed": False}

    def _annotate_items(items: Any) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        source_items = items if isinstance(items, list) else []
        for item in source_items:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            item_id = _journal_item_id(copied)
            rejection = roadmap_patch_rejection_by_id.get(item_id, {})
            if rejection:
                refresh = _evidence_refresh_for_rejection(item_id, rejection)
                copied["roadmap_patch_review_rejection_reason"] = str(
                    rejection.get("latest_reason", "") or ""
                )
                if bool(refresh.get("refreshed", False)):
                    copied["roadmap_patch_review_suppression_lifted"] = True
                    copied["roadmap_patch_review_refresh_reason"] = str(
                        refresh.get("reason", "") or ""
                    )
                    copied["roadmap_patch_review_refresh_timestamp"] = float(
                        refresh.get("latest_timestamp", 0.0) or 0.0
                    )
                    copied["priority_adjustment"] = "roadmap_patch_reproposal_allowed_after_refresh_evidence"
                    policy_summary["roadmap_patch_rejection_refreshed_count"] = (
                        int(policy_summary["roadmap_patch_rejection_refreshed_count"]) + 1
                    )
                else:
                    copied["roadmap_patch_review_suppressed"] = True
                    copied["requires_additional_evidence"] = True
                    copied["roadmap_patch_review_suppression_reason"] = str(
                        refresh.get("reason", "needs_additional_evidence") or "needs_additional_evidence"
                    )
                    copied["priority_adjustment"] = (
                        "requires_alternative_probe_or_additional_evidence_after_roadmap_patch_rejection"
                    )
                    policy_summary["roadmap_patch_rejection_suppressed_count"] = (
                        int(policy_summary["roadmap_patch_rejection_suppressed_count"]) + 1
                    )
                policy_summary["annotated_item_count"] = int(policy_summary["annotated_item_count"]) + 1
            trend = trend_by_id.get(item_id, {})
            if trend:
                trend_label = str(trend.get("trend", "") or "unknown")
                copied["remeasure_trend"] = trend_label
                copied["remeasure_latest_status"] = str(trend.get("latest_status", "") or "")
                copied["remeasure_success_count"] = int(trend.get("success_count", 0) or 0)
                copied["remeasure_failed_count"] = int(trend.get("failed_count", 0) or 0)
                copied["remeasure_skipped_count"] = int(trend.get("skipped_count", 0) or 0)
                policy_summary["annotated_item_count"] = int(policy_summary["annotated_item_count"]) + 1
                if trend_label in {"recovered", "confirmed"}:
                    copied["priority"] = _lower_priority(str(copied.get("priority", "medium")))
                    copied["priority_adjustment"] = "deprioritized_after_remeasure_recovery"
                    copied["recommended_remeasure_interval_seconds"] = RECOVERED_REMEASURE_INTERVAL_SECONDS
                    policy_summary["deprioritized_recovered_count"] = (
                        int(policy_summary["deprioritized_recovered_count"]) + 1
                    )
                elif trend_label in {"still_failing", "regressed_after_success"}:
                    copied["priority"] = "high"
                    copied["priority_adjustment"] = "escalated_after_remeasure_failure"
                    copied["recommended_remeasure_interval_seconds"] = FAILED_REMEASURE_INTERVAL_SECONDS
                    policy_summary["escalated_still_failing_count"] = (
                        int(policy_summary["escalated_still_failing_count"]) + 1
                    )
                elif trend_label == "skipped":
                    copied["priority_adjustment"] = "retry_after_skipped_remeasure"
                    copied["recommended_remeasure_interval_seconds"] = SKIPPED_REMEASURE_INTERVAL_SECONDS
            annotated.append(copied)
        return annotated

    updated = dict(planner)
    for key in ("next_hypotheses", "regression_watchlist", "negative_results"):
        updated[key] = _annotate_items(updated.get(key, []))
    cause_boundary_tasks: List[Dict[str, Any]] = []
    targeted_fixture_tasks: List[Dict[str, Any]] = []
    evidence_collection_tasks: List[Dict[str, Any]] = []
    evidence_collection_candidate_ids = sorted(
        str(item.get("id", "") or "")
        for item in roadmap_patch_rejected_items
        if isinstance(item, dict) and str(item.get("id", "") or "")
    )
    if evidence_next_required_kind and evidence_collection_candidate_ids:
        for item_id in evidence_collection_candidate_ids:
            if (item_id, evidence_next_required_kind) in completed_evidence_keys:
                continue
            evidence_collection_tasks.append(
                {
                    "id": item_id,
                    "priority": "high" if evidence_next_required_kind == "real_data_fixture" else "medium",
                    "evidence_kind": evidence_next_required_kind,
                    "evidence_kind_counts": dict(evidence_kind_counts),
                    "description": (
                        "Collect the next evidence kind required by the roadmap patch refresh policy before resurfacing the same proposal."
                    ),
                }
            )
        policy_summary["roadmap_patch_evidence_collection_count"] = len(evidence_collection_tasks)
    for item_id, trend in sorted(alternative_probe_trend_by_id.items()):
        trend_label = str(trend.get("trend", "") or "unknown")
        latest_status = str(trend.get("latest_status", "") or "")
        latest_command = str(trend.get("latest_command", "") or "")
        latest_timestamp = _safe_float(trend.get("latest_timestamp", 0.0), 0.0)
        previous_refresh = roadmap_patch_refresh_by_id.get(item_id, {})
        previous_refresh_at = _safe_float(previous_refresh.get("latest_refresh_timestamp", 0.0), 0.0)
        if previous_refresh_at >= latest_timestamp > 0:
            continue
        if evidence_next_required_kind and item_id in evidence_collection_candidate_ids:
            continue
        if trend_label == "targeted_probe_passed":
            if item_id in completed_cause_boundary_ids:
                continue
            cause_boundary_tasks.append(
                {
                    "id": item_id,
                    "priority": "medium",
                    "alternative_probe_trend": trend_label,
                    "latest_status": latest_status,
                    "latest_command": latest_command,
                    "description": "Document the boundary narrowed by the targeted probe before rerunning the full benchmark.",
                }
            )
            policy_summary["cause_boundary_documentation_count"] = (
                int(policy_summary["cause_boundary_documentation_count"]) + 1
            )
        elif trend_label == "targeted_probe_failed":
            if item_id in completed_targeted_fixture_ids:
                continue
            targeted_fixture_tasks.append(
                {
                    "id": item_id,
                    "priority": "high",
                    "alternative_probe_trend": trend_label,
                    "latest_status": latest_status,
                    "latest_command": latest_command,
                    "description": "Add or repair the minimal targeted fixture before spending another full-benchmark run.",
                }
            )
            policy_summary["targeted_fixture_repair_count"] = (
                int(policy_summary["targeted_fixture_repair_count"]) + 1
            )
    updated["cause_boundary_documentation_tasks"] = cause_boundary_tasks
    updated["targeted_fixture_repair_tasks"] = targeted_fixture_tasks
    updated["roadmap_patch_evidence_collection_tasks"] = evidence_collection_tasks
    updated["remeasure_priority_policy"] = policy_summary
    return updated


def _linear_snn_signal(phase3_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(phase3_report, dict):
        return {
            "available": False,
            "metrics": {},
            "minimum_value": 0.0,
            "ready": False,
            "regression_count": 0,
            "regressions": [],
            "failures": [],
        }

    metrics = extract_cognitive_linear_snn_fusion_metrics(phase3_report)
    minimum_value = min(metrics.values()) if metrics else 0.0
    trend = phase3_report.get("linear_snn_fusion_observed_trend", {})
    if not isinstance(trend, dict):
        trend = {}
    regressions = [
        dict(item)
        for item in trend.get("regressions", [])
        if isinstance(item, dict)
    ] if isinstance(trend.get("regressions", []), list) else []
    regression_count = int(_safe_float(trend.get("regression_count", len(regressions)), 0.0))
    failures = _metric_failures(metrics, LINEAR_SNN_READY_THRESHOLD)
    return {
        "available": bool(metrics),
        "metrics": metrics,
        "required_metrics": list(COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES),
        "minimum_value": float(minimum_value),
        "ready": bool(metrics) and minimum_value >= LINEAR_SNN_READY_THRESHOLD and regression_count == 0,
        "regression_count": regression_count,
        "regressions": regressions,
        "failures": failures,
    }


def _stage_e_architecture_signal(phase3_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(phase3_report, dict):
        return {
            "available": False,
            "metrics": {},
            "minimum_value": 0.0,
            "ready": False,
            "regression_count": 0,
            "regressions": [],
            "failures": [],
        }

    metrics = extract_cognitive_stage_e_architecture_integration_metrics(phase3_report)
    minimum_value = min(metrics.values()) if metrics else 0.0
    trend = phase3_report.get("stage_e_architecture_integration_observed_trend", {})
    if not isinstance(trend, dict):
        trend = {}
    regressions = [
        dict(item)
        for item in trend.get("regressions", [])
        if isinstance(item, dict)
    ] if isinstance(trend.get("regressions", []), list) else []
    regression_count = int(_safe_float(trend.get("regression_count", len(regressions)), 0.0))
    failures = _metric_failures(metrics, STAGE_E_ARCHITECTURE_READY_THRESHOLD)
    for failure in failures:
        failure["hypothesis"] = (
            "Repair Stage E architecture integration observed metrics while preserving sparse event budgets."
        )
    return {
        "available": bool(metrics),
        "metrics": metrics,
        "required_metrics": list(COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES),
        "minimum_value": float(minimum_value),
        "ready": bool(metrics) and minimum_value >= STAGE_E_ARCHITECTURE_READY_THRESHOLD,
        "regression_count": regression_count,
        "regressions": regressions,
        "failures": failures,
    }


def _stage_e_observed_acceptance_signal(phase3_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(phase3_report, dict):
        return {
            "available": False,
            "candidate_count": 0,
            "ready_count": 0,
            "failure_count": 0,
            "ready": True,
            "failures": [],
            "required_metrics": list(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES),
        }
    stage_e = phase3_report.get("stage_e_readiness", {})
    if not isinstance(stage_e, dict) or "observed_acceptance_candidate_count" not in stage_e:
        return {
            "available": False,
            "candidate_count": 0,
            "ready_count": 0,
            "failure_count": 0,
            "ready": True,
            "failures": [],
            "required_metrics": list(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES),
        }
    failures = [
        {
            **dict(item),
            "id": str(item.get("metric", item.get("check", "")) or ""),
            "hypothesis": (
                "Repair Stage E observed acceptance candidate while keeping it observed-only until stability review."
            ),
        }
        for item in (
            stage_e.get("observed_acceptance_candidate_failures", [])
            if isinstance(stage_e.get("observed_acceptance_candidate_failures", []), list)
            else []
        )
        if isinstance(item, dict)
    ]
    failure_count = int(stage_e.get("observed_acceptance_candidate_failure_count", len(failures)) or 0)
    return {
        "available": True,
        "candidate_count": int(stage_e.get("observed_acceptance_candidate_count", 0) or 0),
        "ready_count": int(stage_e.get("observed_acceptance_candidate_ready_count", 0) or 0),
        "failure_count": failure_count,
        "ready": bool(stage_e.get("observed_acceptance_candidates_ready", failure_count == 0)) and failure_count == 0,
        "failures": failures,
        "required_metrics": list(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES),
    }


def _neuromorphic_signal(phase3_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(phase3_report, dict):
        return {
            "available": False,
            "regression_count": 0,
            "policy_change_count": 0,
            "compact_trend": compact_neuromorphic_profile_trend({}),
            "ready": False,
        }
    component_reports = phase3_report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    energy_report = component_reports.get("energy_efficiency", {})
    if not isinstance(energy_report, dict):
        energy_report = {}
    trend = energy_report.get("neuromorphic_profile_trend", {})
    if not isinstance(trend, dict):
        trend = {}
    regression_count = int(_safe_float(trend.get("regression_count", 0), 0.0))
    policy_change_count = int(_safe_float(trend.get("policy_change_count", 0), 0.0))
    return {
        "available": bool(trend),
        "regression_count": regression_count,
        "policy_change_count": policy_change_count,
        "compact_trend": compact_neuromorphic_profile_trend(trend),
        "ready": bool(trend) and regression_count == 0,
    }


def _sara_policy_alignment_signal(phase3_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(phase3_report, dict):
        return {
            "available": False,
            "dimensions": {},
            "minimum_value": 0.0,
            "ready": False,
            "failures": [],
        }

    linear_metrics = extract_cognitive_linear_snn_fusion_metrics(phase3_report)
    architecture_metrics = extract_cognitive_stage_e_architecture_integration_metrics(phase3_report)
    plastic_metrics = extract_cognitive_plastic_submodel_metrics(phase3_report)
    combined_metrics = {
        **linear_metrics,
        **architecture_metrics,
        **plastic_metrics,
    }
    dimension_metric_names = {
        "no_backprop_alignment": [
            "forward_only_local_update_stability_observed",
            "block_independent_local_update_budget_observed",
            "submodel_credit_assignment_trace_integrity_observed",
        ],
        "sparse_event_alignment": [
            "predictive_spike_entropy_reduction_observed",
            "phase_binding_coincidence_integrity_observed",
            "micro_turn_event_budget_observed",
            "phase_assigned_submodel_route_observed",
        ],
        "local_learning_alignment": [
            "submodel_credit_assignment_trace_integrity_observed",
            "submodel_credit_selectivity_observed",
            "runtime_submodel_local_credit_assignment_observed",
            "runtime_submodel_feedback_trace_observed",
        ],
        "interpretability_trace_coverage": [
            "interpretable_submodel_concept_trace_observed",
            "runtime_submodel_concept_trace_observed",
            "submodel_intervention_trace_integrity_observed",
            "submodel_ablation_effect_observed",
            "submodel_scientific_hypothesis_trace_integrity_observed",
            "submodel_counterexample_revision_observed",
        ],
        "submodel_integration_impact": [
            "plastic_submodel_registry_integrity_observed",
            "dynamic_submodel_route_integrity_observed",
            "runtime_submodel_route_action_grounding_observed",
            "runtime_submodel_counterfactual_route_separation_observed",
            "submodel_hypothesis_bank_integrity_observed",
            "submodel_open_ended_selection_observed",
        ],
    }
    dimensions: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    for dimension_name, metric_names in dimension_metric_names.items():
        values = [
            float(combined_metrics.get(metric_name, 0.0) or 0.0)
            for metric_name in metric_names
        ]
        minimum_value = min(values) if values else 0.0
        ready = minimum_value >= SARA_POLICY_ALIGNMENT_THRESHOLD
        failed_metrics = [
            {
                "metric": metric_name,
                "value": float(combined_metrics.get(metric_name, 0.0) or 0.0),
                "threshold": float(SARA_POLICY_ALIGNMENT_THRESHOLD),
                "dimension": dimension_name,
                "hypothesis": (
                    "Restore SARA policy alignment without adding backprop, dense matrix, or GPU-dependent runtime paths."
                ),
            }
            for metric_name in metric_names
            if float(combined_metrics.get(metric_name, 0.0) or 0.0) < SARA_POLICY_ALIGNMENT_THRESHOLD
        ]
        dimensions[dimension_name] = {
            "score": 1.0 if ready else 0.0,
            "status": "pass" if ready else "needs_review",
            "minimum_value": float(minimum_value),
            "metric_names": list(metric_names),
            "failed_metrics": failed_metrics,
        }
        failures.extend(failed_metrics)

    minimum_value = (
        min(float(item.get("minimum_value", 0.0) or 0.0) for item in dimensions.values())
        if dimensions
        else 0.0
    )
    return {
        "available": bool(combined_metrics),
        "dimensions": dimensions,
        "minimum_value": float(minimum_value),
        "ready": bool(dimensions) and not failures,
        "failures": failures,
        "required_metric_count": int(
            len(
                sorted(
                    {
                        metric_name
                        for metric_names in dimension_metric_names.values()
                        for metric_name in metric_names
                    }
                )
            )
        ),
    }


def _release_safety_signal(
    phase3_report: Optional[Dict[str, Any]],
    release_soak_report: Optional[Dict[str, Any]],
    operational_report: Optional[Dict[str, Any]],
    *,
    require_operational_readiness: bool = True,
) -> Dict[str, Any]:
    release_gate = (
        release_soak_report.get("release_gate", {})
        if isinstance(release_soak_report, dict)
        and isinstance(release_soak_report.get("release_gate"), dict)
        else {}
    )
    checks = {
        "phase3_passed": bool(isinstance(phase3_report, dict) and phase3_report.get("passed", False)),
        "release_soak_passed": bool(
            isinstance(release_soak_report, dict)
            and (
                release_soak_report.get("passed", False)
                or release_gate.get("passed", False)
            )
        ),
    }
    if require_operational_readiness:
        checks["operational_readiness_passed"] = bool(
            isinstance(operational_report, dict) and operational_report.get("passed", False)
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "ready": not failed,
    }


def _score_review_dimension(ready: bool, available: bool = True) -> Dict[str, Any]:
    if not available:
        return {"score": 0.0, "status": "missing_input"}
    return {"score": 1.0 if ready else 0.0, "status": "pass" if ready else "needs_review"}


def build_research_review_report(
    *,
    phase3_report: Optional[Dict[str, Any]],
    release_soak_report: Optional[Dict[str, Any]],
    operational_report: Optional[Dict[str, Any]],
    input_snapshots: Iterable[Dict[str, Any]],
    generated_at: Optional[float] = None,
    require_operational_readiness: bool = True,
    research_journal_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    linear_signal = _linear_snn_signal(phase3_report)
    stage_e_architecture_signal = _stage_e_architecture_signal(phase3_report)
    stage_e_observed_acceptance_signal = _stage_e_observed_acceptance_signal(phase3_report)
    neuromorphic_signal = _neuromorphic_signal(phase3_report)
    sara_policy_signal = _sara_policy_alignment_signal(phase3_report)
    release_safety = _release_safety_signal(
        phase3_report,
        release_soak_report,
        operational_report,
        require_operational_readiness=bool(require_operational_readiness),
    )

    next_hypotheses: List[Dict[str, Any]] = []
    stable_hypotheses: List[Dict[str, Any]] = []
    regression_watchlist: List[Dict[str, Any]] = []
    negative_results: List[Dict[str, Any]] = []

    def _extend_negative_results_unique(items: List[Dict[str, Any]]) -> None:
        existing_ids = {
            _journal_item_id(item)
            for item in negative_results
            if isinstance(item, dict)
        }
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = _journal_item_id(item)
            if item_id and item_id in existing_ids:
                continue
            negative_results.append(item)
            if item_id:
                existing_ids.add(item_id)

    if linear_signal["ready"]:
        stable_hypotheses.append(
            {
                "id": "linear_snn_fusion_observed_metrics",
                "status": "stable_observed",
                "evidence": linear_signal["metrics"],
            }
        )
    else:
        next_hypotheses.append(
            {
                "id": "linear_snn_fusion_metric_recovery",
                "priority": "high",
                "description": "Run the CPU-first linear RNN + SNN fusion toy loop until observed metrics recover without becoming release-gate blocking.",
                "signals": linear_signal["failures"],
            }
        )
        _extend_negative_results_unique(linear_signal["failures"])

    if linear_signal["regression_count"] > 0:
        regression_watchlist.append(
            {
                "id": "linear_snn_fusion_observed_regression",
                "priority": "high",
                "regressions": linear_signal["regressions"],
            }
        )

    if stage_e_architecture_signal["ready"]:
        stable_hypotheses.append(
            {
                "id": "stage_e_architecture_integration_observed_metrics",
                "status": "stable_observed",
                "evidence": stage_e_architecture_signal["metrics"],
            }
        )
    else:
        next_hypotheses.append(
            {
                "id": "stage_e_architecture_integration_metric_recovery",
                "priority": "medium",
                "description": "Repair Stage E micro-turn and phase-block observed metrics without adding dense runtime paths.",
                "signals": stage_e_architecture_signal["failures"],
            }
        )
        _extend_negative_results_unique(stage_e_architecture_signal["failures"])

    if stage_e_architecture_signal["regression_count"] > 0:
        regression_watchlist.append(
            {
                "id": "stage_e_architecture_integration_observed_regression",
                "priority": "medium",
                "regressions": stage_e_architecture_signal["regressions"],
            }
        )

    if stage_e_observed_acceptance_signal["available"] and not stage_e_observed_acceptance_signal["ready"]:
        next_hypotheses.append(
            {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "priority": "medium",
                "description": "Repair Stage E observed acceptance candidates without moving them directly into the minimum gate.",
                "signals": stage_e_observed_acceptance_signal["failures"],
            }
        )
        regression_watchlist.append(
            {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "priority": "medium",
                "failures": stage_e_observed_acceptance_signal["failures"],
            }
        )
        _extend_negative_results_unique(stage_e_observed_acceptance_signal["failures"])

    if neuromorphic_signal["ready"]:
        stable_hypotheses.append(
            {
                "id": "neuromorphic_profile_compatibility",
                "status": "stable_observed",
                "evidence": neuromorphic_signal["compact_trend"],
            }
        )
    else:
        next_hypotheses.append(
            {
                "id": "neuromorphic_profile_regression_review",
                "priority": "medium",
                "description": "Review Loihi/Lava, SpiNNaker, and Akida adapter compatibility before promoting edge profiles.",
                "signals": neuromorphic_signal["compact_trend"],
            }
        )

    if not release_safety["ready"]:
        regression_watchlist.append(
            {
                "id": "release_gate_safety_review",
                "priority": "high",
                "failed_checks": release_safety["failed_checks"],
            }
        )

    if sara_policy_signal["ready"]:
        stable_hypotheses.append(
            {
                "id": "sara_policy_alignment_observed_metrics",
                "status": "stable_observed",
                "evidence": {
                    name: {
                        "minimum_value": float(value.get("minimum_value", 0.0) or 0.0),
                        "status": str(value.get("status", "") or ""),
                    }
                    for name, value in sara_policy_signal["dimensions"].items()
                    if isinstance(value, dict)
                },
            }
        )
    else:
        next_hypotheses.append(
            {
                "id": "sara_policy_alignment_recovery",
                "priority": "high",
                "description": "Repair SARA policy alignment dimensions without adding backprop, dense matrix, or GPU-dependent runtime paths.",
                "signals": sara_policy_signal["failures"],
            }
        )
        _extend_negative_results_unique(sara_policy_signal["failures"])

    review_dimensions = {
        "novelty": {
            **_score_review_dimension(linear_signal["available"], linear_signal["available"]),
            "note": "Novelty is credited only when SNN-specific observed metrics are present.",
        },
        "reproducibility": {
            **_score_review_dimension(release_safety["ready"], True),
            "note": "Reproducibility follows Phase 3, release soak, and operational readiness pass/fail.",
        },
        "energy_impact": {
            **_score_review_dimension(neuromorphic_signal["ready"], neuromorphic_signal["available"]),
            "note": "Energy impact is reviewed through neuromorphic profile trend stability.",
        },
        "stage_e_architecture_integration": {
            **_score_review_dimension(
                stage_e_architecture_signal["ready"],
                stage_e_architecture_signal["available"],
            ),
            "note": "Stage E architecture integration is credited only when micro-turn and phase-block observed metrics remain stable.",
        },
        "stage_e_observed_acceptance_candidates": {
            **_score_review_dimension(
                stage_e_observed_acceptance_signal["ready"],
                True,
            ),
            "note": "Stage E observed acceptance candidates are review-only and must not bypass human promotion review.",
        },
        "release_gate_safety": {
            **_score_review_dimension(release_safety["ready"], True),
            "note": "Research automation must not bypass release gate checks.",
        },
        "neuromorphic_compatibility": {
            **_score_review_dimension(neuromorphic_signal["ready"], neuromorphic_signal["available"]),
            "note": "Compatibility is observed-only until hardware-backed adapters are available.",
        },
    }
    for dimension_name, dimension in sara_policy_signal["dimensions"].items():
        if not isinstance(dimension, dict):
            continue
        review_dimensions[f"sara_policy_{dimension_name}"] = {
            "score": float(dimension.get("score", 0.0) or 0.0),
            "status": str(dimension.get("status", "") or ""),
            "minimum_value": float(dimension.get("minimum_value", 0.0) or 0.0),
            "failed_metric_count": len(dimension.get("failed_metrics", []))
            if isinstance(dimension.get("failed_metrics", []), list)
            else 0,
            "note": "SARA policy reviewer dimension; observed-only and non-release-gate blocking.",
        }
    review_score = sum(item["score"] for item in review_dimensions.values()) / max(len(review_dimensions), 1)

    experiment_planner = _apply_remeasure_trends_to_experiment_planner(
        {
            "next_hypotheses": next_hypotheses,
            "stable_hypotheses": stable_hypotheses,
            "regression_watchlist": regression_watchlist,
            "negative_results": negative_results,
        },
        research_journal_summary,
    )
    experiment_planner["bounded_experiment_graph"] = build_bounded_experiment_graph(
        experiment_planner,
        research_journal_summary,
    )
    experiment_planner["experiment_status_summary"] = classify_experiment_graph_status(
        experiment_planner,
        research_journal_summary,
    )
    experiment_planner["experiment_priority_plan"] = build_experiment_status_priority_plan(
        experiment_planner["experiment_status_summary"],
    )
    experiment_planner["experiment_promotion_target_plan"] = build_experiment_promotion_target_plan(
        experiment_planner["experiment_status_summary"],
    )

    return {
        "schema": "sara-research-review-report-v1",
        "generated_at": float(generated_at if generated_at is not None else time.time()),
        "inputs": list(input_snapshots),
        "passed": bool(review_score >= 0.8 and release_safety["ready"]),
        "review_score": float(review_score),
        "review_dimensions": review_dimensions,
        "signals": {
            "linear_snn_fusion": linear_signal,
            "stage_e_architecture_integration": stage_e_architecture_signal,
            "stage_e_observed_acceptance_candidates": stage_e_observed_acceptance_signal,
            "neuromorphic_profile": neuromorphic_signal,
            "sara_policy_alignment": sara_policy_signal,
            "release_safety": release_safety,
        },
        "policy": {
            "require_operational_readiness": bool(require_operational_readiness),
            "release_gate_blocking": False,
            "human_approval_required_for_roadmap_patch": True,
        },
        "experiment_planner": experiment_planner,
    }


def compact_research_review_report(report: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    if not isinstance(report, dict):
        report = {}
    planner = report.get("experiment_planner", {})
    if not isinstance(planner, dict):
        planner = {}
    dimensions = report.get("review_dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
    planner_policy = (
        planner.get("remeasure_priority_policy", {})
        if isinstance(planner.get("remeasure_priority_policy"), dict)
        else {}
    )
    experiment_graph = (
        planner.get("bounded_experiment_graph", {})
        if isinstance(planner.get("bounded_experiment_graph"), dict)
        else {}
    )
    experiment_status = (
        planner.get("experiment_status_summary", {})
        if isinstance(planner.get("experiment_status_summary"), dict)
        else {}
    )
    experiment_priority_plan = (
        planner.get("experiment_priority_plan", {})
        if isinstance(planner.get("experiment_priority_plan"), dict)
        else {}
    )
    experiment_promotion_target_plan = (
        planner.get("experiment_promotion_target_plan", {})
        if isinstance(planner.get("experiment_promotion_target_plan"), dict)
        else {}
    )

    def _ids(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        output: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "") or "")
            if item_id:
                output.append(item_id)
        return output[: max(1, int(limit))]

    dimension_statuses = {
        str(name): str(value.get("status", "") or "")
        for name, value in dimensions.items()
        if isinstance(value, dict)
    }
    sara_policy_dimension_statuses = {
        name: status
        for name, status in dimension_statuses.items()
        if name.startswith("sara_policy_")
    }
    return {
        "schema": "sara-research-review-compact-v1",
        "passed": bool(report.get("passed", False)),
        "review_score": float(report.get("review_score", 0.0) or 0.0),
        "dimension_statuses": dimension_statuses,
        "sara_policy_dimension_statuses": sara_policy_dimension_statuses,
        "sara_policy_dimension_count": int(len(sara_policy_dimension_statuses)),
        "sara_policy_needs_review_count": len(
            [
                status
                for status in sara_policy_dimension_statuses.values()
                if status != "pass"
            ]
        ),
        "next_hypothesis_count": len(planner.get("next_hypotheses", []))
        if isinstance(planner.get("next_hypotheses", []), list)
        else 0,
        "stable_hypothesis_count": len(planner.get("stable_hypotheses", []))
        if isinstance(planner.get("stable_hypotheses", []), list)
        else 0,
        "regression_watchlist_count": len(planner.get("regression_watchlist", []))
        if isinstance(planner.get("regression_watchlist", []), list)
        else 0,
        "negative_result_count": len(planner.get("negative_results", []))
        if isinstance(planner.get("negative_results", []), list)
        else 0,
        "cause_boundary_documentation_count": len(planner.get("cause_boundary_documentation_tasks", []))
        if isinstance(planner.get("cause_boundary_documentation_tasks", []), list)
        else 0,
        "targeted_fixture_repair_count": len(planner.get("targeted_fixture_repair_tasks", []))
        if isinstance(planner.get("targeted_fixture_repair_tasks", []), list)
        else 0,
        "roadmap_patch_evidence_collection_count": len(planner.get("roadmap_patch_evidence_collection_tasks", []))
        if isinstance(planner.get("roadmap_patch_evidence_collection_tasks", []), list)
        else 0,
        "next_hypothesis_ids": _ids(planner.get("next_hypotheses", [])),
        "regression_watchlist_ids": _ids(planner.get("regression_watchlist", [])),
        "cause_boundary_documentation_ids": _ids(planner.get("cause_boundary_documentation_tasks", [])),
        "targeted_fixture_repair_ids": _ids(planner.get("targeted_fixture_repair_tasks", [])),
        "roadmap_patch_evidence_collection_ids": _ids(planner.get("roadmap_patch_evidence_collection_tasks", [])),
        "bounded_experiment_graph_node_count": int(
            experiment_graph.get("node_count", 0) or 0
        ),
        "bounded_experiment_graph_edge_count": int(
            experiment_graph.get("edge_count", 0) or 0
        ),
        "bounded_experiment_graph_stage_counts": (
            dict(experiment_graph.get("stage_counts", {}))
            if isinstance(experiment_graph.get("stage_counts"), dict)
            else {}
        ),
        "experiment_status_summary": dict(experiment_status),
        "experiment_adoption_candidate_count": int(
            experiment_status.get("adoption_candidate_count", 0) or 0
        ),
        "experiment_regressing_item_count": int(
            experiment_status.get("regressing_item_count", 0) or 0
        ),
        "experiment_falsified_item_count": int(
            experiment_status.get("falsified_item_count", 0) or 0
        ),
        "experiment_human_review_pending_count": int(
            experiment_status.get("human_review_pending_count", 0) or 0
        ),
        "experiment_adoption_candidate_ids": (
            list(experiment_status.get("adoption_candidate_ids", []))
            if isinstance(experiment_status.get("adoption_candidate_ids", []), list)
            else []
        ),
        "experiment_regressing_item_ids": (
            list(experiment_status.get("regressing_item_ids", []))
            if isinstance(experiment_status.get("regressing_item_ids", []), list)
            else []
        ),
        "experiment_falsified_item_ids": (
            list(experiment_status.get("falsified_item_ids", []))
            if isinstance(experiment_status.get("falsified_item_ids", []), list)
            else []
        ),
        "experiment_human_review_pending_ids": (
            list(experiment_status.get("human_review_pending_ids", []))
            if isinstance(experiment_status.get("human_review_pending_ids", []), list)
            else []
        ),
        "experiment_priority_plan": dict(experiment_priority_plan),
        "experiment_priority_action_count": int(
            experiment_priority_plan.get("action_count", 0) or 0
        ),
        "experiment_top_priority_source": str(
            experiment_priority_plan.get("top_priority_source", "") or ""
        ),
        "experiment_top_priority_category": str(
            experiment_priority_plan.get("top_priority_category", "") or ""
        ),
        "experiment_promotion_target_plan": dict(experiment_promotion_target_plan),
        "experiment_promotion_target_candidate_count": int(
            experiment_promotion_target_plan.get("candidate_count", 0) or 0
        ),
        "experiment_promotion_target_review_action_count": int(
            experiment_promotion_target_plan.get("review_action_count", 0) or 0
        ),
        "experiment_promotion_target_stage_counts": (
            dict(experiment_promotion_target_plan.get("target_stage_counts", {}))
            if isinstance(experiment_promotion_target_plan.get("target_stage_counts"), dict)
            else {}
        ),
        "roadmap_patch_rejection_suppressed_count": int(
            planner_policy.get("roadmap_patch_rejection_suppressed_count", 0) or 0
        ),
        "roadmap_patch_rejection_refreshed_count": int(
            planner_policy.get("roadmap_patch_rejection_refreshed_count", 0) or 0
        ),
        "release_gate_blocking": False,
        "requires_human_approval": True,
    }


def build_roadmap_patch_suggestion(review_report: Dict[str, Any]) -> Dict[str, Any]:
    planner = review_report.get("experiment_planner", {})
    if not isinstance(planner, dict):
        planner = {}
    def _is_not_suppressed(item: Dict[str, Any]) -> bool:
        return not bool(item.get("roadmap_patch_review_suppressed", False))

    suppressed_items = [
        {
            "id": _journal_item_id(item),
            "reason": str(item.get("roadmap_patch_review_rejection_reason", "") or ""),
        }
        for key in ("next_hypotheses", "regression_watchlist", "negative_results")
        for item in (planner.get(key, []) if isinstance(planner.get(key, []), list) else [])
        if isinstance(item, dict)
        and _journal_item_id(item)
        and bool(item.get("roadmap_patch_review_suppressed", False))
    ]
    next_hypotheses = [
        str(item.get("id", ""))
        for item in planner.get("next_hypotheses", [])
        if isinstance(item, dict) and str(item.get("id", "")) and _is_not_suppressed(item)
    ]
    watchlist = [
        str(item.get("id", ""))
        for item in planner.get("regression_watchlist", [])
        if isinstance(item, dict) and str(item.get("id", "")) and _is_not_suppressed(item)
    ]
    boundary_docs = [
        str(item.get("id", ""))
        for item in planner.get("cause_boundary_documentation_tasks", [])
        if isinstance(item, dict) and str(item.get("id", ""))
    ]
    fixture_repairs = [
        str(item.get("id", ""))
        for item in planner.get("targeted_fixture_repair_tasks", [])
        if isinstance(item, dict) and str(item.get("id", ""))
    ]
    evidence_tasks = [
        {
            "id": str(item.get("id", "") or ""),
            "kind": str(item.get("evidence_kind", "") or ""),
        }
        for item in planner.get("roadmap_patch_evidence_collection_tasks", [])
        if isinstance(item, dict) and str(item.get("id", "") or "")
    ]
    suggestions = []
    for hypothesis_id in next_hypotheses:
        suggestions.append(f"NEXT: validate `{hypothesis_id}` with CPU-first observed-only benchmarks.")
    for watch_id in watchlist:
        suggestions.append(f"REVIEW: inspect `{watch_id}` before changing release-gate thresholds.")
    for boundary_id in boundary_docs:
        suggestions.append(f"DOC: document targeted-probe boundary for `{boundary_id}` before another full benchmark.")
    for repair_id in fixture_repairs:
        suggestions.append(f"FIXTURE: add or repair minimal targeted fixture for `{repair_id}` before another full benchmark.")
    for evidence in evidence_tasks:
        suggestions.append(
            f"EVIDENCE: collect `{evidence['kind']}` for `{evidence['id']}` before resurfacing the roadmap patch."
        )
    if not suggestions:
        suggestions.append("KEEP: maintain current research automation loop and continue collecting stable evidence.")
    return {
        "schema": "sara-roadmap-patch-suggestion-v1",
        "apply_automatically": False,
        "requires_human_approval": True,
        "source_report_schema": str(review_report.get("schema", "")),
        "source_generated_at": review_report.get("generated_at"),
        "suggestions": suggestions,
        "suppressed_rejected_items": suppressed_items,
    }


def _journal_item_ids(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    output: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "") or item.get("metric", "") or item.get("check", "") or "")
        if item_id:
            output.append(item_id)
    return sorted(set(output))


def build_research_journal_entry(review_report: Dict[str, Any]) -> Dict[str, Any]:
    planner = review_report.get("experiment_planner", {})
    if not isinstance(planner, dict):
        planner = {}
    negative_results = planner.get("negative_results", [])
    regression_watchlist = planner.get("regression_watchlist", [])
    next_hypotheses = planner.get("next_hypotheses", [])
    evidence_collection_tasks = planner.get("roadmap_patch_evidence_collection_tasks", [])
    bounded_experiment_graph = (
        planner.get("bounded_experiment_graph", {})
        if isinstance(planner.get("bounded_experiment_graph"), dict)
        else {}
    )
    key_parts = [
        "passed=" + str(bool(review_report.get("passed", False))),
        "negative=" + ",".join(_journal_item_ids(negative_results)),
        "regression=" + ",".join(_journal_item_ids(regression_watchlist)),
        "next=" + ",".join(_journal_item_ids(next_hypotheses)),
    ]
    return {
        "schema": "sara-research-journal-entry-v1",
        "generated_at": review_report.get("generated_at", time.time()),
        "review_score": review_report.get("review_score", 0.0),
        "passed": bool(review_report.get("passed", False)),
        "negative_results": negative_results,
        "regression_watchlist": regression_watchlist,
        "next_hypotheses": next_hypotheses,
        "roadmap_patch_evidence_collection_tasks": evidence_collection_tasks if isinstance(evidence_collection_tasks, list) else [],
        "bounded_experiment_graph": {
            "schema": str(bounded_experiment_graph.get("schema", "")),
            "node_count": int(bounded_experiment_graph.get("node_count", 0) or 0),
            "edge_count": int(bounded_experiment_graph.get("edge_count", 0) or 0),
            "stage_counts": (
                dict(bounded_experiment_graph.get("stage_counts", {}))
                if isinstance(bounded_experiment_graph.get("stage_counts"), dict)
                else {}
            ),
        },
        "dedupe_key": "|".join(key_parts),
    }


def load_research_journal_entries(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _prune_research_journal_entries(
    entries: List[Dict[str, Any]],
    *,
    now_timestamp: float,
    max_entries: int,
    max_age_seconds: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    age_limit = float(max(max_age_seconds, 0.0))
    entry_limit = int(max(max_entries, 1))
    kept: List[Dict[str, Any]] = []
    pruned_by_age = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        generated_at = _safe_float(entry.get("generated_at", 0.0), 0.0)
        if age_limit > 0 and generated_at > 0 and now_timestamp - generated_at > age_limit:
            pruned_by_age += 1
            continue
        kept.append(dict(entry))
    pruned_by_limit = max(len(kept) - entry_limit, 0)
    if pruned_by_limit > 0:
        kept = kept[-entry_limit:]
    return kept, {
        "pruned_by_age": int(pruned_by_age),
        "pruned_by_limit": int(pruned_by_limit),
        "kept_count": int(len(kept)),
        "max_entries": int(entry_limit),
        "max_age_seconds": float(age_limit),
    }


def write_research_journal_entries(path: str, entries: List[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def _journal_entry_item_ids(entry: Dict[str, Any]) -> List[str]:
    if not isinstance(entry, dict):
        return []
    item_ids: List[str] = []
    item_ids.extend(_journal_item_ids(entry.get("negative_results", [])))
    item_ids.extend(_journal_item_ids(entry.get("regression_watchlist", [])))
    item_ids.extend(_journal_item_ids(entry.get("next_hypotheses", [])))
    return sorted(set(item_ids))


def attach_remeasure_results_to_research_journal_entries(
    entries: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [dict(item) for item in entries if isinstance(item, dict)]
    linked_count = 0
    skipped_duplicate_count = 0
    status_counts: Dict[str, int] = {}
    result_counts: Dict[str, int] = {}

    for repair in repair_entries if isinstance(repair_entries, list) else []:
        if not isinstance(repair, dict):
            continue
        source = str(repair.get("source", "") or "")
        checks = (
            [str(item).strip() for item in repair.get("covered_checks", []) if str(item).strip()]
            if isinstance(repair.get("covered_checks"), list)
            else []
        )
        if "research_journal_alternative_probe" in source or "remeasure_quota_hold" in checks:
            continue
        if "research_journal_remeasure" not in source and "research_journal_summary" not in checks:
            continue
        status = str(repair.get("status", "") or "").strip().lower()
        if status not in {"success", "failed", "skipped", "timeout", "error"}:
            continue
        target_ids = sorted({item for item in checks if item != "research_journal_summary"})
        if not target_ids:
            continue
        result = {
            "command": str(repair.get("command", "") or ""),
            "status": status,
            "source": source,
            "resolved_timestamp": _safe_float(
                repair.get("resolved_timestamp", repair.get("timestamp", 0.0)),
                0.0,
            ),
            "target_ids": target_ids,
        }
        result_key = json.dumps(result, sort_keys=True)
        for entry in output:
            entry_ids = set(_journal_entry_item_ids(entry))
            if not entry_ids.intersection(target_ids):
                continue
            existing = (
                entry.get("remeasure_results", [])
                if isinstance(entry.get("remeasure_results", []), list)
                else []
            )
            existing_keys = {
                json.dumps(item, sort_keys=True)
                for item in existing
                if isinstance(item, dict)
            }
            if result_key in existing_keys:
                skipped_duplicate_count += 1
                continue
            entry["remeasure_results"] = [*existing, dict(result)]
            linked_count += 1
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            result_counts["linked"] = int(result_counts.get("linked", 0)) + 1

    return output, {
        "linked_count": int(linked_count),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "status_counts": dict(sorted(status_counts.items())),
        "result_counts": dict(sorted(result_counts.items())),
    }


def attach_alternative_probe_results_to_research_journal_entries(
    entries: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [dict(item) for item in entries if isinstance(item, dict)]
    linked_count = 0
    skipped_duplicate_count = 0
    status_counts: Dict[str, int] = {}

    for repair in repair_entries if isinstance(repair_entries, list) else []:
        if not isinstance(repair, dict):
            continue
        source = str(repair.get("source", "") or "")
        checks = (
            [str(item).strip() for item in repair.get("covered_checks", []) if str(item).strip()]
            if isinstance(repair.get("covered_checks"), list)
            else []
        )
        if "research_journal_alternative_probe" not in source and "remeasure_quota_hold" not in checks:
            continue
        status = str(repair.get("status", "") or "").strip().lower()
        if status not in {"success", "failed", "skipped", "timeout", "error"}:
            continue
        target_ids = sorted(
            {
                item
                for item in checks
                if item not in {"research_journal_summary", "remeasure_quota_hold"}
            }
        )
        if not target_ids:
            continue
        result = {
            "command": str(repair.get("command", "") or ""),
            "status": status,
            "source": source,
            "resolved_timestamp": _safe_float(
                repair.get("resolved_timestamp", repair.get("timestamp", 0.0)),
                0.0,
            ),
            "target_ids": target_ids,
            "probe_type": "alternative_benchmark",
        }
        result_key = json.dumps(result, sort_keys=True)
        for entry in output:
            entry_ids = set(_journal_entry_item_ids(entry))
            if not entry_ids.intersection(target_ids):
                continue
            existing = (
                entry.get("alternative_probe_results", [])
                if isinstance(entry.get("alternative_probe_results", []), list)
                else []
            )
            existing_keys = {
                json.dumps(item, sort_keys=True)
                for item in existing
                if isinstance(item, dict)
            }
            if result_key in existing_keys:
                skipped_duplicate_count += 1
                continue
            entry["alternative_probe_results"] = [*existing, dict(result)]
            linked_count += 1
            status_counts[status] = int(status_counts.get(status, 0)) + 1

    return output, {
        "linked_count": int(linked_count),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "status_counts": dict(sorted(status_counts.items())),
    }


def attach_research_planner_task_completions_to_research_journal_entries(
    entries: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [dict(item) for item in entries if isinstance(item, dict)]
    linked_count = 0
    skipped_duplicate_count = 0
    task_type_counts: Dict[str, int] = {}

    for repair in repair_entries if isinstance(repair_entries, list) else []:
        if not isinstance(repair, dict):
            continue
        status = str(repair.get("status", "") or "").strip().lower()
        if status not in {"success", "skipped"}:
            continue
        source = str(repair.get("source", "") or "").strip()
        checks = (
            [str(item).strip() for item in repair.get("covered_checks", []) if str(item).strip()]
            if isinstance(repair.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        task_type = ""
        if "cause_boundary_documentation" in source or "cause_boundary_documentation" in check_set:
            task_type = "cause_boundary_documentation"
        elif "targeted_fixture_repair" in source or "targeted_fixture_repair" in check_set:
            task_type = "targeted_fixture_repair"
        if not task_type:
            continue
        target_ids = sorted(
            item
            for item in check_set
            if item
            and item
            not in {
                "research_review",
                "roadmap_patch_suggestion",
                "cause_boundary_documentation",
                "targeted_fixture_repair",
            }
        )
        if not target_ids:
            explicit_id = str(repair.get("task_id", "") or repair.get("id", "") or "").strip()
            if explicit_id:
                target_ids = [explicit_id]
        if not target_ids:
            continue
        result = {
            "task_type": task_type,
            "status": status,
            "source": source,
            "command": str(repair.get("command", "") or ""),
            "resolved_timestamp": _safe_float(
                repair.get("resolved_timestamp", repair.get("timestamp", 0.0)),
                0.0,
            ),
            "target_ids": target_ids,
        }
        result_key = json.dumps(result, sort_keys=True)
        for entry in output:
            entry_ids = set(_journal_entry_item_ids(entry))
            if not entry_ids.intersection(target_ids):
                continue
            existing = (
                entry.get("completed_research_planner_tasks", [])
                if isinstance(entry.get("completed_research_planner_tasks", []), list)
                else []
            )
            existing_keys = {
                json.dumps(item, sort_keys=True)
                for item in existing
                if isinstance(item, dict)
            }
            if result_key in existing_keys:
                skipped_duplicate_count += 1
                continue
            entry["completed_research_planner_tasks"] = [*existing, dict(result)]
            if task_type == "cause_boundary_documentation":
                previous = (
                    entry.get("completed_cause_boundary_documentation_ids", [])
                    if isinstance(entry.get("completed_cause_boundary_documentation_ids", []), list)
                    else []
                )
                entry["completed_cause_boundary_documentation_ids"] = sorted(
                    {str(item).strip() for item in [*previous, *target_ids] if str(item).strip()}
                )
            else:
                previous = (
                    entry.get("completed_targeted_fixture_repair_ids", [])
                    if isinstance(entry.get("completed_targeted_fixture_repair_ids", []), list)
                    else []
                )
                entry["completed_targeted_fixture_repair_ids"] = sorted(
                    {str(item).strip() for item in [*previous, *target_ids] if str(item).strip()}
                )
            linked_count += 1
            task_type_counts[task_type] = int(task_type_counts.get(task_type, 0)) + 1

    return output, {
        "linked_count": int(linked_count),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "task_type_counts": dict(sorted(task_type_counts.items())),
    }


def attach_roadmap_patch_evidence_collection_completions_to_research_journal_entries(
    entries: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [dict(item) for item in entries if isinstance(item, dict)]
    linked_count = 0
    skipped_duplicate_count = 0
    evidence_kind_counts: Dict[str, int] = {}

    def _evidence_kind(command: str, source: str, checks: List[str]) -> str:
        haystack = " ".join([command, source, *checks]).lower()
        if "real_data" in haystack or "real-data" in haystack or "fixture" in haystack:
            return "real_data_fixture"
        if "release_soak" in haystack or "release-soak" in haystack or "soak_trend" in haystack:
            return "release_soak_trend"
        if "targeted_probe" in haystack or "targeted-probe" in haystack or "probe" in haystack:
            return "targeted_probe"
        return ""

    ignored_checks = {
        "research_review",
        "research_journal_summary",
        "roadmap_patch_refresh_policy",
        "roadmap_patch_suggestion",
        "evidence_collection",
        "roadmap_patch_evidence_collection",
        "targeted_probe",
        "real_data_fixture",
        "release_soak_trend",
    }
    for repair in repair_entries if isinstance(repair_entries, list) else []:
        if not isinstance(repair, dict):
            continue
        status = str(repair.get("status", "") or "").strip().lower()
        if status not in {"success", "skipped"}:
            continue
        source = str(repair.get("source", "") or "").strip()
        command = str(repair.get("command", "") or "")
        checks = (
            [str(item).strip() for item in repair.get("covered_checks", []) if str(item).strip()]
            if isinstance(repair.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        if (
            "evidence_collection" not in source
            and "evidence_collection" not in command
            and "evidence_collection" not in check_set
        ):
            continue
        evidence_kind = _evidence_kind(command, source, checks)
        if not evidence_kind:
            continue
        target_ids = sorted(item for item in check_set if item and item not in ignored_checks)
        if not target_ids:
            explicit_id = str(repair.get("task_id", "") or repair.get("id", "") or "").strip()
            if explicit_id:
                target_ids = [explicit_id]
        if not target_ids:
            continue
        result = {
            "task_type": "roadmap_patch_evidence_collection",
            "status": status,
            "source": source,
            "command": command,
            "resolved_timestamp": _safe_float(
                repair.get("resolved_timestamp", repair.get("timestamp", 0.0)),
                0.0,
            ),
            "target_ids": target_ids,
            "evidence_kind": evidence_kind,
        }
        result_key = json.dumps(result, sort_keys=True)
        for entry in output:
            task_ids = {
                str(item.get("id", "") or "").strip()
                for item in (
                    entry.get("roadmap_patch_evidence_collection_tasks", [])
                    if isinstance(entry.get("roadmap_patch_evidence_collection_tasks", []), list)
                    else []
                )
                if isinstance(item, dict) and str(item.get("id", "") or "").strip()
            }
            entry_ids = set(_journal_entry_item_ids(entry)).union(task_ids)
            if not entry_ids.intersection(target_ids):
                continue
            existing = (
                entry.get("completed_roadmap_patch_evidence_collection_tasks", [])
                if isinstance(entry.get("completed_roadmap_patch_evidence_collection_tasks", []), list)
                else []
            )
            existing_keys = {
                json.dumps(item, sort_keys=True)
                for item in existing
                if isinstance(item, dict)
            }
            if result_key in existing_keys:
                skipped_duplicate_count += 1
                continue
            entry["completed_roadmap_patch_evidence_collection_tasks"] = [*existing, dict(result)]
            previous_keys = (
                entry.get("completed_roadmap_patch_evidence_collection_keys", [])
                if isinstance(entry.get("completed_roadmap_patch_evidence_collection_keys", []), list)
                else []
            )
            entry["completed_roadmap_patch_evidence_collection_keys"] = sorted(
                {
                    str(item).strip()
                    for item in [
                        *previous_keys,
                        *[f"{target_id}:{evidence_kind}" for target_id in target_ids],
                    ]
                    if str(item).strip()
                }
            )
            linked_count += 1
            evidence_kind_counts[evidence_kind] = int(evidence_kind_counts.get(evidence_kind, 0)) + 1

    return output, {
        "linked_count": int(linked_count),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "evidence_kind_counts": dict(sorted(evidence_kind_counts.items())),
    }


def attach_stage_e_observed_candidate_recovery_reviews_to_research_journal_entries(
    entries: List[Dict[str, Any]],
    repair_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [dict(item) for item in entries if isinstance(item, dict)]
    linked_count = 0
    skipped_duplicate_count = 0
    status_counts: Dict[str, int] = {}

    for repair in repair_entries if isinstance(repair_entries, list) else []:
        if not isinstance(repair, dict):
            continue
        status = str(repair.get("status", "") or "").strip().lower()
        if status not in {"pending", "success", "skipped", "failed", "timeout", "error"}:
            continue
        source = str(repair.get("source", "") or "").strip()
        command = str(repair.get("command", "") or "")
        checks = (
            [str(item).strip() for item in repair.get("covered_checks", []) if str(item).strip()]
            if isinstance(repair.get("covered_checks"), list)
            else []
        )
        check_set = set(checks)
        if not (
            "stage_e_observed_acceptance_candidate_recovery_review" in source
            or "stage_e_observed_acceptance_candidate_recovery_review" in command
            or "stage_e_observed_acceptance_candidate_repair_recovery" in check_set
        ):
            continue
        review_type = (
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck" in source
                or "recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe" in command
            )
            else
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe" in source
                or "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck" in source
                or "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection" in source
                or "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence" in command
            )
            else
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation" in source
                or "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry" in command
            )
            else "stage_e_observed_acceptance_candidate_recovery_review_followup_retry"
            if (
                "stage_e_observed_acceptance_candidate_recovery_review_followup_retry" in source
                or "retry_stage_e_observed_acceptance_candidate_recovery_review_followup" in command
            )
            else (
                "stage_e_observed_acceptance_candidate_recovery_review_followup"
                if (
                    "stage_e_observed_acceptance_candidate_recovery_review_followup" in source
                    or "followup_stage_e_observed_acceptance_candidate_recovery_review" in command
                )
                else "stage_e_observed_acceptance_candidate_recovery_review"
            )
        )
        result = {
            "status": status,
            "source": source,
            "command": command,
            "resolved_timestamp": _safe_float(
                repair.get("resolved_timestamp", repair.get("timestamp", 0.0)),
                0.0,
            ),
            "target_ids": [STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID],
            "review_type": review_type,
        }
        result_key = json.dumps(result, sort_keys=True)
        for entry in output:
            entry_ids = set(_journal_entry_item_ids(entry))
            if STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID not in entry_ids:
                continue
            existing = (
                entry.get("stage_e_observed_acceptance_candidate_recovery_review_results", [])
                if isinstance(
                    entry.get("stage_e_observed_acceptance_candidate_recovery_review_results", []),
                    list,
                )
                else []
            )
            existing_keys = {
                json.dumps(item, sort_keys=True)
                for item in existing
                if isinstance(item, dict)
            }
            if result_key in existing_keys:
                skipped_duplicate_count += 1
                continue
            entry["stage_e_observed_acceptance_candidate_recovery_review_results"] = [
                *existing,
                dict(result),
            ]
            linked_count += 1
            status_counts[status] = int(status_counts.get(status, 0)) + 1

    return output, {
        "linked_count": int(linked_count),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "status_counts": dict(sorted(status_counts.items())),
    }


def summarize_research_journal_entries(
    entries: List[Dict[str, Any]],
    *,
    limit: int = 5,
    now_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    normalized = [dict(item) for item in entries if isinstance(item, dict)]
    now = _safe_float(now_timestamp, time.time()) if now_timestamp is not None else time.time()
    top_limit = max(1, int(limit))
    negative_counts: Dict[str, int] = {}
    regression_counts: Dict[str, int] = {}
    next_counts: Dict[str, int] = {}
    rejected_patch_count = 0
    approved_patch_count = 0
    roadmap_patch_rejection_reason_counts: Dict[str, int] = {}
    roadmap_patch_rejected_item_counts: Dict[str, int] = {}
    roadmap_patch_rejected_items_by_id: Dict[str, Dict[str, Any]] = {}
    roadmap_patch_refreshed_items_by_id: Dict[str, Dict[str, Any]] = {}
    remeasure_result_count = 0
    remeasure_status_counts: Dict[str, int] = {}
    remeasure_id_counts: Dict[str, int] = {}
    remeasure_trends_by_id: Dict[str, Dict[str, Any]] = {}
    alternative_probe_result_count = 0
    alternative_probe_status_counts: Dict[str, int] = {}
    alternative_probe_trends_by_id: Dict[str, Dict[str, Any]] = {}
    stage_e_recovery_review_entries: List[Dict[str, Any]] = []
    completed_cause_boundary_ids: set[str] = set()
    completed_targeted_fixture_ids: set[str] = set()
    completed_research_planner_tasks: List[Dict[str, Any]] = []
    completed_evidence_collection_tasks: List[Dict[str, Any]] = []
    completed_evidence_collection_keys: set[str] = set()
    oldest_at = 0.0
    newest_at = 0.0
    total_seen_count = 0

    def _add_counts(counter: Dict[str, int], items: Any) -> None:
        for item_id in _journal_item_ids(items):
            counter[item_id] = int(counter.get(item_id, 0)) + 1

    for entry in normalized:
        generated_at = _safe_float(entry.get("generated_at", 0.0), 0.0)
        if generated_at > 0:
            oldest_at = generated_at if oldest_at <= 0 else min(oldest_at, generated_at)
            newest_at = max(newest_at, generated_at)
        total_seen_count += int(entry.get("seen_count", 1) or 1)
        _add_counts(negative_counts, entry.get("negative_results", []))
        _add_counts(regression_counts, entry.get("regression_watchlist", []))
        _add_counts(next_counts, entry.get("next_hypotheses", []))
        for source_key in ("negative_results", "regression_watchlist", "next_hypotheses"):
            source_items = entry.get(source_key, [])
            if not isinstance(source_items, list):
                continue
            for item in source_items:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("roadmap_patch_review_suppression_lifted", False)):
                    continue
                item_id = _journal_item_id(item)
                if not item_id:
                    continue
                refresh_at = _safe_float(item.get("roadmap_patch_review_refresh_timestamp", 0.0), 0.0)
                refreshed = roadmap_patch_refreshed_items_by_id.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "count": 0,
                        "latest_reason": "",
                        "latest_refresh_timestamp": 0.0,
                        "latest_seen_generated_at": 0.0,
                    },
                )
                refreshed["count"] = int(refreshed.get("count", 0) or 0) + 1
                if refresh_at >= float(refreshed.get("latest_refresh_timestamp", 0.0) or 0.0):
                    refreshed["latest_reason"] = str(
                        item.get("roadmap_patch_review_refresh_reason", "") or ""
                    )
                    refreshed["latest_refresh_timestamp"] = float(refresh_at)
                    refreshed["latest_seen_generated_at"] = float(generated_at)
        decision = str(entry.get("roadmap_patch_review_decision", "") or "").strip().lower()
        if decision == "rejected":
            rejected_patch_count += 1
            reason = str(entry.get("roadmap_patch_review_reason", "") or "").strip()
            reason_key = reason or "unspecified"
            roadmap_patch_rejection_reason_counts[reason_key] = (
                int(roadmap_patch_rejection_reason_counts.get(reason_key, 0)) + 1
            )
            rejected_item_ids = sorted(
                {
                    *_journal_item_ids(entry.get("negative_results", [])),
                    *_journal_item_ids(entry.get("regression_watchlist", [])),
                    *_journal_item_ids(entry.get("next_hypotheses", [])),
                }
            )
            for item_id in rejected_item_ids:
                roadmap_patch_rejected_item_counts[item_id] = (
                    int(roadmap_patch_rejected_item_counts.get(item_id, 0)) + 1
                )
                item = roadmap_patch_rejected_items_by_id.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "count": 0,
                        "latest_reason": "",
                        "latest_timestamp": 0.0,
                    },
                )
                item["count"] = int(item.get("count", 0) or 0) + 1
                if generated_at >= float(item.get("latest_timestamp", 0.0) or 0.0):
                    item["latest_reason"] = reason
                    item["latest_timestamp"] = float(generated_at)
        elif decision == "approved":
            approved_patch_count += 1
        remeasure_results = (
            entry.get("remeasure_results", [])
            if isinstance(entry.get("remeasure_results", []), list)
            else []
        )
        for result in remeasure_results:
            if not isinstance(result, dict):
                continue
            status = str(result.get("status", "") or "").strip().lower() or "unknown"
            remeasure_result_count += 1
            remeasure_status_counts[status] = int(remeasure_status_counts.get(status, 0)) + 1
            target_ids = (
                [str(item).strip() for item in result.get("target_ids", []) if str(item).strip()]
                if isinstance(result.get("target_ids"), list)
                else []
            )
            timestamp = _safe_float(result.get("resolved_timestamp", 0.0), 0.0)
            for target_id in target_ids:
                remeasure_id_counts[target_id] = int(remeasure_id_counts.get(target_id, 0)) + 1
                trend = remeasure_trends_by_id.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "success_count": 0,
                        "failed_count": 0,
                        "skipped_count": 0,
                        "latest_status": "",
                        "latest_timestamp": 0.0,
                        "latest_command": "",
                    },
                )
                if status == "success":
                    trend["success_count"] = int(trend.get("success_count", 0) or 0) + 1
                elif status in {"failed", "error", "timeout"}:
                    trend["failed_count"] = int(trend.get("failed_count", 0) or 0) + 1
                elif status == "skipped":
                    trend["skipped_count"] = int(trend.get("skipped_count", 0) or 0) + 1
                if timestamp >= float(trend.get("latest_timestamp", 0.0) or 0.0):
                    trend["latest_status"] = status
                    trend["latest_timestamp"] = float(timestamp)
                    trend["latest_command"] = str(result.get("command", "") or "")
        completed_tasks = (
            entry.get("completed_research_planner_tasks", [])
            if isinstance(entry.get("completed_research_planner_tasks", []), list)
            else []
        )
        for task in completed_tasks:
            if not isinstance(task, dict):
                continue
            task_type = str(task.get("task_type", "") or "")
            target_ids = (
                [str(item).strip() for item in task.get("target_ids", []) if str(item).strip()]
                if isinstance(task.get("target_ids"), list)
                else []
            )
            for target_id in target_ids:
                if task_type == "cause_boundary_documentation":
                    completed_cause_boundary_ids.add(target_id)
                elif task_type == "targeted_fixture_repair":
                    completed_targeted_fixture_ids.add(target_id)
            completed_research_planner_tasks.append(dict(task))
        completed_cause_boundary_ids.update(
            str(item).strip()
            for item in (
                entry.get("completed_cause_boundary_documentation_ids", [])
                if isinstance(entry.get("completed_cause_boundary_documentation_ids", []), list)
                else []
            )
            if str(item).strip()
        )
        completed_targeted_fixture_ids.update(
            str(item).strip()
            for item in (
                entry.get("completed_targeted_fixture_repair_ids", [])
                if isinstance(entry.get("completed_targeted_fixture_repair_ids", []), list)
                else []
            )
            if str(item).strip()
        )
        evidence_tasks = (
            entry.get("completed_roadmap_patch_evidence_collection_tasks", [])
            if isinstance(entry.get("completed_roadmap_patch_evidence_collection_tasks", []), list)
            else []
        )
        for task in evidence_tasks:
            if not isinstance(task, dict):
                continue
            evidence_kind = str(task.get("evidence_kind", "") or "").strip()
            target_ids = (
                [str(item).strip() for item in task.get("target_ids", []) if str(item).strip()]
                if isinstance(task.get("target_ids"), list)
                else []
            )
            if evidence_kind:
                completed_evidence_collection_keys.update(
                    f"{target_id}:{evidence_kind}" for target_id in target_ids
                )
            completed_evidence_collection_tasks.append(dict(task))
        completed_evidence_collection_keys.update(
            str(item).strip()
            for item in (
                entry.get("completed_roadmap_patch_evidence_collection_keys", [])
                if isinstance(entry.get("completed_roadmap_patch_evidence_collection_keys", []), list)
                else []
            )
            if str(item).strip()
        )
        alternative_probe_results = (
            entry.get("alternative_probe_results", [])
            if isinstance(entry.get("alternative_probe_results", []), list)
            else []
        )
        for result in alternative_probe_results:
            if not isinstance(result, dict):
                continue
            status = str(result.get("status", "") or "").strip().lower() or "unknown"
            alternative_probe_result_count += 1
            alternative_probe_status_counts[status] = (
                int(alternative_probe_status_counts.get(status, 0)) + 1
            )
            target_ids = (
                [str(item).strip() for item in result.get("target_ids", []) if str(item).strip()]
                if isinstance(result.get("target_ids"), list)
                else []
            )
            timestamp = _safe_float(result.get("resolved_timestamp", 0.0), 0.0)
            for target_id in target_ids:
                trend = alternative_probe_trends_by_id.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "success_count": 0,
                        "failed_count": 0,
                        "skipped_count": 0,
                        "latest_status": "",
                        "latest_timestamp": 0.0,
                        "latest_command": "",
                    },
                )
                if status == "success":
                    trend["success_count"] = int(trend.get("success_count", 0) or 0) + 1
                elif status in {"failed", "error", "timeout"}:
                    trend["failed_count"] = int(trend.get("failed_count", 0) or 0) + 1
                elif status == "skipped":
                    trend["skipped_count"] = int(trend.get("skipped_count", 0) or 0) + 1
                if timestamp >= float(trend.get("latest_timestamp", 0.0) or 0.0):
                    trend["latest_status"] = status
                    trend["latest_timestamp"] = float(timestamp)
                    trend["latest_command"] = str(result.get("command", "") or "")
        recovery_review_results = (
            entry.get("stage_e_observed_acceptance_candidate_recovery_review_results", [])
            if isinstance(
                entry.get("stage_e_observed_acceptance_candidate_recovery_review_results", []),
                list,
            )
            else []
        )
        for result in recovery_review_results:
            if not isinstance(result, dict):
                continue
            stage_e_recovery_review_entries.append(dict(result))

    def _top(counter: Dict[str, int]) -> List[Dict[str, Any]]:
        return [
            {"id": key, "count": int(value)}
            for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_limit]
        ]

    def _trend_label(item: Dict[str, Any]) -> str:
        latest = str(item.get("latest_status", "") or "")
        failed_count = int(item.get("failed_count", 0) or 0)
        success_count = int(item.get("success_count", 0) or 0)
        if latest == "success" and failed_count > 0:
            return "recovered"
        if latest == "success":
            return "confirmed"
        if latest in {"failed", "error", "timeout"} and success_count > 0:
            return "regressed_after_success"
        if latest in {"failed", "error", "timeout"}:
            return "still_failing"
        if latest == "skipped":
            return "skipped"
        return "unknown"

    def _trend_interval_seconds(trend_label: str) -> float:
        if trend_label in {"recovered", "confirmed"}:
            return RECOVERED_REMEASURE_INTERVAL_SECONDS
        if trend_label in {"still_failing", "regressed_after_success"}:
            return FAILED_REMEASURE_INTERVAL_SECONDS
        if trend_label == "skipped":
            return SKIPPED_REMEASURE_INTERVAL_SECONDS
        return 0.0

    remeasure_trends = []
    for item in sorted(
        remeasure_trends_by_id.values(),
        key=lambda value: (-float(value.get("latest_timestamp", 0.0) or 0.0), str(value.get("id", ""))),
    )[:top_limit]:
        remeasure_trends.append(
            {
                **item,
                "trend": _trend_label(item),
            }
        )

    alternative_probe_trends = []
    for item in sorted(
        alternative_probe_trends_by_id.values(),
        key=lambda value: (-float(value.get("latest_timestamp", 0.0) or 0.0), str(value.get("id", ""))),
    )[:top_limit]:
        trend_label = _trend_label(item)
        alternative_probe_trends.append(
            {
                **item,
                "trend": (
                    "targeted_probe_passed"
                    if trend_label in {"confirmed", "recovered"}
                    else "targeted_probe_failed"
                    if trend_label in {"still_failing", "regressed_after_success"}
                    else trend_label
                ),
            }
        )

    stage_e_repair_remeasure = remeasure_trends_by_id.get(
        STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID,
        {},
    )
    stage_e_repair_alternative_probe = alternative_probe_trends_by_id.get(
        STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID,
        {},
    )
    stage_e_repair_remeasure_trend = (
        _trend_label(stage_e_repair_remeasure)
        if stage_e_repair_remeasure
        else ""
    )
    stage_e_repair_alternative_probe_base_trend = (
        _trend_label(stage_e_repair_alternative_probe)
        if stage_e_repair_alternative_probe
        else ""
    )
    stage_e_repair_alternative_probe_trend = (
        "targeted_probe_passed"
        if stage_e_repair_alternative_probe_base_trend in {"confirmed", "recovered"}
        else "targeted_probe_failed"
        if stage_e_repair_alternative_probe_base_trend in {"still_failing", "regressed_after_success"}
        else stage_e_repair_alternative_probe_base_trend
    )
    trend_lookup = {str(item.get("id", "") or ""): item for item in remeasure_trends if isinstance(item, dict)}
    recommended_actions: List[Dict[str, Any]] = []
    suppressed_actions: List[Dict[str, Any]] = []
    seen_commands: set[str] = set()
    combined_top = [
        ("negative_result", item)
        for item in _top(negative_counts)
    ] + [
        ("regression_watchlist", item)
        for item in _top(regression_counts)
    ] + [
        ("next_hypothesis", item)
        for item in _top(next_counts)
    ]
    for source, item in combined_top:
        item_id = str(item.get("id", "") or "")
        command = RESEARCH_JOURNAL_BENCHMARK_COMMANDS.get(item_id, "")
        if not command or command in seen_commands:
            continue
        trend = trend_lookup.get(item_id, {})
        trend_label = str(trend.get("trend", "") or "")
        interval_seconds = _trend_interval_seconds(trend_label)
        latest_timestamp = _safe_float(trend.get("latest_timestamp", 0.0), 0.0)
        elapsed_seconds = max(now - latest_timestamp, 0.0) if latest_timestamp > 0 else 0.0
        remaining_seconds = max(interval_seconds - elapsed_seconds, 0.0) if interval_seconds > 0 else 0.0
        if trend_label in {"recovered", "confirmed", "skipped"} and remaining_seconds > 0:
            suppressed_actions.append(
                {
                    "id": item_id,
                    "source": source,
                    "command": command,
                    "priority": "low",
                    "count": int(item.get("count", 0) or 0),
                    "remeasure_trend": trend_label,
                    "seconds_until_next_remeasure": float(remaining_seconds),
                }
            )
            continue
        if trend_label in {"still_failing", "regressed_after_success"} and remaining_seconds > 0:
            suppressed_actions.append(
                {
                    "id": item_id,
                    "source": source,
                    "command": command,
                    "priority": "high",
                    "count": int(item.get("count", 0) or 0),
                    "remeasure_trend": trend_label,
                    "seconds_until_next_remeasure": float(remaining_seconds),
                }
            )
            continue
        seen_commands.add(command)
        recommended_actions.append(
            {
                "id": item_id,
                "source": source,
                "command": command,
                "priority": "high" if source != "next_hypothesis" else "medium",
                "count": int(item.get("count", 0) or 0),
                "remeasure_trend": trend_label,
                "recommended_remeasure_interval_seconds": float(interval_seconds),
                "seconds_since_latest_remeasure": float(elapsed_seconds),
            }
        )

    stage_e_repair_remeasure_recommended = any(
        str(item.get("id", "") or "") == STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID
        for item in recommended_actions
        if isinstance(item, dict)
    )
    stage_e_repair_remeasure_suppressed = any(
        str(item.get("id", "") or "") == STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID
        for item in suppressed_actions
        if isinstance(item, dict)
    )
    stage_e_repair_recovery_sources = []
    if stage_e_repair_remeasure_trend in {"recovered", "confirmed"}:
        stage_e_repair_recovery_sources.append("remeasure")
    if stage_e_repair_alternative_probe_trend == "targeted_probe_passed":
        stage_e_repair_recovery_sources.append("alternative_probe")
    stage_e_repair_recovery_confirmed = bool(stage_e_repair_recovery_sources)
    stage_e_recovery_review_status_counts: Dict[str, int] = {}
    stage_e_recovery_review_latest: Dict[str, Any] = {}
    for review in stage_e_recovery_review_entries:
        status = str(review.get("status", "") or "").strip().lower() or "unknown"
        stage_e_recovery_review_status_counts[status] = (
            int(stage_e_recovery_review_status_counts.get(status, 0)) + 1
        )
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest = review
    stage_e_recovery_review_latest_status = str(
        stage_e_recovery_review_latest.get("status", "") or ""
    )
    stage_e_recovery_review_completed = stage_e_recovery_review_latest_status in {
        "success",
        "skipped",
    }
    stage_e_recovery_review_in_progress = stage_e_recovery_review_latest_status == "pending"
    stage_e_recovery_review_followup_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        in {
            "stage_e_observed_acceptance_candidate_recovery_review_followup",
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
            "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
            "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
            "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck",
        }
    ]
    stage_e_recovery_review_followup_retry_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry"
    ]
    stage_e_recovery_review_followup_retry_escalation_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation"
    ]
    stage_e_recovery_review_evidence_collection_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection"
    ]
    stage_e_recovery_review_evidence_recheck_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck"
    ]
    stage_e_recovery_review_targeted_probe_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe"
    ]
    stage_e_recovery_review_targeted_probe_recheck_entries = [
        item
        for item in stage_e_recovery_review_entries
        if str(item.get("review_type", "") or "")
        == "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck"
    ]
    stage_e_recovery_review_latest_followup: Dict[str, Any] = {}
    for review in stage_e_recovery_review_followup_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_followup.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest_followup = review
    stage_e_recovery_review_followup_latest_status = str(
        stage_e_recovery_review_latest_followup.get("status", "") or ""
    )
    stage_e_recovery_review_followup_in_progress = (
        stage_e_recovery_review_followup_latest_status == "pending"
    )
    stage_e_recovery_review_followup_completed = stage_e_recovery_review_followup_latest_status in {
        "success",
        "skipped",
    }
    stage_e_recovery_review_followup_failed = stage_e_recovery_review_followup_latest_status in {
        "failed",
        "timeout",
        "error",
    }
    stage_e_recovery_review_latest_followup_retry: Dict[str, Any] = {}
    for review in stage_e_recovery_review_followup_retry_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_followup_retry.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest_followup_retry = review
    stage_e_recovery_review_followup_retry_latest_status = str(
        stage_e_recovery_review_latest_followup_retry.get("status", "") or ""
    )
    stage_e_recovery_review_followup_retry_in_progress = (
        stage_e_recovery_review_followup_retry_latest_status == "pending"
    )
    stage_e_recovery_review_followup_retry_completed = (
        stage_e_recovery_review_followup_retry_latest_status in {"success", "skipped"}
    )
    stage_e_recovery_review_followup_retry_failed = (
        stage_e_recovery_review_followup_retry_latest_status in {"failed", "timeout", "error"}
    )
    stage_e_recovery_review_latest_followup_retry_escalation: Dict[str, Any] = {}
    for review in stage_e_recovery_review_followup_retry_escalation_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_followup_retry_escalation.get(
                "resolved_timestamp",
                0.0,
            ),
            0.0,
        ):
            stage_e_recovery_review_latest_followup_retry_escalation = review
    stage_e_recovery_review_followup_retry_escalation_latest_status = str(
        stage_e_recovery_review_latest_followup_retry_escalation.get("status", "") or ""
    )
    stage_e_recovery_review_followup_retry_escalation_in_progress = (
        stage_e_recovery_review_followup_retry_escalation_latest_status == "pending"
    )
    stage_e_recovery_review_followup_retry_escalation_completed = (
        stage_e_recovery_review_followup_retry_escalation_latest_status
        in {"success", "skipped"}
    )
    stage_e_recovery_review_followup_retry_escalation_failed = (
        stage_e_recovery_review_followup_retry_escalation_latest_status
        in {"failed", "timeout", "error"}
    )
    stage_e_recovery_review_latest_evidence_collection: Dict[str, Any] = {}
    for review in stage_e_recovery_review_evidence_collection_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_evidence_collection.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest_evidence_collection = review
    stage_e_recovery_review_evidence_collection_latest_status = str(
        stage_e_recovery_review_latest_evidence_collection.get("status", "") or ""
    )
    stage_e_recovery_review_evidence_collection_in_progress = (
        stage_e_recovery_review_evidence_collection_latest_status == "pending"
    )
    stage_e_recovery_review_evidence_collection_completed = (
        stage_e_recovery_review_evidence_collection_latest_status in {"success", "skipped"}
    )
    stage_e_recovery_review_evidence_collection_failed = (
        stage_e_recovery_review_evidence_collection_latest_status
        in {"failed", "timeout", "error"}
    )
    stage_e_recovery_review_latest_evidence_recheck: Dict[str, Any] = {}
    for review in stage_e_recovery_review_evidence_recheck_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_evidence_recheck.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest_evidence_recheck = review
    stage_e_recovery_review_evidence_recheck_latest_status = str(
        stage_e_recovery_review_latest_evidence_recheck.get("status", "") or ""
    )
    stage_e_recovery_review_evidence_recheck_in_progress = (
        stage_e_recovery_review_evidence_recheck_latest_status == "pending"
    )
    stage_e_recovery_review_evidence_recheck_completed = (
        stage_e_recovery_review_evidence_recheck_latest_status in {"success", "skipped"}
    )
    stage_e_recovery_review_evidence_recheck_failed = (
        stage_e_recovery_review_evidence_recheck_latest_status in {"failed", "timeout", "error"}
    )
    stage_e_recovery_review_latest_targeted_probe: Dict[str, Any] = {}
    for review in stage_e_recovery_review_targeted_probe_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_targeted_probe.get("resolved_timestamp", 0.0),
            0.0,
        ):
            stage_e_recovery_review_latest_targeted_probe = review
    stage_e_recovery_review_targeted_probe_latest_status = str(
        stage_e_recovery_review_latest_targeted_probe.get("status", "") or ""
    )
    stage_e_recovery_review_targeted_probe_in_progress = (
        stage_e_recovery_review_targeted_probe_latest_status == "pending"
    )
    stage_e_recovery_review_targeted_probe_completed = (
        stage_e_recovery_review_targeted_probe_latest_status in {"success", "skipped"}
    )
    stage_e_recovery_review_targeted_probe_failed = (
        stage_e_recovery_review_targeted_probe_latest_status in {"failed", "timeout", "error"}
    )
    stage_e_recovery_review_latest_targeted_probe_recheck: Dict[str, Any] = {}
    for review in stage_e_recovery_review_targeted_probe_recheck_entries:
        if _safe_float(review.get("resolved_timestamp", 0.0), 0.0) >= _safe_float(
            stage_e_recovery_review_latest_targeted_probe_recheck.get(
                "resolved_timestamp",
                0.0,
            ),
            0.0,
        ):
            stage_e_recovery_review_latest_targeted_probe_recheck = review
    stage_e_recovery_review_targeted_probe_recheck_latest_status = str(
        stage_e_recovery_review_latest_targeted_probe_recheck.get("status", "") or ""
    )
    stage_e_recovery_review_targeted_probe_recheck_in_progress = (
        stage_e_recovery_review_targeted_probe_recheck_latest_status == "pending"
    )
    stage_e_recovery_review_targeted_probe_recheck_completed = (
        stage_e_recovery_review_targeted_probe_recheck_latest_status in {"success", "skipped"}
    )
    stage_e_recovery_review_targeted_probe_recheck_failed = (
        stage_e_recovery_review_targeted_probe_recheck_latest_status
        in {"failed", "timeout", "error"}
    )
    stage_e_repair_needs_followup = bool(
        stage_e_repair_remeasure_recommended
        or stage_e_repair_remeasure_trend in {"still_failing", "regressed_after_success", "skipped"}
        or stage_e_repair_alternative_probe_trend == "targeted_probe_failed"
    )
    stage_e_repair_promotion_review_recommended = bool(
        stage_e_repair_recovery_confirmed
        and not stage_e_repair_needs_followup
        and not stage_e_recovery_review_completed
        and not stage_e_recovery_review_in_progress
    )
    stage_e_repair_loop = {
        "schema": "sara-stage-e-observed-acceptance-candidate-repair-loop-v1",
        "id": STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID,
        "negative_result_count": int(
            negative_counts.get(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID, 0)
        ),
        "regression_watchlist_count": int(
            regression_counts.get(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID, 0)
        ),
        "next_hypothesis_count": int(
            next_counts.get(STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_REPAIR_ID, 0)
        ),
        "remeasure_recommended": bool(stage_e_repair_remeasure_recommended),
        "remeasure_suppressed": bool(stage_e_repair_remeasure_suppressed),
        "latest_remeasure_trend": stage_e_repair_remeasure_trend,
        "latest_remeasure_status": str(
            stage_e_repair_remeasure.get("latest_status", "") or ""
        ),
        "latest_alternative_probe_trend": stage_e_repair_alternative_probe_trend,
        "latest_alternative_probe_status": str(
            stage_e_repair_alternative_probe.get("latest_status", "") or ""
        ),
        "alternative_probe_recommended": False,
        "recovery_confirmed": stage_e_repair_recovery_confirmed,
        "recovery_source": ",".join(stage_e_repair_recovery_sources),
        "promotion_review_recommended": stage_e_repair_promotion_review_recommended,
        "promotion_review_completed": stage_e_recovery_review_completed,
        "promotion_review_in_progress": stage_e_recovery_review_in_progress,
        "promotion_review_followup_in_progress": stage_e_recovery_review_followup_in_progress,
        "promotion_review_followup_completed": stage_e_recovery_review_followup_completed,
        "promotion_review_followup_failed": stage_e_recovery_review_followup_failed,
        "promotion_review_followup_latest_status": stage_e_recovery_review_followup_latest_status,
        "promotion_review_followup_retry_in_progress": (
            stage_e_recovery_review_followup_retry_in_progress
        ),
        "promotion_review_followup_retry_completed": (
            stage_e_recovery_review_followup_retry_completed
        ),
        "promotion_review_followup_retry_failed": stage_e_recovery_review_followup_retry_failed,
        "promotion_review_followup_retry_latest_status": (
            stage_e_recovery_review_followup_retry_latest_status
        ),
        "promotion_review_followup_retry_escalation_in_progress": (
            stage_e_recovery_review_followup_retry_escalation_in_progress
        ),
        "promotion_review_followup_retry_escalation_completed": (
            stage_e_recovery_review_followup_retry_escalation_completed
        ),
        "promotion_review_followup_retry_escalation_failed": (
            stage_e_recovery_review_followup_retry_escalation_failed
        ),
        "promotion_review_followup_retry_escalation_latest_status": (
            stage_e_recovery_review_followup_retry_escalation_latest_status
        ),
        "promotion_review_evidence_collection_in_progress": (
            stage_e_recovery_review_evidence_collection_in_progress
        ),
        "promotion_review_evidence_collection_completed": (
            stage_e_recovery_review_evidence_collection_completed
        ),
        "promotion_review_evidence_collection_failed": (
            stage_e_recovery_review_evidence_collection_failed
        ),
        "promotion_review_evidence_collection_latest_status": (
            stage_e_recovery_review_evidence_collection_latest_status
        ),
        "promotion_review_evidence_recheck_in_progress": (
            stage_e_recovery_review_evidence_recheck_in_progress
        ),
        "promotion_review_evidence_recheck_completed": (
            stage_e_recovery_review_evidence_recheck_completed
        ),
        "promotion_review_evidence_recheck_failed": (
            stage_e_recovery_review_evidence_recheck_failed
        ),
        "promotion_review_evidence_recheck_latest_status": (
            stage_e_recovery_review_evidence_recheck_latest_status
        ),
        "promotion_review_targeted_probe_in_progress": (
            stage_e_recovery_review_targeted_probe_in_progress
        ),
        "promotion_review_targeted_probe_completed": (
            stage_e_recovery_review_targeted_probe_completed
        ),
        "promotion_review_targeted_probe_failed": stage_e_recovery_review_targeted_probe_failed,
        "promotion_review_targeted_probe_latest_status": (
            stage_e_recovery_review_targeted_probe_latest_status
        ),
        "promotion_review_targeted_probe_recheck_in_progress": (
            stage_e_recovery_review_targeted_probe_recheck_in_progress
        ),
        "promotion_review_targeted_probe_recheck_completed": (
            stage_e_recovery_review_targeted_probe_recheck_completed
        ),
        "promotion_review_targeted_probe_recheck_failed": (
            stage_e_recovery_review_targeted_probe_recheck_failed
        ),
        "promotion_review_targeted_probe_recheck_latest_status": (
            stage_e_recovery_review_targeted_probe_recheck_latest_status
        ),
        "promotion_review_latest_status": stage_e_recovery_review_latest_status,
        "promotion_review_completed_count": int(
            stage_e_recovery_review_status_counts.get("success", 0)
            + stage_e_recovery_review_status_counts.get("skipped", 0)
        ),
        "next_review_action": (
            "stage_e_observed_acceptance_candidate_stability"
            if stage_e_repair_promotion_review_recommended
            else ""
        ),
        "needs_followup": stage_e_repair_needs_followup,
    }
    stale_age_seconds = max(now - newest_at, 0.0) if newest_at > 0 else 0.0
    rejected_item_count = int(len(roadmap_patch_rejected_items_by_id))
    refreshed_item_count = int(len(roadmap_patch_refreshed_items_by_id))
    refresh_to_rejection_ratio = (
        float(refreshed_item_count) / float(rejected_item_count)
        if rejected_item_count > 0
        else 0.0
    )
    journal_experiment_status = classify_experiment_graph_status(
        {
            "stable_hypotheses": [],
            "regression_watchlist": [
                {"id": item["id"]}
                for item in _top(regression_counts)
            ],
            "negative_results": [
                {"id": item["id"]}
                for item in _top(negative_counts)
            ],
            "roadmap_patch_evidence_collection_tasks": [],
        },
        {
            "remeasure_trends": remeasure_trends,
            "roadmap_patch_rejected_items": [
                dict(item)
                for item in sorted(
                    roadmap_patch_rejected_items_by_id.values(),
                    key=lambda value: (
                        -int(value.get("count", 0) or 0),
                        str(value.get("id", "")),
                    ),
                )[:top_limit]
            ],
            "roadmap_patch_refreshed_items": [
                dict(item)
                for item in sorted(
                    roadmap_patch_refreshed_items_by_id.values(),
                    key=lambda value: (
                        -int(value.get("count", 0) or 0),
                        str(value.get("id", "")),
                    ),
                )[:top_limit]
            ],
            "completed_roadmap_patch_evidence_collection_keys": sorted(
                completed_evidence_collection_keys
            ),
        },
        limit=top_limit,
    )
    journal_experiment_priority_plan = build_experiment_status_priority_plan(
        journal_experiment_status,
        limit=top_limit,
    )
    journal_experiment_promotion_target_plan = build_experiment_promotion_target_plan(
        journal_experiment_status,
        limit=top_limit,
    )
    return {
        "schema": "sara-research-journal-summary-v1",
        "entry_count": int(len(normalized)),
        "total_seen_count": int(total_seen_count),
        "oldest_generated_at": float(oldest_at),
        "newest_generated_at": float(newest_at),
        "stale_age_seconds": float(stale_age_seconds),
        "top_negative_results": _top(negative_counts),
        "top_regression_watchlist": _top(regression_counts),
        "top_next_hypotheses": _top(next_counts),
        "recommended_benchmark_actions": recommended_actions[:top_limit],
        "suppressed_benchmark_actions": suppressed_actions[:top_limit],
        "stage_e_observed_acceptance_candidate_repair_loop": stage_e_repair_loop,
        "remeasure_result_count": int(remeasure_result_count),
        "remeasure_status_counts": dict(sorted(remeasure_status_counts.items())),
        "top_remeasured_ids": _top(remeasure_id_counts),
        "remeasure_trends": remeasure_trends,
        "alternative_probe_result_count": int(alternative_probe_result_count),
        "alternative_probe_status_counts": dict(sorted(alternative_probe_status_counts.items())),
        "alternative_probe_trends": alternative_probe_trends,
        "stage_e_observed_acceptance_candidate_recovery_review_entries": stage_e_recovery_review_entries[:top_limit],
        "stage_e_observed_acceptance_candidate_recovery_review_count": int(
            len(stage_e_recovery_review_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_status_counts": dict(
            sorted(stage_e_recovery_review_status_counts.items())
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_latest_status": stage_e_recovery_review_latest_status,
        "stage_e_observed_acceptance_candidate_recovery_review_completed": bool(
            stage_e_recovery_review_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_in_progress": bool(
            stage_e_recovery_review_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_count": int(
            len(stage_e_recovery_review_followup_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_in_progress": bool(
            stage_e_recovery_review_followup_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_completed": bool(
            stage_e_recovery_review_followup_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_failed": bool(
            stage_e_recovery_review_followup_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_latest_status": (
            stage_e_recovery_review_followup_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_count": int(
            len(stage_e_recovery_review_followup_retry_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_in_progress": bool(
            stage_e_recovery_review_followup_retry_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_completed": bool(
            stage_e_recovery_review_followup_retry_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_failed": bool(
            stage_e_recovery_review_followup_retry_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_latest_status": (
            stage_e_recovery_review_followup_retry_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_count": int(
            len(stage_e_recovery_review_followup_retry_escalation_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_in_progress": bool(
            stage_e_recovery_review_followup_retry_escalation_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_completed": bool(
            stage_e_recovery_review_followup_retry_escalation_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_failed": bool(
            stage_e_recovery_review_followup_retry_escalation_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_latest_status": (
            stage_e_recovery_review_followup_retry_escalation_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_count": int(
            len(stage_e_recovery_review_evidence_collection_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_in_progress": bool(
            stage_e_recovery_review_evidence_collection_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_completed": bool(
            stage_e_recovery_review_evidence_collection_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_failed": bool(
            stage_e_recovery_review_evidence_collection_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_latest_status": (
            stage_e_recovery_review_evidence_collection_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_count": int(
            len(stage_e_recovery_review_evidence_recheck_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_in_progress": bool(
            stage_e_recovery_review_evidence_recheck_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_completed": bool(
            stage_e_recovery_review_evidence_recheck_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_failed": bool(
            stage_e_recovery_review_evidence_recheck_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_latest_status": (
            stage_e_recovery_review_evidence_recheck_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_count": int(
            len(stage_e_recovery_review_targeted_probe_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_in_progress": bool(
            stage_e_recovery_review_targeted_probe_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_completed": bool(
            stage_e_recovery_review_targeted_probe_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_failed": bool(
            stage_e_recovery_review_targeted_probe_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_latest_status": (
            stage_e_recovery_review_targeted_probe_latest_status
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_count": int(
            len(stage_e_recovery_review_targeted_probe_recheck_entries)
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_in_progress": bool(
            stage_e_recovery_review_targeted_probe_recheck_in_progress
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_completed": bool(
            stage_e_recovery_review_targeted_probe_recheck_completed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_failed": bool(
            stage_e_recovery_review_targeted_probe_recheck_failed
        ),
        "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_latest_status": (
            stage_e_recovery_review_targeted_probe_recheck_latest_status
        ),
        "completed_cause_boundary_documentation_ids": sorted(completed_cause_boundary_ids),
        "completed_targeted_fixture_repair_ids": sorted(completed_targeted_fixture_ids),
        "completed_research_planner_task_count": int(len(completed_research_planner_tasks)),
        "completed_research_planner_tasks": completed_research_planner_tasks[:top_limit],
        "completed_roadmap_patch_evidence_collection_count": int(len(completed_evidence_collection_tasks)),
        "completed_roadmap_patch_evidence_collection_keys": sorted(completed_evidence_collection_keys),
        "completed_roadmap_patch_evidence_collection_tasks": completed_evidence_collection_tasks[:top_limit],
        "roadmap_patch_review_approved_count": int(approved_patch_count),
        "roadmap_patch_review_rejected_count": int(rejected_patch_count),
        "roadmap_patch_rejected_item_count": rejected_item_count,
        "roadmap_patch_refreshed_item_count": refreshed_item_count,
        "roadmap_patch_refresh_to_rejection_ratio": float(refresh_to_rejection_ratio),
        "roadmap_patch_rejection_reasons": [
            {"reason": key, "count": int(value)}
            for key, value in sorted(
                roadmap_patch_rejection_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:top_limit]
        ],
        "roadmap_patch_rejected_items": [
            dict(item)
            for item in sorted(
                roadmap_patch_rejected_items_by_id.values(),
                key=lambda value: (
                    -int(value.get("count", 0) or 0),
                    str(value.get("id", "")),
                ),
            )[:top_limit]
        ],
        "roadmap_patch_refreshed_items": [
            dict(item)
            for item in sorted(
                roadmap_patch_refreshed_items_by_id.values(),
                key=lambda value: (
                    -int(value.get("count", 0) or 0),
                    str(value.get("id", "")),
                ),
            )[:top_limit]
        ],
        "experiment_status_summary": journal_experiment_status,
        "experiment_priority_plan": journal_experiment_priority_plan,
        "experiment_promotion_target_plan": journal_experiment_promotion_target_plan,
    }


def append_research_journal_entry(
    path: str,
    review_report: Dict[str, Any],
    *,
    dedupe_window_seconds: float = DEFAULT_JOURNAL_DEDUPE_WINDOW_SECONDS,
    max_entries: int = DEFAULT_JOURNAL_MAX_ENTRIES,
    max_age_seconds: float = DEFAULT_JOURNAL_MAX_AGE_SECONDS,
) -> Dict[str, Any]:
    entry = build_research_journal_entry(review_report)
    now_timestamp = _safe_float(entry.get("generated_at", time.time()), time.time())
    entries = load_research_journal_entries(path)
    dedupe_key = str(entry.get("dedupe_key", "") or "")
    duplicate_suppressed = False
    if dedupe_key and dedupe_window_seconds > 0:
        for previous in reversed(entries):
            if str(previous.get("dedupe_key", "") or "") != dedupe_key:
                continue
            previous_at = _safe_float(previous.get("generated_at", 0.0), 0.0)
            if previous_at > 0 and now_timestamp - previous_at <= float(dedupe_window_seconds):
                duplicate_suppressed = True
                previous["last_seen_at"] = now_timestamp
                previous["seen_count"] = int(previous.get("seen_count", 1) or 1) + 1
                previous["last_review_score"] = entry.get("review_score", 0.0)
                break
    if not duplicate_suppressed:
        entry["seen_count"] = 1
        entry["last_seen_at"] = now_timestamp
        entries.append(entry)
    entries, prune_summary = _prune_research_journal_entries(
        entries,
        now_timestamp=now_timestamp,
        max_entries=max_entries,
        max_age_seconds=max_age_seconds,
    )
    resolved = write_research_journal_entries(path, entries)
    return {
        "path": resolved,
        "appended": not duplicate_suppressed,
        "duplicate_suppressed": bool(duplicate_suppressed),
        "entry_count": int(len(entries)),
        "dedupe_key": dedupe_key,
        **prune_summary,
    }


def save_json_report(path: str, payload: Dict[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return resolved


def run_research_automation_benchmark(
    *,
    phase3_report_path: str = DEFAULT_PHASE3_REPORT_PATH,
    release_soak_report_path: str = DEFAULT_RELEASE_SOAK_REPORT_PATH,
    operational_report_path: str = DEFAULT_OPERATIONAL_REPORT_PATH,
    report_path: str = DEFAULT_RESEARCH_REVIEW_REPORT_PATH,
    roadmap_patch_suggestion_path: str = DEFAULT_ROADMAP_PATCH_SUGGESTION_PATH,
    journal_path: str = DEFAULT_RESEARCH_JOURNAL_PATH,
    append_journal: bool = False,
    generated_at: Optional[float] = None,
    require_operational_readiness: bool = True,
    journal_dedupe_window_seconds: float = DEFAULT_JOURNAL_DEDUPE_WINDOW_SECONDS,
    journal_max_entries: int = DEFAULT_JOURNAL_MAX_ENTRIES,
    journal_max_age_seconds: float = DEFAULT_JOURNAL_MAX_AGE_SECONDS,
) -> Dict[str, Any]:
    phase3_report, phase3_snapshot = _load_json_object_if_present(phase3_report_path)
    release_soak_report, release_snapshot = _load_json_object_if_present(release_soak_report_path)
    operational_report, operational_snapshot = _load_json_object_if_present(operational_report_path)
    journal_summary = summarize_research_journal_entries(
        load_research_journal_entries(journal_path),
        now_timestamp=float(generated_at if generated_at is not None else time.time()),
    )

    review_report = build_research_review_report(
        phase3_report=phase3_report,
        release_soak_report=release_soak_report,
        operational_report=operational_report,
        input_snapshots=[phase3_snapshot, release_snapshot, operational_snapshot],
        generated_at=generated_at,
        require_operational_readiness=bool(require_operational_readiness),
        research_journal_summary=journal_summary,
    )
    patch_suggestion = build_roadmap_patch_suggestion(review_report)
    review_report["artifacts"] = {
        "research_review_report_path": save_json_report(report_path, review_report),
        "roadmap_patch_suggestion_path": save_json_report(
            roadmap_patch_suggestion_path,
            patch_suggestion,
        ),
        "research_journal_path": os.path.abspath(journal_path),
        "research_journal_summary": journal_summary,
        "journal_appended": False,
        "journal_duplicate_suppressed": False,
        "journal_entry_count": 0,
    }
    if append_journal:
        journal_result = append_research_journal_entry(
            journal_path,
            review_report,
            dedupe_window_seconds=float(journal_dedupe_window_seconds),
            max_entries=int(journal_max_entries),
            max_age_seconds=float(journal_max_age_seconds),
        )
        review_report["artifacts"]["research_journal_path"] = journal_result["path"]
        review_report["artifacts"]["journal_appended"] = bool(journal_result.get("appended", False))
        review_report["artifacts"]["journal_duplicate_suppressed"] = bool(
            journal_result.get("duplicate_suppressed", False)
        )
        review_report["artifacts"]["journal_entry_count"] = int(journal_result.get("entry_count", 0) or 0)
        review_report["artifacts"]["journal_pruned_by_age"] = int(journal_result.get("pruned_by_age", 0) or 0)
        review_report["artifacts"]["journal_pruned_by_limit"] = int(journal_result.get("pruned_by_limit", 0) or 0)
        save_json_report(report_path, review_report)
    return review_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SARA research automation review report.")
    parser.add_argument("--phase3-report-path", default=DEFAULT_PHASE3_REPORT_PATH)
    parser.add_argument("--release-soak-report-path", default=DEFAULT_RELEASE_SOAK_REPORT_PATH)
    parser.add_argument("--operational-report-path", default=DEFAULT_OPERATIONAL_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_RESEARCH_REVIEW_REPORT_PATH)
    parser.add_argument("--roadmap-patch-suggestion-path", default=DEFAULT_ROADMAP_PATCH_SUGGESTION_PATH)
    parser.add_argument("--journal-path", default=DEFAULT_RESEARCH_JOURNAL_PATH)
    parser.add_argument("--append-journal", action="store_true")
    parser.add_argument("--journal-dedupe-window-seconds", type=float, default=DEFAULT_JOURNAL_DEDUPE_WINDOW_SECONDS)
    parser.add_argument("--journal-max-entries", type=int, default=DEFAULT_JOURNAL_MAX_ENTRIES)
    parser.add_argument("--journal-max-age-seconds", type=float, default=DEFAULT_JOURNAL_MAX_AGE_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_research_automation_benchmark(
        phase3_report_path=args.phase3_report_path,
        release_soak_report_path=args.release_soak_report_path,
        operational_report_path=args.operational_report_path,
        report_path=args.report_path,
        roadmap_patch_suggestion_path=args.roadmap_patch_suggestion_path,
        journal_path=args.journal_path,
        append_journal=bool(args.append_journal),
        journal_dedupe_window_seconds=float(args.journal_dedupe_window_seconds),
        journal_max_entries=int(args.journal_max_entries),
        journal_max_age_seconds=float(args.journal_max_age_seconds),
    )
    print("Research automation benchmark completed.")
    print(f"Review score: {report['review_score']:.3f}")
    print(f"Passed: {bool(report['passed'])}")
    print(f"Report: {report['artifacts']['research_review_report_path']}")
    print(f"Roadmap suggestion: {report['artifacts']['roadmap_patch_suggestion_path']}")


if __name__ == "__main__":
    main()
