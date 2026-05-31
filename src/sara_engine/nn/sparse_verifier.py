# Directory Path: src/sara_engine/nn/sparse_verifier.py
# English Title: Sparse Verifier
# Purpose/Content: Scores sparse candidate branches with local grounding, trace, energy, and uncertainty checks without dense model inference.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _tokens(text: str) -> set[str]:
    cleaned = str(text or "").lower()
    for char in ":;,.()[]{}|/_-":
        cleaned = cleaned.replace(char, " ")
    return {token for token in cleaned.split() if token}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class SparseVerifierThresholds:
    """Thresholds for local branch verification."""

    min_grounding: float = 0.25
    min_trace_integrity: float = 1.0
    min_energy_score: float = 0.50
    max_uncertainty: float = 0.55


class SparseVerifier:
    """Verifies sparse reasoning candidates using traceable local evidence."""

    def __init__(
        self,
        *,
        thresholds: SparseVerifierThresholds | None = None,
        max_energy_budget: float = 6.0,
    ) -> None:
        self.thresholds = thresholds or SparseVerifierThresholds()
        self.max_energy_budget = max(float(max_energy_budget), 1.0)

    def verify_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        evidence_texts: Sequence[str],
        competing_candidates: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        action = str(candidate.get("action", "") or "")
        projected_state = str(candidate.get("projected_state", "") or "")
        branch_id = str(candidate.get("branch_id", "") or "")
        relation_trace = candidate.get("relation_trace", {})
        if not isinstance(relation_trace, Mapping):
            relation_trace = {}
        causal_trace = candidate.get("causal_trace", {})
        if not isinstance(causal_trace, Mapping):
            causal_trace = {}
        budget = candidate.get("budget", {})
        if not isinstance(budget, Mapping):
            budget = {}

        candidate_tokens = _tokens(" ".join([action, projected_state]))
        evidence_tokens = set()
        for text in evidence_texts:
            evidence_tokens.update(_tokens(text))
        grounded_tokens = sorted(candidate_tokens.intersection(evidence_tokens))
        grounding_score = (
            float(len(grounded_tokens)) / float(max(len(candidate_tokens), 1))
            if candidate_tokens
            else 0.0
        )

        relation_branch = str(relation_trace.get("branch_id", "") or "")
        causal_branch = str(causal_trace.get("branch_id", "") or "")
        trace_integrity = 1.0 if (
            branch_id
            and relation_branch == branch_id
            and causal_branch == branch_id
            and bool(relation_trace.get("trace_complete", False))
            and bool(causal_trace.get("trace_complete", False))
        ) else 0.0

        budget_value = _safe_float(budget.get("budget", self.max_energy_budget), self.max_energy_budget)
        budget_bounded = bool(budget.get("bounded", budget_value <= self.max_energy_budget))
        energy_score = max(0.0, min(1.0, 1.0 - (budget_value / (self.max_energy_budget * 2.0))))
        if not budget_bounded:
            energy_score = min(energy_score, 0.25)

        candidate_score = _safe_float(candidate.get("score", 0.0), 0.0)
        competitor_scores = [
            _safe_float(item.get("score", 0.0), 0.0)
            for item in competing_candidates
            if isinstance(item, Mapping) and str(item.get("branch_id", "") or "") != branch_id
        ]
        best_competitor = max(competitor_scores) if competitor_scores else 0.0
        score_gap = max(candidate_score - best_competitor, 0.0)
        confidence_evidence = min(
            1.0,
            score_gap
            + grounding_score * 0.35
            + trace_integrity * 0.10
            + energy_score * 0.15,
        )
        uncertainty = max(0.0, min(1.0, 1.0 - confidence_evidence))

        passed = bool(
            grounding_score >= self.thresholds.min_grounding
            and trace_integrity >= self.thresholds.min_trace_integrity
            and energy_score >= self.thresholds.min_energy_score
            and uncertainty <= self.thresholds.max_uncertainty
        )
        verifier_score = (
            grounding_score
            + trace_integrity
            + energy_score
            + (1.0 - uncertainty)
        ) / 4.0
        return {
            "branch_id": branch_id,
            "action": action,
            "grounding_score": float(grounding_score),
            "trace_integrity": float(trace_integrity),
            "energy_score": float(energy_score),
            "uncertainty": float(uncertainty),
            "score_gap": float(score_gap),
            "verifier_score": float(verifier_score),
            "passed": passed,
            "grounded_tokens": grounded_tokens,
            "evidence_token_count": int(len(evidence_tokens)),
            "thresholds": {
                "min_grounding": float(self.thresholds.min_grounding),
                "min_trace_integrity": float(self.thresholds.min_trace_integrity),
                "min_energy_score": float(self.thresholds.min_energy_score),
                "max_uncertainty": float(self.thresholds.max_uncertainty),
            },
        }

    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        evidence_texts: Sequence[str],
    ) -> Dict[str, Any]:
        results = [
            self.verify_candidate(
                candidate,
                evidence_texts=evidence_texts,
                competing_candidates=candidates,
            )
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        ranked = sorted(
            results,
            key=lambda item: (
                bool(item.get("passed", False)),
                float(item.get("verifier_score", 0.0)),
                -float(item.get("uncertainty", 1.0)),
                str(item.get("branch_id", "")),
            ),
            reverse=True,
        )
        selected = ranked[0] if ranked else {}
        return {
            "schema": "sara-sparse-verifier-v1",
            "candidate_count": int(len(results)),
            "selected_branch": str(selected.get("branch_id", "") or ""),
            "selected_passed": bool(selected.get("passed", False)),
            "ranked_candidates": ranked,
            "observed_only": True,
        }


def evaluate_sparse_verifier_trace(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates verifier behavior as an observed-only trace for benchmarks."""

    verifier = SparseVerifier(max_energy_budget=max_energy_budget)
    ranked = verifier.rank_candidates(candidates, evidence_texts=evidence_texts)
    selected = (
        ranked.get("ranked_candidates", [])[0]
        if isinstance(ranked.get("ranked_candidates", []), list) and ranked.get("ranked_candidates")
        else {}
    )
    grounding_ok = float(selected.get("grounding_score", 0.0) or 0.0) >= verifier.thresholds.min_grounding
    trace_ok = float(selected.get("trace_integrity", 0.0) or 0.0) >= verifier.thresholds.min_trace_integrity
    energy_ok = float(selected.get("energy_score", 0.0) or 0.0) >= verifier.thresholds.min_energy_score
    uncertainty_ok = float(selected.get("uncertainty", 1.0) or 1.0) <= verifier.thresholds.max_uncertainty
    selection_ok = (
        bool(expected_branch_id)
        and str(ranked.get("selected_branch", "") or "") == str(expected_branch_id)
        and bool(ranked.get("selected_passed", False))
    )
    return {
        **ranked,
        "expected_branch_id": str(expected_branch_id),
        "metrics": {
            "sparse_verifier_grounding_observed": 1.0 if grounding_ok else 0.0,
            "sparse_verifier_trace_integrity_observed": 1.0 if trace_ok else 0.0,
            "sparse_verifier_energy_budget_observed": 1.0 if energy_ok else 0.0,
            "sparse_verifier_uncertainty_observed": 1.0 if uncertainty_ok else 0.0,
            "sparse_verifier_selection_observed": 1.0 if selection_ok else 0.0,
        },
    }


def evaluate_sparse_best_of_n_trace(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    summary_text: str = "",
    max_n: int = 3,
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates a bounded Best-of-N path using the sparse verifier."""

    bounded_candidates = [
        dict(candidate)
        for candidate in candidates[: max(1, int(max_n))]
        if isinstance(candidate, Mapping)
    ]
    verifier_trace = evaluate_sparse_verifier_trace(
        bounded_candidates,
        evidence_texts=evidence_texts,
        expected_branch_id=expected_branch_id,
        max_energy_budget=max_energy_budget,
    )
    selected_branch = str(verifier_trace.get("selected_branch", "") or "")
    selected_action = ""
    for candidate in bounded_candidates:
        if str(candidate.get("branch_id", "") or "") == selected_branch:
            selected_action = str(candidate.get("action", "") or "")
            break
    summary_tokens = _tokens(summary_text)
    action_tokens = _tokens(selected_action)
    summary_matches_selection = bool(
        selected_branch
        and selected_action
        and action_tokens
        and len(action_tokens.intersection(summary_tokens)) / max(len(action_tokens), 1) >= 0.5
    )
    branch_diversity = len(
        {
            str(candidate.get("branch_id", "") or "")
            for candidate in bounded_candidates
            if str(candidate.get("branch_id", "") or "")
        }
    )
    bounded_ok = bool(1 <= len(bounded_candidates) <= max(1, int(max_n)))
    diversity_ok = bool(branch_diversity >= min(len(bounded_candidates), 3))
    selection_ok = bool(
        verifier_trace.get("selected_passed", False)
        and (
            not expected_branch_id
            or selected_branch == str(expected_branch_id)
        )
    )
    return {
        "schema": "sara-sparse-best-of-n-v1",
        "candidate_count": int(len(bounded_candidates)),
        "max_n": int(max(1, int(max_n))),
        "selected_branch": selected_branch,
        "selected_action": selected_action,
        "summary_matches_selection": summary_matches_selection,
        "branch_diversity": int(branch_diversity),
        "verifier_trace": verifier_trace,
        "observed_only": True,
        "metrics": {
            "sparse_best_of_n_bounded_count_observed": 1.0 if bounded_ok else 0.0,
            "sparse_best_of_n_branch_diversity_observed": 1.0 if diversity_ok else 0.0,
            "sparse_best_of_n_verifier_selection_observed": 1.0 if selection_ok else 0.0,
            "sparse_best_of_n_summary_alignment_observed": 1.0 if summary_matches_selection else 0.0,
        },
    }


def evaluate_self_correction_trace(
    initial_candidate: Mapping[str, Any],
    correction_candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    max_loops: int = 2,
    min_improvement: float = 0.05,
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates a bounded draft-verify-repair path without unbounded recursion."""

    loop_budget = max(0, int(max_loops))
    verifier = SparseVerifier(max_energy_budget=max_energy_budget)
    candidates = [
        dict(candidate)
        for candidate in [initial_candidate, *correction_candidates[:loop_budget]]
        if isinstance(candidate, Mapping)
    ]
    if not candidates:
        return {
            "schema": "sara-self-correction-trace-v1",
            "observed_only": True,
            "max_loops": int(loop_budget),
            "loop_count": 0,
            "initial_branch": "",
            "selected_branch": "",
            "rollback_reason": "empty_candidates",
            "verifier_failure_reason": "empty_candidates",
            "improvement": 0.0,
            "correction_applied": False,
            "loops": [],
            "metrics": {
                "self_correction_bounded_loop_observed": 0.0,
                "self_correction_improvement_observed": 0.0,
                "self_correction_rollback_reason_observed": 1.0,
                "self_correction_verifier_failure_observed": 1.0,
            },
        }

    initial_result = verifier.verify_candidate(
        candidates[0],
        evidence_texts=evidence_texts,
        competing_candidates=candidates,
    )
    current_result = dict(initial_result)
    loops: List[Dict[str, Any]] = []
    rollback_reason = ""
    verifier_failure_reason = "" if bool(initial_result.get("passed", False)) else "initial_verifier_failed"

    for index, candidate in enumerate(candidates[1 : loop_budget + 1], start=1):
        candidate_result = verifier.verify_candidate(
            candidate,
            evidence_texts=evidence_texts,
            competing_candidates=candidates,
        )
        improvement = float(candidate_result.get("verifier_score", 0.0)) - float(
            current_result.get("verifier_score", 0.0)
        )
        accepted = bool(candidate_result.get("passed", False) and improvement >= float(min_improvement))
        reason = "accepted" if accepted else "insufficient_improvement"
        if not bool(candidate_result.get("passed", False)):
            reason = "verifier_failed"
            verifier_failure_reason = reason
        if accepted:
            current_result = dict(candidate_result)
            verifier_failure_reason = ""
        else:
            rollback_reason = reason
        loops.append(
            {
                "loop_index": int(index),
                "branch_id": str(candidate_result.get("branch_id", "") or ""),
                "verifier_score": float(candidate_result.get("verifier_score", 0.0)),
                "uncertainty": float(candidate_result.get("uncertainty", 1.0)),
                "improvement": float(improvement),
                "accepted": accepted,
                "rollback_reason": "" if accepted else reason,
            }
        )

    total_improvement = max(
        0.0,
        float(current_result.get("verifier_score", 0.0)) - float(initial_result.get("verifier_score", 0.0)),
    )
    selected_branch = str(current_result.get("branch_id", "") or "")
    selection_ok = (
        not expected_branch_id
        or selected_branch == str(expected_branch_id)
    )
    correction_applied = selected_branch != str(initial_result.get("branch_id", "") or "")
    bounded_ok = len(loops) <= loop_budget
    improvement_ok = bool(
        correction_applied
        and total_improvement >= float(min_improvement)
        and bool(current_result.get("passed", False))
        and selection_ok
    )
    rollback_reason_ok = bool(rollback_reason or all(loop.get("accepted", False) for loop in loops) or not loops)
    verifier_failure_ok = bool(not verifier_failure_reason or rollback_reason == "verifier_failed")

    return {
        "schema": "sara-self-correction-trace-v1",
        "observed_only": True,
        "max_loops": int(loop_budget),
        "loop_count": int(len(loops)),
        "initial_branch": str(initial_result.get("branch_id", "") or ""),
        "selected_branch": selected_branch,
        "expected_branch_id": str(expected_branch_id),
        "initial_score": float(initial_result.get("verifier_score", 0.0)),
        "selected_score": float(current_result.get("verifier_score", 0.0)),
        "improvement": float(total_improvement),
        "correction_applied": correction_applied,
        "rollback_reason": rollback_reason,
        "verifier_failure_reason": verifier_failure_reason,
        "loops": loops,
        "selected_verification": current_result,
        "initial_verification": initial_result,
        "metrics": {
            "self_correction_bounded_loop_observed": 1.0 if bounded_ok else 0.0,
            "self_correction_improvement_observed": 1.0 if improvement_ok else 0.0,
            "self_correction_rollback_reason_observed": 1.0 if rollback_reason_ok else 0.0,
            "self_correction_verifier_failure_observed": 1.0 if verifier_failure_ok else 0.0,
        },
    }


def evaluate_bounded_tree_search_trace(
    tree_candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    max_depth: int = 2,
    max_branch_factor: int = 2,
    max_event_budget: int = 6,
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates a small bounded search tree with sparse verifier scoring."""

    depth_limit = max(0, int(max_depth))
    branch_limit = max(1, int(max_branch_factor))
    event_budget_limit = max(1, int(max_event_budget))
    verifier = SparseVerifier(max_energy_budget=max_energy_budget)
    accepted: List[Dict[str, Any]] = []
    children_by_parent: Dict[str, int] = {}
    dropped: List[Dict[str, Any]] = []

    for candidate in tree_candidates:
        if not isinstance(candidate, Mapping):
            continue
        branch_id = str(candidate.get("branch_id", "") or "")
        parent_id = str(candidate.get("parent_branch_id", "") or "root")
        depth = int(_safe_float(candidate.get("depth", 0), 0.0))
        event_cost = int(_safe_float(candidate.get("event_cost", 1), 1.0))
        reason = ""
        if depth > depth_limit:
            reason = "depth_limit"
        elif children_by_parent.get(parent_id, 0) >= branch_limit:
            reason = "branch_factor_limit"
        elif sum(int(item.get("event_cost", 1)) for item in accepted) + event_cost > event_budget_limit:
            reason = "event_budget_limit"
        if reason:
            dropped.append(
                {
                    "branch_id": branch_id,
                    "parent_branch_id": parent_id,
                    "depth": int(depth),
                    "event_cost": int(event_cost),
                    "drop_reason": reason,
                }
            )
            continue
        copied = dict(candidate)
        copied["parent_branch_id"] = parent_id
        copied["depth"] = int(depth)
        copied["event_cost"] = int(event_cost)
        accepted.append(copied)
        children_by_parent[parent_id] = children_by_parent.get(parent_id, 0) + 1

    verifier_trace = evaluate_sparse_verifier_trace(
        accepted,
        evidence_texts=evidence_texts,
        expected_branch_id=expected_branch_id,
        max_energy_budget=max_energy_budget,
    )
    selected_branch = str(verifier_trace.get("selected_branch", "") or "")
    max_observed_depth = max((int(item.get("depth", 0)) for item in accepted), default=0)
    max_observed_branch_factor = max(children_by_parent.values(), default=0)
    event_budget_used = sum(int(item.get("event_cost", 1)) for item in accepted)
    depth_ok = max_observed_depth <= depth_limit
    branch_factor_ok = max_observed_branch_factor <= branch_limit
    event_budget_ok = event_budget_used <= event_budget_limit
    selection_ok = bool(
        verifier_trace.get("selected_passed", False)
        and (
            not expected_branch_id
            or selected_branch == str(expected_branch_id)
        )
    )
    return {
        "schema": "sara-bounded-tree-search-trace-v1",
        "observed_only": True,
        "max_depth": int(depth_limit),
        "max_branch_factor": int(branch_limit),
        "max_event_budget": int(event_budget_limit),
        "event_budget_used": int(event_budget_used),
        "candidate_count": int(len(accepted)),
        "dropped_count": int(len(dropped)),
        "selected_branch": selected_branch,
        "accepted_candidates": accepted,
        "dropped_candidates": dropped,
        "verifier_trace": verifier_trace,
        "metrics": {
            "bounded_tree_search_depth_observed": 1.0 if depth_ok else 0.0,
            "bounded_tree_search_branch_factor_observed": 1.0 if branch_factor_ok else 0.0,
            "bounded_tree_search_event_budget_observed": 1.0 if event_budget_ok else 0.0,
            "bounded_tree_search_verifier_selection_observed": 1.0 if selection_ok else 0.0,
        },
    }


def evaluate_reasoning_forest_lane_trace(
    lanes: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    max_lanes: int = 3,
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates a small read-only reasoning forest without state mutation."""

    lane_limit = max(1, int(max_lanes))
    bounded_lanes = [
        dict(lane)
        for lane in lanes[:lane_limit]
        if isinstance(lane, Mapping)
    ]
    verifier_trace = evaluate_sparse_verifier_trace(
        bounded_lanes,
        evidence_texts=evidence_texts,
        expected_branch_id=expected_branch_id,
        max_energy_budget=max_energy_budget,
    )
    selected_branch = str(verifier_trace.get("selected_branch", "") or "")
    lane_summaries: List[Dict[str, Any]] = []
    for lane in bounded_lanes:
        branch_id = str(lane.get("branch_id", "") or "")
        snapshot = lane.get("snapshot", {})
        if not isinstance(snapshot, Mapping):
            snapshot = {}
        selection_reason = str(lane.get("selection_reason", "") or "")
        lane_summaries.append(
            {
                "lane_id": str(lane.get("lane_id", branch_id) or branch_id),
                "branch_id": branch_id,
                "snapshot_read_only": bool(snapshot.get("read_only", False)),
                "snapshot_mutation_count": int(_safe_float(snapshot.get("mutation_count", 0), 0.0)),
                "selection_reason": selection_reason,
                "selection_reason_matches": bool(
                    selected_branch
                    and branch_id == selected_branch
                    and selected_branch in _tokens(selection_reason)
                ),
            }
        )
    read_only_ok = bool(
        lane_summaries
        and all(
            bool(item.get("snapshot_read_only", False))
            and int(item.get("snapshot_mutation_count", 1)) == 0
            for item in lane_summaries
        )
    )
    bounded_ok = 1 <= len(bounded_lanes) <= lane_limit
    diversity = len({str(lane.get("branch_id", "") or "") for lane in bounded_lanes})
    diversity_ok = diversity >= min(len(bounded_lanes), 2)
    verifier_selection_ok = bool(
        verifier_trace.get("selected_passed", False)
        and (
            not expected_branch_id
            or selected_branch == str(expected_branch_id)
        )
    )
    selection_reason_ok = any(
        bool(item.get("selection_reason_matches", False))
        for item in lane_summaries
    )
    return {
        "schema": "sara-reasoning-forest-lane-trace-v1",
        "observed_only": True,
        "max_lanes": int(lane_limit),
        "lane_count": int(len(bounded_lanes)),
        "lane_diversity": int(diversity),
        "selected_branch": selected_branch,
        "lane_summaries": lane_summaries,
        "verifier_trace": verifier_trace,
        "metrics": {
            "reasoning_forest_lane_bounded_count_observed": 1.0 if bounded_ok else 0.0,
            "reasoning_forest_lane_read_only_snapshot_observed": 1.0 if read_only_ok else 0.0,
            "reasoning_forest_lane_diversity_observed": 1.0 if diversity_ok else 0.0,
            "reasoning_forest_lane_verifier_selection_observed": 1.0 if verifier_selection_ok else 0.0,
            "reasoning_forest_lane_selection_reason_observed": 1.0 if selection_reason_ok else 0.0,
        },
    }


def evaluate_hierarchical_reasoning_trace(
    instruction_event: Mapping[str, Any],
    execution_candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_texts: Sequence[str],
    expected_branch_id: str = "",
    max_execution_steps: int = 3,
    max_energy_budget: float = 6.0,
) -> Dict[str, Any]:
    """Evaluates instruction, execution, and verification as sparse trace layers."""

    instruction = dict(instruction_event) if isinstance(instruction_event, Mapping) else {}
    step_limit = max(1, int(max_execution_steps))
    execution_trace = [
        dict(candidate)
        for candidate in execution_candidates[:step_limit]
        if isinstance(candidate, Mapping)
    ]
    verifier_trace = evaluate_sparse_verifier_trace(
        execution_trace,
        evidence_texts=evidence_texts,
        expected_branch_id=expected_branch_id,
        max_energy_budget=max_energy_budget,
    )
    selected_branch = str(verifier_trace.get("selected_branch", "") or "")
    target_branch = str(instruction.get("target_branch_id", "") or expected_branch_id)
    instruction_tokens = _tokens(str(instruction.get("instruction", "") or ""))
    selected_action = ""
    for candidate in execution_trace:
        if str(candidate.get("branch_id", "") or "") == selected_branch:
            selected_action = str(candidate.get("action", "") or "")
            break
    action_tokens = _tokens(selected_action)
    instruction_integrity = bool(
        instruction.get("event_type", "") == "instruction_event"
        and str(instruction.get("instruction_id", "") or "")
        and target_branch
    )
    execution_integrity = bool(
        execution_trace
        and len(execution_trace) <= step_limit
        and all(str(candidate.get("branch_id", "") or "") for candidate in execution_trace)
    )
    verification_integrity = bool(verifier_trace.get("selected_passed", False))
    plan_execution_alignment = bool(
        selected_branch
        and target_branch
        and selected_branch == target_branch
        and (
            not instruction_tokens
            or not action_tokens
            or bool(instruction_tokens.intersection(action_tokens))
        )
    )
    return {
        "schema": "sara-hierarchical-reasoning-trace-v1",
        "observed_only": True,
        "instruction_event": instruction,
        "execution_trace": execution_trace,
        "verification_trace": verifier_trace,
        "max_execution_steps": int(step_limit),
        "selected_branch": selected_branch,
        "target_branch_id": target_branch,
        "plan_execution_alignment": plan_execution_alignment,
        "metrics": {
            "hierarchical_reasoning_instruction_observed": 1.0 if instruction_integrity else 0.0,
            "hierarchical_reasoning_execution_trace_observed": 1.0 if execution_integrity else 0.0,
            "hierarchical_reasoning_verification_trace_observed": 1.0 if verification_integrity else 0.0,
            "hierarchical_reasoning_plan_alignment_observed": 1.0 if plan_execution_alignment else 0.0,
        },
    }
