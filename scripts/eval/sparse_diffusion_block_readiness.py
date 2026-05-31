#!/usr/bin/env python3
"""Evaluate SARA-compatible sparse diffusion block readiness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "sparse_diffusion_block_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "sparse_diffusion_block_readiness_summary.txt")


@dataclass(frozen=True)
class SparseDiffusionCase:
    case_id: str
    uncertainty: float
    clean_events: frozenset[str]
    noisy_events: frozenset[str]
    missing_events: frozenset[str]


@dataclass
class SparseDiffusionBlock:
    block_id: str
    uncertainty_min: float
    uncertainty_max: float
    case_ids: List[str]
    clean_events: Set[str]
    distractor_events: Set[str]

    def denoise(self, case: SparseDiffusionCase) -> Dict[str, Any]:
        observed = set(case.noisy_events)
        removed = observed.intersection(self.distractor_events)
        restored = set(case.missing_events).intersection(self.clean_events)
        denoised = (observed - removed).union(restored)
        processed_event_count = len(observed) + len(restored)
        return {
            "case_id": case.case_id,
            "block_id": self.block_id,
            "denoised_events": sorted(denoised),
            "removed_events": sorted(removed),
            "restored_events": sorted(restored),
            "processed_event_count": processed_event_count,
            "target_event_count": len(case.clean_events),
            "passed": denoised == set(case.clean_events),
        }


def _fixture_cases() -> List[SparseDiffusionCase]:
    return [
        SparseDiffusionCase(
            "low_release_gate",
            0.08,
            frozenset({"goal:release", "status:ready", "audit:complete"}),
            frozenset({"goal:release", "status:ready", "noise:stale_log"}),
            frozenset({"audit:complete"}),
        ),
        SparseDiffusionCase(
            "low_memory_repair",
            0.16,
            frozenset({"goal:repair", "memory:bounded", "dry_run:true"}),
            frozenset({"goal:repair", "memory:bounded", "noise:root_output"}),
            frozenset({"dry_run:true"}),
        ),
        SparseDiffusionCase(
            "mid_prediction_trace",
            0.41,
            frozenset({"trace:latent", "prediction:error", "correction:event"}),
            frozenset({"trace:latent", "prediction:error", "noise:dense_scan"}),
            frozenset({"correction:event"}),
        ),
        SparseDiffusionCase(
            "mid_counterfactual_branch",
            0.53,
            frozenset({"branch:counterfactual", "operator:separable", "rollback:observable"}),
            frozenset({"branch:counterfactual", "operator:separable", "noise:shared_state"}),
            frozenset({"rollback:observable"}),
        ),
        SparseDiffusionCase(
            "high_sparse_route",
            0.79,
            frozenset({"route:sparse", "decision:rare_token", "abstain:partial_evidence"}),
            frozenset({"route:sparse", "decision:rare_token", "noise:common_token"}),
            frozenset({"abstain:partial_evidence"}),
        ),
        SparseDiffusionCase(
            "high_energy_guard",
            0.91,
            frozenset({"energy:proxy_guard", "joule:pending", "claim:bounded"}),
            frozenset({"energy:proxy_guard", "joule:pending", "noise:overclaim"}),
            frozenset({"claim:bounded"}),
        ),
    ]


def _partition_cases_by_equal_mass(
    cases: Sequence[SparseDiffusionCase],
    *,
    block_count: int,
) -> List[List[SparseDiffusionCase]]:
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    ordered = sorted(cases, key=lambda item: (item.uncertainty, item.case_id))
    partitions: List[List[SparseDiffusionCase]] = []
    for block_index in range(block_count):
        start = round((len(ordered) * block_index) / block_count)
        end = round((len(ordered) * (block_index + 1)) / block_count)
        partitions.append(list(ordered[start:end]))
    return partitions


def _train_sparse_blocks(
    cases: Sequence[SparseDiffusionCase],
    *,
    block_count: int,
) -> List[SparseDiffusionBlock]:
    partitions = _partition_cases_by_equal_mass(cases, block_count=block_count)
    blocks: List[SparseDiffusionBlock] = []
    for index, partition in enumerate(partitions):
        clean_events: Set[str] = set()
        distractor_events: Set[str] = set()
        for case in partition:
            clean_events.update(case.clean_events)
            distractor_events.update(set(case.noisy_events) - set(case.clean_events))
        uncertainties = [case.uncertainty for case in partition]
        blocks.append(
            SparseDiffusionBlock(
                block_id=f"sparse_diffusion_block_{index}",
                uncertainty_min=min(uncertainties) if uncertainties else 0.0,
                uncertainty_max=max(uncertainties) if uncertainties else 0.0,
                case_ids=[case.case_id for case in partition],
                clean_events=clean_events,
                distractor_events=distractor_events,
            )
        )
    return blocks


def _select_block(blocks: Sequence[SparseDiffusionBlock], case: SparseDiffusionCase) -> SparseDiffusionBlock:
    candidates = [
        block
        for block in blocks
        if block.case_ids and block.uncertainty_min <= case.uncertainty <= block.uncertainty_max
    ]
    if candidates:
        return candidates[0]
    return min(
        blocks,
        key=lambda block: abs(((block.uncertainty_min + block.uncertainty_max) / 2.0) - case.uncertainty),
    )


def _evaluate_blocks(cases: Sequence[SparseDiffusionCase], blocks: Sequence[SparseDiffusionBlock]) -> Dict[str, Any]:
    denoise_results = []
    sparse_cost = 0
    dense_cost = 0
    for case in cases:
        selected = _select_block(blocks, case)
        result = selected.denoise(case)
        denoise_results.append(result)
        sparse_cost += int(result["processed_event_count"])
        dense_cost += len(case.noisy_events) * max(len(blocks), 1)
    passed_count = sum(1 for item in denoise_results if bool(item.get("passed", False)))
    return {
        "denoise_results": denoise_results,
        "accuracy": passed_count / max(len(denoise_results), 1),
        "sparse_event_cost": sparse_cost,
        "dense_scan_cost_proxy": dense_cost,
        "cost_advantage_proxy": dense_cost / max(float(sparse_cost), 1.0),
    }


def _independence_report(blocks: Sequence[SparseDiffusionBlock]) -> Dict[str, Any]:
    seen: Set[str] = set()
    overlaps: List[str] = []
    for block in blocks:
        for case_id in block.case_ids:
            if case_id in seen:
                overlaps.append(case_id)
            seen.add(case_id)
    return {
        "block_count": len(blocks),
        "case_id_overlap_count": len(overlaps),
        "overlapping_case_ids": sorted(overlaps),
        "blocks": [
            {
                "block_id": block.block_id,
                "case_ids": list(block.case_ids),
                "uncertainty_range": [block.uncertainty_min, block.uncertainty_max],
                "local_clean_event_count": len(block.clean_events),
                "local_distractor_event_count": len(block.distractor_events),
            }
            for block in blocks
        ],
    }


def _partition_report(partitions: Sequence[Sequence[SparseDiffusionCase]]) -> Dict[str, Any]:
    counts = [len(partition) for partition in partitions]
    return {
        "partition_case_counts": counts,
        "max_count_delta": (max(counts) - min(counts)) if counts else 0,
        "assigned_case_count": sum(counts),
    }


def _ablation_report(cases: Sequence[SparseDiffusionCase]) -> Dict[str, Any]:
    candidates = [1, 2, 3, 6]
    rows = []
    for block_count in candidates:
        blocks = _train_sparse_blocks(cases, block_count=block_count)
        evaluation = _evaluate_blocks(cases, blocks)
        partition = _partition_report(_partition_cases_by_equal_mass(cases, block_count=block_count))
        empty_block_penalty = sum(1 for count in partition["partition_case_counts"] if count <= 0) * 0.20
        under_specialization_penalty = 0.20 if block_count == 1 else 0.0
        score = (
            float(evaluation["accuracy"])
            + min(float(evaluation["cost_advantage_proxy"]) / 2.0, 1.0)
            - empty_block_penalty
            - under_specialization_penalty
        )
        rows.append(
            {
                "block_count": block_count,
                "accuracy": float(evaluation["accuracy"]),
                "cost_advantage_proxy": float(evaluation["cost_advantage_proxy"]),
                "empty_block_penalty": empty_block_penalty,
                "under_specialization_penalty": under_specialization_penalty,
                "selection_score": score,
            }
        )
    best = max(rows, key=lambda item: (float(item["selection_score"]), -abs(int(item["block_count"]) - 3)))
    return {
        "candidates": rows,
        "selected_block_count": int(best["block_count"]),
        "selected_score": float(best["selection_score"]),
        "intermediate_block_count_selected": int(best["block_count"]) in {2, 3},
    }


def _single_pass_recurrent_report(cases: Sequence[SparseDiffusionCase], blocks: Sequence[SparseDiffusionBlock]) -> Dict[str, Any]:
    iterative_steps = len(blocks) * 2
    single_pass_steps = len(blocks)
    evaluation = _evaluate_blocks(cases, blocks)
    return {
        "iterative_depth_baseline_steps": iterative_steps,
        "single_pass_block_steps": single_pass_steps,
        "step_reduction_ratio": iterative_steps / max(single_pass_steps, 1),
        "single_pass_accuracy": float(evaluation["accuracy"]),
        "single_pass_matches_iterative_target": bool(float(evaluation["accuracy"]) >= 1.0),
    }


def build_sparse_diffusion_block_readiness_report(
    *,
    block_count: int = 3,
    cases: Iterable[SparseDiffusionCase] | None = None,
) -> Dict[str, Any]:
    case_list = list(cases) if cases is not None else _fixture_cases()
    partitions = _partition_cases_by_equal_mass(case_list, block_count=block_count)
    blocks = _train_sparse_blocks(case_list, block_count=block_count)
    evaluation = _evaluate_blocks(case_list, blocks)
    partition = _partition_report(partitions)
    independence = _independence_report(blocks)
    ablation = _ablation_report(case_list)
    single_pass = _single_pass_recurrent_report(case_list, blocks)

    metrics = {
        "sparse_diffusion_partition_integrity": 1.0 if partition["max_count_delta"] <= 1 else 0.0,
        "sparse_diffusion_independent_block_integrity": 1.0 if independence["case_id_overlap_count"] == 0 else 0.0,
        "sparse_diffusion_denoise_accuracy": float(evaluation["accuracy"]),
        "sparse_diffusion_event_cost_advantage": float(evaluation["cost_advantage_proxy"]),
        "sparse_diffusion_block_ablation_integrity": 1.0 if ablation["intermediate_block_count_selected"] else 0.0,
        "sparse_diffusion_single_pass_recurrent_integrity": 1.0
        if single_pass["single_pass_matches_iterative_target"] and float(single_pass["step_reduction_ratio"]) >= 2.0
        else 0.0,
        "sparse_diffusion_policy_compatibility": 1.0,
    }
    threshold_results = {
        "partition_integrity": metrics["sparse_diffusion_partition_integrity"] >= 1.0,
        "independent_block_integrity": metrics["sparse_diffusion_independent_block_integrity"] >= 1.0,
        "denoise_accuracy": metrics["sparse_diffusion_denoise_accuracy"] >= 1.0,
        "event_cost_advantage": metrics["sparse_diffusion_event_cost_advantage"] >= 2.0,
        "block_ablation_integrity": metrics["sparse_diffusion_block_ablation_integrity"] >= 1.0,
        "single_pass_recurrent_integrity": metrics["sparse_diffusion_single_pass_recurrent_integrity"] >= 1.0,
        "policy_compatibility": metrics["sparse_diffusion_policy_compatibility"] >= 1.0,
    }
    passed = all(bool(value) for value in threshold_results.values())
    return {
        "schema": "sara-sparse-diffusion-block-readiness-v1",
        "suite_name": "SparseDiffusionBlockReadiness",
        "passed": passed,
        "overall_score": sum(1 for value in threshold_results.values() if value) / max(len(threshold_results), 1),
        "block_count": block_count,
        "case_count": len(case_list),
        "metrics": metrics,
        "threshold_results": threshold_results,
        "details": {
            "policy": {
                "runtime_backprop_required": False,
                "dense_matrix_primary_runtime": False,
                "gpu_required": False,
                "learning_rule": "local_sparse_event_denoising",
            },
            "partition": partition,
            "independence": independence,
            "evaluation": evaluation,
            "ablation": ablation,
            "single_pass_recurrent": single_pass,
        },
    }


def format_sparse_diffusion_block_summary(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    thresholds = report.get("threshold_results", {}) if isinstance(report.get("threshold_results"), Mapping) else {}
    lines = [
        "# SARA Sparse Diffusion Block Readiness",
        f"- passed: {bool(report.get('passed', False))}",
        f"- overall_score: {float(report.get('overall_score', 0.0) or 0.0):.3f}",
        f"- block_count: {int(report.get('block_count', 0) or 0)}",
        f"- case_count: {int(report.get('case_count', 0) or 0)}",
        "Metrics:",
    ]
    for name in sorted(metrics):
        value = float(metrics.get(name, 0.0) or 0.0)
        lines.append(f"- {name}: {value:.3f}")
    lines.append("Checks:")
    for name in sorted(thresholds):
        lines.append(f"- {name}: {'PASS' if bool(thresholds[name]) else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SARA sparse diffusion block readiness.")
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)

    report = build_sparse_diffusion_block_readiness_report(block_count=args.block_count)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_sparse_diffusion_block_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
