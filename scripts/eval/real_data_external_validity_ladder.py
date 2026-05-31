# Directory Path: scripts/eval/real_data_external_validity_ladder.py
# English Title: Real-Data External Validity Scale Ladder
# Purpose/Content: Runs small/medium/large external-validity profiles and aggregates sparse-vs-ANN energy-quality evidence for Phase 5.5 scaling.

import argparse
import importlib.util
import json
import os
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PROFILE_SPECS = [
    {"name": "small", "max_docs": 256, "max_cases": 24},
    {"name": "medium", "max_docs": 1024, "max_cases": 64},
    {"name": "large", "max_docs": 4096, "max_cases": 128},
]


def _load_external_validity_module() -> Any:
    module_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "real_data_external_validity.py")
    spec = importlib.util.spec_from_file_location("real_data_external_validity_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load external validity benchmark: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


external_validity = _load_external_validity_module()
ensure_parent_directory = external_validity.ensure_parent_directory
processed_data_path = external_validity.processed_data_path
workspace_path = external_validity.workspace_path

DEFAULT_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity_ladder.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "real_data_external_validity_ladder_summary.txt")


def _profile_artifact_path(profile_name: str, suffix: str) -> str:
    safe_name = "".join(char for char in profile_name.lower() if char.isalnum() or char in ("_", "-"))
    if not safe_name:
        safe_name = "profile"
    return workspace_path("evaluation", f"real_data_external_validity_{safe_name}{suffix}")


def parse_profile_specs(raw_specs: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    if not raw_specs:
        return [dict(item) for item in DEFAULT_PROFILE_SPECS]
    profiles: List[Dict[str, Any]] = []
    for raw_spec in raw_specs:
        parts = [part.strip() for part in str(raw_spec).split(":")]
        if len(parts) != 3 or not parts[0]:
            raise ValueError(f"Invalid profile spec '{raw_spec}'. Expected name:max_docs:max_cases.")
        profiles.append(
            {
                "name": parts[0],
                "max_docs": max(int(parts[1]), 1),
                "max_cases": max(int(parts[2]), 1),
            }
        )
    return profiles


def _numeric_metric(report: Dict[str, Any], metric_name: str) -> float:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    value = metrics.get(metric_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_ladder_metrics(profile_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    profile_count = len(profile_reports)
    passed_profile_count = sum(1 for report in profile_reports if bool(report.get("passed", False)))
    if not profile_reports:
        return {
            "profile_count": 0,
            "passed_profile_count": 0,
            "min_real_data_qa_accuracy": 0.0,
            "min_ann_cost_advantage_proxy": 0.0,
            "min_performance_energy_ratio_proxy": 0.0,
            "min_negative_control_abstention_integrity": 0.0,
            "min_negative_control_cost_advantage_proxy": 0.0,
            "min_partial_evidence_abstention_integrity": 0.0,
            "min_partial_evidence_cost_advantage_proxy": 0.0,
            "min_contrastive_control_accuracy": 0.0,
            "min_contrastive_control_cost_advantage_proxy": 0.0,
            "min_dense_embedding_ann_cost_advantage_proxy": 0.0,
            "min_sparse_diffusion_real_data_denoise_accuracy": 0.0,
            "min_sparse_diffusion_real_data_event_cost_advantage": 0.0,
            "min_sparse_diffusion_real_data_partition_integrity": 0.0,
            "min_sparse_diffusion_real_data_single_pass_integrity": 0.0,
            "avg_ann_cost_advantage_proxy": 0.0,
            "avg_performance_energy_ratio_proxy": 0.0,
        }

    ann_advantages = [_numeric_metric(report, "ann_cost_advantage_proxy") for report in profile_reports]
    performance_ratios = [_numeric_metric(report, "performance_energy_ratio_proxy") for report in profile_reports]
    qa_scores = [_numeric_metric(report, "real_data_qa_accuracy") for report in profile_reports]
    negative_control_scores = [
        _numeric_metric(report, "negative_control_abstention_integrity")
        for report in profile_reports
    ]
    negative_control_cost_advantages = [
        _numeric_metric(report, "negative_control_cost_advantage_proxy")
        for report in profile_reports
    ]
    partial_evidence_scores = [
        _numeric_metric(report, "partial_evidence_abstention_integrity")
        for report in profile_reports
    ]
    partial_evidence_cost_advantages = [
        _numeric_metric(report, "partial_evidence_cost_advantage_proxy")
        for report in profile_reports
    ]
    contrastive_accuracies = [
        _numeric_metric(report, "contrastive_control_accuracy")
        for report in profile_reports
    ]
    contrastive_cost_advantages = [
        _numeric_metric(report, "contrastive_control_cost_advantage_proxy")
        for report in profile_reports
    ]
    dense_embedding_cost_advantages = [
        _numeric_metric(report, "dense_embedding_ann_cost_advantage_proxy")
        for report in profile_reports
    ]
    sparse_diffusion_denoise_accuracies = [
        _numeric_metric(report, "sparse_diffusion_real_data_denoise_accuracy")
        for report in profile_reports
    ]
    sparse_diffusion_cost_advantages = [
        _numeric_metric(report, "sparse_diffusion_real_data_event_cost_advantage")
        for report in profile_reports
    ]
    sparse_diffusion_partition_scores = [
        _numeric_metric(report, "sparse_diffusion_real_data_partition_integrity")
        for report in profile_reports
    ]
    sparse_diffusion_single_pass_scores = [
        _numeric_metric(report, "sparse_diffusion_real_data_single_pass_integrity")
        for report in profile_reports
    ]
    return {
        "profile_count": profile_count,
        "passed_profile_count": passed_profile_count,
        "min_real_data_qa_accuracy": min(qa_scores),
        "min_ann_cost_advantage_proxy": min(ann_advantages),
        "min_performance_energy_ratio_proxy": min(performance_ratios),
        "min_negative_control_abstention_integrity": min(negative_control_scores),
        "min_negative_control_cost_advantage_proxy": min(negative_control_cost_advantages),
        "min_partial_evidence_abstention_integrity": min(partial_evidence_scores),
        "min_partial_evidence_cost_advantage_proxy": min(partial_evidence_cost_advantages),
        "min_contrastive_control_accuracy": min(contrastive_accuracies),
        "min_contrastive_control_cost_advantage_proxy": min(contrastive_cost_advantages),
        "min_dense_embedding_ann_cost_advantage_proxy": min(dense_embedding_cost_advantages),
        "min_sparse_diffusion_real_data_denoise_accuracy": min(sparse_diffusion_denoise_accuracies),
        "min_sparse_diffusion_real_data_event_cost_advantage": min(sparse_diffusion_cost_advantages),
        "min_sparse_diffusion_real_data_partition_integrity": min(sparse_diffusion_partition_scores),
        "min_sparse_diffusion_real_data_single_pass_integrity": min(sparse_diffusion_single_pass_scores),
        "avg_ann_cost_advantage_proxy": sum(ann_advantages) / max(profile_count, 1),
        "avg_performance_energy_ratio_proxy": sum(performance_ratios) / max(profile_count, 1),
    }


def _build_ladder_checks(
    profiles: Sequence[Dict[str, Any]],
    profile_reports: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, bool]:
    doc_counts = [int(report.get("doc_count", 0) or 0) for report in profile_reports]
    profile_names = {str(profile.get("name", "")).lower() for profile in profiles}
    default_scale_plan_started = {"small", "medium"}.issubset(profile_names)
    trend_failures = [
        report
        for report in profile_reports
        if isinstance(report.get("checks"), dict)
        and not bool(report["checks"].get("trend.no_regressions", False))
    ]
    return {
        "all_profiles_passed": int(metrics.get("passed_profile_count", 0) or 0) == len(profile_reports),
        "profile_count_matches_plan": len(profile_reports) == len(profiles) and len(profile_reports) > 0,
        "scale_doc_counts_monotonic": doc_counts == sorted(doc_counts),
        "large_profile_present": (not default_scale_plan_started) or "large" in profile_names,
        "ann_cost_advantage_all_profiles": float(metrics.get("min_ann_cost_advantage_proxy", 0.0) or 0.0) >= 2.0,
        "performance_energy_ratio_all_profiles": (
            float(metrics.get("min_performance_energy_ratio_proxy", 0.0) or 0.0) >= 2.0
        ),
        "negative_control_abstention_all_profiles": (
            float(metrics.get("min_negative_control_abstention_integrity", 0.0) or 0.0) >= 1.0
        ),
        "negative_control_cost_advantage_all_profiles": (
            float(metrics.get("min_negative_control_cost_advantage_proxy", 0.0) or 0.0) >= 2.0
        ),
        "partial_evidence_abstention_all_profiles": (
            float(metrics.get("min_partial_evidence_abstention_integrity", 0.0) or 0.0) >= 1.0
        ),
        "partial_evidence_cost_advantage_all_profiles": (
            float(metrics.get("min_partial_evidence_cost_advantage_proxy", 0.0) or 0.0) >= 2.0
        ),
        "contrastive_control_accuracy_all_profiles": (
            float(metrics.get("min_contrastive_control_accuracy", 0.0) or 0.0) >= 1.0
        ),
        "contrastive_control_cost_advantage_all_profiles": (
            float(metrics.get("min_contrastive_control_cost_advantage_proxy", 0.0) or 0.0) >= 2.0
        ),
        "dense_embedding_cost_advantage_all_profiles": (
            float(metrics.get("min_dense_embedding_ann_cost_advantage_proxy", 0.0) or 0.0) >= 2.0
        ),
        "sparse_diffusion_real_data_denoise_all_profiles": (
            float(metrics.get("min_sparse_diffusion_real_data_denoise_accuracy", 0.0) or 0.0) >= 1.0
        ),
        "sparse_diffusion_real_data_cost_advantage_all_profiles": (
            float(metrics.get("min_sparse_diffusion_real_data_event_cost_advantage", 0.0) or 0.0) >= 2.0
        ),
        "sparse_diffusion_real_data_partition_all_profiles": (
            float(metrics.get("min_sparse_diffusion_real_data_partition_integrity", 0.0) or 0.0) >= 1.0
        ),
        "sparse_diffusion_real_data_single_pass_all_profiles": (
            float(metrics.get("min_sparse_diffusion_real_data_single_pass_integrity", 0.0) or 0.0) >= 1.0
        ),
        "no_trend_regressions_all_profiles": len(trend_failures) == 0,
    }


def _compact_profile_report(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    trend = report.get("trend", {}) if isinstance(report.get("trend"), dict) else {}
    benchmark_context = (
        report.get("benchmark_context", {})
        if isinstance(report.get("benchmark_context"), dict)
        else {}
    )
    return {
        "profile": report.get("profile", ""),
        "passed": bool(report.get("passed", False)),
        "report_path": report.get("report_path", ""),
        "summary_path": report.get("summary_path", ""),
        "history_path": report.get("history_path", ""),
        "doc_count": int(report.get("doc_count", 0) or 0),
        "task_count": int(report.get("task_count", 0) or 0),
        "metrics": {
            "real_data_qa_accuracy": float(metrics.get("real_data_qa_accuracy", 0.0) or 0.0),
            "ann_proxy_qa_accuracy": float(metrics.get("ann_proxy_qa_accuracy", 0.0) or 0.0),
            "dense_embedding_ann_proxy_qa_accuracy": float(
                metrics.get("dense_embedding_ann_proxy_qa_accuracy", 0.0) or 0.0
            ),
            "real_data_summary_keyword_coverage": float(
                metrics.get("real_data_summary_keyword_coverage", 0.0) or 0.0
            ),
            "continual_memory_hit_rate": float(metrics.get("continual_memory_hit_rate", 0.0) or 0.0),
            "ann_cost_advantage_proxy": float(metrics.get("ann_cost_advantage_proxy", 0.0) or 0.0),
            "performance_energy_ratio_proxy": float(
                metrics.get("performance_energy_ratio_proxy", 0.0) or 0.0
            ),
            "negative_control_abstention_integrity": float(
                metrics.get("negative_control_abstention_integrity", 0.0) or 0.0
            ),
            "negative_control_cost_advantage_proxy": float(
                metrics.get("negative_control_cost_advantage_proxy", 0.0) or 0.0
            ),
            "partial_evidence_abstention_integrity": float(
                metrics.get("partial_evidence_abstention_integrity", 0.0) or 0.0
            ),
            "partial_evidence_cost_advantage_proxy": float(
                metrics.get("partial_evidence_cost_advantage_proxy", 0.0) or 0.0
            ),
            "contrastive_control_accuracy": float(
                metrics.get("contrastive_control_accuracy", 0.0) or 0.0
            ),
            "contrastive_control_cost_advantage_proxy": float(
                metrics.get("contrastive_control_cost_advantage_proxy", 0.0) or 0.0
            ),
            "dense_embedding_ann_cost_advantage_proxy": float(
                metrics.get("dense_embedding_ann_cost_advantage_proxy", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_denoise_accuracy": float(
                metrics.get("sparse_diffusion_real_data_denoise_accuracy", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_event_cost_advantage": float(
                metrics.get("sparse_diffusion_real_data_event_cost_advantage", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_partition_integrity": float(
                metrics.get("sparse_diffusion_real_data_partition_integrity", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_single_pass_integrity": float(
                metrics.get("sparse_diffusion_real_data_single_pass_integrity", 0.0) or 0.0
            ),
        },
        "checks": {str(name): bool(value) for name, value in checks.items()},
        "trend": {
            "comparison_active": bool(trend.get("comparison_active", False)),
            "comparison_skipped_reason": str(trend.get("comparison_skipped_reason", "") or ""),
            "regression_count": int(trend.get("regression_count", 0) or 0),
        },
        "benchmark_context": {
            "corpus_sha256": str(benchmark_context.get("corpus_sha256", "") or ""),
            "task_sha256": str(benchmark_context.get("task_sha256", "") or ""),
            "max_docs": int(benchmark_context.get("max_docs", 0) or 0),
            "max_cases": int(benchmark_context.get("max_cases", 0) or 0),
            "retriever_strategy": str(benchmark_context.get("retriever_strategy", "") or ""),
        },
    }


def run_real_data_external_validity_ladder(
    *,
    corpus_path: str = processed_data_path("corpus.txt"),
    profiles: Sequence[Dict[str, Any]] | None = None,
    regression_tolerance: float = 0.05,
    update_history: bool = True,
) -> Dict[str, Any]:
    selected_profiles = [dict(item) for item in (profiles or DEFAULT_PROFILE_SPECS)]
    profile_reports: List[Dict[str, Any]] = []
    for profile in selected_profiles:
        profile_name = str(profile.get("name", "profile"))
        history_path = _profile_artifact_path(profile_name, "_history.json")
        history = external_validity.load_external_validity_history(history_path)
        report = external_validity.run_real_data_external_validity(
            corpus_path=corpus_path,
            max_docs=int(profile.get("max_docs", 1) or 1),
            max_cases=int(profile.get("max_cases", 1) or 1),
            history=history,
            regression_tolerance=max(float(regression_tolerance), 0.0),
        )
        report["profile"] = profile_name
        report_path = ensure_parent_directory(_profile_artifact_path(profile_name, ".json"))
        summary_path = ensure_parent_directory(_profile_artifact_path(profile_name, "_summary.txt"))
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(external_validity.format_real_data_external_validity_summary(report))
        report["report_path"] = report_path
        report["summary_path"] = summary_path
        if update_history:
            report["history_path"] = external_validity.append_external_validity_history(history_path, report)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
        profile_reports.append(report)

    metrics = _build_ladder_metrics(profile_reports)
    checks = _build_ladder_checks(selected_profiles, profile_reports, metrics)
    return {
        "suite_name": "RealDataExternalValidityLadder",
        "passed": all(checks.values()),
        "corpus_path": os.path.abspath(corpus_path),
        "profiles": selected_profiles,
        "metrics": metrics,
        "checks": checks,
        "profile_reports": [_compact_profile_report(report) for report in profile_reports],
    }


def format_real_data_external_validity_ladder_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    profile_reports = report.get("profile_reports", [])
    lines = [
        "Real Data External Validity Ladder Summary",
        f"- passed: {bool(report.get('passed', False))}",
        f"- corpus_path: {report.get('corpus_path', '')}",
        f"- profile_count: {int(metrics.get('profile_count', 0) or 0)}",
        f"- passed_profile_count: {int(metrics.get('passed_profile_count', 0) or 0)}",
        f"- min_real_data_qa_accuracy: {float(metrics.get('min_real_data_qa_accuracy', 0.0) or 0.0):.3f}",
        f"- min_ann_cost_advantage_proxy: {float(metrics.get('min_ann_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- min_performance_energy_ratio_proxy: {float(metrics.get('min_performance_energy_ratio_proxy', 0.0) or 0.0):.3f}",
        f"- min_negative_control_abstention_integrity: {float(metrics.get('min_negative_control_abstention_integrity', 0.0) or 0.0):.3f}",
        f"- min_negative_control_cost_advantage_proxy: {float(metrics.get('min_negative_control_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- min_partial_evidence_abstention_integrity: {float(metrics.get('min_partial_evidence_abstention_integrity', 0.0) or 0.0):.3f}",
        f"- min_partial_evidence_cost_advantage_proxy: {float(metrics.get('min_partial_evidence_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- min_contrastive_control_accuracy: {float(metrics.get('min_contrastive_control_accuracy', 0.0) or 0.0):.3f}",
        f"- min_contrastive_control_cost_advantage_proxy: {float(metrics.get('min_contrastive_control_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- min_dense_embedding_ann_cost_advantage_proxy: {float(metrics.get('min_dense_embedding_ann_cost_advantage_proxy', 0.0) or 0.0):.3f}",
        f"- min_sparse_diffusion_real_data_denoise_accuracy: {float(metrics.get('min_sparse_diffusion_real_data_denoise_accuracy', 0.0) or 0.0):.3f}",
        f"- min_sparse_diffusion_real_data_event_cost_advantage: {float(metrics.get('min_sparse_diffusion_real_data_event_cost_advantage', 0.0) or 0.0):.3f}",
        f"- min_sparse_diffusion_real_data_partition_integrity: {float(metrics.get('min_sparse_diffusion_real_data_partition_integrity', 0.0) or 0.0):.3f}",
        f"- min_sparse_diffusion_real_data_single_pass_integrity: {float(metrics.get('min_sparse_diffusion_real_data_single_pass_integrity', 0.0) or 0.0):.3f}",
        "Profiles:",
    ]
    if isinstance(profile_reports, list):
        for profile_report in profile_reports:
            if not isinstance(profile_report, dict):
                continue
            profile_metrics = (
                profile_report.get("metrics", {})
                if isinstance(profile_report.get("metrics"), dict)
                else {}
            )
            lines.append(
                "- "
                f"{profile_report.get('profile', '')}: "
                f"passed={bool(profile_report.get('passed', False))}, "
                f"docs={int(profile_report.get('doc_count', 0) or 0)}, "
                f"tasks={int(profile_report.get('task_count', 0) or 0)}, "
                f"qa={float(profile_metrics.get('real_data_qa_accuracy', 0.0) or 0.0):.3f}, "
                f"ann_cost_advantage={float(profile_metrics.get('ann_cost_advantage_proxy', 0.0) or 0.0):.3f}, "
                f"performance_energy_ratio={float(profile_metrics.get('performance_energy_ratio_proxy', 0.0) or 0.0):.3f}, "
                f"sparse_diffusion_cost_advantage={float(profile_metrics.get('sparse_diffusion_real_data_event_cost_advantage', 0.0) or 0.0):.3f}"
            )
    lines.append("Checks:")
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if checks[name] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-data external validity scale ladder.")
    parser.add_argument("--corpus", default=processed_data_path("corpus.txt"))
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Profile spec as name:max_docs:max_cases. May be passed multiple times.",
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--regression-tolerance", type=float, default=0.05)
    parser.add_argument("--no-history-update", action="store_true")
    args = parser.parse_args()

    profiles = parse_profile_specs(args.profile)
    report = run_real_data_external_validity_ladder(
        corpus_path=str(args.corpus),
        profiles=profiles,
        regression_tolerance=max(float(args.regression_tolerance), 0.0),
        update_history=not bool(args.no_history_update),
    )
    report_path = ensure_parent_directory(str(args.report_path))
    summary_path = ensure_parent_directory(str(args.summary_path))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_real_data_external_validity_ladder_summary(report))

    print(f"Saved ladder report: {report_path}")
    print(f"Saved ladder summary: {summary_path}")
    if not bool(report.get("passed", False)):
        print("Real-data external validity ladder failed.")
        return 1
    print("Real-data external validity ladder passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
