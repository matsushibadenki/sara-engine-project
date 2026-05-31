# Directory Path: scripts/eval/task_switch_adaptation_benchmark.py
# English Title: Task-Switch Adaptation Benchmark
# Purpose/Content: Runs a lightweight adaptation benchmark for SaraInference session-memory updates and next-step suggestions under task switches.

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _build_engine() -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    return engine


def _run_english_switch_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.generate("You: I want to finish this project\nSARA:")
    engine.generate("You: I am working on the sara engine\nSARA:")
    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: I am working on release preparation\nSARA:")
    response = engine.generate("You: What should I do next?\nSARA:")
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)

    success = all(
        token in response
        for token in [
            "Step 1:",
            "Step 2:",
            "release preparation",
            "ship the release",
        ]
    ) and "sara engine" not in response

    return {
        "success": success,
        "response": response,
        "memory_hit": diagnostics[0]["memory_hit"] if diagnostics else "",
        "description": "Latest English goal/task should override older session-memory planning context.",
    }


def _run_japanese_switch_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.generate("You: 私は安定化したいです\nSARA:")
    engine.generate("You: 私はテスト確認をしています\nSARA:")
    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 次に何をすればいい？\nSARA:")
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)

    success = all(
        token in response
        for token in [
            "Step 1:",
            "Step 2:",
            "「リリース準備」",
            "「公開」",
        ]
    ) and "「テスト確認」" not in response

    return {
        "success": success,
        "response": response,
        "memory_hit": diagnostics[0]["memory_hit"] if diagnostics else "",
        "description": "Latest Japanese goal/task should override older session-memory planning context.",
    }


def _run_meta_adaptation_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.generate("You: I want to finish this project\nSARA:")
    engine.generate("You: I am working on the sara engine\nSARA:")
    first_response = engine.generate("You: What should I do next?\nSARA:")
    second_response = engine.generate("You: What should I do next?\nSARA:")
    adaptation_state = dict(getattr(engine, "adaptation_state", {}))

    success = (
        "Do this now:" not in first_response
        and "Do this now:" in second_response
        and adaptation_state.get("response_mode") == "directive"
        and adaptation_state.get("next_step_requests", 0) >= 2
    )

    return {
        "success": success,
        "first_response": first_response,
        "second_response": second_response,
        "adaptation_state": adaptation_state,
        "description": "Repeated next-step prompts should shift the lightweight adaptation loop from guided mode to directive mode.",
    }


def _run_temporal_self_distillation_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.generate("You: I want to finish this project\nSARA:")
    engine.generate("You: I am working on release preparation\nSARA:")
    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: What should I do next?\nSARA:")
    engine.generate("You: What should I do next?\nSARA:")
    teacher_state = dict(getattr(engine, "adaptation_state", {}))
    distilled_response = engine.generate("You: What should I do next?\nSARA:")
    student_state = dict(getattr(engine, "adaptation_state", {}))

    teacher_conf = float(teacher_state.get("planning_confidence", 0.0) or 0.0)
    teacher_weight = float(teacher_state.get("memory_weight", 0.0) or 0.0)
    teacher_relax = float(teacher_state.get("fallback_relaxation", 0.0) or 0.0)
    student_conf = float(student_state.get("planning_confidence", 0.0) or 0.0)
    student_weight = float(student_state.get("memory_weight", 0.0) or 0.0)
    student_relax = float(student_state.get("fallback_relaxation", 0.0) or 0.0)
    state_drift = (
        abs(student_conf - teacher_conf)
        + abs(student_weight - teacher_weight)
        + abs(student_relax - teacher_relax)
    )

    success = (
        "Step 1:" in distilled_response
        and "ship the release" in distilled_response
        and str(student_state.get("response_mode", "")) == "directive"
        and state_drift <= 0.25
    )
    return {
        "success": success,
        "distilled_response": distilled_response,
        "teacher_state": teacher_state,
        "student_state": student_state,
        "state_drift": float(state_drift),
        "description": "Temporal self-distillation should preserve next-step behavior under paraphrased prompts without adaptation-state collapse.",
    }


def _adaptation_parameter_integrity_score(case: Dict[str, Any]) -> float:
    adaptation_state = case.get("adaptation_state", {})
    if not isinstance(adaptation_state, dict):
        return 0.0
    response_mode = str(adaptation_state.get("response_mode", ""))
    planning_confidence = float(adaptation_state.get("planning_confidence", 0.0) or 0.0)
    memory_weight = float(adaptation_state.get("memory_weight", 0.0) or 0.0)
    fallback_relaxation = float(adaptation_state.get("fallback_relaxation", 0.0) or 0.0)
    if response_mode != "directive":
        return 0.0
    checks = [
        planning_confidence >= 0.8,
        1.1 <= memory_weight <= 1.5,
        0.02 <= fallback_relaxation <= 0.12,
    ]
    return 1.0 if all(checks) else 0.0


def run_task_switch_adaptation_benchmark() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = [
        _run_english_switch_case(),
        _run_japanese_switch_case(),
        _run_meta_adaptation_case(),
        _run_temporal_self_distillation_case(),
    ]
    adaptation_scores = [1.0 if case["success"] else 0.0 for case in cases]
    memory_hit_scores = [
        1.0 if case.get("memory_hit") == "session_memory" else 0.0
        for case in cases
        if "memory_hit" in case
    ]
    meta_adaptation_scores = [
        1.0
        if isinstance(case.get("adaptation_state"), dict)
        and str(case["adaptation_state"].get("response_mode", "")) == "directive"
        else 0.0
        for case in cases
        if "adaptation_state" in case
    ]
    adaptation_parameter_integrity_scores = [
        _adaptation_parameter_integrity_score(case)
        for case in cases
        if "adaptation_state" in case
    ]
    temporal_self_distillation_scores = [
        1.0 if case.get("success", False) else 0.0
        for case in cases
        if "state_drift" in case
    ]

    metrics = {
        "task_switch_adaptation": sum(adaptation_scores) / max(len(adaptation_scores), 1),
        "session_memory_switch_grounding": sum(memory_hit_scores) / max(len(memory_hit_scores), 1),
        "meta_adaptation_loop": sum(meta_adaptation_scores) / max(len(meta_adaptation_scores), 1),
        "meta_adaptation_parameter_integrity": sum(adaptation_parameter_integrity_scores)
        / max(len(adaptation_parameter_integrity_scores), 1),
        "temporal_self_distillation_stability": sum(temporal_self_distillation_scores)
        / max(len(temporal_self_distillation_scores), 1),
    }
    thresholds = {
        "task_switch_adaptation": 1.0,
        "session_memory_switch_grounding": 1.0,
        "meta_adaptation_loop": 1.0,
        "meta_adaptation_parameter_integrity": 1.0,
        "temporal_self_distillation_stability": 1.0,
    }
    threshold_results = {
        name: metrics.get(name, 0.0) >= threshold
        for name, threshold in thresholds.items()
    }

    return {
        "evaluator_name": "TaskSwitchAdaptationBenchmark",
        "overall_score": sum(metrics.values()) / max(len(metrics), 1),
        "metrics": metrics,
        "details": {"test_results": cases},
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight task-switch adaptation benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "task_switch_adaptation_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    args = parser.parse_args()

    report = run_task_switch_adaptation_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Task-switch adaptation benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
