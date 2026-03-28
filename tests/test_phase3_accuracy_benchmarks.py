import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.evaluation.phase3_tracking import (
    append_phase3_history,
    build_phase3_trend,
    load_phase3_history,
)
from sara_engine.utils.project_paths import workspace_path


def _load_script(script_name: str):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", script_name)
    )
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_accuracy_benchmark_returns_expected_metrics():
    module = _load_script("inference_accuracy_benchmark.py")
    report = module.run_inference_accuracy_benchmark()

    assert report["evaluator_name"] == "InferenceSequenceEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["one_shot_accuracy"] == 1.0
    assert report["metrics"]["few_shot_accuracy"] == 1.0
    assert report["metrics"]["fuzzy_retrieval_accuracy"] == 1.0
    assert report["metrics"]["continual_retention"] == 1.0
    assert report["metrics"]["long_horizon_retention"] == 1.0


def test_spiking_llm_accuracy_benchmark_returns_expected_metrics():
    module = _load_script("spiking_llm_accuracy_benchmark.py")
    report = module.run_spiking_llm_accuracy_benchmark()

    assert report["evaluator_name"] == "SpikingLLMSequenceEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["next_token_accuracy"] == 1.0
    assert report["metrics"]["few_shot_context_accuracy"] == 1.0
    assert report["metrics"]["stream_completion_rate"] == 1.0
    assert report["metrics"]["continual_memory_retention"] == 1.0
    assert report["metrics"]["long_horizon_memory_retention"] == 1.0


def test_phase3_accuracy_suite_aggregates_component_reports():
    module = _load_script("phase3_accuracy_suite.py")
    report = module.run_phase3_accuracy_suite()

    assert report["suite_name"] == "Phase3AccuracySuite"
    assert report["passed"] is True
    assert "agent_dialogue" in report["component_reports"]
    assert "sara_inference" in report["component_reports"]
    assert "spiking_llm" in report["component_reports"]
    assert "focus_summary" in report
    assert report["focus_summary"]["few_shot"]["score"] == 1.0
    assert report["focus_summary"]["continual"]["score"] == 1.0
    assert "trend" in report


def test_phase3_accuracy_suite_formats_human_readable_summary():
    module = _load_script("phase3_accuracy_suite.py")
    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 1.0,
        "passed": True,
        "trend": {"regression_count": 0},
        "focus_summary": {
            "few_shot": {"passed": True, "score": 1.0},
            "continual": {"passed": True, "score": 1.0},
        },
        "component_reports": {
            "agent_dialogue": {"passed": True, "overall_score": 0.9},
            "sara_inference": {"passed": True, "overall_score": 1.0},
            "spiking_llm": {"passed": True, "overall_score": 1.0},
        },
    }

    summary = module.format_phase3_accuracy_summary(report)

    assert "SARA Engine Phase 3 Accuracy Summary" in summary
    assert "overall_status: PASS" in summary
    assert "- few_shot_status: PASS" in summary
    assert "- continual_status: PASS" in summary
    assert "- sara_inference: PASS score=1.000" in summary


def test_phase3_tracking_detects_metric_regression():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.9,
                "metrics": {"response_keyword_recall": 0.8},
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.90,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.85,
                "metrics": {"response_keyword_recall": 0.7},
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["has_previous"] is True
    assert trend["regression_count"] >= 1
    assert any(
        item["metric"] == "agent_dialogue.response_keyword_recall"
        for item in trend["regressions"]
    )


def test_phase3_tracking_flattens_focus_summary_metrics():
    from sara_engine.evaluation.phase3_tracking import flatten_phase3_metrics

    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 1.0,
        "component_reports": {},
        "focus_summary": {
            "few_shot": {
                "score": 1.0,
                "metrics": {
                    "sara_inference.few_shot_accuracy": 1.0,
                },
            },
            "continual": {
                "score": 0.9,
                "metrics": {
                    "spiking_llm.long_horizon_memory_retention": 0.9,
                },
            },
        },
    }

    flattened = flatten_phase3_metrics(report)

    assert flattened["focus.few_shot.score"] == 1.0
    assert flattened["focus.few_shot.sara_inference.few_shot_accuracy"] == 1.0
    assert flattened["focus.continual.score"] == 0.9
    assert flattened["focus.continual.spiking_llm.long_horizon_memory_retention"] == 0.9


def test_phase3_accuracy_suite_persists_managed_history():
    module = _load_script("phase3_accuracy_suite.py")
    history_path = workspace_path("tests", "phase3_accuracy_history_test.json")
    if os.path.exists(history_path):
        os.remove(history_path)

    first_report = module.run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=True,
        history_limit=2,
    )
    second_report = module.run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=True,
        history_limit=2,
    )
    history = load_phase3_history(history_path)

    assert first_report["history_length"] == 1
    assert second_report["history_length"] == 2
    assert len(history) == 2
    assert history[-1]["suite_name"] == "Phase3AccuracySuite"


def test_phase3_history_limit_trims_old_entries():
    history_path = workspace_path("tests", "phase3_accuracy_history_trim_test.json")
    if os.path.exists(history_path):
        os.remove(history_path)

    for idx in range(3):
        append_phase3_history(
            history_path=history_path,
            report={"suite_name": "Phase3AccuracySuite", "overall_score": float(idx)},
            max_entries=2,
        )

    history = load_phase3_history(history_path)

    assert len(history) == 2
    assert history[0]["overall_score"] == 1.0
    assert history[1]["overall_score"] == 2.0
