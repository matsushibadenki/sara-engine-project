import importlib.util
import os


def _load_benchmark_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "agent_dialogue_benchmark.py")
    )
    spec = importlib.util.spec_from_file_location("agent_dialogue_benchmark_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_dialogue_benchmark_returns_expected_metric_structure():
    module = _load_benchmark_module()
    report = module.run_agent_dialogue_benchmark()

    assert report["evaluator_name"] == "AgentDialogueEvaluator"
    assert 0.0 <= report["overall_score"] <= 1.0
    assert "response_keyword_recall" in report["metrics"]
    assert "fallback_control" in report["metrics"]
    assert "retrieval_grounding" in report["metrics"]
    assert report["details"]["test_results"]
    assert "threshold_results" in report
    assert "passed" in report
    assert len(report["details"]["test_results"]) >= 4


def test_agent_dialogue_benchmark_thresholds_are_in_unit_range():
    module = _load_benchmark_module()
    report = module.run_agent_dialogue_benchmark()

    for value in report["thresholds"].values():
        assert 0.0 <= value <= 1.0
    assert set(report["threshold_results"].keys()) == set(report["thresholds"].keys())
