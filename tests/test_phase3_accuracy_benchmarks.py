import importlib.util
import os


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
    assert report["metrics"]["fuzzy_retrieval_accuracy"] == 1.0
    assert report["metrics"]["continual_retention"] == 1.0


def test_spiking_llm_accuracy_benchmark_returns_expected_metrics():
    module = _load_script("spiking_llm_accuracy_benchmark.py")
    report = module.run_spiking_llm_accuracy_benchmark()

    assert report["evaluator_name"] == "SpikingLLMSequenceEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["next_token_accuracy"] == 1.0
    assert report["metrics"]["stream_completion_rate"] == 1.0
    assert report["metrics"]["continual_memory_retention"] == 1.0


def test_phase3_accuracy_suite_aggregates_component_reports():
    module = _load_script("phase3_accuracy_suite.py")
    report = module.run_phase3_accuracy_suite()

    assert report["suite_name"] == "Phase3AccuracySuite"
    assert report["passed"] is True
    assert "agent_dialogue" in report["component_reports"]
    assert "sara_inference" in report["component_reports"]
    assert "spiking_llm" in report["component_reports"]
