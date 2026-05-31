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


def test_nested_memory_controller_passes_readiness_fixture() -> None:
    module = _load_script("nested_memory_readiness_benchmark.py")

    report = module.run_nested_memory_readiness_benchmark()
    summary = module.format_nested_memory_readiness_summary(report)

    assert report["evaluator_name"] == "NestedMemoryReadinessBenchmark"
    assert report["passed"] is True
    assert report["threshold_results"]["multi_rate_update_integrity"] is True
    assert report["threshold_results"]["continuum_memory_transfer_stability"] is True
    assert report["threshold_results"]["scheduler_energy_budget_integrity"] is True
    assert report["threshold_results"]["catastrophic_interference_guard"] is True
    assert "- status: PASS" in summary


def test_nested_memory_controller_blocks_slow_updates_during_interference() -> None:
    nested_module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "sara_engine", "memory", "nested_continual.py")
    )
    spec = importlib.util.spec_from_file_location("nested_continual_module", nested_module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    controller = module.NestedContinualMemoryController()
    for _ in range(16):
        controller.observe(signal_strength=0.9, interference=0.7, novelty=0.2, urgency=0.1)

    snapshot = controller.snapshot()
    assert snapshot["guard_events"] == 16
    assert snapshot["update_counts"]["ltm"] == 0
    assert snapshot["update_counts"]["structural"] == 0
