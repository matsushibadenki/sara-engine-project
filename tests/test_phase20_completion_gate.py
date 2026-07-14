import importlib.util
import json
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase20_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase20_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase20_gate_passes_observed_evidence():
    report = _load_module().build_report()
    assert report["phase20_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase20_gate_rejects_missing_abstention(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({
        "passed": True,
        "observed_only": True,
        "case_count": 5,
        "metrics": {
            "semantic_echo_improves_single": 1,
            "semantic_echo_improves_multiscale": 1,
            "abstention_integrity": 0,
            "idle_spikes": 0,
            "max_active_echoes": 3,
            "max_comparisons": 3,
            "max_updates": 3,
        },
        "policy_notes": ["sparse CPU-first no dense Attention without an external parser or LLM backpropagation fixed single-decay observed-only"],
    }), encoding="utf-8")
    report = _load_module().build_report(benchmark_path=str(path))
    assert report["phase20_complete"] is False
    assert report["checks"]["abstention_integrity"] is False
