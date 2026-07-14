import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase14_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase14_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase14_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase14_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase14_gate_rejects_unbounded_state_budget(tmp_path):
    module = _load_module()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        '{"passed": true, "observed_only": true, "case_count": 1, "metrics": '
        '{"own_latent_sample_efficiency_ok": 1, "own_latent_event_cost_bounded": 1, '
        '"own_latent_max_state_budget_units": 999}}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"passed": true, "observed_only": true, "manifest_count": 1, "manifest_path": "' + str(tmp_path / "manifest.jsonl") + '", "policy_notes": ["sparse dense backpropagation"]}', encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text('{"case_id": "x"}\n', encoding="utf-8")
    report = module.build_report(benchmark_path=str(benchmark), manifest_path=str(manifest), fixture_path=str(fixture))
    assert report["phase14_complete"] is False
    assert report["checks"]["state_budget_bounded"] is False
