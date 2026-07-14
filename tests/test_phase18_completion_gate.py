import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase18_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase18_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase18_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase18_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase18_gate_rejects_linear_state_growth(tmp_path):
    module = _load_module()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text('{"passed":true,"observed_only":true,"candidate_count":4,"metrics":{"fixed_delayed_recall":0,"logarithmic_delayed_recall":1,"logarithmic_to_linear_state_ratio":1.2,"logarithmic_negative_abstention":1,"blocked_decision_integrity":1,"logarithmic_max_retrieval_event_cost":2,"logarithmic_entry_count":2},"profiles":{"none":{},"fixed":{},"linear":{},"logarithmic":{}},"policy_notes":["verified sparse bounded dense retrieval; does not alter production memory"]}', encoding="utf-8")
    integration = tmp_path / "integration.json"
    integration.write_text('{"passed":true,"observed_only":true,"metrics":{"round_trip_integrity":1,"corrupted_state_rejection":1,"source_revision_integrity":1,"reactivation_hint_integrity":1,"missing_report_freeze_integrity":1,"source_aware_fixed_delayed_recall":0,"source_aware_logarithmic_delayed_recall":1,"max_retrieval_event_cost":2},"policy_notes":["verified sparse bounded; does not alter production memory"]}', encoding="utf-8")
    report = module.build_report(benchmark_path=str(benchmark), integration_path=str(integration))
    assert report["phase18_complete"] is False
    assert report["checks"]["logarithmic_state_growth_bounded"] is False
