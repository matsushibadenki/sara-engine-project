import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase17_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase17_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase17_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase17_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase17_gate_rejects_missing_integration_evidence(tmp_path):
    module = _load_module()
    credit = tmp_path / "credit.json"
    credit.write_text('{"passed":true,"observed_only":true,"metrics":{"decision_integrity":1,"harmful_update_suppression":1,"naive_reward_harmful_update_count":1,"resonance_update_count":1,"max_event_cost":2,"max_state_budget_units":2},"rows":[{"decision":"freeze_contradiction"}],"policy_notes":["sparse local CPU-first observed-only does not alter production learning"]}', encoding="utf-8")
    integration = tmp_path / "integration.json"
    integration.write_text('{"passed":false,"observed_only":true,"metrics":{},"source_paths":[]}', encoding="utf-8")
    report = module.build_report(credit_path=str(credit), integration_path=str(integration))
    assert report["phase17_complete"] is False
    assert report["checks"]["integration_passed"] is False
