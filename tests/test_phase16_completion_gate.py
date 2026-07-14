import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase16_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase16_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase16_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase16_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase16_gate_rejects_missing_modality_without_abstention(tmp_path):
    module = _load_module()
    path = tmp_path / "benchmark.json"
    path.write_text(
        '{"passed": true, "observed_only": true, "case_count": 4, "selected_window_ms": 32, '
        '"window_profiles":[{"window_ms":25},{"window_ms":32},{"window_ms":40}], '
        '"metrics":{"max_event_cost":4,"max_state_budget_units":8,'
        '"adapter_ir_integrity":1,"temporal_alignment_quality":1,"cross_modal_link_precision":1,'
        '"plug_swap_integrity":1,"missing_modality_abstention_integrity":1,"non_language_route_usefulness":1,'
        '"bundle_integrity":1,"binding_audit_coverage":1,"route_traceability":1}, '
        '"missing_modality_results":[], "policy_notes":["sparse events, dense universal embeddings are forbidden; bounded; preserve modality-local payloads"]}',
        encoding="utf-8",
    )
    report = module.build_report(benchmark_path=str(path))
    assert report["phase16_complete"] is False
    assert report["checks"]["missing_modality_prediction_labeled"] is False
