"""Two-command temporal-transfer capture never evaluates before publication."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import tempfile

import pytest

from sara_engine.research import DiscoveryCaptureLog
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/temporal_transfer_capture.py"


def _module():
    spec = importlib.util.spec_from_file_location("temporal_transfer_capture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_and_separate_head_gate(monkeypatch):
    module = _module()
    assert sha256(module.PROTOCOL_PATH.read_bytes()).hexdigest() == module.PROTOCOL_SHA256
    audit = module.preflight(module._protocol())[2]
    assert audit == {
        "training_count": 200,
        "development_count": 100,
        "exact_labeled_pattern_overlap": 0,
        "episode_identity_overlap": 0,
        "nuisance_route_overlap": 0,
        "input_event_count": 900,
    }
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = str(Path(temporary) / "transfer.jsonl")
        prepared = module.prepare_next(path)
        assert prepared["action"] == "B_compact_event"
        assert prepared["evaluator_invoked"] is False
        assert DiscoveryCaptureLog(path).load().pending_node_ids == ("B_compact_event",)
        calls = []

        def fake_evaluator(arm, training, development, *, intervention):
            calls.append((arm, intervention, len(training), len(development)))
            assert DiscoveryCaptureLog(path).load().pending_node_ids == ("B_compact_event",)
            return {
                "accuracy": 1.0,
                "prediction_rows": [(row.identity, row.label) for row in development],
                "prediction_trace_sha256": "f" * 64,
                "maximum_event_work": 8,
                "predictions": len(development),
                "state_bytes": 128,
            }

        monkeypatch.setattr(module, "run_frozen_development_arm", fake_evaluator)
        with pytest.raises(ValueError, match="Published intent head"):
            module.execute_pending(path, published_intent_head_sha256="0" * 64)
        assert calls == []
        result = module.execute_pending(
            path, published_intent_head_sha256=prepared["intent_head_sha256"])
        assert calls == [("B_compact_event", "none", 200, 100)]
        assert result["score"] == 1.0
        assert result["input_event_count"] == 900
        assert all(value == 1.0 for value in result["per_seed_accuracy"].values())
        assert DiscoveryCaptureLog(path).load().pending_node_ids == ()
        with pytest.raises(ValueError, match="no sole pending"):
            module.execute_pending(path, published_intent_head_sha256=prepared["intent_head_sha256"])
        second = module.prepare_next(path)
        assert second["action"] == "C_temporal_state"


def test_evaluator_failure_leaves_pending_intent(monkeypatch):
    module = _module()
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = str(Path(temporary) / "transfer.jsonl")
        prepared = module.prepare_next(path)

        def fail(*args, **kwargs):
            raise RuntimeError("evaluator interrupted")

        monkeypatch.setattr(module, "run_frozen_development_arm", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            module.execute_pending(
                path, published_intent_head_sha256=prepared["intent_head_sha256"])
        view = DiscoveryCaptureLog(path).load()
        assert view.pending_node_ids == ("B_compact_event",)
        assert view.head_sha256 == prepared["intent_head_sha256"]


def test_verifier_rejects_wrong_trace_and_keeps_failed_gate_unapproved(monkeypatch):
    module = _module()
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = str(Path(temporary) / "transfer.jsonl")

        def fake_evaluator(arm, training, development, *, intervention):
            return {
                "accuracy": 1.0,
                "prediction_rows": [(row.identity, row.label) for row in development],
                "prediction_trace_sha256": "f" * 64,
                "maximum_event_work": 8,
                "predictions": len(development),
                "state_bytes": 128,
            }

        monkeypatch.setattr(module, "run_frozen_development_arm", fake_evaluator)
        intent_heads = []
        for _ in range(4):
            prepared = module.prepare_next(path)
            intent_heads.append(prepared["intent_head_sha256"])
            final = module.execute_pending(
                path, published_intent_head_sha256=prepared["intent_head_sha256"])
        traces = {node: "f" * 64 for node in (
            "B_compact_event", "C_temporal_state", "C_time_shuffle", "C_state_reset")}
        verified = module.verify_capture(
            path, expected_capture_head_sha256=final["outcome_head_sha256"],
            expected_traces=traces)
        assert verified["checks"]["exact_prediction_trace_replay"] is True
        assert verified["diagnostic_gate_passed"] is False
        assert verified["export_approved"] is False
        projection = DiscoveryCaptureLog(path).project_source_snapshot(
            expected_head_sha256=final["outcome_head_sha256"])
        lineage = module.audit_lineage(
            path,
            expected_capture_file_sha256=sha256(Path(path).read_bytes()).hexdigest(),
            expected_capture_head_sha256=final["outcome_head_sha256"],
            expected_snapshot_sha256=projection.snapshot_sha256,
            expected_intent_heads=intent_heads,
        )
        assert lineage["split_audit"]["exact_labeled_pattern_overlap"] == 0
        assert lineage["export_approved"] is False
        with pytest.raises(ValueError, match="Intent heads differ"):
            module.audit_lineage(
                path,
                expected_capture_file_sha256=sha256(Path(path).read_bytes()).hexdigest(),
                expected_capture_head_sha256=final["outcome_head_sha256"],
                expected_snapshot_sha256=projection.snapshot_sha256,
                expected_intent_heads=["0" * 64, *intent_heads[1:]],
            )
        traces["C_temporal_state"] = "e" * 64
        with pytest.raises(ValueError, match="prediction trace differs"):
            module.verify_capture(
                path, expected_capture_head_sha256=final["outcome_head_sha256"],
                expected_traces=traces)
