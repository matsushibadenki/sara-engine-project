"""Static pre-run checks for the prospective development pair protocol."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile

import pytest

from sara_engine.evaluation.event_unit_causal_isolation import FAMILIES, generate_episodes
from sara_engine.research import DiscoveryCaptureLog
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/discovery_capture_pair_v1.json"
PRIOR = ROOT / "data/processed/benchmark_fixtures/event_unit_causal_isolation_v2.json"
SCRIPT = ROOT / "scripts/eval/discovery_capture_event_unit_pair.py"


def _module():
    spec = importlib.util.spec_from_file_location("discovery_capture_pair", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_protocol_identity_and_split_isolation():
    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    source = ROOT / config["candidate_source"]["path"]
    assert sha256(source.read_bytes()).hexdigest() == config["candidate_source"]["sha256"]
    assert sha256(PRIOR.read_bytes()).hexdigest() == config["prior_evidence"]["event_unit_v2_protocol_sha256"]

    generator = config["generator"]
    training_seeds = generator["training_seeds"]
    development_seeds = generator["development_seeds"]
    prior_seeds = prior["fresh_identity"]["seeds"]
    assert len(training_seeds) == len(development_seeds) == 5
    assert len(set(training_seeds + development_seeds + prior_seeds + [92017])) == 16
    assert generator["heldout_generated"] is False
    assert generator["namespace"] == "capture-pair-v1"
    training = generate_episodes(
        seeds=training_seeds,
        count_per_family=generator["training_count_per_family_per_seed"],
        split="training",
        namespace=generator["namespace"],
    )
    development = generate_episodes(
        seeds=development_seeds,
        count_per_family=generator["development_count_per_family_per_seed"],
        split="development",
        namespace=generator["namespace"],
    )
    assert len(training) == 5 * len(FAMILIES) * 20
    assert len(development) == 5 * len(FAMILIES) * 10
    assert not {row.identity for row in training} & {row.identity for row in development}


def test_pair_protocol_has_fixed_actions_and_no_promotion():
    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert [(row["arm"], row["intervention"]) for row in config["fixed_action_order"]] == [
        ("B_compact_event", "none"),
        ("C_temporal_state", "none"),
        ("C_temporal_state", "time_shuffle"),
        ("C_temporal_state", "state_reset"),
    ]
    assert config["capture"]["require_intent_fsync_before_evaluator"] is True
    assert config["capture"]["require_head_published_outside_capture_log_before_evaluator"] is True
    assert config["capture"]["automatic_export_or_approval"] is False
    assert config["boundaries"] == {
        "development_only": True,
        "BPI_Sepsis_frozen_test_opened": False,
        "event_unit_v1_v2_heldout_opened": False,
        "production_authorized": False,
        "meta_policy_gain_claim_allowed": False,
        "spike_specific_gain_claim_allowed": False,
    }


def test_pair_runner_requires_separate_published_head_before_evaluation(monkeypatch):
    module = _module()
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == module.PROTOCOL_SHA256
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = str(Path(temporary) / "pair.jsonl")
        prepared = module.prepare_next(path)
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
                "maximum_event_work": 4,
                "predictions": len(development),
                "state_bytes": 120,
            }

        monkeypatch.setattr(module, "run_v2_development_arm", fake_evaluator)
        with pytest.raises(ValueError, match="Published intent head"):
            module.execute_pending(path, published_intent_head_sha256="0" * 64)
        assert calls == []
        observed = module.execute_pending(
            path, published_intent_head_sha256=prepared["intent_head_sha256"])
        assert calls == [("B_compact_event", "none", 500, 250)]
        assert observed["score"] == 1.0
        assert observed["unapproved"] is True
        assert all(value == 1.0 for value in observed["per_seed_accuracy"].values())
        assert DiscoveryCaptureLog(path).load().pending_node_ids == ()
        with pytest.raises(ValueError, match="no sole pending"):
            module.execute_pending(path, published_intent_head_sha256=prepared["intent_head_sha256"])
        second = module.prepare_next(path)
        assert second["action"] == "C_temporal_state"
        assert DiscoveryCaptureLog(path).load().pending_node_ids == ("C_temporal_state",)


def test_pair_verifier_rejects_incomplete_or_unpublished_prediction_trace(monkeypatch):
    module = _module()
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = str(Path(temporary) / "pair.jsonl")

        def fake_evaluator(arm, training, development, *, intervention):
            return {
                "accuracy": 1.0,
                "prediction_rows": [(row.identity, row.label) for row in development],
                "prediction_trace_sha256": "f" * 64,
                "maximum_event_work": 4,
                "predictions": len(development),
                "state_bytes": 120,
            }

        monkeypatch.setattr(module, "run_v2_development_arm", fake_evaluator)
        first = module.prepare_next(path)
        with pytest.raises(ValueError, match="incomplete"):
            module.verify_capture(
                path, expected_capture_head_sha256=first["intent_head_sha256"],
                expected_traces={row["node_id"]: "f" * 64
                                 for row in json.loads(PROTOCOL.read_text())["fixed_action_order"]},
            )
        prepared_heads = []
        for index in range(4):
            prepared = first if index == 0 else module.prepare_next(path)
            prepared_heads.append(prepared["intent_head_sha256"])
            final = module.execute_pending(
                path, published_intent_head_sha256=prepared["intent_head_sha256"])
        traces = {row["node_id"]: "f" * 64
                  for row in json.loads(PROTOCOL.read_text())["fixed_action_order"]}
        verified = module.verify_capture(
            path, expected_capture_head_sha256=final["outcome_head_sha256"],
            expected_traces=traces,
        )
        assert verified["checks"]["exact_prediction_trace_replay"] is True
        assert verified["diagnostic_gate_passed"] is False
        projection = DiscoveryCaptureLog(path).project_source_snapshot(
            expected_head_sha256=final["outcome_head_sha256"])
        audit = module.audit_pair_lineage(
            path,
            expected_capture_file_sha256=sha256(Path(path).read_bytes()).hexdigest(),
            expected_capture_head_sha256=final["outcome_head_sha256"],
            expected_snapshot_sha256=projection.snapshot_sha256,
            expected_intent_heads=prepared_heads,
        )
        assert audit["development_examples_with_training_labeled_pattern"] == 250
        assert audit["training_development_identity_overlap"] == 0
        assert audit["source_pattern_isolation_passed"] is False
        assert audit["export_approved"] is False
        with pytest.raises(ValueError, match="Intent heads differ"):
            module.audit_pair_lineage(
                path,
                expected_capture_file_sha256=sha256(Path(path).read_bytes()).hexdigest(),
                expected_capture_head_sha256=final["outcome_head_sha256"],
                expected_snapshot_sha256=projection.snapshot_sha256,
                expected_intent_heads=["0" * 64, *prepared_heads[1:]],
            )
        traces["C_temporal_state"] = "e" * 64
        with pytest.raises(ValueError, match="prediction trace differs"):
            module.verify_capture(
                path, expected_capture_head_sha256=final["outcome_head_sha256"],
                expected_traces=traces,
            )
