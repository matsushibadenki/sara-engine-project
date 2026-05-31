import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_memory_health_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "utils", "memory_health.py")
    )
    spec = importlib.util.spec_from_file_location("memory_health_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_inference_memory_builds_health_report_and_writes_managed_output():
    module = _load_memory_health_module()
    memory_path = model_path("tests", "memory_health.msgpack")
    report_path = workspace_path("tests", "memory_health_report.json")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (111,): {7: 1.0, 8: 2.0},
        (222,): {9: 3.0},
    }
    writer.context_index = {
        (111,): (1, 2),
        (222,): (3, 4),
    }
    writer.retrieval_diagnostics = [
        {
            "source": "inference_direct_map",
            "memory_hit": "retrieval",
            "content_preview": "1 2",
            "base_score": 1.0,
            "stability_score": 0.9,
            "suffix_match": 1.0,
            "drift_penalty": 0.0,
            "metadata_keyword_overlap": 1.0,
            "context_match": True,
            "role_match": True,
        }
    ]
    writer.refractory_buffer = []
    writer.session_memory = {"location": "Tokyo", "name": "Alex"}
    writer.predictor_state = {}
    writer.adaptation_state = {}
    writer.lif_network = None
    writer.save_pretrained(memory_path)

    report = module.inspect_inference_memory(memory_path, report_path)

    assert report["artifact_generation"] == "indexed"
    assert report["pattern_count"] == 2
    assert report["context_count"] == 2
    assert report["avg_branching_factor"] == 1.5
    assert report["retrieval_diagnostics_count"] == 1
    assert report["retrieval_stability_average"] == 0.9
    assert report["session_memory_keys"] == ["location", "name"]
    assert report["session_memory_snapshot"]["location"] == "Tokyo"
    assert report["predictor_state_keys"] == []
    assert report["predictor_state_snapshot"] == {}
    assert report["adaptation_state_keys"] == []
    assert report["adaptation_state_snapshot"] == {}
    assert report["diagnostic_memory_hits"] == ["retrieval"]
    assert report["conversational_readiness"]["profile_memory_ready"] is True
    assert report["conversational_readiness"]["next_step_ready"] is False
    assert report["conversational_readiness"]["predictor_state_ready"] is False
    assert report["conversational_readiness"]["predictive_simulation_ready"] is False
    assert report["conversational_readiness"]["meta_adaptation_ready"] is False
    assert report["conversational_readiness"]["session_memory_observable"] is False
    assert report["conversational_readiness"]["operator_trace_ready"] is False
    assert report["conversational_readiness"]["speculative_trace_ready"] is False
    assert report["conversational_readiness"]["fluid_trace_ready"] is False
    assert report["health_checks"]["has_patterns"] is True
    assert report["health_checks"]["contexts_cover_patterns"] is True
    assert report["health_checks"]["supports_fuzzy_retrieval"] is True
    assert report["health_checks"]["diagnostics_schema_ok"] is True
    assert report["recommendations"]
    assert report["report_path"] == os.path.abspath(report_path)
    assert os.path.exists(report_path)


def test_inspect_inference_memory_detects_turboquant_payload():
    module = _load_memory_health_module()
    memory_path = model_path("tests", "memory_health_turboquant.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (333,): {5: 0.5, 6: 1.5, 7: 2.5},
    }
    writer.context_index = {
        (333,): (7, 8, 9),
    }
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.session_memory = {}
    writer.predictor_state = {}
    writer.adaptation_state = {}
    writer.lif_network = None
    writer.quantization_enabled = True
    writer.save_pretrained(memory_path)

    report = module.inspect_inference_memory(memory_path)

    assert report["quantization_enabled"] is True
    assert report["pattern_count"] == 1


def test_inspect_inference_memory_marks_next_step_readiness_when_goal_and_task_exist():
    module = _load_memory_health_module()
    memory_path = model_path("tests", "memory_health_conversation.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (555,): {11: 1.0},
    }
    writer.context_index = {
        (555,): (1, 2, 3),
    }
    writer.retrieval_diagnostics = [
        {
            "source": "inference_fast_path",
            "memory_hit": "session_memory",
            "content_preview": "What should I do next?",
            "base_score": 1.0,
            "stability_score": 1.0,
            "suffix_match": 1.0,
            "drift_penalty": 0.0,
            "metadata_keyword_overlap": 1.0,
            "context_match": True,
            "role_match": True,
        }
    ]
    writer.refractory_buffer = []
    writer.session_memory = {"goal": "finish this project", "task": "the sara engine"}
    writer.predictor_state = {
        "category": "development",
        "action": "choose one concrete change to make in the sara engine",
        "target_state": "finish this project",
        "command": "pytest -q",
        "confidence": 1.0,
        "alternative_action": "compare one alternative small action for the sara engine",
        "alternative_target_state": "finish this project",
        "alternative_command": "",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "finish one low-dependency small action first for the sara engine",
        "secondary_alternative_target_state": "finish this project",
        "secondary_alternative_command": "",
        "secondary_alternative_confidence": 0.4,
        "branch_candidates": [
            {"kind": "primary", "label": "primary", "response": "Step 1: choose one concrete change to make in the sara engine.", "confidence": 1.0},
            {"kind": "alternative", "label": "alternative", "response": "An alternative next step is to compare one alternative small action for the sara engine.", "confidence": 0.5},
            {"kind": "secondary", "label": "secondary", "response": "A second alternative next step is to finish one low-dependency small action first for the sara engine.", "confidence": 0.4},
        ],
        "ranked_branch_candidates": [
            {"kind": "alternative", "label": "alternative", "response": "An alternative next step is to compare one alternative small action for the sara engine.", "confidence": 0.5},
            {"kind": "primary", "label": "primary", "response": "Step 1: choose one concrete change to make in the sara engine.", "confidence": 1.0},
            {"kind": "secondary", "label": "secondary", "response": "A second alternative next step is to finish one low-dependency small action first for the sara engine.", "confidence": 0.4},
        ],
        "simulated_branch_candidates": [
            {"kind": "alternative", "label": "alternative", "simulation_score": 0.88},
            {"kind": "primary", "label": "primary", "simulation_score": 0.82},
            {"kind": "secondary", "label": "secondary", "simulation_score": 0.71},
        ],
        "best_simulated_branch": "alternative",
        "preferred_branch": "alternative",
        "transition_operator": "development.execute",
        "speculative_trace": {
            "predicted_operator": "development.execute",
            "verified_operator": "development.execute",
            "operator_match": True,
            "draft_verify_accepted": True,
            "rollback_observable": True,
            "counterfactual_branch_viable": True,
        },
        "fluid_trace": {
            "bounded": True,
            "support_score": 0.75,
            "active_columns": 6,
            "total_spikes": 18,
        },
    }
    writer.adaptation_state = {
        "adaptation_turns": 3,
        "next_step_requests": 2,
        "memory_requests": 1,
        "response_mode": "directive",
        "command_preference": True,
        "planning_confidence": 1.0,
        "memory_weight": 1.5,
        "fallback_relaxation": 0.1,
        "last_intent": "next_step",
    }
    writer.lif_network = None
    writer.future_state_runtime_state = {
        "transition_count": 2,
        "stable_transition_count": 1,
        "shift_count": 0,
        "stability_ratio": 1.0,
        "previous_category": "development",
        "previous_target_state": "finish this project",
        "last_shift_from": "",
        "last_shift_to": "",
        "last_category": "development",
        "last_target_state": "finish this project",
        "last_language": "en",
        "last_branch_count": 3,
        "last_branch_labels": ["primary", "alternative", "secondary"],
        "last_preferred_branch": "alternative",
        "last_simulated_branch_count": 3,
        "last_best_simulated_branch": "alternative",
        "last_best_simulation_score": 0.88,
    }
    writer.save_pretrained(memory_path)

    report = module.inspect_inference_memory(memory_path)

    assert report["conversational_readiness"]["next_step_ready"] is True
    assert report["conversational_readiness"]["predictor_state_ready"] is True
    assert report["conversational_readiness"]["predictive_branching_ready"] is True
    assert report["conversational_readiness"]["predictive_simulation_ready"] is True
    assert report["conversational_readiness"]["meta_adaptation_ready"] is True
    assert report["conversational_readiness"]["session_memory_observable"] is True
    assert report["conversational_readiness"]["operator_trace_ready"] is True
    assert report["conversational_readiness"]["speculative_trace_ready"] is True
    assert report["conversational_readiness"]["fluid_trace_ready"] is True
    assert "action" in report["predictor_state_keys"]
    assert "secondary_alternative_action" in report["predictor_state_keys"]
    assert "transition_operator" in report["predictor_state_keys"]
    assert "speculative_trace" in report["predictor_state_keys"]
    assert "branch_candidates" in report["predictor_state_keys"]
    assert "simulated_branch_candidates" in report["predictor_state_keys"]
    assert "best_simulated_branch" in report["predictor_state_keys"]
    assert "preferred_branch" in report["predictor_state_keys"]
    assert "response_mode" in report["adaptation_state_keys"]
    assert report["predictor_state_snapshot"]["target_state"] == "finish this project"
    assert report["predictor_state_snapshot"]["secondary_alternative_target_state"] == "finish this project"
    assert report["predictor_state_snapshot"]["preferred_branch"] == "primary"
    assert report["predictor_state_snapshot"]["best_simulated_branch"] == "primary"
    assert report["predictor_state_snapshot"]["transition_operator"]
    assert isinstance(report["predictor_state_snapshot"]["speculative_trace"], dict)
    assert report["adaptation_state_snapshot"]["response_mode"] == "directive"
    assert report["adaptation_state_snapshot"]["memory_weight"] == 1.5
    assert report["future_state_runtime_state"]["last_branch_count"] == 3
    assert report["future_state_runtime_state"]["last_preferred_branch"] == "primary"
    assert report["future_state_runtime_state"]["last_simulated_branch_count"] == 3
    assert report["future_state_runtime_state"]["last_best_simulated_branch"] == "primary"
    assert report["future_state_runtime_state"]["last_transition_operator"]
    assert report["future_state_runtime_state"]["operator_consistency_ratio"] >= 0.0
    assert "session_memory" in report["diagnostic_memory_hits"]


def test_inspect_inference_memory_flags_legacy_direct_map_only_artifact():
    module = _load_memory_health_module()
    memory_path = model_path("tests", "memory_health_legacy.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (444,): {10: 1.0},
    }
    writer.context_index = {}
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.session_memory = {}
    writer.predictor_state = {}
    writer.adaptation_state = {}
    writer.lif_network = None
    writer.context_encoding = "legacy_python_hash"
    writer.save_pretrained(memory_path)

    report = module.inspect_inference_memory(memory_path)

    assert report["artifact_generation"] == "legacy_direct_map_only"
    assert report["context_encoding"] == "legacy_python_hash"
    assert report["health_checks"]["supports_fuzzy_retrieval"] is False
    assert report["health_checks"]["contexts_cover_patterns"] is False
    assert any("context_index" in item for item in report["recommendations"])
    assert any("legacy_python_hash" in item for item in report["recommendations"])
