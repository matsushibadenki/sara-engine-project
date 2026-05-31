import os
import sys
import msgpack

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path


def test_inference_save_and_load_memory_roundtrip():
    memory_path = model_path("tests", "sara_inference_memory.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (12345,): {7: 1.0, 8: 2.0},
        (67890,): {9: 3.5},
    }
    writer.context_index = {
        (12345,): (1, 2, 3),
        (67890,): (4, 5),
    }
    writer.retrieval_diagnostics = [
        {
            "source": "inference_direct_map",
            "content_preview": "1 2 3",
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
        "preferred_branch": "alternative",
    }
    writer.adaptation_state = {
        "adaptation_turns": 3,
        "next_step_requests": 2,
        "memory_requests": 0,
        "response_mode": "directive",
        "command_preference": True,
        "planning_confidence": 0.85,
        "memory_weight": 1.36,
        "fallback_relaxation": 0.08,
        "last_intent": "next_step",
    }
    writer.lif_network = None
    writer.save_pretrained(memory_path)

    reader = SaraInference.__new__(SaraInference)
    reader.model_path = memory_path
    reader.direct_map = {}
    reader.context_index = {}
    reader.retrieval_diagnostics = []
    reader.refractory_buffer = []
    reader.session_memory = {}
    reader.predictor_state = {}
    reader.adaptation_state = {}
    reader.lif_network = None
    reader._load_memory()

    assert reader.quantization_enabled is False
    assert reader.direct_map == writer.direct_map
    assert reader.context_index == writer.context_index
    assert reader.retrieval_diagnostics == writer.retrieval_diagnostics
    assert reader.session_memory == writer.session_memory
    assert reader.predictor_state == writer.predictor_state
    assert reader.adaptation_state == writer.adaptation_state


def test_inference_save_and_load_memory_roundtrip_with_turboquant():
    memory_path = model_path("tests", "sara_inference_memory_turboquant.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (12345,): {7: 1.0, 8: 2.0, 9: 3.0},
        (67890,): {5: 0.5, 6: 1.5},
    }
    writer.context_index = {
        (12345,): (1, 2, 3),
        (67890,): (4, 5),
    }
    writer.retrieval_diagnostics = [
        {
            "source": "inference_direct_map",
            "content_preview": "4 5",
            "base_score": 0.8,
            "stability_score": 0.7,
            "suffix_match": 0.5,
            "drift_penalty": 0.1,
            "metadata_keyword_overlap": 0.5,
            "context_match": True,
            "role_match": False,
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
        "preferred_branch": "alternative",
    }
    writer.adaptation_state = {
        "adaptation_turns": 2,
        "next_step_requests": 1,
        "memory_requests": 1,
        "response_mode": "guided",
        "command_preference": True,
        "planning_confidence": 0.7,
        "memory_weight": 1.3,
        "fallback_relaxation": 0.05,
        "last_intent": "memory",
    }
    writer.lif_network = None
    writer.quantization_enabled = True
    writer.save_pretrained(memory_path)

    reader = SaraInference.__new__(SaraInference)
    reader.model_path = memory_path
    reader.direct_map = {}
    reader.context_index = {}
    reader.retrieval_diagnostics = []
    reader.refractory_buffer = []
    reader.session_memory = {}
    reader.predictor_state = {}
    reader.adaptation_state = {}
    reader.lif_network = None
    reader._load_memory()

    assert reader.quantization_enabled is True
    assert reader.context_index == writer.context_index
    assert reader.retrieval_diagnostics == writer.retrieval_diagnostics
    assert reader.session_memory == writer.session_memory
    assert reader.predictor_state == writer.predictor_state
    assert reader.adaptation_state == writer.adaptation_state
    assert set(reader.direct_map.keys()) == set(writer.direct_map.keys())
    for key, values in writer.direct_map.items():
        restored = reader.direct_map[key]
        assert set(restored.keys()) == set(values.keys())
        for token_id, weight in values.items():
            assert abs(restored[token_id] - weight) < 1.5


def test_inference_save_persists_quantization_flag_in_payload():
    memory_path = model_path("tests", "sara_inference_quantization_flag.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (555,): {1: 1.0, 2: 2.0},
    }
    writer.context_index = {
        (555,): (9, 10),
    }
    writer.retrieval_diagnostics = []
    writer.refractory_buffer = []
    writer.session_memory = {}
    writer.predictor_state = {}
    writer.adaptation_state = {}
    writer.lif_network = None
    writer.quantization_enabled = True
    writer.save_pretrained(memory_path)

    with open(memory_path, "rb") as handle:
        payload = msgpack.unpack(handle, raw=False)

    assert payload["quantization_enabled"] is True
    assert payload["predictor_state"] == {}
    assert payload["adaptation_state"] == {}
