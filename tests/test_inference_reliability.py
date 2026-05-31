import os
import sys
import types
from typing import Any, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path


def test_inference_missing_memory_file_starts_empty():
    missing_path = model_path("tests", "missing_inference_memory.msgpack")
    if os.path.exists(missing_path):
        os.remove(missing_path)

    engine = SaraInference.__new__(SaraInference)
    engine.model_path = missing_path
    engine.direct_map = cast(dict[tuple[int, ...], dict[int, float]], {})
    engine.refractory_buffer = []
    engine.lif_network = None
    engine._load_memory()

    assert engine.direct_map == {}


def test_inference_learn_sequence_uses_tuple_keys_and_integer_tokens():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    engine.learn_sequence([10, 11, 12])

    assert engine.direct_map
    assert engine.context_index
    for key, values in engine.direct_map.items():
        assert isinstance(key, tuple)
        assert all(isinstance(item, int) for item in key)
        assert all(isinstance(token_id, int) for token_id in values.keys())


def test_inference_fuzzy_context_match_recovers_nearby_sequence():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    engine.learn_sequence([101, 102, 103, 999])

    matched_key = engine._find_best_matching_key([102, 103])

    assert matched_key is not None
    assert matched_key in engine.direct_map
    assert 999 in engine.direct_map[matched_key]


def test_inference_direct_context_alignment_penalizes_drift_and_rewards_suffix_match():
    engine = SaraInference.__new__(SaraInference)

    aligned = engine._score_direct_context_alignment([10, 20], [10, 20])
    drifted = engine._score_direct_context_alignment([10, 20], [10, 99, 20])

    assert aligned["suffix_match"] == 1.0
    assert drifted["suffix_match"] < 1.0
    assert aligned["drift_penalty"] < drifted["drift_penalty"]
    assert aligned["stability_score"] > drifted["stability_score"]


def test_inference_fuzzy_match_prefers_low_drift_candidate_from_context_index():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {
        (111,): {30: 2.0},
        (222,): {40: 2.0},
    }
    engine.context_index = {
        (111,): (10, 20),
        (222,): (10, 99, 20),
    }
    engine.refractory_buffer = []
    engine.lif_network = None

    matched_key = engine._find_best_matching_key([10, 20])

    assert matched_key == (111,)


def test_inference_records_diagnostics_in_common_format():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {
        (111,): {30: 2.0},
    }
    engine.context_index = {
        (111,): (10, 20),
    }
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.lif_network = None
    engine.tokenizer = types.SimpleNamespace(
        decode=lambda token_ids: " ".join(f"tok{token_id}" for token_id in token_ids)
    )

    matched_key = engine._find_best_matching_key([10, 20])
    diagnostics = engine.get_recent_retrieval_diagnostics()
    formatted = engine.format_recent_retrieval_diagnostics()

    assert matched_key == (111,)
    assert diagnostics
    assert diagnostics[0]["source"] == "inference_direct_map"
    assert diagnostics[0]["memory_hit"] == "retrieval"
    assert diagnostics[0]["content_preview"] == "tok10 tok20"
    stability_score = diagnostics[0]["stability_score"]
    assert isinstance(stability_score, (int, float))
    assert float(stability_score) >= 1.0
    assert "source=inference_direct_map" in formatted
    assert "memory=retrieval" in formatted


def test_inference_diagnostics_decode_token_preview_when_tokenizer_is_available():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {(111,): {30: 2.0}}
    engine.context_index = {(111,): (10, 20)}
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.lif_network = None
    engine.tokenizer = types.SimpleNamespace(
        decode=lambda token_ids: " ".join(f"tok{token_id}" for token_id in token_ids)
    )

    engine._find_best_matching_key([10, 20])

    diagnostics = engine.get_recent_retrieval_diagnostics()
    assert diagnostics[0]["content_preview"] == "tok10 tok20"


def test_inference_relevance_gate_replaces_irrelevant_identity_answer():
    engine = SaraInference.__new__(SaraInference)

    response = engine._apply_practical_relevance_gate(
        "You: Who are you?\nSARA:",
        '"The model parameters ... are converted ahead of time and stored in INT8 form."',
    )

    assert response == "I am SARA, a CPU-first spiking neural network assistant."


def test_inference_relevance_gate_replaces_irrelevant_greeting_answer():
    engine = SaraInference.__new__(SaraInference)

    response = engine._apply_practical_relevance_gate(
        "You: Hello\nSARA:",
        ". 2025年11月25日閲覧。",
    )

    assert response == "Hello. I am SARA. How can I help you?"


def test_inference_relevance_gate_preserves_relevant_response():
    engine = SaraInference.__new__(SaraInference)

    response = engine._apply_practical_relevance_gate(
        "You: Who are you?\nSARA:",
        "I am SARA, a CPU-first spiking neural network assistant.",
    )

    assert response == "I am SARA, a CPU-first spiking neural network assistant."


def test_inference_selects_more_relevant_opening_candidate_before_fallback():
    engine = cast(Any, SaraInference.__new__(SaraInference))
    engine.refractory_buffer = []

    engine._prompt_needs_relevance_assist = lambda _prompt: True
    engine._rank_next_token_candidates = lambda _sdr_k, _penalty, refractory_buffer=None: [
        (101, 5.0),
        (202, 4.0),
    ]
    previews = {
        101: ". 2025年11月25日閲覧。",
        202: "I am SARA, a CPU-first spiking neural network assistant.",
    }
    engine._preview_response_from_candidate = (
        lambda current_tokens, candidate_id, max_new_tokens, stop_conditions, refractory_penalty: previews[candidate_id]
    )

    selected = engine._select_best_opening_candidate(
        prompt="You: Who are you?\nSARA:",
        current_tokens=[1, 2, 3],
        sdr_k=(111,),
        sampled_next_id=101,
        max_new_tokens=12,
        stop_conditions=["."],
        refractory_penalty=1.2,
    )

    assert selected == 202


def test_inference_keeps_original_opening_when_no_better_candidate_is_found():
    engine = cast(Any, SaraInference.__new__(SaraInference))
    engine.refractory_buffer = []

    engine._prompt_needs_relevance_assist = lambda _prompt: True
    engine._rank_next_token_candidates = lambda _sdr_k, _penalty, refractory_buffer=None: [
        (101, 5.0),
        (202, 4.0),
    ]
    engine._preview_response_from_candidate = (
        lambda current_tokens, candidate_id, max_new_tokens, stop_conditions, refractory_penalty: "Unrelated fragment."
    )

    selected = engine._select_best_opening_candidate(
        prompt="You: Who are you?\nSARA:",
        current_tokens=[1, 2, 3],
        sdr_k=(111,),
        sampled_next_id=101,
        max_new_tokens=12,
        stop_conditions=["."],
        refractory_penalty=1.2,
    )

    assert selected == 101


def test_inference_fast_intent_response_handles_memory_question():
    engine = SaraInference.__new__(SaraInference)

    response = engine._fast_intent_response("You: Do you remember me?\nSARA:")

    assert response is not None
    assert "current conversation" in response


def test_inference_fast_intent_response_handles_location_disclosure():
    engine = SaraInference.__new__(SaraInference)

    response = engine._fast_intent_response("You: I live in Tokyo\nSARA:")

    assert response == "Thank you for telling me. In this conversation, I understand that you live in Tokyo."


def test_inference_fast_intent_response_handles_japanese_capability_question():
    engine = SaraInference.__new__(SaraInference)

    response = engine._fast_intent_response("You: 日本語はわかりますか？\nSARA:")

    assert response is not None
    assert "Japanese and English" in response


def test_inference_session_memory_recalls_location_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: I live in Tokyo\nSARA:")
    second_response = engine.generate("You: Where do I live?\nSARA:")

    assert "Tokyo" in first_response
    assert second_response == "In this conversation, you told me that you live in Tokyo."


def test_inference_session_memory_personalizes_remember_me_response():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {"location": "Tokyo"}
    engine.lif_network = None

    response = engine.generate("You: Do you remember me?\nSARA:")

    assert response == "Yes. In this conversation, I remember that you live in Tokyo."


def test_inference_session_memory_remember_me_prioritizes_core_facts():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {
        "name": "Alex",
        "location": "Tokyo",
        "task": "the sara engine",
        "goal": "finish this project",
        "preference": "sushi",
    }
    engine.lif_network = None

    response = engine.generate("You: Do you remember me?\nSARA:")

    assert "your name is Alex" in response
    assert "you live in Tokyo" in response
    assert "you are working on the sara engine" in response
    assert "you want to finish this project" in response
    assert "you like sushi" not in response


def test_inference_session_memory_recalls_name_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: My name is Alex\nSARA:")
    second_response = engine.generate("You: What is my name?\nSARA:")

    assert "Alex" in first_response
    assert second_response == "In this conversation, you told me that your name is Alex."


def test_inference_session_memory_recalls_goal_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: I want to finish this project\nSARA:")
    second_response = engine.generate("You: What is my goal?\nSARA:")

    assert "goal" in first_response.lower()
    assert "you want to finish this project" in second_response
    assert "smaller steps" in second_response


def test_inference_session_memory_recalls_task_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: I am working on the sara engine\nSARA:")
    second_response = engine.generate("You: What am I working on?\nSARA:")

    assert "current task" in first_response.lower()
    assert "you are working on the sara engine" in second_response
    assert "next step" in second_response


def test_inference_session_memory_links_goal_and_task_in_suggestions():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {
        "goal": "finish this project",
        "task": "the sara engine",
    }
    engine.lif_network = None

    goal_response = engine.generate("You: What is my goal?\nSARA:")
    task_response = engine.generate("You: What am I working on?\nSARA:")

    assert "concrete change" in goal_response
    assert "the sara engine" in goal_response
    assert "move you toward finish this project" in task_response


def test_inference_session_memory_suggests_next_step_from_goal_and_task():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {
        "goal": "finish this project",
        "task": "the sara engine",
    }
    engine.lif_network = None

    response = engine.generate("You: What should I do next?\nSARA:")

    assert "Step 1:" in response
    assert "Step 2:" in response
    assert "concrete change" in response
    assert "the sara engine" in response
    assert "finish this project" in response


def test_inference_session_memory_suggests_next_step_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {
        "goal": "このプロジェクトを終える",
        "task": "SARA Engine の改善",
    }
    engine.lif_network = None

    response = engine.generate("You: 次に何をすればいい？\nSARA:")

    assert "Step 1:" in response
    assert "Step 2:" in response
    assert "変更点" in response
    assert "「SARA Engine の改善」" in response
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    assert diagnostics[0]["memory_hit"] == "session_memory"


def test_inference_formats_ascii_session_memory_for_japanese_responses():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "the sara engine", "goal": "SARA Engine の改善"}

    assert engine._format_session_value("task", language="ja") == "英語の「the sara engine」"
    assert engine._format_session_value("goal", language="ja") == "「SARA Engine の改善」"


def test_inference_formats_japanese_goal_and_task_labels_for_inline_responses():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "テスト確認", "goal": "品質を上げる"}

    assert engine._format_session_label_ja("task") == "「テスト確認」"
    assert engine._format_session_label_ja("goal") == "「品質を上げる」"


def test_inference_builds_future_state_label_from_goal_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"goal": "finish this project", "task": "the sara engine"}

    assert engine._build_future_state_label() == "finish this project"


def test_inference_builds_future_state_label_from_goal_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"goal": "品質を上げる", "task": "テスト確認"}

    assert engine._build_future_state_label(language="ja") == "「品質を上げる」"


def test_inference_builds_future_state_label_from_task_when_goal_is_missing():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "the sara engine"}

    assert engine._build_future_state_label() == "making progress on the sara engine"


def test_inference_predicts_future_state_transition_for_english_engine_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "the sara engine", "goal": "finish this project"}

    transition = engine._predict_future_state_transition()

    assert transition["action"] == "choose one concrete change to make in the sara engine"
    assert transition["target_state"] == "finish this project"


def test_inference_predicts_lightweight_future_state_for_release_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.future_state_runtime_state = {}

    prediction = engine._predict_lightweight_future_state()

    assert prediction["category"] == "release"
    assert prediction["action"] == "choose one release check to complete for pytest release checks"
    assert prediction["target_state"] == "ship the release"
    assert prediction["command"] == "python scripts/eval/release_soak.py --include-accuracy"
    assert prediction["confidence"] == 1.0


def test_inference_tracks_future_state_runtime_snapshot():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}

    engine._refresh_predictor_state()
    snapshot = engine._get_future_state_runtime_snapshot()

    assert snapshot["transition_count"] >= 1
    assert snapshot["last_category"] == "release"
    assert snapshot["last_target_state"] == "ship the release"
    assert snapshot["last_branch_count"] == 3
    assert snapshot["last_preferred_branch"] == "alternative"
    assert snapshot["last_branch_labels"] == ["primary", "alternative", "secondary"]
    assert snapshot["last_transition_operator"] == "release.check"
    assert snapshot["last_verified_operator"] == "release.check"
    assert snapshot["last_operator_match"] is True
    assert snapshot["last_speculative_acceptance"] is True
    assert snapshot["last_rollback_observable"] is True
    assert snapshot["last_counterfactual_viable"] is True
    assert snapshot["operator_consistency_ratio"] == 1.0
    assert snapshot["speculative_acceptance_ratio"] == 1.0
    assert snapshot["speculative_rollback_ratio"] == 1.0
    assert snapshot["counterfactual_viability_ratio"] == 1.0


def test_inference_refresh_predictor_state_populates_branch_candidates():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}

    engine._refresh_predictor_state()

    branch_candidates = engine.predictor_state["branch_candidates"]
    ranked_branch_candidates = engine.predictor_state["ranked_branch_candidates"]

    assert isinstance(branch_candidates, list)
    assert len(branch_candidates) == 3
    assert [candidate["label"] for candidate in branch_candidates] == ["primary", "alternative", "secondary"]
    assert isinstance(ranked_branch_candidates, list)
    assert ranked_branch_candidates[0]["label"] == "alternative"
    assert engine.predictor_state["preferred_branch"] == "alternative"
    assert engine.predictor_state["transition_operator"] == "release.check"
    assert engine.predictor_state["alternative_transition_operator"] == "release.risk_prioritize"
    assert engine.predictor_state["secondary_alternative_transition_operator"] == "release.rollback_guard"
    assert engine.predictor_state["speculative_trace"]["operator_match"] is True
    assert engine.predictor_state["speculative_trace"]["draft_verify_accepted"] is True
    assert isinstance(engine.predictor_state["refinement_trace"], dict)
    assert engine.predictor_state["refinement_trace"]["loop_count"] >= 1
    assert engine.predictor_state["refinement_trace"]["selected_branch_after"] == "alternative"
    depth_budget = engine.predictor_state["refinement_trace"]["adaptive_depth_budget"]
    assert depth_budget["base_loop_budget"] == 1
    assert depth_budget["allocated_loop_budget"] in {1, 2}
    assert depth_budget["allocated_loop_budget"] <= depth_budget["max_loop_budget"]
    assert engine.predictor_state["fluid_trace"]["bounded"] is True
    assert engine.predictor_state["fluid_trace"]["active_columns"] >= 1
    assert engine.predictor_state["fluid_trace"]["total_spikes"] > 0


def test_inference_describes_future_state_shift_after_goal_change():
    engine = SaraInference.__new__(SaraInference)
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}

    engine._refresh_predictor_state()
    engine.session_memory = {"task": "research neuromorphic hardware", "goal": "improve the design"}
    engine._refresh_predictor_state()

    snapshot = engine._get_future_state_runtime_snapshot()
    summary = engine._describe_future_state_shift()

    assert snapshot["shift_count"] >= 1
    assert snapshot["last_shift_from"] == "ship the release"
    assert snapshot["last_shift_to"] == "improve the design"
    assert "shifted from ship the release to improve the design" in summary


def test_inference_next_step_mentions_shift_after_goal_change():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {"goal": "ship the release", "task": "pytest release checks"}

    engine._refresh_predictor_state()
    engine.session_memory = {"goal": "improve the design", "task": "research neuromorphic hardware"}
    engine._refresh_predictor_state()
    response = engine.generate("You: What should I do next?\nSARA:")

    assert "research neuromorphic hardware" in response
    assert "improve the design" in response
    assert "shifted from ship the release to improve the design" in response


def test_inference_adaptation_state_becomes_directive_after_repeated_next_step_requests():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: I want to finish this project\nSARA:")
    engine.generate("You: I am working on the sara engine\nSARA:")
    first_response = engine.generate("You: What should I do next?\nSARA:")
    second_response = engine.generate("You: What should I do next?\nSARA:")

    assert "Do this now:" not in first_response
    assert "Do this now:" in second_response
    assert engine.adaptation_state["response_mode"] == "directive"
    assert engine.adaptation_state["next_step_requests"] >= 2
    assert engine.adaptation_state["memory_weight"] > 1.0
    assert engine.adaptation_state["fallback_relaxation"] > 0.0


def test_inference_adaptation_state_tracks_memory_requests():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {"name": "Alex"}

    response = engine.generate("You: What is my name?\nSARA:")

    assert "Alex" in response
    assert engine.adaptation_state["memory_requests"] >= 1
    assert engine.adaptation_state["last_intent"] == "memory"


def test_inference_response_relevance_score_gets_adaptation_bonus_for_memory_reply():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {"goal": "finish this project", "task": "the sara engine"}
    engine.adaptation_state = {
        "adaptation_turns": 4,
        "next_step_requests": 2,
        "memory_requests": 1,
        "response_mode": "directive",
        "command_preference": True,
        "planning_confidence": 0.85,
        "memory_weight": 1.48,
        "fallback_relaxation": 0.10,
        "last_intent": "next_step",
    }

    prompt = "You: What should I do next?\nSARA:"
    response = "Step 1: choose one concrete change to make in the sara engine. Step 2: finish it and check that it moves you toward finish this project."
    boosted_score = engine._response_relevance_score(prompt, response)
    engine.adaptation_state = {}
    baseline_score = engine._response_relevance_score(prompt, response)

    assert boosted_score > baseline_score


def test_inference_practical_relevance_gate_keeps_memory_like_response_when_adaptation_is_ready():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {"goal": "finish this project", "task": "the sara engine"}
    engine.adaptation_state = {
        "adaptation_turns": 4,
        "next_step_requests": 2,
        "memory_requests": 1,
        "response_mode": "directive",
        "command_preference": True,
        "planning_confidence": 0.85,
        "memory_weight": 1.48,
        "fallback_relaxation": 0.10,
        "last_intent": "next_step",
    }

    prompt = "You: What should I do next?\nSARA:"
    response = "Step 1: choose one concrete change to make in the sara engine. Step 2: finish it and check that it moves you toward finish this project."

    assert engine._apply_practical_relevance_gate(prompt, response) == response


def test_inference_predicts_future_state_transition_for_japanese_release_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "リリース準備", "goal": "公開する"}

    transition = engine._predict_future_state_transition(language="ja")

    assert transition["action"] == "「リリース準備」で最初に確認するリリース確認項目を1つ決める"
    assert transition["target_state"] == "「公開する」"


def test_inference_predicts_operational_command_hint_for_release_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}

    assert engine._predict_operational_command_hint() == "python scripts/eval/release_soak.py --include-accuracy"


def test_inference_predicts_operational_command_hint_for_research_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "research neuromorphic hardware", "goal": "improve the design"}

    assert engine._predict_operational_command_hint() == "python scripts/sara_cli.py db-list --category research --format json"


def test_inference_builds_alternative_next_step_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.predictor_state = {
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
    }

    response = engine._build_alternative_next_step_response()

    assert "alternative next step" in response
    assert "prioritize the highest-risk release check in pytest release checks" in response
    assert "ship the release" in response
    assert "release_soak.py --include-accuracy" in response


def test_inference_builds_alternative_next_step_response_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.predictor_state = {
        "alternative_action": "「リリース準備」で影響が最も大きいリリース確認項目を1つ先に確認する",
        "alternative_target_state": "「公開」",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
    }

    response = engine._build_alternative_next_step_response(language="ja")

    assert "別案としては" in response
    assert "リリース確認項目" in response
    assert "「公開」" in response
    assert "release_soak.py --include-accuracy" in response


def test_inference_builds_secondary_alternative_next_step_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.predictor_state = {
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
    }

    response = engine._build_secondary_alternative_next_step_response()

    assert "second alternative next step" in response
    assert "rollback condition" in response
    assert "release_gate.py" in response


def test_inference_fast_intent_returns_secondary_alternative_next_step_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: I am working on pytest release checks\nSARA:")
    response = engine.generate("You: What is a second alternative next step?\nSARA:")

    assert "second alternative next step" in response
    assert "rollback condition" in response


def test_inference_fast_intent_returns_alternative_next_step_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: I am working on pytest release checks\nSARA:")
    response = engine.generate("You: What else could I do next?\nSARA:")

    assert "alternative next step" in response
    assert "highest-risk release check" in response


def test_inference_fast_intent_returns_alternative_next_step_in_japanese_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 別の次の一歩は何？\nSARA:")

    assert "別案としては" in response
    assert "リリース確認項目" in response


def test_inference_fast_intent_returns_secondary_alternative_next_step_in_japanese_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: もう一つの別案は何？\nSARA:")

    assert "もう一つの別案としては" in response
    assert "ロールバック条件" in response


def test_inference_builds_next_step_comparison_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_comparison_response()

    assert response.startswith("Primary:")
    assert "\nAlternative:" in response
    assert "pytest release checks" in response
    assert "highest-risk release check" in response


def test_inference_builds_next_step_comparison_response_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "リリース準備", "goal": "公開"}
    engine.predictor_state = {
        "alternative_action": "「リリース準備」で影響が最も大きいリリース確認項目を1つ先に確認する",
        "alternative_target_state": "「公開」",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_comparison_response(language="ja")

    assert response.startswith("主案:")
    assert "\n別案:" in response
    assert "リリース確認項目" in response


def test_inference_builds_next_step_choice_response_preferring_alternative_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "category": "release",
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_choice_response()

    assert response.startswith("I would start with the alternative plan:")
    assert "highest-risk release check" in response
    assert "Reason:" in response


def test_inference_builds_next_step_choice_response_preferring_primary_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "開発作業", "goal": "改善する"}
    engine.predictor_state = {
        "category": "development",
        "alternative_action": "「開発作業」で別の小さな進め方を1つ比較する",
        "alternative_target_state": "「改善する」",
        "alternative_command": "pytest -q",
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_choice_response(language="ja")

    assert response.startswith("まずは主案から進めるのがよいです:")
    assert "開発作業" in response
    assert "理由:" in response


def test_inference_builds_next_step_options_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_options_response()

    assert response.startswith("Primary:")
    assert "\nAlternative:" in response
    assert "\nAdditional:" in response


def test_inference_fast_intent_returns_next_step_options_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 次の一歩の候補を見せて\nSARA:")

    assert response.startswith("主案:")
    assert "\n別案:" in response
    assert "\n追加案:" in response


def test_inference_builds_ranked_next_step_options_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "confidence": 1.0,
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
        "secondary_alternative_confidence": 0.4,
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_ranked_next_step_options_response()

    assert response.startswith("1. Alternative:")
    assert "\n2. Primary:" in response
    assert "\n3. Additional:" in response


def test_inference_ranks_future_state_branch_candidates_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "category": "release",
        "confidence": 1.0,
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
        "secondary_alternative_confidence": 0.4,
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    ranked = engine._rank_future_state_branch_candidates()

    assert [item["label"] for item in ranked] == ["alternative", "primary", "secondary"]


def test_inference_simulates_future_state_branch_candidates_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "category": "release",
        "confidence": 1.0,
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
        "secondary_alternative_confidence": 0.4,
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    simulated = engine._simulate_future_state_branch_candidates()

    assert simulated[0]["label"] == "alternative"
    assert simulated[0]["simulation_score"] >= simulated[1]["simulation_score"]
    assert "progress_score" in simulated[0]
    assert "risk_reduction_score" in simulated[0]
    assert "reversibility_score" in simulated[0]


def test_inference_builds_next_step_simulation_response_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "category": "release",
        "confidence": 1.0,
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
        "secondary_alternative_confidence": 0.4,
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_simulation_response()

    assert response.startswith("Lightweight simulation:")
    assert "- Alternative: score=" in response
    assert "progress=" in response
    assert "risk=" in response
    assert "reversible=" in response


def test_inference_fast_intent_returns_ranked_next_step_options_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 候補に順位を付けて\nSARA:")

    assert response.startswith("1位 (別案):")
    assert "\n2位 (主案):" in response


def test_inference_fast_intent_returns_next_step_simulation_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 次の一歩をシミュレーションして\nSARA:")

    assert response.startswith("軽量シミュレーション:")
    assert "別案" in response


def test_inference_builds_next_step_decision_brief_in_english():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}
    engine.predictor_state = {
        "category": "release",
        "confidence": 1.0,
        "alternative_action": "prioritize the highest-risk release check in pytest release checks",
        "alternative_target_state": "ship the release",
        "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
        "alternative_confidence": 0.5,
        "secondary_alternative_action": "check one rollback condition first in pytest release checks",
        "secondary_alternative_target_state": "ship the release",
        "secondary_alternative_command": "python scripts/eval/release_gate.py",
        "secondary_alternative_confidence": 0.4,
    }
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}

    response = engine._build_next_step_decision_brief()

    assert response.startswith("Decision brief:")
    assert "I would start with the alternative plan:" in response
    assert "1. Alternative:" in response


def test_inference_fast_intent_returns_next_step_decision_brief_in_japanese():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 次の一歩を要約して\nSARA:")

    assert response.startswith("判断メモ:")
    assert "まずは別案から進めるのがよいです:" in response
    assert "1位 (別案):" in response


def test_inference_fast_intent_returns_next_step_comparison_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: I am working on pytest release checks\nSARA:")
    response = engine.generate("You: Compare the next steps.\nSARA:")

    assert response.startswith("Primary:")
    assert "\nAlternative:" in response


def test_inference_fast_intent_returns_next_step_choice_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: I want to ship the release\nSARA:")
    engine.generate("You: I am working on pytest release checks\nSARA:")
    response = engine.generate("You: Which next step should I choose?\nSARA:")

    assert response.startswith("I would start with the alternative plan:")
    assert "highest-risk release check" in response


def test_inference_fast_intent_returns_next_step_comparison_in_japanese_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: 次の一歩を比較して\nSARA:")

    assert response.startswith("主案:")
    assert "\n別案:" in response


def test_inference_fast_intent_returns_next_step_choice_in_japanese_when_requested():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.predictor_state = {}
    engine.adaptation_state = {}
    engine.future_state_runtime_state = {}
    engine.lif_network = None
    engine.session_memory = {}

    engine.generate("You: 私は公開したいです\nSARA:")
    engine.generate("You: 私はリリース準備をしています\nSARA:")
    response = engine.generate("You: どちらを選ぶべき？\nSARA:")

    assert response.startswith("まずは別案から進めるのがよいです:")
    assert "リリース確認項目" in response


def test_inference_task_hint_specializes_writing_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "blog article", "goal": "publish a draft"}

    response = engine._build_next_step_response()

    assert "Step 1:" in response
    assert "first heading or paragraph" in response
    assert "blog article" in response


def test_inference_task_hint_specializes_illustration_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "character illustration", "goal": "finish the concept"}

    response = engine._build_next_step_response()

    assert "Step 1:" in response
    assert "rough sketch" in response
    assert "character illustration" in response


def test_inference_task_hint_specializes_testing_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "pytest release checks", "goal": "ship the release"}

    response = engine._build_next_step_response()

    assert "Step 1:" in response
    assert "verify" in response
    assert "pytest release checks" in response
    assert "Suggested command: `python scripts/eval/release_soak.py --include-accuracy`" in response


def test_inference_task_hint_specializes_japanese_testing_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "テスト確認", "goal": "品質を上げる"}

    response = engine._build_next_step_response(language="ja")

    assert "Step 1:" in response
    assert "確認する1つのケース" in response
    assert "「テスト確認」" in response
    assert "「品質を上げる」" in response


def test_inference_task_hint_specializes_debugging_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "debug memory regression", "goal": "stabilize the runtime"}

    response = engine._build_next_step_response()

    assert "Step 1:" in response
    assert "reproducible failure" in response
    assert "debug memory regression" in response


def test_inference_task_hint_specializes_japanese_debugging_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "デバッグ作業", "goal": "安定化する"}

    response = engine._build_next_step_response(language="ja")

    assert "Step 1:" in response
    assert "再現できる不具合" in response
    assert "「デバッグ作業」" in response
    assert "「安定化する」" in response


def test_inference_task_hint_specializes_research_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "research neuromorphic hardware", "goal": "improve the design"}

    response = engine._build_next_step_response()

    assert "Step 1:" in response
    assert "one question" in response
    assert "research neuromorphic hardware" in response


def test_inference_task_hint_specializes_japanese_research_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "論文調査", "goal": "設計を改善する"}

    response = engine._build_next_step_response(language="ja")

    assert "Step 1:" in response
    assert "答えたい問いを1つに絞ります" in response
    assert "「論文調査」" in response
    assert "「設計を改善する」" in response


def test_inference_task_hint_specializes_japanese_release_task():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "リリース準備", "goal": "公開する"}

    response = engine._build_next_step_response(language="ja")

    assert "Step 1:" in response
    assert "リリース確認項目" in response
    assert "「リリース準備」" in response
    assert "「公開する」" in response
    assert "提案コマンド: `python scripts/eval/release_soak.py --include-accuracy`" in response


def test_inference_future_state_label_mentions_progress_when_only_task_exists():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "the sara engine"}

    response = engine._build_future_state_label()

    assert response == "making progress on the sara engine"


def test_inference_next_step_uses_predicted_transition_when_only_task_exists():
    engine = SaraInference.__new__(SaraInference)
    engine.session_memory = {"task": "misc runtime cleanup"}

    response = engine._build_next_step_response()

    assert "The next step is to choose one small unfinished action for misc runtime cleanup." in response
    assert "future state of making progress on misc runtime cleanup" in response


def test_inference_japanese_goal_recall_uses_inline_label_format():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {"goal": "品質を上げる", "task": "テスト確認"}
    engine.lif_network = None

    response = engine.generate("You: 目標は何？\nSARA:")

    assert "「品質を上げる」" in response
    assert "「テスト確認」" in response


def test_inference_japanese_task_recall_uses_inline_label_format():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {"goal": "公開する", "task": "リリース準備"}
    engine.lif_network = None

    response = engine.generate("You: 今の作業は何？\nSARA:")

    assert "「リリース準備」" in response
    assert "「公開する」" in response


def test_inference_session_memory_recalls_preference_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: I like sushi\nSARA:")
    second_response = engine.generate("You: What do I like?\nSARA:")

    assert "context" in first_response.lower() or "thank you" in first_response.lower()
    assert second_response == "In this conversation, you told me that you like sushi."


def test_inference_session_memory_recalls_japanese_location_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: 私は東京に住んでいます\nSARA:")
    second_response = engine.generate("You: どこに住んでいますか？\nSARA:")

    assert "文脈" in first_response
    assert second_response == "この会話では、あなたは東京に住んでいると教えてくれました。"


def test_inference_session_memory_recalls_japanese_name_within_conversation():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    first_response = engine.generate("You: 私の名前はアレックスです\nSARA:")
    second_response = engine.generate("You: 名前は何ですか？\nSARA:")

    assert "アレックス" in first_response
    assert second_response == "この会話では、あなたの名前はアレックスだと教えてくれました。"


def test_inference_session_memory_does_not_treat_preference_as_profession():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    engine.generate("You: 私は寿司が好きです\nSARA:")

    assert engine.session_memory["preference"] == "寿司"
    assert "profession" not in engine.session_memory


def test_inference_sanitizes_legacy_session_memory_profession_on_runtime_init():
    engine = SaraInference.__new__(SaraInference)
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.session_memory = {"profession": "寿司が好き", "preference": "寿司"}
    engine.context_encoding = "stable_v1"

    engine._ensure_runtime_state()

    assert engine.session_memory["preference"] == "寿司"
    assert "profession" not in engine.session_memory


def test_inference_generate_returns_fast_intent_response_without_tokenizer_work():
    engine = SaraInference.__new__(SaraInference)
    engine.retrieval_diagnostics = []
    engine.session_memory = {}
    engine.tokenizer = types.SimpleNamespace(
        __call__=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tokenizer should not run"))
    )

    response = engine.generate("You: Who are you?\nSARA:")

    assert response == "I am SARA, a CPU-first spiking neural network assistant."
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    assert diagnostics[0]["source"] == "inference_fast_path"
    assert diagnostics[0]["memory_hit"] == "fast_path"


def test_inference_session_memory_diagnostic_marks_memory_hit():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {"location": "Tokyo"}
    engine.lif_network = None

    response = engine.generate("You: Where do I live?\nSARA:")

    assert "Tokyo" in response
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    assert diagnostics[0]["source"] == "inference_fast_path"
    assert diagnostics[0]["memory_hit"] == "session_memory"


def test_inference_session_memory_disclosure_marks_fast_path_hit():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.lif_network = None

    response = engine.generate("You: I want to finish this project\nSARA:")

    assert "goal as context" in response
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    assert diagnostics[0]["memory_hit"] == "fast_path"


def test_inference_fast_intent_path_skips_session_update_work():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.fast_intent_cache = {}
    engine.lif_network = None

    def _blocked_update(_: str) -> None:
        raise AssertionError("session update should not run on fast intent path")

    engine._update_session_memory = _blocked_update

    response = engine.generate("You: Who are you?\nSARA:")

    assert response == "I am SARA, a CPU-first spiking neural network assistant."


def test_inference_ultra_fast_greeting_skips_session_update_work():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.retrieval_diagnostics = []
    engine.direct_map = {}
    engine.context_index = {}
    engine.session_memory = {}
    engine.fast_intent_cache = {}
    engine.lif_network = None

    def _blocked_update(_: str) -> None:
        raise AssertionError("session update should not run on ultra-fast greeting path")

    engine._update_session_memory = _blocked_update

    response = engine.generate("You: hello\nSARA:")

    assert "Hello. I am SARA." in response
