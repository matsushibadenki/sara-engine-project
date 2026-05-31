from sara_engine.utils.project_paths import model_path, workspace_path
from sara_engine.inference import SaraInference
from sara_engine.agent.sara_agent import SaraAgent
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


def _load_release_soak_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",
                     "scripts", "eval", "release_soak.py")
    )
    spec = importlib.util.spec_from_file_location(
        "release_soak_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_soak_repair_log_roundtrip_json_entries():
    module = _load_release_soak_module()
    path = workspace_path("tests", "release_soak_repair_log_roundtrip.json")
    entries = []
    module.append_repair_execution_entry(
        entries,
        command="python scripts/eval/release_soak.py --profile release --include-accuracy",
        status="success",
        covered_checks=["soak.duration_seconds"],
        title="manual",
        source="test",
    )
    saved = module.save_repair_execution_log(path, entries)
    loaded = module.load_repair_execution_log(saved)

    assert len(loaded) == 1
    assert loaded[0]["status"] == "success"
    assert "soak.duration_seconds" in loaded[0]["covered_checks"]


def test_release_soak_appends_iterative_next_actions_without_duplicates():
    module = _load_release_soak_module()
    entries = []
    iterative = {
        "next_actions": [
            {
                "title": "Embed Accuracy Into Soak Report",
                "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
                "affected_checks": ["release_gate.embedded_accuracy_present"],
            }
        ]
    }

    first = module.append_iterative_next_actions_to_repair_log(
        entries, iterative)
    second = module.append_iterative_next_actions_to_repair_log(
        entries, iterative)

    assert first == 1
    assert second == 0
    assert len(entries) == 1
    assert entries[0]["status"] == "pending"
    assert entries[0]["source"] == "iterative_next_action"


def test_release_soak_finalize_pending_repair_entries_updates_existing_pending():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "status": "pending",
            "covered_checks": ["release_gate.embedded_accuracy_present"],
            "title": "pending_step",
            "source": "iterative_next_action",
            "timestamp": 1.0,
        }
    ]

    updated = module.finalize_pending_repair_entries(
        entries,
        command="python scripts/eval/release_soak.py --profile release --include-accuracy",
        status="success",
        covered_checks=["release_gate.accuracy_required"],
        title="manual_repair_completion",
        source="manual_cli_completion",
    )

    assert updated == 1
    assert entries[0]["status"] == "success"
    assert "release_gate.embedded_accuracy_present" in entries[0]["covered_checks"]
    assert "release_gate.accuracy_required" in entries[0]["covered_checks"]
    assert entries[0]["source"] == "manual_cli_completion"
    assert "resolved_timestamp" in entries[0]


def test_release_soak_append_entry_rolls_forward_pending_before_appending():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "covered_checks": ["release_gate.errors"],
            "title": "pending_gate",
            "source": "iterative_next_action",
            "timestamp": 1.0,
        }
    ]

    ok = module.append_repair_execution_entry(
        entries,
        command="python scripts/eval/release_gate.py",
        status="failed",
        covered_checks=["release_gate.errors"],
        title="manual_repair_completion",
        source="manual_cli_completion",
    )

    assert ok is True
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert entries[0]["source"] == "manual_cli_completion"


def test_release_soak_expires_pending_entries_with_ttl():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "status": "pending",
            "covered_checks": ["release_gate.embedded_accuracy_present"],
            "timestamp": 10.0,
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "covered_checks": ["release_gate.errors"],
            "timestamp": 95.0,
        },
    ]

    expired = module.expire_pending_repair_entries(
        entries,
        ttl_seconds=30.0,
        now_timestamp=100.0,
    )

    assert expired == 1
    assert entries[0]["status"] == "timeout"
    assert entries[0]["source"] == "pending_ttl_timeout"
    assert entries[1]["status"] == "pending"


def test_release_soak_builds_retry_queue_from_timeout_and_failed():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "status": "timeout",
            "covered_checks": ["release_gate.embedded_accuracy_present"],
            "timestamp": 10.0,
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "failed",
            "covered_checks": ["release_gate.errors"],
            "timestamp": 20.0,
        },
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "status": "success",
            "covered_checks": ["stage_b.minimum_checks"],
            "timestamp": 30.0,
        },
    ]

    queue = module.build_retry_queue_from_repair_log(entries, max_attempts=2)

    assert len(queue) == 2
    assert any(item["reason"] == "timeout" for item in queue)
    assert any(item["reason"] == "failed" for item in queue)
    assert all(item["next_attempt"] == 2 for item in queue)


def test_release_soak_retry_queue_excludes_commands_over_attempt_budget():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "failed",
            "covered_checks": ["release_gate.errors"],
            "timestamp": 10.0,
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "timeout",
            "covered_checks": ["release_gate.errors"],
            "timestamp": 20.0,
        },
    ]

    queue = module.build_retry_queue_from_repair_log(entries, max_attempts=2)

    assert queue == []


def test_release_soak_retry_queue_respects_cooldown_window():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "failed",
            "covered_checks": ["release_gate.errors"],
            "timestamp": 99.0,
        },
    ]

    queue = module.build_retry_queue_from_repair_log(
        entries,
        max_attempts=3,
        cooldown_seconds=10.0,
        now_timestamp=100.0,
    )
    blocked = module.build_retry_cooldown_blocked_from_repair_log(
        entries,
        max_attempts=3,
        cooldown_seconds=10.0,
        now_timestamp=100.0,
    )

    assert queue == []
    assert len(blocked) == 1
    assert blocked[0]["command"] == "python scripts/eval/release_gate.py"
    assert blocked[0]["next_attempt"] == 2
    assert blocked[0]["cooldown_remaining_seconds"] > 0.0


def test_release_soak_prioritizes_retry_queue_by_overlap_and_reason():
    module = _load_release_soak_module()
    queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "reason": "failed",
            "covered_checks": ["soak.duration_seconds"],
            "attempts_used": 1,
            "max_attempts": 2,
            "next_attempt": 2,
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "reason": "timeout",
            "covered_checks": ["release_gate.errors"],
            "attempts_used": 1,
            "max_attempts": 2,
            "next_attempt": 2,
        },
    ]
    iterative = {
        "remaining_checks": ["soak.duration_seconds"],
    }

    prioritized = module.prioritize_retry_queue(
        queue, iterative_plan=iterative)

    assert len(prioritized) == 2
    assert prioritized[0]["command"] == "python scripts/eval/release_soak.py --profile extended --include-accuracy"
    assert prioritized[0]["priority_tier"] in {"high", "medium"}
    assert float(prioritized[0]["priority_score"]) >= float(
        prioritized[1]["priority_score"])


def test_release_soak_selects_dispatch_batch_with_priority_threshold():
    module = _load_release_soak_module()
    prioritized_queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "priority_tier": "high",
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "priority_tier": "medium",
        },
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "priority_tier": "low",
        },
    ]

    batch = module.select_retry_dispatch_batch(
        prioritized_queue,
        max_dispatch=2,
        min_priority_tier="medium",
    )

    assert batch["min_priority_tier"] == "medium"
    assert batch["eligible_count"] == 2
    assert batch["selected_count"] == 2
    assert [item["command"] for item in batch["selected"]] == [
        "python scripts/eval/release_soak.py --profile extended --include-accuracy",
        "python scripts/eval/release_gate.py",
    ]
    assert batch["skipped_low_priority_count"] == 1
    assert batch["skipped_low_priority_commands"] == [
        "python scripts/eval/phase3_accuracy_suite.py"
    ]


def test_release_soak_selects_dispatch_batch_with_check_diversification():
    module = _load_release_soak_module()
    prioritized_queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "priority_tier": "high",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "priority_tier": "high",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "priority_tier": "medium",
            "covered_checks": ["release_gate.errors"],
        },
    ]

    batch = module.select_retry_dispatch_batch(
        prioritized_queue,
        max_dispatch=2,
        min_priority_tier="low",
        diversify_checks=True,
    )

    assert batch["selection_mode"] == "priority_diversified"
    assert batch["selected_count"] == 2
    assert batch["selected_unique_check_count"] == 2
    selected_commands = [item["command"] for item in batch["selected"]]
    assert "python scripts/eval/release_soak.py --profile extended --include-accuracy" in selected_commands
    assert "python scripts/eval/release_gate.py" in selected_commands


def test_release_soak_selects_dispatch_batch_with_per_check_quota():
    module = _load_release_soak_module()
    prioritized_queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "priority_tier": "high",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "priority_tier": "high",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "priority_tier": "medium",
            "covered_checks": ["release_gate.errors"],
        },
    ]

    batch = module.select_retry_dispatch_batch(
        prioritized_queue,
        max_dispatch=3,
        min_priority_tier="low",
        diversify_checks=False,
        max_per_check=1,
    )

    assert batch["max_per_check"] == 1
    assert batch["selected_count"] == 2
    assert batch["selected_unique_check_count"] == 2
    selected_commands = [item["command"] for item in batch["selected"]]
    assert "python scripts/eval/release_soak.py --profile extended --include-accuracy" in selected_commands
    assert "python scripts/eval/release_gate.py" in selected_commands
    assert batch["skipped_check_quota_count"] == 1
    assert batch["skipped_check_quota_commands"] == [
        "python scripts/eval/release_soak.py --profile release --include-accuracy"
    ]


def test_release_soak_dispatch_retry_queue_to_pending_with_limit():
    module = _load_release_soak_module()
    entries = []
    retry_queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "title": "Run Extended Release Soak With Accuracy",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "title": "Re-run Release Gate",
            "covered_checks": ["release_gate.errors"],
        },
    ]

    dispatched = module.dispatch_retry_queue_to_pending(
        entries,
        retry_queue,
        max_dispatch=1,
    )

    assert dispatched == 1
    assert len(entries) == 1
    assert entries[0]["status"] == "pending"
    assert entries[0]["source"] == "retry_queue_dispatch"
    assert entries[0]["title"] == "Run Extended Release Soak With Accuracy"


def test_release_soak_dispatch_retry_queue_report_includes_skip_reasons():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "covered_checks": ["release_gate.errors"],
            "title": "existing_pending",
            "source": "iterative_next_action",
            "timestamp": 1.0,
        }
    ]
    retry_queue = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "title": "Run Extended Release Soak With Accuracy",
            "covered_checks": ["soak.duration_seconds"],
        },
        {
            "command": "python scripts/eval/release_gate.py",
            "title": "Re-run Release Gate",
            "covered_checks": ["release_gate.errors"],
        },
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "title": "Re-run Phase 3 Accuracy Suite",
            "covered_checks": ["stage_b.minimum_checks"],
        },
    ]

    dispatch_report = module.dispatch_retry_queue_to_pending_with_report(
        entries,
        retry_queue,
        max_dispatch=1,
    )

    assert dispatch_report["requested"] == 1
    assert dispatch_report["candidate_count"] == 3
    assert dispatch_report["dispatched"] == 1
    assert dispatch_report["dispatched_commands"] == [
        "python scripts/eval/release_soak.py --profile extended --include-accuracy"
    ]
    assert dispatch_report["skipped_pending_commands"] == [
        "python scripts/eval/release_gate.py"
    ]
    assert dispatch_report["skipped_limit_commands"] == [
        "python scripts/eval/phase3_accuracy_suite.py"
    ]


def test_release_soak_dispatch_retry_queue_skips_existing_pending_command():
    module = _load_release_soak_module()
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "covered_checks": ["release_gate.errors"],
            "title": "existing_pending",
            "source": "iterative_next_action",
            "timestamp": 1.0,
        }
    ]
    retry_queue = [
        {
            "command": "python scripts/eval/release_gate.py",
            "title": "Re-run Release Gate",
            "covered_checks": ["release_gate.errors"],
        }
    ]

    dispatched = module.dispatch_retry_queue_to_pending(
        entries,
        retry_queue,
        max_dispatch=1,
    )

    assert dispatched == 0
    assert len(entries) == 1
    assert entries[0]["source"] == "iterative_next_action"


def test_agent_soak_dialogue_keeps_bounded_state():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    def calculator(_: str) -> str:
        return "5"

    agent.register_tool("<CALC>", calculator)

    for turn in range(24):
        teaching_text = f"Python の補足知識 {turn} は 可読性 を高めます。"
        agent.chat(teaching_text, teaching_mode=True)
        response = agent.chat(f"この要点を教えて <CALC> {turn}", teaching_mode=False)
        assert response

    assert len(agent.dialogue_history) <= agent.max_history_turns * 2
    assert len(agent.get_recent_issues(limit=50)) <= 20

    session_path = workspace_path("tests", "soak_agent_session.pkl")
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    agent.save_session(session_path)

    restored = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    restored.load_session(session_path)

    assert len(restored.dialogue_history) <= restored.max_history_turns * 2
    assert restored.topic_tracker.active_terms(limit=3)


def test_inference_soak_learning_and_memory_roundtrip():
    memory_path = model_path("tests", "release_soak_inference.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    engine = SaraInference.__new__(SaraInference)
    engine.model_path = memory_path
    engine.direct_map = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    for offset in range(32):
        engine.learn_sequence([offset, offset + 1, offset + 2, offset + 3])

    assert engine.direct_map
    assert len(engine.direct_map) >= 16
    assert all(isinstance(key, tuple) for key in engine.direct_map.keys())

    engine.save_pretrained(memory_path)

    reloaded = SaraInference.__new__(SaraInference)
    reloaded.model_path = memory_path
    reloaded.direct_map = {}
    reloaded.refractory_buffer = []
    reloaded.lif_network = None
    reloaded._load_memory()

    assert reloaded.direct_map == engine.direct_map


def test_release_soak_sections_report_minimum_workload_flags():
    module = _load_release_soak_module()
    agent_report = module.run_agent_soak(
        duration_seconds=0.5, max_turns=12, min_turns=4)
    inference_report = module.run_inference_soak(
        duration_seconds=0.5, max_iterations=16, min_iterations=6)

    assert agent_report["turns"] >= 4
    assert agent_report["meets_min_turns"] is True
    assert agent_report["min_turns_required"] == 4

    assert inference_report["iterations"] >= 6
    assert inference_report["meets_min_iterations"] is True
    assert inference_report["min_iterations_required"] == 6
    assert "memory_health" in inference_report
    assert "conversational_readiness" in inference_report["memory_health"]
    assert "predictor_state_keys" in inference_report["memory_health"]
    assert "predictor_state_snapshot" in inference_report["memory_health"]
    assert "adaptation_state_keys" in inference_report["memory_health"]
    assert "adaptation_state_snapshot" in inference_report["memory_health"]
    assert "future_state_runtime_state" in inference_report["memory_health"]
    assert inference_report["memory_health"]["conversational_readiness"]["profile_memory_ready"] is True
    assert inference_report["memory_health"]["conversational_readiness"]["next_step_ready"] is True
    assert inference_report["memory_health"]["conversational_readiness"]["predictor_state_ready"] is True
    assert inference_report["memory_health"]["conversational_readiness"]["meta_adaptation_ready"] is True
    assert inference_report["memory_health"]["conversational_readiness"]["session_memory_observable"] is True
    assert "action" in inference_report["memory_health"]["predictor_state_keys"]
    assert "response_mode" in inference_report["memory_health"]["adaptation_state_keys"]
    assert inference_report["memory_health"]["future_state_runtime_state"]["transition_count"] >= 1


def test_release_soak_accuracy_embedding_keeps_stage_gate_data():
    module = _load_release_soak_module()

    expected_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.97,
        "passed": True,
        "trend": {"regression_count": 0},
        "component_reports": {"future_state_consistency": {"passed": True}},
        "focus_summary": {"predictive_readiness": {"passed": True, "score": 1.0}},
        "focus_trend": {"predictive_readiness": {"status": "UP", "delta": 0.1}},
        "stage_a_acceptance": {"passed": True},
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
        },
        "history_length": 3,
    }

    original_runner = module.run_phase3_accuracy_suite
    module.run_phase3_accuracy_suite = lambda **_: expected_report
    try:
        embedded = module.run_accuracy_soak(
            history_path=workspace_path("evaluation", "phase3_history.json"))
    finally:
        module.run_phase3_accuracy_suite = original_runner

    assert embedded["focus_trend"] == expected_report["focus_trend"]
    assert embedded["stage_a_acceptance"] == expected_report["stage_a_acceptance"]
    assert embedded["stage_b_readiness"] == expected_report["stage_b_readiness"]
    assert embedded["history_length"] == 3


def test_release_soak_profile_resolution_supports_extended_shipping_profile():
    module = _load_release_soak_module()
    settings = module.resolve_soak_profile(
        profile_name="extended",
        duration_seconds=None,
        max_agent_turns=None,
        min_agent_turns=None,
        max_inference_iterations=None,
        min_inference_iterations=None,
    )

    assert settings["profile_name"] == "extended"
    assert settings["duration_seconds"] == 30.0
    assert settings["min_agent_turns"] == 60
    assert settings["min_inference_iterations"] == 96
    assert settings["shipping_ready"] is True


def test_release_soak_profile_resolution_downgrades_shipping_ready_when_thresholds_are_lowered():
    module = _load_release_soak_module()
    settings = module.resolve_soak_profile(
        profile_name="extended",
        duration_seconds=1.0,
        max_agent_turns=8,
        min_agent_turns=4,
        max_inference_iterations=12,
        min_inference_iterations=6,
    )

    assert settings["profile_name"] == "extended"
    assert settings["shipping_ready"] is False


def test_release_soak_accuracy_summary_embeds_phase3_suite(monkeypatch):
    module = _load_release_soak_module()

    monkeypatch.setattr(
        module,
        "run_phase3_accuracy_suite",
        lambda history_path, persist_history, history_limit, stage_b_promotion_required_streak=3: {
            "suite_name": "Phase3AccuracySuite",
            "overall_score": 0.92,
            "passed": True,
            "trend": {"has_previous": True, "regression_count": 0},
            "component_reports": {"agent_dialogue": {"passed": True}},
            "focus_summary": {
                "few_shot": {"score": 1.0, "passed": True},
                "continual": {"score": 1.0, "passed": True},
                "retrieval_hygiene": {"score": 0.8, "passed": True},
                "adaptive_readiness": {"score": 1.0, "passed": True},
                "predictive_readiness": {"score": 1.0, "passed": True},
                "efficiency_readiness": {"score": 0.95, "passed": True},
            },
            "stage_c_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_d_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_e_readiness": {"passed": True, "minimum_requirements_passed": True},
            "phase3_completion": {"passed": True, "completion_score": 1.0},
            "history_length": 3,
        },
    )

    report = module.run_accuracy_soak(
        history_path=workspace_path(
            "tests", "release_soak_accuracy_history.json"),
        history_limit=5,
    )

    assert report["suite_name"] == "Phase3AccuracySuite"
    assert report["passed"] is True
    assert report["history_length"] == 3
    assert report["trend"]["regression_count"] == 0
    assert report["focus_summary"]["few_shot"]["score"] == 1.0
    assert report["focus_summary"]["retrieval_hygiene"]["score"] == 0.8
    assert report["stage_c_readiness"]["minimum_requirements_passed"] is True
    assert report["stage_d_readiness"]["minimum_requirements_passed"] is True
    assert report["stage_e_readiness"]["minimum_requirements_passed"] is True
    assert report["phase3_completion"]["completion_score"] == 1.0


def test_release_soak_accuracy_status_uses_gate_regression_count():
    module = _load_release_soak_module()

    assert module._accuracy_status(
        {
            "passed": True,
            "trend": {
                "regression_count": 2,
                "gate_regression_count": 0,
            },
        }
    ) is True
    assert module._accuracy_status(
        {
            "passed": True,
            "trend": {
                "regression_count": 2,
                "gate_regression_count": 1,
            },
        }
    ) is False


def test_release_soak_collects_release_metadata():
    module = _load_release_soak_module()

    metadata = module.collect_release_metadata()

    assert metadata["pyproject_version"] == metadata["cargo_version"]
    assert metadata["versions_match"] is True
    assert "sara-chat" in metadata["console_scripts"]
    assert "sara-train" in metadata["console_scripts"]
    assert metadata["release_notes_heading"] == "Current v1.1 Release Candidate"
    assert "Highlights" in metadata["release_note_sections"]


def test_release_soak_collects_release_gate_feedback():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": False,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
    }

    feedback = module.collect_release_gate_feedback(report)

    assert feedback["passed"] is True
    assert feedback["error_count"] == 0
    assert feedback["errors"] == []
    assert feedback["accuracy_required"] is False
    assert feedback["embedded_accuracy_present"] is False

    assert feedback["stage_a_passed"] is False
    assert feedback["stage_b_passed"] is False
    assert feedback["stage_b_minimum_requirements_passed"] is False
    assert feedback["stage_b_minimum_failure_count"] == 0
    assert feedback["stage_b_minimum_failures"] == []
    assert feedback["stage_b_promotion_next_step_hint"] == ""
    assert feedback["stage_b_promotion_actions"] == []
    assert feedback["stage_c_passed"] is False
    assert feedback["stage_c_minimum_requirements_passed"] is False
    assert feedback["stage_c_minimum_failure_count"] == 0
    assert feedback["stage_c_minimum_failures"] == []
    assert feedback["stage_d_passed"] is False
    assert feedback["stage_d_minimum_requirements_passed"] is False
    assert feedback["stage_d_minimum_failure_count"] == 0
    assert feedback["stage_d_minimum_failures"] == []
    assert feedback["stage_d_readiness_score"] == 0.0
    assert feedback["stage_d_replay_recovery_integrity"] == 0.0
    assert feedback["stage_d_replay_upgrade_reindex_integrity"] == 0.0
    assert feedback["stage_d_memory_health_index_integrity"] == 0.0
    assert feedback["stage_d_replay_noise_resilience_integrity"] == 0.0
    assert feedback["stage_d_astro_modulation_stability"] == 0.0
    assert feedback["stage_d_manifold_continual_retention_observed"] == 0.0
    assert feedback["stage_d_manifold_capacity_pressure_recall_observed"] == 0.0
    assert feedback["stage_d_manifold_capacity_pressure_scan_reduction_observed"] == 0.0
    assert feedback["stage_d_manifold_replay_refresh_retention_observed"] == 0.0
    assert feedback["stage_d_manifold_replay_refresh_eviction_integrity_observed"] == 0.0
    assert feedback["stage_e_passed"] is False
    assert feedback["stage_e_minimum_requirements_passed"] is False
    assert feedback["stage_e_minimum_failure_count"] == 0
    assert feedback["stage_e_minimum_failures"] == []
    assert feedback["stage_e_readiness_score"] == 0.0
    assert feedback["stage_e_common_spike_space_integrity"] == 0.0
    assert feedback["stage_e_temporal_compression_efficiency"] == 0.0
    assert feedback["stage_e_modality_temporal_budget_integrity"] == 0.0
    assert feedback["stage_e_dendritic_context_gate_stability"] == 0.0
    assert feedback["stage_e_spiking_hjepa_latent_transition"] == 0.0
    assert feedback["stage_e_reverse_reasoning_trace_integrity"] == 0.0
    assert feedback["stage_e_causal_candidate_trace_integrity"] == 0.0
    assert feedback["stage_e_module_orchestration_integrity"] == 0.0
    assert feedback["stage_e_counterfactual_lane_integrity"] == 0.0
    assert feedback["stage_e_action_trace_observability"] == 0.0
    assert feedback["stage_e_runtime_trace_replay_consistency"] == 0.0
    assert feedback["stage_e_manifold_trace_support_observed"] == 0.0
    assert feedback["stage_e_manifold_trace_recall_observed"] == 0.0
    assert feedback["stage_e_manifold_trace_scan_budget_observed"] == 0.0
    assert feedback["stage_e_manifold_trace_index_scan_reduction_observed"] == 0.0
    assert feedback["stage_e_manifold_trace_candidate_guard_observed"] == 0.0
    assert feedback["stage_e_delta_memory_steering_integrity_observed"] == 0.0
    assert feedback["stage_e_delta_memory_counterfactual_isolation_observed"] == 0.0
    assert feedback["stage_e_delta_memory_trace_observability_observed"] == 0.0
    assert feedback["phase5_entry_passed"] is False
    assert feedback["phase5_entry_readiness_score"] == 0.0
    assert feedback["phase5_latent_transition_alignment"] == 0.0
    assert feedback["phase5_correction_event_coverage"] == 0.0
    assert feedback["phase5_counterfactual_transition_separation"] == 0.0
    assert feedback["phase5_multi_step_latent_chain_integrity"] == 0.0
    assert feedback["phase5_long_horizon_error_correction_convergence"] == 0.0
    assert feedback["error_details"] == []
    assert feedback["error_details_summary"]["total"] == 0
    assert feedback["recovery_actions"] == []
    assert feedback["repair_plan"]["estimated_steps"] == 0
    assert feedback["repair_plan"]["coverage_ratio"] == 1.0
    assert feedback["repair_plan"]["fallback_actions"] == []
    assert feedback["repair_execution_log"] == []
    assert feedback["repair_pending_count"] == 0
    assert feedback["repair_timeout_count"] == 0
    assert feedback["repair_retry_queue_count"] == 0
    assert feedback["repair_retry_queue"] == []
    assert feedback["repair_retry_cooldown_seconds"] == 0.0
    assert feedback["repair_retry_cooldown_blocked_count"] == 0
    assert feedback["repair_retry_cooldown_blocked"] == []
    assert feedback["iterative_repair_plan"]["remaining_checks"] == []
    assert feedback["iterative_repair_plan"]["next_actions"] == []
    assert feedback["iterative_repair_plan"]["completed"] is True
    assert feedback["iterative_repair_plan"]["stop_reason"] == "no_target_checks"
    assert feedback["packaging_metadata_passed"] is True


def test_release_soak_embeds_research_review_summary():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": True,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "passed": True,
            "overall_score": 1.0,
            "trend": {"regression_count": 0},
            "component_reports": {
                "cognitive_runtime": {
                    "passed": True,
                    "overall_score": 1.0,
                        "metrics": {
                            "predictive_spike_entropy_reduction_observed": 1.0,
                            "phase_binding_coincidence_integrity_observed": 1.0,
                            "forward_only_local_update_stability_observed": 1.0,
                            "lejepa_linear_identifiability_proxy_observed": 1.0,
                            "lejepa_latent_whitening_health_observed": 1.0,
                            "lejepa_factor_disentanglement_observed": 1.0,
                            "lejepa_latent_planning_consistency_observed": 1.0,
                            "lejepa_positive_pair_alignment_observed": 1.0,
                            "plastic_submodel_registry_integrity_observed": 1.0,
                        "dynamic_submodel_route_integrity_observed": 1.0,
                        "submodel_relearning_trace_integrity_observed": 1.0,
                        "interpretable_submodel_concept_trace_observed": 1.0,
                        "runtime_submodel_route_action_grounding_observed": 1.0,
                        "runtime_submodel_counterfactual_route_separation_observed": 1.0,
                        "runtime_submodel_concept_trace_observed": 1.0,
                        "submodel_intervention_trace_integrity_observed": 1.0,
                        "submodel_ablation_effect_observed": 1.0,
                        "submodel_reactivation_recovery_observed": 1.0,
                        "submodel_credit_assignment_trace_integrity_observed": 1.0,
                        "submodel_credit_selectivity_observed": 1.0,
                        "submodel_credit_state_budget_observed": 1.0,
                        "runtime_submodel_local_credit_assignment_observed": 1.0,
                        "runtime_submodel_feedback_trace_observed": 1.0,
                        "submodel_structural_adaptation_trace_integrity_observed": 1.0,
                        "submodel_structural_growth_bounded_observed": 1.0,
                        "submodel_structural_pruning_observed": 1.0,
                        "submodel_scientific_hypothesis_trace_integrity_observed": 1.0,
                        "submodel_counterexample_revision_observed": 1.0,
                        "submodel_scientific_model_budget_observed": 1.0,
                        "submodel_hypothesis_bank_integrity_observed": 1.0,
                        "submodel_open_ended_selection_observed": 1.0,
                        "submodel_hypothesis_bank_budget_observed": 1.0,
                        "micro_turn_event_budget_observed": 1.0,
                        "foreground_background_context_handoff_observed": 1.0,
                        "interrupt_recovery_trace_observed": 1.0,
                        "simultaneous_stream_route_integrity_observed": 1.0,
                        "time_aligned_backchannel_policy_observed": 1.0,
                        "phase_assigned_submodel_route_observed": 1.0,
                        "uncertainty_bucket_specialization_observed": 1.0,
                        "denoising_correction_trace_integrity_observed": 1.0,
                        "block_independent_local_update_budget_observed": 1.0,
                    },
                },
                "energy_efficiency": {
                    "passed": True,
                    "overall_score": 1.0,
                    "metrics": {},
                    "neuromorphic_profile_trend": {
                        "has_previous": True,
                        "regression_count": 0,
                        "policy_change_count": 0,
                        "regressions": [],
                        "missing_profiles": [],
                        "policy_changes": [],
                    },
                },
            },
            "focus_summary": {},
            "stage_a_acceptance": {"passed": True},
            "stage_b_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_c_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_d_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_e_readiness": {"passed": True, "minimum_requirements_passed": True},
            "linear_snn_fusion_observed_trend": {
                "has_previous": True,
                "regression_count": 0,
                "release_gate_blocking": False,
            },
            "stage_e_architecture_integration_observed_trend": {
                "has_previous": True,
                "regression_count": 0,
                "release_gate_blocking": False,
            },
        },
    }
    report["release_gate"] = {"passed": True}
    report["release_checklist"] = module.collect_release_checklist_status(
        report,
        report_path=workspace_path("release", "release_soak_report.json"),
        summary_path=workspace_path("release", "release_soak_summary.txt"),
    )
    report["research_review"] = module.build_release_research_review(report)

    summary = module.format_release_summary(report)

    assert report["research_review"]["compact"]["passed"] is True
    assert "- review_score: 1.000" in summary
    assert "- release_gate_blocking: False" in summary
    assert "- requires_human_approval: True" in summary
    assert "- next_hypothesis_count: 0" in summary
    assert "- bounded_experiment_graph_node_count: 4" in summary
    assert "- bounded_experiment_graph_edge_count: 0" in summary
    assert "- sara_policy_dimension_count: 5" in summary
    assert "- sara_policy_needs_review_count: 0" in summary
    assert "- experiment_adoption_candidate_count: 4" in summary
    assert "- experiment_regressing_item_count: 0" in summary
    assert "- experiment_falsified_item_count: 0" in summary
    assert "- experiment_human_review_pending_count: 0" in summary
    assert "- experiment_priority_action_count: 1" in summary
    assert "- experiment_top_priority_source: experiment_adoption_candidate_review" in summary
    assert "- experiment_top_priority_category: adoption_candidate" in summary
    assert "- experiment_promotion_target_candidate_count: 4" in summary
    assert "- experiment_promotion_target_review_action_count: 4" in summary


def test_release_soak_summary_includes_research_planner_cleanup_classification():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "require_phase3_accuracy": False,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "passed": True,
            "overall_score": 1.0,
            "trend": {"regression_count": 0, "gate_regression_count": 0},
            "stage_a_acceptance": {"passed": True, "checks": {}},
            "stage_b_readiness": {"passed": True, "minimum_requirements_passed": True},
            "focus_summary": {},
        },
        "release_gate": {"passed": True, "errors": []},
        "release_checklist": {"passed": True},
        "research_review": {
            "compact": {
                "passed": False,
                "review_score": 0.72,
                "release_gate_blocking": False,
                "requires_human_approval": True,
                "next_hypothesis_count": 1,
                "stable_hypothesis_count": 0,
                "regression_watchlist_count": 0,
                "negative_result_count": 1,
                "bounded_experiment_graph_node_count": 2,
                "bounded_experiment_graph_edge_count": 1,
                "experiment_adoption_candidate_count": 1,
                "experiment_regressing_item_count": 1,
                "experiment_falsified_item_count": 1,
                "experiment_human_review_pending_count": 1,
                "experiment_priority_action_count": 2,
                "experiment_top_priority_source": "experiment_regression_remeasure",
                "experiment_top_priority_category": "regressing",
                "experiment_promotion_target_candidate_count": 1,
                "experiment_promotion_target_review_action_count": 1,
                "roadmap_patch_rejection_suppressed_count": 0,
                "roadmap_patch_rejection_refreshed_count": 1,
                "cause_boundary_documentation_count": 1,
                "targeted_fixture_repair_count": 1,
                "next_hypothesis_ids": ["predictive_spike_entropy_reduction_observed"],
                "regression_watchlist_ids": [],
            },
        },
        "research_journal_summary": {
            "completed_research_planner_task_count": 0,
            "research_planner_task_cleanup_pending_count": 1,
            "research_planner_task_cleanup_success_count": 0,
            "research_planner_task_cleanup_skipped_count": 0,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "recovery_confirmed": True,
                "promotion_review_recommended": False,
                "promotion_review_completed": True,
                "promotion_review_in_progress": False,
                "promotion_review_latest_status": "success",
                "recovery_source": "remeasure,alternative_probe",
                "next_review_action": "",
            },
            "stage_e_observed_acceptance_candidate_recovery_review_count": 1,
            "stage_e_observed_acceptance_candidate_recovery_review_status_counts": {
                "success": 1,
            },
            "stage_e_observed_acceptance_candidate_recovery_review_latest_status": "success",
            "stage_e_observed_acceptance_candidate_recovery_review_completed": True,
            "stage_e_observed_acceptance_candidate_recovery_review_in_progress": False,
            "completed_roadmap_patch_evidence_collection_keys": [
                "predictive_spike_entropy_reduction_observed:real_data_fixture"
            ],
            "roadmap_patch_refreshed_items": [],
        },
    }

    summary = module.format_release_summary(report)

    assert "- planner_task_pending_count: 2" in summary
    assert "- roadmap_patch_rejection_suppressed_count: 0" in summary
    assert "- roadmap_patch_rejection_refreshed_count: 1" in summary
    assert "- bounded_experiment_graph_node_count: 2" in summary
    assert "- bounded_experiment_graph_edge_count: 1" in summary
    assert "- experiment_adoption_candidate_count: 1" in summary
    assert "- experiment_regressing_item_count: 1" in summary
    assert "- experiment_falsified_item_count: 1" in summary
    assert "- experiment_human_review_pending_count: 1" in summary
    assert "- experiment_priority_action_count: 2" in summary
    assert "- experiment_top_priority_source: experiment_regression_remeasure" in summary
    assert "- experiment_top_priority_category: regressing" in summary
    assert "- experiment_promotion_target_candidate_count: 1" in summary
    assert "- experiment_promotion_target_review_action_count: 1" in summary
    assert "- completed_evidence_pending_review_count: 1" in summary
    assert (
        "- completed_evidence_pending_review_keys: "
        "predictive_spike_entropy_reduction_observed:real_data_fixture"
    ) in summary
    assert "- planner_task_completed_count: 0" in summary
    assert "- planner_task_completion_ratio: 0.000" in summary
    assert "- planner_task_cleanup_needed: False" in summary
    assert "- planner_task_cleanup_pending_count: 1" in summary
    assert "- planner_task_cleanup_stalled: True" in summary
    assert "- planner_task_cleanup_stalled_reason: fixture_implementation_wait" in summary
    assert (
        "- planner_task_cleanup_stalled_action_source: research_planner_fixture_repair_followup"
        in summary
    )
    assert "- stage_e_recovery_review_available: True" in summary
    assert "- stage_e_recovery_confirmed: True" in summary
    assert "- stage_e_recovery_review_recommended: False" in summary
    assert "- stage_e_recovery_review_completed: True" in summary
    assert "- stage_e_recovery_review_in_progress: False" in summary
    assert "- stage_e_recovery_review_latest_status: success" in summary
    assert "- stage_e_recovery_review_count: 1" in summary
    assert "- stage_e_recovery_review_success_count: 1" in summary
    assert "- stage_e_recovery_source: remeasure,alternative_probe" in summary


def test_release_stage_e_recovery_review_status_reads_operational_journal():
    module = _load_release_soak_module()
    status = module.compact_release_stage_e_recovery_review_status(
        {
            "operational_readiness": {
                "research_journal_summary": {
                    "stage_e_observed_acceptance_candidate_repair_loop": {
                        "recovery_confirmed": True,
                        "promotion_review_recommended": True,
                        "promotion_review_completed": False,
                        "promotion_review_in_progress": True,
                        "promotion_review_latest_status": "pending",
                        "recovery_source": "remeasure",
                        "next_review_action": "stage_e_observed_acceptance_candidate_stability",
                    },
                    "stage_e_observed_acceptance_candidate_recovery_review_count": 1,
                    "stage_e_observed_acceptance_candidate_recovery_review_status_counts": {
                        "pending": 1,
                    },
                    "stage_e_observed_acceptance_candidate_recovery_review_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_stale": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_latest_age_seconds": 999999.0,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_latest_status": "pending",
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_in_progress": True,
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_failed": False,
                    "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck_latest_status": "pending",
                }
            }
        }
    )

    assert status["available"] is True
    assert status["recovery_confirmed"] is True
    assert status["promotion_review_recommended"] is False
    assert status["promotion_review_in_progress"] is True
    assert status["promotion_review_stale"] is True
    assert status["promotion_review_followup_in_progress"] is True
    assert status["promotion_review_followup_failed"] is False
    assert status["promotion_review_followup_latest_status"] == "pending"
    assert status["promotion_review_followup_retry_in_progress"] is True
    assert status["promotion_review_followup_retry_failed"] is False
    assert status["promotion_review_followup_retry_latest_status"] == "pending"
    assert status["promotion_review_followup_retry_escalation_in_progress"] is True
    assert status["promotion_review_followup_retry_escalation_failed"] is False
    assert status["promotion_review_followup_retry_escalation_latest_status"] == "pending"
    assert status["promotion_review_evidence_collection_in_progress"] is True
    assert status["promotion_review_evidence_collection_failed"] is False
    assert status["promotion_review_evidence_collection_latest_status"] == "pending"
    assert status["promotion_review_evidence_recheck_in_progress"] is True
    assert status["promotion_review_evidence_recheck_failed"] is False
    assert status["promotion_review_evidence_recheck_latest_status"] == "pending"
    assert status["promotion_review_targeted_probe_in_progress"] is True
    assert status["promotion_review_targeted_probe_failed"] is False
    assert status["promotion_review_targeted_probe_latest_status"] == "pending"
    assert status["promotion_review_targeted_probe_recheck_in_progress"] is True
    assert status["promotion_review_targeted_probe_recheck_failed"] is False
    assert status["promotion_review_targeted_probe_recheck_latest_status"] == "pending"
    assert status["promotion_review_latest_status"] == "pending"
    assert status["promotion_review_latest_age_seconds"] == 999999.0
    assert status["promotion_review_pending_count"] == 1
    assert status["next_review_action"] == "stage_e_observed_acceptance_candidate_stability"


def test_release_soak_collects_stage_d_manifold_observations_from_accuracy():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": True,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "passed": True,
            "overall_score": 1.0,
            "trend": {"regression_count": 0},
            "linear_snn_fusion_observed_trend": {
                "has_previous": True,
                "regression_count": 0,
                "release_gate_blocking": False,
            },
            "stage_e_architecture_integration_observed_trend": {
                "has_previous": True,
                "regression_count": 0,
                "release_gate_blocking": False,
            },
            "stage_a_acceptance": {"passed": True},
            "stage_b_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_c_readiness": {"passed": True, "minimum_requirements_passed": True},
            "stage_d_readiness": {
                "passed": True,
                "minimum_requirements_passed": True,
                "readiness_score": 1.0,
                "acceptance_candidate_count": 16,
                "acceptance_candidate_ready_count": 16,
                "acceptance_candidates_ready": True,
                "acceptance_candidate_failure_count": 0,
                "acceptance_candidate_stability": {
                    "consecutive_passes": 3,
                    "required_streak": 3,
                    "recommended": True,
                },
                "delta_memory_candidate_ready": True,
                "delta_memory_candidate_failure_count": 0,
                "delta_memory_candidate_promoted": False,
                "delta_memory_promotion_readiness": {
                    "consecutive_passes": 3,
                    "required_streak": 3,
                    "recommended": True,
                    "promoted_to_minimum": False,
                },
                "metrics": {"replay_recovery_integrity": 1.0},
            },
            "stage_e_readiness": {"passed": True, "minimum_requirements_passed": True},
            "component_reports": {
                "continual_consolidation": {
                    "metrics": {
                        "manifold_continual_retention_observed": 1.0,
                        "manifold_capacity_pressure_recall_observed": 1.0,
                        "manifold_capacity_pressure_scan_reduction_observed": 0.889,
                        "manifold_replay_refresh_retention_observed": 1.0,
                        "manifold_replay_refresh_eviction_integrity_observed": 1.0,
                        "synaptic_tag_integrity_observed": 1.0,
                        "memory_phase_transition_integrity_observed": 1.0,
                        "metabolic_budget_integrity_observed": 1.0,
                        "sleep_consolidation_retention_observed": 1.0,
                        "astro_structural_lock_observed": 1.0,
                        "delta_memory_phase_retention_policy_observed": 1.0,
                        "delta_memory_crystal_retention_observed": 1.0,
                        "delta_memory_multi_history_recall_observed": 1.0,
                        "delta_memory_multi_history_health_observed": 1.0,
                        "delta_memory_erase_write_decoupling_observed": 1.0,
                        "delta_memory_erase_preserves_stable_memory_observed": 1.0,
                        "delta_memory_write_commits_residual_observed": 1.0,
                    }
                },
                "cognitive_runtime": {
                    "metrics": {
                        "manifold_trace_support_observed": 1.0,
                        "manifold_trace_recall_observed": 1.0,
                        "manifold_trace_scan_budget_observed": 1.0,
                        "manifold_trace_index_scan_reduction_observed": 1.0,
                        "manifold_trace_candidate_guard_observed": 1.0,
                        "delta_memory_steering_integrity_observed": 1.0,
                        "delta_memory_counterfactual_isolation_observed": 1.0,
                        "delta_memory_trace_observability_observed": 1.0,
                        "predictive_spike_entropy_reduction_observed": 1.0,
                        "phase_binding_coincidence_integrity_observed": 1.0,
                        "forward_only_local_update_stability_observed": 1.0,
                        "lejepa_linear_identifiability_proxy_observed": 1.0,
                        "lejepa_latent_whitening_health_observed": 1.0,
                        "lejepa_factor_disentanglement_observed": 1.0,
                        "lejepa_latent_planning_consistency_observed": 1.0,
                        "lejepa_positive_pair_alignment_observed": 1.0,
                        "plastic_submodel_registry_integrity_observed": 1.0,
                        "dynamic_submodel_route_integrity_observed": 1.0,
                        "submodel_relearning_trace_integrity_observed": 1.0,
                        "interpretable_submodel_concept_trace_observed": 1.0,
                        "runtime_submodel_route_action_grounding_observed": 1.0,
                        "runtime_submodel_counterfactual_route_separation_observed": 1.0,
                        "runtime_submodel_concept_trace_observed": 1.0,
                        "submodel_intervention_trace_integrity_observed": 1.0,
                        "submodel_ablation_effect_observed": 1.0,
                        "submodel_reactivation_recovery_observed": 1.0,
                        "submodel_credit_assignment_trace_integrity_observed": 1.0,
                        "submodel_credit_selectivity_observed": 1.0,
                        "submodel_credit_state_budget_observed": 1.0,
                        "runtime_submodel_local_credit_assignment_observed": 1.0,
                        "runtime_submodel_feedback_trace_observed": 1.0,
                        "submodel_structural_adaptation_trace_integrity_observed": 1.0,
                        "submodel_structural_growth_bounded_observed": 1.0,
                        "submodel_structural_pruning_observed": 1.0,
                        "submodel_scientific_hypothesis_trace_integrity_observed": 1.0,
                        "submodel_counterexample_revision_observed": 1.0,
                        "submodel_scientific_model_budget_observed": 1.0,
                        "submodel_hypothesis_bank_integrity_observed": 1.0,
                        "submodel_open_ended_selection_observed": 1.0,
                        "submodel_hypothesis_bank_budget_observed": 1.0,
                        "micro_turn_event_budget_observed": 1.0,
                        "foreground_background_context_handoff_observed": 1.0,
                        "interrupt_recovery_trace_observed": 1.0,
                        "simultaneous_stream_route_integrity_observed": 1.0,
                        "time_aligned_backchannel_policy_observed": 1.0,
                        "phase_assigned_submodel_route_observed": 1.0,
                        "uncertainty_bucket_specialization_observed": 1.0,
                        "denoising_correction_trace_integrity_observed": 1.0,
                        "block_independent_local_update_budget_observed": 1.0,
                    }
                },
            },
        },
    }

    feedback = module.collect_release_gate_feedback(report)

    assert feedback["stage_d_manifold_continual_retention_observed"] == 1.0
    assert feedback["stage_d_manifold_capacity_pressure_recall_observed"] == 1.0
    assert feedback["stage_d_manifold_capacity_pressure_scan_reduction_observed"] == 0.889
    assert feedback["stage_d_manifold_replay_refresh_retention_observed"] == 1.0
    assert feedback["stage_d_manifold_replay_refresh_eviction_integrity_observed"] == 1.0
    assert feedback["stage_d_synaptic_tag_integrity_observed"] == 1.0
    assert feedback["stage_d_memory_phase_transition_integrity_observed"] == 1.0
    assert feedback["stage_d_metabolic_budget_integrity_observed"] == 1.0
    assert feedback["stage_d_sleep_consolidation_retention_observed"] == 1.0
    assert feedback["stage_d_astro_structural_lock_observed"] == 1.0
    assert feedback["stage_d_delta_memory_phase_retention_policy_observed"] == 1.0
    assert feedback["stage_d_delta_memory_crystal_retention_observed"] == 1.0
    assert feedback["stage_d_delta_memory_multi_history_recall_observed"] == 1.0
    assert feedback["stage_d_delta_memory_multi_history_health_observed"] == 1.0
    assert feedback["stage_d_delta_memory_erase_write_decoupling_observed"] == 1.0
    assert feedback["stage_d_delta_memory_erase_preserves_stable_memory_observed"] == 1.0
    assert feedback["stage_d_delta_memory_write_commits_residual_observed"] == 1.0
    assert feedback["stage_d_delta_memory_candidate_ready"] is True
    assert feedback["stage_d_delta_memory_candidate_failure_count"] == 0
    assert feedback["stage_d_acceptance_candidate_consecutive_passes"] == 3
    assert feedback["stage_d_acceptance_candidate_stability_recommended"] is True
    assert (
        feedback["stage_d_acceptance_candidate_next_step_hint"]
        == "review_stage_d_acceptance_candidates_for_minimum_promotion"
    )
    assert feedback["stage_d_acceptance_candidate_actions"] == [
        "review stage_d_contract acceptance candidates and choose minimum promotion scope",
        "run python scripts/eval/phase3_accuracy_suite.py with persisted history and verify Stage D stability remains green",
        "run python scripts/eval/release_soak.py and verify operational acceptance-candidate summary remains green",
    ]
    assert feedback["stage_d_acceptance_candidate_action_count"] == 3
    assert feedback["stage_d_delta_memory_consecutive_passes"] == 3
    assert feedback["stage_d_delta_memory_promotion_recommended"] is True
    assert feedback["stage_d_delta_memory_next_step_hint"] == "promote_stage_d_delta_memory_metrics_to_minimum_gate"
    assert feedback["stage_e_manifold_trace_support_observed"] == 1.0
    assert feedback["stage_e_manifold_trace_recall_observed"] == 1.0
    assert feedback["stage_e_manifold_trace_scan_budget_observed"] == 1.0
    assert feedback["stage_e_manifold_trace_index_scan_reduction_observed"] == 1.0
    assert feedback["stage_e_manifold_trace_candidate_guard_observed"] == 1.0
    assert feedback["stage_e_delta_memory_steering_integrity_observed"] == 1.0
    assert feedback["stage_e_delta_memory_counterfactual_isolation_observed"] == 1.0
    assert feedback["stage_e_delta_memory_trace_observability_observed"] == 1.0
    assert feedback["stage_e_linear_snn_fusion_observed_policy"] == "excluded_from_score_and_release_gate"
    assert feedback["stage_e_predictive_spike_entropy_reduction_observed"] == 1.0
    assert feedback["stage_e_phase_binding_coincidence_integrity_observed"] == 1.0
    assert feedback["stage_e_forward_only_local_update_stability_observed"] == 1.0
    assert feedback["stage_e_plastic_submodel_registry_integrity_observed"] == 1.0
    assert feedback["stage_e_runtime_submodel_route_action_grounding_observed"] == 1.0
    assert feedback["stage_e_runtime_submodel_counterfactual_route_separation_observed"] == 1.0
    assert feedback["stage_e_submodel_credit_assignment_trace_integrity_observed"] == 1.0
    assert feedback["stage_e_runtime_submodel_local_credit_assignment_observed"] == 1.0
    assert feedback["stage_e_submodel_structural_growth_bounded_observed"] == 1.0
    assert feedback["stage_e_submodel_structural_pruning_observed"] == 1.0
    assert feedback["stage_e_submodel_scientific_hypothesis_trace_integrity_observed"] == 1.0
    assert feedback["stage_e_submodel_counterexample_revision_observed"] == 1.0
    assert feedback["stage_e_submodel_hypothesis_bank_integrity_observed"] == 1.0
    assert feedback["stage_e_submodel_open_ended_selection_observed"] == 1.0
    assert feedback["stage_e_submodel_intervention_trace_integrity_observed"] == 1.0
    assert feedback["stage_e_submodel_ablation_effect_observed"] == 1.0


def test_release_soak_collects_recovery_actions_when_gate_fails():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 1.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": False,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 1,
            "history_bounded": False,
            "issue_count": 1,
            "meets_min_turns": False,
        },
        "inference": {
            "iterations": 1,
            "roundtrip_ok": False,
            "tuple_keys_only": False,
            "pattern_count": 0,
            "meets_min_iterations": False,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "repair_execution_log": [
            {
                "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
                "status": "failed",
                "covered_checks": ["soak.duration_seconds"],
            },
            {
                "command": "python scripts/eval/release_gate.py",
                "status": "timeout",
                "covered_checks": ["release_gate.errors"],
            },
        ],
    }

    feedback = module.collect_release_gate_feedback(report)

    assert feedback["passed"] is False
    assert feedback["error_count"] > 0
    assert isinstance(feedback["recovery_actions"], list)
    assert feedback["recovery_actions"]
    assert isinstance(feedback["repair_plan"], dict)
    assert feedback["repair_plan"]["estimated_steps"] >= 1
    assert isinstance(feedback["repair_plan"]["selected_actions"], list)
    assert feedback["repair_plan"]["selected_actions"]
    assert isinstance(feedback["repair_plan"]["fallback_actions"], list)
    assert feedback["repair_execution_log"]
    assert feedback["repair_pending_count"] == 0
    assert feedback["repair_timeout_count"] == 1
    assert feedback["repair_retry_queue_count"] == 2
    assert len(feedback["repair_retry_queue"]) == 2
    assert isinstance(feedback["iterative_repair_plan"], dict)
    assert feedback["iterative_repair_plan"]["executed_steps"] >= 1
    assert isinstance(feedback["iterative_repair_plan"]["next_actions"], list)
    assert feedback["iterative_repair_plan"]["next_actions"]
    assert feedback["iterative_repair_plan"]["completed"] is False
    assert all(isinstance(action.get("priority", ""), str) and action.get("priority", "")
               for action in feedback["recovery_actions"] if isinstance(action, dict))
    assert all(
        isinstance(action.get("expected_effect", ""),
                   str) and action.get("expected_effect", "")
        for action in feedback["recovery_actions"]
        if isinstance(action, dict)
    )
    assert all(
        isinstance(action.get("affected_checks", []), list)
        for action in feedback["recovery_actions"]
        if isinstance(action, dict)
    )
    assert any(
        "release_soak.py --profile extended --include-accuracy" in action.get(
            "command", "")
        for action in feedback["recovery_actions"]
        if isinstance(action, dict)
    )


def test_release_soak_feedback_tracks_retry_cooldown_blocked_entries():
    module = _load_release_soak_module()
    now_ts = module.time.time()
    report = {
        "duration_seconds": 1.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": False,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 1,
            "history_bounded": False,
            "issue_count": 1,
            "meets_min_turns": False,
        },
        "inference": {
            "iterations": 1,
            "roundtrip_ok": False,
            "tuple_keys_only": False,
            "pattern_count": 0,
            "meets_min_iterations": False,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "repair_execution_log": [
            {
                "command": "python scripts/eval/release_gate.py",
                "status": "failed",
                "covered_checks": ["release_gate.errors"],
                "timestamp": now_ts,
            }
        ],
    }

    feedback = module.collect_release_gate_feedback(
        report,
        retry_max_attempts=2,
        retry_cooldown_seconds=30.0,
    )

    assert feedback["repair_retry_cooldown_seconds"] == 30.0
    assert feedback["repair_retry_queue_count"] == 0
    assert feedback["repair_retry_cooldown_blocked_count"] == 1
    assert len(feedback["repair_retry_cooldown_blocked"]) == 1
    assert feedback["repair_retry_cooldown_blocked"][0]["command"] == "python scripts/eval/release_gate.py"
    assert isinstance(
        feedback["repair_retry_cooldown_blocked"][0]["priority_tier"], str)
    assert isinstance(
        feedback["repair_retry_cooldown_blocked"][0]["priority_score"], float)


def test_release_soak_collects_release_checklist_status():
    module = _load_release_soak_module()
    report = {
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
        },
        "release_metadata": {
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": True,
        },
    }

    checklist = module.collect_release_checklist_status(
        report,
        report_path=workspace_path("release", "release_soak_report.json"),
        summary_path=workspace_path("release", "release_soak_summary.txt"),
    )

    assert checklist["passed"] is True
    assert checklist["managed_output_paths_ok"] is True
    assert checklist["release_notes_reviewed"] is True
    assert checklist["extended_profile_ready"] is False


def test_release_soak_checklist_is_not_blocked_by_release_gate_status():
    module = _load_release_soak_module()
    report = {
        "criteria": {
            "profile_name": "extended",
            "shipping_ready": True,
        },
        "release_metadata": {
            "release_notes_heading": "Current v1.1 Release Candidate",
        },
        "release_gate": {
            "passed": False,
        },
    }
    checklist = module.collect_release_checklist_status(
        report,
        report_path=workspace_path("release", "release_soak_report.json"),
        summary_path=workspace_path("release", "release_soak_summary.txt"),
    )
    assert checklist["passed"] is True
    assert checklist["extended_profile_ready"] is True


def test_release_soak_formats_human_readable_summary():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
            "require_phase3_accuracy": True,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
            "memory_health": {
                "session_memory_keys": ["goal", "task"],
                "diagnostic_memory_hits": ["fast_path", "session_memory"],
                "predictor_state_keys": ["action", "category", "confidence", "target_state"],
                "predictor_state_snapshot": {
                    "category": "release",
                    "confidence": 1.0,
                    "transition_operator": "release.check",
                    "alternative_transition_operator": "release.risk_prioritize",
                    "secondary_alternative_transition_operator": "release.rollback_guard",
                    "fluid_trace": {
                        "bounded": True,
                        "support_score": 0.75,
                        "active_columns": 6,
                        "total_spikes": 18,
                    },
                    "speculative_trace": {
                        "predicted_operator": "release.check",
                        "verified_operator": "release.check",
                        "operator_match": True,
                        "draft_verify_accepted": True,
                        "rollback_observable": True,
                        "counterfactual_branch_viable": True,
                    },
                },
                "adaptation_state_keys": [
                    "adaptation_turns",
                    "command_preference",
                    "fallback_relaxation",
                    "memory_weight",
                    "next_step_requests",
                    "planning_confidence",
                    "response_mode",
                ],
                "adaptation_state_snapshot": {
                    "response_mode": "directive",
                    "planning_confidence": 1.0,
                    "memory_weight": 1.5,
                    "fallback_relaxation": 0.1,
                },
                "future_state_runtime_state": {
                    "transition_count": 2,
                    "shift_count": 1,
                    "stability_ratio": 1.0,
                    "operator_consistency_ratio": 1.0,
                    "speculative_acceptance_ratio": 1.0,
                    "speculative_rollback_ratio": 1.0,
                    "counterfactual_viability_ratio": 1.0,
                    "rewarded_selection_ratio": 1.0,
                    "policy_stability_ratio": 1.0,
                    "energy_aware_preference_ratio": 1.0,
                    "previous_target_state": "ship the release",
                    "last_transition_operator": "release.check",
                    "last_verified_operator": "release.check",
                    "last_simulated_branch_count": 3,
                    "last_best_simulated_branch": "alternative",
                },
                "conversational_readiness": {
                    "profile_memory_ready": True,
                    "next_step_ready": True,
                    "predictor_state_ready": True,
                    "predictive_simulation_ready": True,
                    "meta_adaptation_ready": True,
                    "session_memory_observable": True,
                    "operator_trace_ready": True,
                    "speculative_trace_ready": True,
                    "fluid_trace_ready": True,
                },
            },
        },
        "release_metadata": {
            "pyproject_version": "1.1.0",
            "versions_match": True,
            "has_expected_console_scripts": True,
            "console_scripts": ["sara-chat", "sara-train"],
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": True,
            "error_count": 0,
            "errors": [],
            "accuracy_required": True,
            "embedded_accuracy_present": True,
            "stage_a_passed": True,
            "stage_b_passed": True,
            "stage_b_minimum_requirements_passed": True,
            "stage_b_minimum_failure_count": 0,
            "stage_b_promotion_candidate_ready": True,
            "stage_b_promotion_candidate_failure_count": 0,
            "stage_b_promotion_consecutive_passes": 3,
            "stage_b_promotion_required_streak": 3,
            "stage_b_promotion_recommended": True,
            "stage_b_promotion_next_step_hint": "promote_stage_b_reward_policy_metrics_to_minimum_gate",
            "stage_b_promotion_actions": [
                "review stage_b_contract minimum list and add the three promotion-candidate metrics"
            ],
            "stage_b_rlm_observation_candidate_ready": True,
            "stage_b_rlm_observation_candidate_failure_count": 0,
            "stage_b_rlm_observation_candidate_promoted": True,
            "stage_b_rlm_observation_consecutive_passes": 0,
            "stage_b_rlm_observation_required_streak": 3,
            "stage_b_rlm_observation_promotion_recommended": False,
            "stage_b_rlm_observation_next_step_hint": "",
            "stage_b_rlm_observation_actions": [],
            "stage_c_passed": True,
            "stage_c_minimum_requirements_passed": True,
            "stage_c_minimum_failure_count": 0,
            "stage_c_minimum_failures": [],
            "stage_d_passed": True,
            "stage_d_minimum_requirements_passed": True,
            "stage_d_minimum_failure_count": 0,
            "stage_d_minimum_failures": [],
            "stage_d_readiness_score": 1.0,
            "stage_d_acceptance_candidate_count": 16,
            "stage_d_acceptance_candidate_ready_count": 16,
            "stage_d_acceptance_candidates_ready": True,
            "stage_d_acceptance_candidate_failure_count": 0,
            "stage_d_acceptance_candidate_consecutive_passes": 3,
            "stage_d_acceptance_candidate_required_streak": 3,
            "stage_d_acceptance_candidate_stability_recommended": True,
            "stage_d_acceptance_candidate_next_step_hint": "review_stage_d_acceptance_candidates_for_minimum_promotion",
            "stage_d_acceptance_candidate_actions": [
                "review stage_d_contract acceptance candidates and choose minimum promotion scope"
            ],
            "stage_d_acceptance_candidate_action_count": 1,
            "stage_d_delta_memory_candidate_ready": True,
            "stage_d_delta_memory_candidate_failure_count": 0,
            "stage_d_delta_memory_candidate_failures": [],
            "stage_d_delta_memory_candidate_promoted": False,
            "stage_d_delta_memory_consecutive_passes": 3,
            "stage_d_delta_memory_required_streak": 3,
            "stage_d_delta_memory_promotion_recommended": True,
            "stage_d_delta_memory_next_step_hint": "promote_stage_d_delta_memory_metrics_to_minimum_gate",
            "stage_d_delta_memory_actions": [
                "review stage_d_contract minimum list and add the delta-memory promotion metrics"
            ],
            "stage_d_replay_recovery_integrity": 1.0,
            "stage_d_replay_upgrade_reindex_integrity": 1.0,
            "stage_d_memory_health_index_integrity": 1.0,
            "stage_d_replay_noise_resilience_integrity": 1.0,
            "stage_d_astro_modulation_stability": 1.0,
            "stage_d_manifold_continual_retention_observed": 1.0,
            "stage_d_manifold_trajectory_case_coverage_observed": 1.0,
            "stage_d_manifold_average_case_recall_observed": 1.0,
            "stage_d_manifold_scan_budget_integrity_observed": 1.0,
            "stage_d_manifold_indexed_candidate_integrity_observed": 1.0,
            "stage_d_manifold_index_scan_reduction_observed": 1.0,
            "stage_d_manifold_capacity_pressure_recall_observed": 1.0,
            "stage_d_manifold_capacity_pressure_scan_reduction_observed": 0.889,
            "stage_d_manifold_replay_refresh_retention_observed": 1.0,
            "stage_d_manifold_replay_refresh_eviction_integrity_observed": 1.0,
            "stage_d_synaptic_tag_integrity_observed": 1.0,
            "stage_d_memory_phase_transition_integrity_observed": 1.0,
            "stage_d_metabolic_budget_integrity_observed": 1.0,
            "stage_d_sleep_consolidation_retention_observed": 1.0,
            "stage_d_astro_structural_lock_observed": 1.0,
            "stage_d_delta_memory_phase_retention_policy_observed": 1.0,
            "stage_d_delta_memory_crystal_retention_observed": 1.0,
            "stage_d_delta_memory_multi_history_recall_observed": 1.0,
            "stage_d_delta_memory_multi_history_health_observed": 1.0,
            "stage_d_delta_memory_erase_write_decoupling_observed": 1.0,
            "stage_d_delta_memory_erase_preserves_stable_memory_observed": 1.0,
            "stage_d_delta_memory_write_commits_residual_observed": 1.0,
            "stage_e_passed": True,
            "stage_e_minimum_requirements_passed": True,
            "stage_e_minimum_failure_count": 0,
            "stage_e_minimum_failures": [],
            "stage_e_readiness_score": 1.0,
            "stage_e_observed_acceptance_candidate_count": 49,
            "stage_e_observed_acceptance_candidate_ready_count": 49,
            "stage_e_observed_acceptance_candidates_ready": True,
            "stage_e_observed_acceptance_candidate_failure_count": 0,
            "stage_e_observed_acceptance_candidate_consecutive_passes": 3,
            "stage_e_observed_acceptance_candidate_required_streak": 3,
            "stage_e_observed_acceptance_candidate_stability_recommended": True,
            "stage_e_common_spike_space_integrity": 1.0,
            "stage_e_temporal_compression_efficiency": 1.0,
            "stage_e_modality_temporal_budget_integrity": 1.0,
            "stage_e_dendritic_context_gate_stability": 1.0,
            "stage_e_spiking_hjepa_latent_transition": 1.0,
            "stage_e_reverse_reasoning_trace_integrity": 1.0,
            "stage_e_causal_candidate_trace_integrity": 1.0,
            "stage_e_module_orchestration_integrity": 1.0,
            "stage_e_counterfactual_lane_integrity": 1.0,
            "stage_e_action_trace_observability": 1.0,
            "stage_e_runtime_trace_replay_consistency": 1.0,
            "stage_e_manifold_trace_support_observed": 1.0,
            "stage_e_manifold_trace_recall_observed": 1.0,
            "stage_e_manifold_trace_scan_budget_observed": 1.0,
            "stage_e_manifold_trace_index_scan_reduction_observed": 1.0,
            "stage_e_manifold_trace_candidate_guard_observed": 1.0,
            "stage_e_delta_memory_steering_integrity_observed": 1.0,
            "stage_e_delta_memory_counterfactual_isolation_observed": 1.0,
            "stage_e_delta_memory_trace_observability_observed": 1.0,
            "stage_e_linear_snn_fusion_observed_policy": "excluded_from_score_and_release_gate",
            "stage_e_linear_snn_fusion_trend_has_previous": True,
            "stage_e_linear_snn_fusion_trend_regression_count": 0,
            "stage_e_linear_snn_fusion_trend_release_gate_blocking": False,
            "stage_e_architecture_integration_observed_policy": "excluded_from_score_and_release_gate",
            "stage_e_architecture_integration_trend_has_previous": True,
            "stage_e_architecture_integration_trend_regression_count": 0,
            "stage_e_architecture_integration_trend_release_gate_blocking": False,
            "stage_e_predictive_spike_entropy_reduction_observed": 1.0,
            "stage_e_phase_binding_coincidence_integrity_observed": 1.0,
            "stage_e_forward_only_local_update_stability_observed": 1.0,
            "stage_e_plastic_submodel_registry_integrity_observed": 1.0,
            "stage_e_dynamic_submodel_route_integrity_observed": 1.0,
            "stage_e_submodel_relearning_trace_integrity_observed": 1.0,
            "stage_e_interpretable_submodel_concept_trace_observed": 1.0,
            "stage_e_runtime_submodel_route_action_grounding_observed": 1.0,
            "stage_e_runtime_submodel_counterfactual_route_separation_observed": 1.0,
            "stage_e_runtime_submodel_concept_trace_observed": 1.0,
            "stage_e_submodel_intervention_trace_integrity_observed": 1.0,
            "stage_e_submodel_ablation_effect_observed": 1.0,
            "stage_e_submodel_reactivation_recovery_observed": 1.0,
            "stage_e_submodel_credit_assignment_trace_integrity_observed": 1.0,
            "stage_e_submodel_credit_selectivity_observed": 1.0,
            "stage_e_submodel_credit_state_budget_observed": 1.0,
            "stage_e_runtime_submodel_local_credit_assignment_observed": 1.0,
            "stage_e_runtime_submodel_feedback_trace_observed": 1.0,
            "stage_e_submodel_structural_adaptation_trace_integrity_observed": 1.0,
            "stage_e_submodel_structural_growth_bounded_observed": 1.0,
            "stage_e_submodel_structural_pruning_observed": 1.0,
            "stage_e_submodel_scientific_hypothesis_trace_integrity_observed": 1.0,
            "stage_e_submodel_counterexample_revision_observed": 1.0,
            "stage_e_submodel_scientific_model_budget_observed": 1.0,
            "stage_e_submodel_hypothesis_bank_integrity_observed": 1.0,
            "stage_e_submodel_open_ended_selection_observed": 1.0,
            "stage_e_submodel_hypothesis_bank_budget_observed": 1.0,
            "stage_e_micro_turn_event_budget_observed": 1.0,
            "stage_e_foreground_background_context_handoff_observed": 1.0,
            "stage_e_interrupt_recovery_trace_observed": 1.0,
            "stage_e_simultaneous_stream_route_integrity_observed": 1.0,
            "stage_e_time_aligned_backchannel_policy_observed": 1.0,
            "stage_e_phase_assigned_submodel_route_observed": 1.0,
            "stage_e_uncertainty_bucket_specialization_observed": 1.0,
            "stage_e_denoising_correction_trace_integrity_observed": 1.0,
            "stage_e_block_independent_local_update_budget_observed": 1.0,
            "phase5_entry_passed": True,
            "phase5_entry_readiness_score": 1.0,
            "phase5_latent_transition_alignment": 1.0,
            "phase5_prediction_error_observability": 1.0,
            "phase5_correction_event_coverage": 1.0,
            "phase5_anti_collapse_event_diversity": 1.0,
                "phase5_counterfactual_transition_separation": 1.0,
                "phase5_multi_step_latent_chain_integrity": 1.0,
                "phase5_long_horizon_error_correction_convergence": 1.0,
                "phase5_horizon_bucket_stability": 1.0,
                "phase5_macro_action_effectiveness": 1.0,
                "phase5_subgoal_decomposition_integrity": 1.0,
                "phase5_depth_selective_routing_integrity": 1.0,
                "phase5_micro_es_policy_refinement_integrity": 1.0,
                "phase5_manifold_candidate_miss_guard_observed": 1.0,
                "packaging_metadata_passed": True,
            },
        "release_checklist": {
            "passed": True,
            "profile_name": "release",
            "managed_output_paths_ok": True,
            "report_summary_review_ready": True,
            "release_notes_reviewed": True,
            "extended_profile_ready": False,
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "passed": True,
            "overall_score": 0.95,
            "trend": {
                "regression_count": 0,
                "gate_regression_count": 0,
                "improvements": [
                    {
                        "metric": "agent_dialogue.direction_shift_following",
                        "delta": 0.1,
                    },
                    {
                        "metric": "future_state_consistency.future_state_command_integrity",
                        "delta": 0.05,
                    },
                    {
                        "metric": "future_state_consistency.future_state_counterfactual_integrity",
                        "delta": 0.03,
                    },
                    {
                        "metric": "future_state_consistency.future_state_counterfactual_usefulness",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_choice_integrity",
                        "delta": 0.01,
                    },
                    {
                        "metric": "future_state_consistency.future_state_choice_reason_integrity",
                        "delta": 0.01,
                    },
                    {
                        "metric": "spiking_llm.hierarchical_context_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "energy_efficiency.memory_per_success_proxy",
                        "delta": 0.03,
                    },
                    {
                        "metric": "energy_efficiency.stochastic_readout_integrity",
                        "delta": 0.04,
                    },
                    {
                        "metric": "future_state_consistency.future_state_branching_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_options_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_ranking_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_decision_brief_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_shift_tracking_integrity",
                        "delta": 0.04,
                    },
                    {
                        "metric": "future_state_consistency.future_state_simulation_integrity",
                        "delta": 0.03,
                    },
                    {
                        "metric": "future_state_consistency.future_state_fluid_trace_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_fluid_support_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_refinement_loop_integrity",
                        "delta": 0.02,
                    },
                    {
                        "metric": "future_state_consistency.future_state_adaptive_refinement",
                        "delta": 0.02,
                    },
                ],
                "regressions": [],
                "unchanged": [],
                "new_metrics": [],
            },
            "stage_a_acceptance": {
                "passed": True,
                "checks": {
                    "overall.acc_target_0_95": True,
                    "trend.zero_regressions": True,
                },
            },
                "stage_b_readiness": {
                    "passed": True,
                    "minimum_requirements_passed": True,
                    "readiness_score": 1.0,
                    "promotion_candidate_ready": True,
                    "promotion_candidate_failure_count": 0,
                    "promotion_readiness": {
                        "consecutive_passes": 3,
                        "required_streak": 3,
                        "recommended": True,
                    },
                    "rlm_observation_candidate_ready": True,
                    "rlm_observation_candidate_failure_count": 0,
                    "rlm_observation_candidate_promoted": True,
                    "rlm_observation_promotion_readiness": {
                        "consecutive_passes": 0,
                        "required_streak": 3,
                        "recommended": False,
                        "promoted_to_minimum": True,
                    },
                    "minimum_checks": {
                    "metric.future_state_transition_integrity": True,
                    "metric.future_state_command_integrity": True,
                    "metric.future_state_predictor_snapshot_integrity": True,
                    "metric.future_state_runtime_tracking_integrity": True,
                    "metric.future_state_shift_tracking_integrity": True,
                    "metric.future_state_transition_operator_coverage": True,
                    "metric.future_state_transition_operator_consistency": True,
                    "metric.future_state_counterfactual_branch_viability": True,
                    "metric.future_state_fluid_trace_integrity": True,
                    "metric.future_state_fluid_support_integrity": True,
                    "metric.future_state_refinement_loop_integrity": True,
                    "metric.future_state_adaptive_refinement": True,
                    "metric.future_state_focused_retrieval_hit_ratio": True,
                    "metric.future_state_branch_level_decision_consistency": True,
                },
                "checks": {
                    "metric.future_state_branching_integrity": True,
                    "metric.future_state_simulation_integrity": True,
                    "metric.future_state_speculative_acceptance_ratio": True,
                    "metric.future_state_speculative_rollback_observability": True,
                    "metric.future_state_focused_retrieval_hit_ratio": True,
                    "metric.future_state_branch_level_decision_consistency": True,
                },
            },
            "focus_summary": {
                "few_shot": {"score": 1.0, "passed": True},
                "continual": {"score": 1.0, "passed": True},
                "retrieval_hygiene": {"score": 0.78, "passed": True},
                "adaptive_readiness": {
                    "score": 1.0,
                    "passed": True,
                    "metrics": {
                        "task_switch_adaptation.meta_adaptation_parameter_integrity": 1.0,
                    },
                },
                "predictive_readiness": {"score": 1.0, "passed": True},
                "efficiency_readiness": {
                    "score": 0.95,
                    "passed": True,
                    "metrics": {
                        "energy_efficiency.energy_per_success_proxy": 1.0,
                        "energy_efficiency.performance_energy_ratio_proxy": 0.22,
                        "energy_efficiency.ann_cost_advantage_proxy": 12.0,
                        "energy_efficiency.sparse_event_cost_score": 1.0,
                        "energy_efficiency.brain_efficiency_alignment_proxy": 0.9,
                        "energy_efficiency.memory_per_success_proxy": 0.0,
                        "energy_efficiency.low_overhead_route_score": 1.0,
                        "energy_efficiency.bounded_latency_score": 0.8,
                        "energy_efficiency.stochastic_readout_integrity": 1.0,
                    },
                },
                "consolidation_readiness": {
                    "score": 1.0,
                    "passed": True,
                    "metrics": {
                        "continual_consolidation.replay_recovery_integrity": 1.0,
                        "continual_consolidation.long_horizon_consolidation_retention": 1.0,
                        "continual_consolidation.counterfactual_replay_selection_integrity": 1.0,
                        "continual_consolidation.replay_upgrade_reindex_integrity": 1.0,
                        "continual_consolidation.memory_health_index_integrity": 1.0,
                        "continual_consolidation.replay_noise_resilience_integrity": 1.0,
                        "continual_consolidation.astro_modulation_stability": 1.0,
                    },
                },
            },
            "focus_trend": {
                "retrieval_hygiene": {"status": "UP", "delta": 0.06},
                "adaptive_readiness": {"status": "NEW", "delta": None},
                "predictive_readiness": {"status": "NEW", "delta": None},
                "efficiency_readiness": {"status": "NEW", "delta": None},
                "consolidation_readiness": {"status": "NEW", "delta": None},
            },
            "component_reports": {
                "agent_dialogue": {
                    "details": {
                        "test_results": [
                            {
                                "shift_from": "Pythonの関数とは？",
                                "user_input": "リスト内包表記のメリットは何ですか？",
                                "shift_following_score": 1.0,
                            }
                        ]
                    },
                    "metrics": {
                        "direction_shift_following": 1.0,
                    }
                },
                "future_state_consistency": {
                    "details": {
                        "test_results": [
                            {
                                "predicted_action": "choose one release check to complete for pytest release checks",
                                "predicted_target_state": "ship the release",
                                "predicted_command": "python scripts/eval/release_soak.py --include-accuracy",
                                "alternative_action": "prioritize the highest-risk release check in pytest release checks",
                                "alternative_target_state": "ship the release",
                                "alternative_command": "python scripts/eval/release_soak.py --include-accuracy",
                                "secondary_alternative_action": "check one rollback condition first in pytest release checks",
                                "secondary_alternative_target_state": "ship the release",
                                "secondary_alternative_command": "python scripts/eval/release_gate.py",
                                "chosen_plan": "alternative",
                                "choice_reason": "Reason: checking the highest-risk path first is more likely to reduce rework.",
                                "choice_response": "I would start with the alternative plan: An alternative next step is to prioritize the highest-risk release check in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`",
                                "options_response": "Primary: Step 1: choose one release check to complete for pytest release checks. Step 2: finish it and check that it moves you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`\nAlternative: An alternative next step is to prioritize the highest-risk release check in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`\nAdditional: A second alternative next step is to check one rollback condition first in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_gate.py`",
                                "ranked_options_response": "1. Alternative: An alternative next step is to prioritize the highest-risk release check in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`\n2. Primary: Step 1: choose one release check to complete for pytest release checks. Step 2: finish it and check that it moves you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`\n3. Additional: A second alternative next step is to check one rollback condition first in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_gate.py`",
                                "decision_brief_response": "Decision brief:\nI would start with the alternative plan: An alternative next step is to prioritize the highest-risk release check in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy` Reason: checking the highest-risk path first is more likely to reduce rework.\n1. Alternative: An alternative next step is to prioritize the highest-risk release check in pytest release checks. That can also move you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`\n2. Primary: Step 1: choose one release check to complete for pytest release checks. Step 2: finish it and check that it moves you toward ship the release. Suggested command: `python scripts/eval/release_soak.py --include-accuracy`",
                                "simulation_response": "Lightweight simulation:\n- Alternative: score=0.900, progress=0.800, risk=0.850, reversible=0.850\n- Primary: score=0.840, progress=1.000, risk=0.600, reversible=0.600\n- Additional: score=0.735, progress=0.700, risk=0.850, reversible=0.850",
                                "predictor_state": {
                                    "category": "release",
                                    "confidence": 1.0,
                                    "transition_operator": "release.check",
                                    "alternative_transition_operator": "release.risk_prioritize",
                                    "secondary_alternative_transition_operator": "release.rollback_guard",
                                    "fluid_trace": {
                                        "bounded": True,
                                        "support_score": 0.75,
                                        "active_columns": 6,
                                        "total_spikes": 18,
                                    },
                                    "speculative_trace": {
                                        "predicted_operator": "release.check",
                                        "verified_operator": "release.check",
                                        "operator_match": True,
                                        "draft_verify_accepted": True,
                                        "rollback_observable": True,
                                        "counterfactual_branch_viable": True,
                                    },
                                    "best_simulated_branch": "alternative",
                                },
                                "runtime_state": {
                                    "stability_ratio": 1.0,
                                    "operator_consistency_ratio": 1.0,
                                    "speculative_acceptance_ratio": 1.0,
                                    "speculative_rollback_ratio": 1.0,
                                    "counterfactual_viability_ratio": 1.0,
                                    "rewarded_selection_ratio": 1.0,
                                    "policy_stability_ratio": 1.0,
                                    "energy_aware_preference_ratio": 1.0,
                                    "last_best_simulated_branch": "alternative",
                                },
                            }
                        ]
                    }
                },
                "energy_efficiency": {
                    "metrics": {
                        "neuromorphic_profile_history_regression_observed": 1.0,
                        "neuromorphic_stage_e_state_trace_ir_observed": 1.0,
                        "neuromorphic_stage_e_routing_hint_coverage_observed": 1.0,
                        "neuromorphic_stage_e_online_update_policy_observed": 1.0,
                        "neuromorphic_stage_e_event_budget_observed": 1.0,
                    },
                    "details": {
                        "average_state_units": 2.5,
                    },
                    "neuromorphic_profile_trend": {
                        "regression_count": 0,
                        "policy_change_count": 0,
                    },
                }
            },
        },
    }

    summary = module.format_release_summary(report)

    assert "SARA Engine Release Soak Summary" in summary
    assert "overall_status: PASS" in summary
    assert "profile: release" in summary
    assert "- status: PASS" in summary
    assert "- turns: 24 / min 24" in summary
    assert "- iterations: 32 / min 32" in summary
    assert "- session_memory_keys: goal, task" in summary
    assert "- diagnostic_memory_hits: fast_path, session_memory" in summary
    assert "- predictor_state_keys: action, category, confidence, target_state" in summary
    assert "- adaptation_state_keys: adaptation_turns, command_preference, fallback_relaxation, memory_weight, next_step_requests, planning_confidence, response_mode" in summary
    assert "- profile_memory_ready: True" in summary
    assert "- next_step_ready: True" in summary
    assert "- predictor_state_ready: True" in summary
    assert "- predictive_simulation_ready: True" in summary
    assert "- meta_adaptation_ready: True" in summary
    assert "- session_memory_observable: True" in summary
    assert "- operator_trace_ready: True" in summary
    assert "- speculative_trace_ready: True" in summary
    assert "- fluid_trace_ready: True" in summary
    assert "- runtime_transition_count: 2" in summary
    assert "- runtime_shift_count: 1" in summary
    assert "- runtime_simulated_branch_count: 3" in summary
    assert "- runtime_best_simulated_branch: alternative" in summary
    assert "- runtime_transition_operator: release.check" in summary
    assert "- runtime_verified_operator: release.check" in summary
    assert "- runtime_operator_consistency_ratio: 1.000" in summary
    assert "- runtime_speculative_acceptance_ratio: 1.000" in summary
    assert "- runtime_speculative_rollback_ratio: 1.000" in summary
    assert "- runtime_counterfactual_viability_ratio: 1.000" in summary
    assert "- runtime_rewarded_selection_ratio: 1.000" in summary
    assert "- runtime_policy_stability_ratio: 1.000" in summary
    assert "- runtime_energy_aware_preference_ratio: 1.000" in summary
    assert "- version: 1.1.0" in summary
    assert "- suite_name: Phase3AccuracySuite" in summary
    assert "- stage_a_status: PASS" in summary
    assert "- stage_a_acc_target_met: True" in summary
    assert "- stage_a_zero_regressions: True" in summary
    assert "- stage_b_status: PASS" in summary
    assert "- stage_b_readiness_score: 1.000" in summary
    assert "- stage_b_minimum_requirements_passed: True" in summary
    assert "- stage_b_transition_ready: True" in summary
    assert "- stage_b_command_ready: True" in summary
    assert "- stage_b_predictor_snapshot_ready: True" in summary
    assert "- stage_b_runtime_tracking_ready: True" in summary
    assert "- stage_b_shift_tracking_ready: True" in summary
    assert "- stage_b_operator_coverage_ready: True" in summary
    assert "- stage_b_operator_consistency_ready: True" in summary
    assert "- stage_b_counterfactual_viability_ready: True" in summary
    assert "- stage_b_fluid_trace_ready: True" in summary
    assert "- stage_b_fluid_support_ready: True" in summary
    assert "- stage_b_refinement_loop_ready: True" in summary
    assert "- stage_b_adaptive_refinement_ready: True" in summary
    assert "- stage_b_focused_retrieval_observed: True" in summary
    assert "- stage_b_branch_decision_consistency_observed: True" in summary
    assert "- stage_b_rlm_observation_candidate_ready: True" in summary
    assert "- stage_b_rlm_observation_candidate_failure_count: 0" in summary
    assert "- stage_b_rlm_observation_candidate_promoted: True" in summary
    assert "- stage_b_rlm_observation_consecutive_passes: 0" in summary
    assert "- stage_b_rlm_observation_required_streak: 3" in summary
    assert "- stage_b_rlm_observation_promotion_recommended: False" in summary
    assert "- stage_b_branching_ready: True" in summary
    assert "- stage_b_simulation_ready: True" in summary
    assert "- stage_b_speculative_acceptance_ready: True" in summary
    assert "- stage_b_speculative_rollback_ready: True" in summary
    assert "- stage_b_promotion_candidate_ready: True" in summary
    assert "- stage_b_promotion_candidate_failure_count: 0" in summary
    assert "- stage_b_promotion_consecutive_passes: 3" in summary
    assert "- stage_b_promotion_required_streak: 3" in summary
    assert "- stage_b_promotion_recommended: True" in summary
    assert "- stage_b_promotion_next_step_hint: promote_stage_b_reward_policy_metrics_to_minimum_gate" in summary
    assert "- stage_b_promotion_action: review stage_b_contract minimum list and add the three promotion-candidate metrics" in summary
    assert "- stage_b_rlm_observation_candidate_ready: True" in summary
    assert "- stage_b_rlm_observation_candidate_failure_count: 0" in summary
    assert "- stage_b_rlm_observation_candidate_promoted: True" in summary
    assert "- stage_b_rlm_observation_consecutive_passes: 0" in summary
    assert "- stage_b_rlm_observation_required_streak: 3" in summary
    assert "- stage_b_rlm_observation_promotion_recommended: False" in summary
    assert "- stage_b_rlm_observation_next_step_hint: " in summary
    assert "Phase 3 Focus" in summary
    assert "- few_shot_status: PASS" in summary
    assert "- hierarchical_context_trend: UP" in summary
    assert "- hierarchical_context_delta: +0.020" in summary
    assert "- continual_status: PASS" in summary
    assert "- retrieval_hygiene_status: PASS" in summary
    assert "- retrieval_hygiene_trend: UP" in summary
    assert "- retrieval_hygiene_delta: +0.060" in summary
    assert "- adaptive_readiness_status: PASS" in summary
    assert "- adaptive_readiness_score: 1.000" in summary
    assert "- adaptive_readiness_trend: NEW" in summary
    assert "- adaptation_parameter_integrity: 1.000" in summary
    assert "- adaptation_parameter_integrity_trend: NEW" in summary
    assert "- adaptation_parameter_integrity_delta: +0.000" in summary
    assert "- direction_shift_following: 1.000" in summary
    assert "- direction_shift_trend: UP" in summary
    assert "- direction_shift_delta: +0.100" in summary
    assert "Dialogue Shift Detail" in summary
    assert "- shift_from: Pythonの関数とは？" in summary
    assert "- shift_query: リスト内包表記のメリットは何ですか？" in summary
    assert "- shift_following_score: 1.000" in summary
    assert "- predictive_readiness_status: PASS" in summary
    assert "- predictive_readiness_score: 1.000" in summary
    assert "- predictive_readiness_trend: NEW" in summary
    assert "- predictive_command_trend: UP" in summary
    assert "- predictive_command_delta: +0.050" in summary
    assert "- predictive_counterfactual_trend: UP" in summary
    assert "- predictive_counterfactual_delta: +0.030" in summary
    assert "- predictive_counterfactual_usefulness_trend: UP" in summary
    assert "- predictive_counterfactual_usefulness_delta: +0.020" in summary
    assert "- predictive_choice_trend: UP" in summary
    assert "- predictive_choice_delta: +0.010" in summary
    assert "- predictive_choice_reason_trend: UP" in summary
    assert "- predictive_choice_reason_delta: +0.010" in summary
    assert "- predictive_branching_trend: UP" in summary
    assert "- predictive_branching_delta: +0.020" in summary
    assert "- predictive_options_trend: UP" in summary
    assert "- predictive_options_delta: +0.020" in summary
    assert "- predictive_ranking_trend: UP" in summary
    assert "- predictive_ranking_delta: +0.020" in summary
    assert "- predictive_decision_brief_trend: UP" in summary
    assert "- predictive_decision_brief_delta: +0.020" in summary
    assert "- predictive_shift_trend: UP" in summary
    assert "- predictive_shift_delta: +0.040" in summary
    assert "- predictive_simulation_trend: UP" in summary
    assert "- predictive_simulation_delta: +0.030" in summary
    assert "- predictive_fluid_trace_trend: UP" in summary
    assert "- predictive_fluid_trace_delta: +0.020" in summary
    assert "- predictive_fluid_support_trend: UP" in summary
    assert "- predictive_fluid_support_delta: +0.020" in summary
    assert "- predictive_refinement_loop_trend: UP" in summary
    assert "- predictive_refinement_loop_delta: +0.020" in summary
    assert "- predictive_adaptive_refinement_trend: UP" in summary
    assert "- predictive_adaptive_refinement_delta: +0.020" in summary
    assert "- adaptation_response_mode: directive" in summary
    assert "- adaptation_planning_confidence: 1.000" in summary
    assert "- adaptation_memory_weight: 1.500" in summary
    assert "- adaptation_fallback_relaxation: 0.100" in summary
    assert "- alternative_action: prioritize the highest-risk release check in pytest release checks" in summary
    assert "- alternative_target_state: ship the release" in summary
    assert "- alternative_command: python scripts/eval/release_soak.py --include-accuracy" in summary
    assert "- secondary_alternative_action: check one rollback condition first in pytest release checks" in summary
    assert "- secondary_alternative_target_state: ship the release" in summary
    assert "- secondary_alternative_command: python scripts/eval/release_gate.py" in summary
    assert "- chosen_plan: alternative" in summary
    assert "- choice_reason: Reason: checking the highest-risk path first is more likely to reduce rework." in summary
    assert "- choice_response: I would start with the alternative plan:" in summary
    assert "- options_response: Primary: Step 1: choose one release check to complete for pytest release checks." in summary
    assert "- ranked_options_response: 1. Alternative: An alternative next step is to prioritize the highest-risk release check in pytest release checks." in summary
    assert "- decision_brief_response: Decision brief:" in summary
    assert "- simulation_response: Lightweight simulation:" in summary
    assert "- best_simulated_branch: alternative" in summary
    assert "- efficiency_readiness_status: PASS" in summary
    assert "- efficiency_readiness_score: 0.950" in summary
    assert "- energy_per_success_proxy: 1.000" in summary
    assert "- performance_energy_ratio_proxy: 0.220" in summary
    assert "- ann_cost_advantage_proxy: 12.000" in summary
    assert "- sparse_event_cost_score: 1.000" in summary
    assert "- brain_efficiency_alignment_proxy: 0.900" in summary
    assert "- memory_per_success_proxy: 0.000" in summary
    assert "- low_overhead_route_score: 1.000" in summary
    assert "- bounded_latency_score: 0.800" in summary
    assert "- stochastic_readout_integrity: 1.000" in summary
    assert "- neuromorphic_stage_e_state_trace_ir_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_routing_hint_coverage_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_online_update_policy_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_event_budget_observed: 1.000" in summary
    assert "- neuromorphic_profile_history_regression_observed: 1.000" in summary
    assert "- neuromorphic_profile_trend_regression_count: 0" in summary
    assert "- neuromorphic_profile_trend_policy_change_count: 0" in summary
    assert "- neuromorphic_profile_trend_regression_details: none" in summary
    assert "- neuromorphic_profile_trend_policy_change_details: none" in summary
    assert "- average_state_units: 2.500" in summary
    assert "- memory_per_success_trend: UP" in summary
    assert "- memory_per_success_delta: +0.030" in summary
    assert "- stochastic_readout_trend: UP" in summary
    assert "- stochastic_readout_delta: +0.040" in summary
    assert "- efficiency_readiness_trend: NEW" in summary
    assert "- consolidation_readiness_status: PASS" in summary
    assert "- consolidation_readiness_score: 1.000" in summary
    assert "- consolidation_replay_recovery_integrity: 1.000" in summary
    assert "- consolidation_replay_upgrade_reindex_integrity: 1.000" in summary
    assert "- consolidation_memory_health_index_integrity: 1.000" in summary
    assert "- consolidation_replay_noise_resilience_integrity: 1.000" in summary
    assert "- consolidation_astro_modulation_stability: 1.000" in summary
    assert "- consolidation_readiness_trend: NEW" in summary
    assert "Predictive Detail" in summary
    assert "- predicted_action: choose one release check to complete for pytest release checks" in summary
    assert "- predicted_target_state: ship the release" in summary
    assert "- predicted_command: python scripts/eval/release_soak.py --include-accuracy" in summary
    assert "- predictor_category: release" in summary
    assert "- predictor_confidence: 1.000" in summary
    assert "- transition_operator: release.check" in summary
    assert "- alternative_transition_operator: release.risk_prioritize" in summary
    assert "- secondary_alternative_transition_operator: release.rollback_guard" in summary
    assert "- speculative_predicted_operator: release.check" in summary
    assert "- speculative_verified_operator: release.check" in summary
    assert "- speculative_operator_match: True" in summary
    assert "- speculative_acceptance: True" in summary
    assert "- speculative_rollback_observable: True" in summary
    assert "- speculative_counterfactual_viable: True" in summary
    assert "- refinement_triggered: False" in summary
    assert "- refinement_loop_count: 0" in summary
    assert "- fluid_bounded: True" in summary
    assert "- fluid_support_score: 0.750" in summary
    assert "- fluid_active_columns: 6" in summary
    assert "- fluid_total_spikes: 18" in summary
    assert "- runtime_stability_ratio: 1.000" in summary
    assert "- runtime_operator_consistency_ratio: 1.000" in summary
    assert "- runtime_speculative_acceptance_ratio: 1.000" in summary
    assert "- runtime_speculative_rollback_ratio: 1.000" in summary
    assert "- runtime_counterfactual_viability_ratio: 1.000" in summary
    assert "- runtime_rewarded_selection_ratio: 1.000" in summary
    assert "- runtime_policy_stability_ratio: 1.000" in summary
    assert "- runtime_energy_aware_preference_ratio: 1.000" in summary
    assert "- previous_target_state: ship the release" in summary
    assert "Gate" in summary
    assert "- error_count: 0" in summary
    assert "- accuracy_required: True" in summary
    assert "- embedded_accuracy_present: True" in summary
    assert "- stage_a_passed: True" in summary
    assert "- stage_b_passed: True" in summary
    assert "- stage_b_minimum_requirements_passed: True" in summary
    assert "- stage_b_minimum_failure_count: 0" in summary
    assert "- stage_b_promotion_candidate_ready: True" in summary
    assert "- stage_b_promotion_candidate_failure_count: 0" in summary
    assert "- stage_b_promotion_consecutive_passes: 3" in summary
    assert "- stage_b_promotion_required_streak: 3" in summary
    assert "- stage_b_promotion_recommended: True" in summary
    assert "- stage_b_promotion_next_step_hint: promote_stage_b_reward_policy_metrics_to_minimum_gate" in summary
    assert "- stage_c_passed: True" in summary
    assert "- stage_c_minimum_requirements_passed: True" in summary
    assert "- stage_c_minimum_failure_count: 0" in summary
    assert "- stage_d_passed: True" in summary
    assert "- stage_d_minimum_requirements_passed: True" in summary
    assert "- stage_d_minimum_failure_count: 0" in summary
    assert "- stage_d_readiness_score: 1.000" in summary
    assert "- stage_d_acceptance_candidate_count: 16" in summary
    assert "- stage_d_acceptance_candidate_ready_count: 16" in summary
    assert "- stage_d_acceptance_candidates_ready: True" in summary
    assert "- stage_d_acceptance_candidate_failure_count: 0" in summary
    assert "- stage_d_acceptance_candidate_consecutive_passes: 3" in summary
    assert "- stage_d_acceptance_candidate_required_streak: 3" in summary
    assert "- stage_d_acceptance_candidate_stability_recommended: True" in summary
    assert "- stage_d_acceptance_candidate_next_step_hint: review_stage_d_acceptance_candidates_for_minimum_promotion" in summary
    assert "- stage_d_acceptance_candidate_action_count: 1" in summary
    assert "- stage_d_acceptance_candidate_action: review stage_d_contract acceptance candidates and choose minimum promotion scope" in summary
    assert "- stage_d_delta_memory_candidate_ready: True" in summary
    assert "- stage_d_delta_memory_candidate_failure_count: 0" in summary
    assert "- stage_d_delta_memory_candidate_promoted: False" in summary
    assert "- stage_d_delta_memory_consecutive_passes: 3" in summary
    assert "- stage_d_delta_memory_required_streak: 3" in summary
    assert "- stage_d_delta_memory_promotion_recommended: True" in summary
    assert "- stage_d_delta_memory_next_step_hint: promote_stage_d_delta_memory_metrics_to_minimum_gate" in summary
    assert "- stage_d_replay_recovery_integrity: 1.000" in summary
    assert "- stage_d_replay_upgrade_reindex_integrity: 1.000" in summary
    assert "- stage_d_memory_health_index_integrity: 1.000" in summary
    assert "- stage_d_replay_noise_resilience_integrity: 1.000" in summary
    assert "- stage_d_astro_modulation_stability: 1.000" in summary
    assert "- stage_d_manifold_continual_retention_observed: 1.000" in summary
    assert "- stage_d_manifold_capacity_pressure_recall_observed: 1.000" in summary
    assert "- stage_d_manifold_capacity_pressure_scan_reduction_observed: 0.889" in summary
    assert "- stage_d_manifold_replay_refresh_retention_observed: 1.000" in summary
    assert "- stage_d_manifold_replay_refresh_eviction_integrity_observed: 1.000" in summary
    assert "- stage_d_synaptic_tag_integrity_observed: 1.000" in summary
    assert "- stage_d_memory_phase_transition_integrity_observed: 1.000" in summary
    assert "- stage_d_metabolic_budget_integrity_observed: 1.000" in summary
    assert "- stage_d_sleep_consolidation_retention_observed: 1.000" in summary
    assert "- stage_d_astro_structural_lock_observed: 1.000" in summary
    assert "- stage_d_delta_memory_phase_retention_policy_observed: 1.000" in summary
    assert "- stage_d_delta_memory_crystal_retention_observed: 1.000" in summary
    assert "- stage_d_delta_memory_multi_history_recall_observed: 1.000" in summary
    assert "- stage_d_delta_memory_multi_history_health_observed: 1.000" in summary
    assert "- stage_d_delta_memory_erase_write_decoupling_observed: 1.000" in summary
    assert "- stage_d_delta_memory_erase_preserves_stable_memory_observed: 1.000" in summary
    assert "- stage_d_delta_memory_write_commits_residual_observed: 1.000" in summary
    assert "- stage_d_delta_memory_action: review stage_d_contract minimum list and add the delta-memory promotion metrics" in summary
    assert "- stage_e_passed: True" in summary
    assert "- stage_e_minimum_requirements_passed: True" in summary
    assert "- stage_e_minimum_failure_count: 0" in summary
    assert "- stage_e_readiness_score: 1.000" in summary
    assert "- stage_e_observed_acceptance_candidate_count: 49" in summary
    assert "- stage_e_observed_acceptance_candidate_ready_count: 49" in summary
    assert "- stage_e_observed_acceptance_candidates_ready: True" in summary
    assert "- stage_e_observed_acceptance_candidate_failure_count: 0" in summary
    assert "- stage_e_observed_acceptance_candidate_consecutive_passes: 3" in summary
    assert "- stage_e_observed_acceptance_candidate_required_streak: 3" in summary
    assert "- stage_e_observed_acceptance_candidate_stability_recommended: True" in summary
    assert "- stage_e_common_spike_space_integrity: 1.000" in summary
    assert "- stage_e_temporal_compression_efficiency: 1.000" in summary
    assert "- stage_e_modality_temporal_budget_integrity: 1.000" in summary
    assert "- stage_e_dendritic_context_gate_stability: 1.000" in summary
    assert "- stage_e_spiking_hjepa_latent_transition: 1.000" in summary
    assert "- stage_e_reverse_reasoning_trace_integrity: 1.000" in summary
    assert "- stage_e_causal_candidate_trace_integrity: 1.000" in summary
    assert "- stage_e_module_orchestration_integrity: 1.000" in summary
    assert "- stage_e_counterfactual_lane_integrity: 1.000" in summary
    assert "- stage_e_action_trace_observability: 1.000" in summary
    assert "- stage_e_runtime_trace_replay_consistency: 1.000" in summary
    assert "- stage_e_manifold_trace_support_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_recall_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_scan_budget_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_index_scan_reduction_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_candidate_guard_observed: 1.000" in summary
    assert "- stage_e_delta_memory_steering_integrity_observed: 1.000" in summary
    assert "- stage_e_delta_memory_counterfactual_isolation_observed: 1.000" in summary
    assert "- stage_e_delta_memory_trace_observability_observed: 1.000" in summary
    assert "- stage_e_linear_snn_fusion_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- stage_e_linear_snn_fusion_trend_has_previous: True" in summary
    assert "- stage_e_linear_snn_fusion_trend_regression_count: 0" in summary
    assert "- stage_e_linear_snn_fusion_trend_release_gate_blocking: False" in summary
    assert "- stage_e_architecture_integration_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- stage_e_architecture_integration_trend_has_previous: True" in summary
    assert "- stage_e_architecture_integration_trend_regression_count: 0" in summary
    assert "- stage_e_architecture_integration_trend_release_gate_blocking: False" in summary
    assert "- stage_e_predictive_spike_entropy_reduction_observed: 1.000" in summary
    assert "- stage_e_phase_binding_coincidence_integrity_observed: 1.000" in summary
    assert "- stage_e_forward_only_local_update_stability_observed: 1.000" in summary
    assert "- stage_e_plastic_submodel_registry_integrity_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_route_action_grounding_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_counterfactual_route_separation_observed: 1.000" in summary
    assert "- stage_e_submodel_intervention_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_ablation_effect_observed: 1.000" in summary
    assert "- stage_e_submodel_credit_assignment_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_local_credit_assignment_observed: 1.000" in summary
    assert "- stage_e_submodel_structural_adaptation_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_structural_growth_bounded_observed: 1.000" in summary
    assert "- stage_e_submodel_scientific_hypothesis_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_counterexample_revision_observed: 1.000" in summary
    assert "- stage_e_submodel_hypothesis_bank_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_open_ended_selection_observed: 1.000" in summary
    assert "- stage_e_micro_turn_event_budget_observed: 1.000" in summary
    assert "- stage_e_foreground_background_context_handoff_observed: 1.000" in summary
    assert "- stage_e_interrupt_recovery_trace_observed: 1.000" in summary
    assert "- stage_e_simultaneous_stream_route_integrity_observed: 1.000" in summary
    assert "- stage_e_time_aligned_backchannel_policy_observed: 1.000" in summary
    assert "- stage_e_phase_assigned_submodel_route_observed: 1.000" in summary
    assert "- stage_e_uncertainty_bucket_specialization_observed: 1.000" in summary
    assert "- stage_e_denoising_correction_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_block_independent_local_update_budget_observed: 1.000" in summary
    assert "- phase5_entry_passed: True" in summary
    assert "- phase5_entry_readiness_score: 1.000" in summary
    assert "- phase5_latent_transition_alignment: 1.000" in summary
    assert "- phase5_correction_event_coverage: 1.000" in summary
    assert "- phase5_counterfactual_transition_separation: 1.000" in summary
    assert "- phase5_multi_step_latent_chain_integrity: 1.000" in summary
    assert "- phase5_long_horizon_error_correction_convergence: 1.000" in summary
    assert "- phase5_horizon_bucket_stability: 1.000" in summary
    assert "- phase5_macro_action_effectiveness: 1.000" in summary
    assert "- phase5_subgoal_decomposition_integrity: 1.000" in summary
    assert "- phase5_depth_selective_routing_integrity: 1.000" in summary
    assert "- phase5_micro_es_policy_refinement_integrity: 1.000" in summary
    assert "- phase5_manifold_candidate_miss_guard_observed: 1.000" in summary
    assert "- gate_regression_count: 0" in summary
    assert "- packaging_metadata_passed: True" in summary
    assert "Checklist" in summary
    assert "- managed_output_paths_ok: True" in summary
    assert "- extended_profile_ready: False" in summary


def test_release_soak_summary_includes_neuromorphic_profile_regression_details():
    module = _load_release_soak_module()
    report = {
        "criteria": {
            "profile_name": "release",
            "min_turns": 24,
            "min_iterations": 32,
            "require_accuracy": True,
        },
        "passed": True,
        "turns": 24,
        "iterations": 32,
        "memory_checks": {
            "profile_memory_ready": True,
            "next_step_ready": True,
            "predictor_state_ready": True,
            "predictive_simulation_ready": True,
            "meta_adaptation_ready": True,
            "session_memory_observable": True,
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "focus_summary": {
                "efficiency_readiness": {
                    "score": 0.95,
                    "passed": True,
                    "metrics": {
                        "energy_efficiency.energy_per_success_proxy": 1.0,
                        "energy_efficiency.performance_energy_ratio_proxy": 0.22,
                        "energy_efficiency.ann_cost_advantage_proxy": 12.0,
                        "energy_efficiency.sparse_event_cost_score": 1.0,
                        "energy_efficiency.brain_efficiency_alignment_proxy": 0.9,
                        "energy_efficiency.memory_per_success_proxy": 1.0,
                        "energy_efficiency.low_overhead_route_score": 1.0,
                        "energy_efficiency.bounded_latency_score": 0.8,
                        "energy_efficiency.stochastic_readout_integrity": 1.0,
                    },
                },
            },
            "focus_trend": {"efficiency_readiness": {"status": "NEW", "delta": None}},
            "component_reports": {
                "energy_efficiency": {
                    "metrics": {"neuromorphic_profile_history_regression_observed": 0.0},
                    "details": {"average_state_units": 2.5},
                    "neuromorphic_profile_trend": {
                        "regression_count": 2,
                        "policy_change_count": 1,
                        "regressions": [
                            {"profile": "akida", "kind": "compatibility_regression"},
                            {
                                "profile": "akida",
                                "kind": "check_regression",
                                "check": "low_precision_weight_ok",
                            },
                        ],
                        "policy_changes": [
                            {
                                "profile": "akida",
                                "previous": "freeze_state_for_inference_profile",
                                "current": "native_online_update",
                            }
                        ],
                    },
                }
            },
        },
    }

    summary = module.format_release_summary(report)

    assert "- neuromorphic_profile_trend_regression_count: 2" in summary
    assert (
        "- neuromorphic_profile_trend_regression_details: "
        "akida:compatibility_regression,akida:check_regression:low_precision_weight_ok"
        in summary
    )
    assert (
        "- neuromorphic_profile_trend_policy_change_details: "
        "akida:freeze_state_for_inference_profile->native_online_update"
        in summary
    )


def test_release_soak_summary_warns_when_required_accuracy_is_missing():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
            "require_phase3_accuracy": True,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "pyproject_version": "1.1.0",
            "versions_match": True,
            "has_expected_console_scripts": True,
            "console_scripts": ["sara-chat", "sara-train"],
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": False,
            "error_count": 1,
            "errors": ["Release soak report requires embedded Phase 3 accuracy results."],
            "error_details": [
                {
                    "index": 1,
                    "error": "Release soak report requires embedded Phase 3 accuracy results.",
                    "inferred_checks": ["release_gate.embedded_accuracy_present"],
                    "category": "release_gate.embedded_accuracy_present",
                    "type": "general_error",
                }
            ],
            "error_details_summary": {
                "total": 1,
                "by_type": {"general_error": 1},
                "by_category": {"release_gate.embedded_accuracy_present": 1},
                "by_metric": {},
                "top_types": [{"name": "general_error", "count": 1}],
                "top_categories": [{"name": "release_gate.embedded_accuracy_present", "count": 1}],
                "top_metrics": [],
            },
            "failure_focus": {
                "primary_category": "release_gate.embedded_accuracy_present",
                "secondary_category": "",
                "primary_metric": "",
                "confidence": 1.0,
                "primary_action": {
                    "title": "Embed Accuracy Into Soak Report",
                    "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
                    "priority": "high",
                },
            },
            "recovery_actions": [
                {
                    "title": "Embed Accuracy Into Soak Report",
                    "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
                    "reason": "Attach Phase 3 report to the release soak artifact.",
                    "priority": "high",
                    "expected_effect": "Restores missing embedded Phase 3 evidence in the soak report.",
                    "affected_checks": [
                        "release_gate.embedded_accuracy_present",
                        "release_gate.accuracy_required",
                    ],
                }
            ],
            "repair_plan": {
                "selected_actions": [
                    {
                        "step": 1,
                        "title": "Embed Accuracy Into Soak Report",
                        "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
                        "affected_checks": [
                            "release_gate.embedded_accuracy_present",
                            "release_gate.accuracy_required",
                        ],
                    }
                ],
                "covered_checks": [
                    "release_gate.embedded_accuracy_present",
                    "release_gate.accuracy_required",
                ],
                "uncovered_checks": [],
                "fallback_actions": [],
                "coverage_ratio": 1.0,
                "estimated_steps": 1,
            },
            "iterative_repair_plan": {
                "remaining_checks": [
                    "release_gate.embedded_accuracy_present",
                    "release_gate.accuracy_required",
                ],
                "next_actions": [
                    {
                        "step": 1,
                        "title": "Embed Accuracy Into Soak Report",
                        "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
                        "affected_checks": [
                            "release_gate.embedded_accuracy_present",
                            "release_gate.accuracy_required",
                        ],
                    }
                ],
                "completed": False,
                "stop_reason": "pending_actions",
                "next_step_hint": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            },
            "accuracy_required": True,
            "embedded_accuracy_present": False,
            "stage_a_passed": False,
            "stage_b_passed": False,
            "stage_b_minimum_requirements_passed": False,
            "stage_c_passed": False,
            "stage_c_minimum_requirements_passed": False,
            "stage_d_passed": False,
            "stage_d_minimum_requirements_passed": False,
            "stage_d_delta_memory_candidate_failures": [
                {
                    "check": "metric.delta_memory_multi_history_recall_observed",
                    "metric": "delta_memory_multi_history_recall_observed",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "stage_d_acceptance_candidate_failures": [
                {
                    "check": "metric.delta_memory_erase_write_decoupling_observed",
                    "metric": "delta_memory_erase_write_decoupling_observed",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "stage_e_passed": False,
            "stage_e_minimum_requirements_passed": False,
            "stage_e_observed_acceptance_candidate_failures": [
                {
                    "check": "metric.micro_turn_event_budget_observed",
                    "metric": "micro_turn_event_budget_observed",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "packaging_metadata_passed": True,
        },
        "release_checklist": {
            "passed": False,
            "profile_name": "release",
            "managed_output_paths_ok": True,
            "report_summary_review_ready": True,
            "release_notes_reviewed": True,
            "extended_profile_ready": False,
        },
    }

    summary = module.format_release_summary(report)

    assert "overall_status: WARN" in summary
    assert "Accuracy" in summary
    assert "- status: WARN" in summary
    assert "- suite_name: missing" in summary
    assert "- stage_a_status: WARN" in summary
    assert "- stage_a_acc_target_met: False" in summary
    assert "- stage_a_zero_regressions: False" in summary
    assert "- stage_b_status: WARN" in summary
    assert "- stage_b_readiness_score: 0.000" in summary
    assert "- stage_b_branching_ready: False" in summary
    assert "- stage_b_simulation_ready: False" in summary
    assert "- stage_b_operator_coverage_ready: False" in summary
    assert "- stage_b_operator_consistency_ready: False" in summary
    assert "- stage_b_counterfactual_viability_ready: False" in summary
    assert "- stage_b_speculative_acceptance_ready: False" in summary
    assert "- stage_b_speculative_rollback_ready: False" in summary
    assert "- stage_b_focused_retrieval_observed: False" in summary
    assert "- stage_b_branch_decision_consistency_observed: False" in summary
    assert "- stage_b_promotion_candidate_ready: False" in summary
    assert "- stage_b_promotion_candidate_failure_count: 3" in summary
    assert "- stage_b_promotion_consecutive_passes: 0" in summary
    assert "- stage_b_promotion_required_streak: 3" in summary
    assert "- stage_b_promotion_recommended: False" in summary
    assert "- stage_b_promotion_next_step_hint: " in summary
    assert "Phase 3 Focus" in summary
    assert "- few_shot_status: WARN" in summary
    assert "Gate" in summary
    assert "- error_count: 1" in summary
    assert "- accuracy_required: True" in summary
    assert "- embedded_accuracy_present: False" in summary
    assert "- stage_a_passed: False" in summary
    assert "- stage_b_passed: False" in summary
    assert "- stage_b_minimum_requirements_passed: False" in summary
    assert "- stage_c_passed: False" in summary
    assert "- stage_c_minimum_requirements_passed: False" in summary
    assert "- stage_d_passed: False" in summary
    assert "- stage_d_minimum_requirements_passed: False" in summary
    assert "- stage_d_readiness_score: 0.000" in summary
    assert "- stage_d_replay_noise_resilience_integrity: 0.000" in summary
    assert "- stage_d_astro_modulation_stability: 0.000" in summary
    assert "- stage_d_delta_memory_candidate_failure: metric.delta_memory_multi_history_recall_observed value=0.000 required>=1.000 description=delta-memory multi-history recall" in summary
    assert "- stage_d_acceptance_candidate_failure: metric.delta_memory_erase_write_decoupling_observed value=0.000 required>=1.000 description=delta-memory separate erase and write gates" in summary
    assert "- stage_e_passed: False" in summary
    assert "- stage_e_minimum_requirements_passed: False" in summary
    assert "- stage_e_readiness_score: 0.000" in summary
    assert "- stage_e_observed_acceptance_candidate_failure: metric.micro_turn_event_budget_observed value=0.000 required>=1.000 description=micro turn event budget observed" in summary
    assert "- stage_e_causal_candidate_trace_integrity: 0.000" in summary
    assert "- stage_e_module_orchestration_integrity: 0.000" in summary
    assert "- stage_e_counterfactual_lane_integrity: 0.000" in summary
    assert "- stage_e_action_trace_observability: 0.000" in summary
    assert "- stage_e_runtime_trace_replay_consistency: 0.000" in summary
    assert "- stage_e_manifold_trace_support_observed: 0.000" in summary
    assert "- stage_e_manifold_trace_recall_observed: 0.000" in summary
    assert "- stage_e_manifold_trace_scan_budget_observed: 0.000" in summary
    assert "- stage_e_manifold_trace_index_scan_reduction_observed: 0.000" in summary
    assert "- stage_e_manifold_trace_candidate_guard_observed: 0.000" in summary
    assert "- stage_e_predictive_spike_entropy_reduction_observed: 0.000" in summary
    assert "- stage_e_phase_binding_coincidence_integrity_observed: 0.000" in summary
    assert "- stage_e_forward_only_local_update_stability_observed: 0.000" in summary
    assert "- packaging_metadata_passed: True" in summary
    assert "- repair_pending_count: 0" in summary
    assert "- repair_timeout_count: 0" in summary
    assert "- repair_retry_queue_count: 0" in summary
    assert "- repair_plan_steps: 1" in summary
    assert "- repair_plan_coverage: 2/2" in summary
    assert "- repair_step: 1 Embed Accuracy Into Soak Report -> python scripts/eval/release_soak.py --profile release --include-accuracy (covers=release_gate.embedded_accuracy_present, release_gate.accuracy_required)" in summary
    assert "- fallback_plan_steps: 0" in summary
    assert "- iterative_remaining_checks: 2" in summary
    assert "- iterative_next_steps: 1" in summary
    assert "- iterative_completed: False" in summary
    assert "- iterative_stop_reason: pending_actions" in summary
    assert "- iterative_next_step_hint: python scripts/eval/release_soak.py --profile release --include-accuracy" in summary
    assert "- iterative_next_step: 1 Embed Accuracy Into Soak Report -> python scripts/eval/release_soak.py --profile release --include-accuracy (covers=release_gate.embedded_accuracy_present, release_gate.accuracy_required)" in summary
    assert "- recovery_action: Embed Accuracy Into Soak Report -> python scripts/eval/release_soak.py --profile release --include-accuracy (priority=high, effect=Restores missing embedded Phase 3 evidence in the soak report., affected_checks=release_gate.embedded_accuracy_present, release_gate.accuracy_required, reason=Attach Phase 3 report to the release soak artifact.)" in summary
    assert "embedded Phase 3 accuracy" in summary
    assert "- error_detail_count: 1" in summary
    assert "- error_detail: type=general_error, category=release_gate.embedded_accuracy_present" in summary
    assert "- error_detail_total: 1" in summary
    assert "- error_detail_type_count: general_error=1" in summary
    assert "- error_detail_category_count: release_gate.embedded_accuracy_present=1" in summary
    assert "- failure_focus_primary_category: release_gate.embedded_accuracy_present" in summary
    assert "- failure_focus_confidence: 1.000" in summary
    assert "- failure_focus_primary_action_title: Embed Accuracy Into Soak Report" in summary
    assert "- retrieval_hygiene_status: WARN" in summary
    assert "- retrieval_hygiene_trend: NEW" in summary
    assert "- adaptive_readiness_status: WARN" in summary
    assert "- adaptive_readiness_trend: NEW" in summary
    assert "- adaptation_parameter_integrity: 0.000" in summary
    assert "- adaptation_parameter_integrity_trend: NEW" in summary
    assert "- adaptation_parameter_integrity_delta: +0.000" in summary
    assert "- direction_shift_following: 0.000" in summary
    assert "- direction_shift_trend: NEW" in summary
    assert "- direction_shift_delta: +0.000" in summary
    assert "- predictive_readiness_status: WARN" in summary
    assert "- predictive_readiness_trend: NEW" in summary
    assert "- predictive_command_trend: NEW" in summary
    assert "- predictive_command_delta: +0.000" in summary
    assert "- predictive_shift_trend: NEW" in summary
    assert "- predictive_shift_delta: +0.000" in summary
    assert "- predictive_simulation_trend: NEW" in summary
    assert "- predictive_simulation_delta: +0.000" in summary
    assert "- efficiency_readiness_status: WARN" in summary
    assert "- performance_energy_ratio_proxy: 0.000" in summary
    assert "- ann_cost_advantage_proxy: 0.000" in summary
    assert "- sparse_event_cost_score: 0.000" in summary
    assert "- brain_efficiency_alignment_proxy: 0.000" in summary
    assert "- memory_per_success_proxy: 0.000" in summary
    assert "- memory_per_success_trend: NEW" in summary
    assert "- stochastic_readout_integrity: 0.000" in summary
    assert "- stochastic_readout_trend: NEW" in summary
    assert "- stochastic_readout_delta: +0.000" in summary
    assert "- memory_per_success_delta: +0.000" in summary
    assert "- efficiency_readiness_trend: NEW" in summary
    assert "- consolidation_readiness_status: WARN" in summary
    assert "- consolidation_readiness_score: 0.000" in summary
    assert "- consolidation_replay_noise_resilience_integrity: 0.000" in summary
    assert "- consolidation_astro_modulation_stability: 0.000" in summary
    assert "- consolidation_readiness_trend: NEW" in summary
    assert "Checklist" in summary
    assert "- status: WARN" in summary


def test_release_soak_summary_includes_auto_dispatch_status():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
            "require_phase3_accuracy": False,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "pyproject_version": "1.1.0",
            "versions_match": True,
            "has_expected_console_scripts": True,
            "console_scripts": ["sara-chat", "sara-train"],
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": True,
            "error_count": 0,
            "errors": [],
            "accuracy_required": False,
            "embedded_accuracy_present": False,
            "stage_a_passed": False,
            "stage_b_passed": False,
            "stage_b_minimum_requirements_passed": False,
            "stage_c_passed": False,
            "stage_c_minimum_requirements_passed": False,
            "stage_d_passed": False,
            "stage_d_minimum_requirements_passed": False,
            "packaging_metadata_passed": True,
            "repair_pending_count": 1,
            "repair_timeout_count": 0,
            "repair_retry_queue_count": 1,
            "repair_retry_queue": [
                {
                    "command": "python scripts/eval/release_gate.py",
                    "reason": "failed",
                    "next_attempt": 2,
                    "max_attempts": 2,
                    "covered_checks": ["release_gate.errors"],
                }
            ],
            "repair_plan": {
                "estimated_steps": 0,
                "covered_checks": [],
                "uncovered_checks": [],
                "selected_actions": [],
                "fallback_actions": [],
            },
            "iterative_repair_plan": {
                "remaining_checks": [],
                "next_actions": [],
                "completed": True,
                "stop_reason": "no_target_checks",
            },
            "recovery_actions": [],
            "stage_b_minimum_failure_count": 0,
            "stage_b_minimum_failures": [],
            "stage_c_minimum_failure_count": 0,
            "stage_c_minimum_failures": [],
            "stage_d_minimum_failure_count": 0,
            "stage_d_minimum_failures": [],
        },
        "repair_auto_dispatch": {
            "requested": 2,
            "candidate_count": 3,
            "eligible_count": 2,
            "selected_count": 2,
            "selected_unique_check_count": 2,
            "min_priority_tier": "medium",
            "selection_mode": "priority_diversified",
            "max_per_check": 1,
            "dispatched": 1,
            "dispatched_commands": [
                "python scripts/eval/release_soak.py --profile extended --include-accuracy"
            ],
            "skipped_pending_commands": [
                "python scripts/eval/release_gate.py"
            ],
            "skipped_limit_commands": [
                "python scripts/eval/phase3_accuracy_suite.py"
            ],
            "skipped_low_priority_commands": [
                "python scripts/eval/future_state_consistency.py"
            ],
            "skipped_low_priority_count": 1,
            "skipped_check_quota_commands": [
                "python scripts/eval/release_soak.py --profile release --include-accuracy"
            ],
            "skipped_check_quota_count": 1,
        },
        "release_checklist": {
            "passed": True,
            "profile_name": "release",
            "managed_output_paths_ok": True,
            "report_summary_review_ready": True,
            "release_notes_reviewed": True,
            "extended_profile_ready": False,
        },
    }

    summary = module.format_release_summary(report)

    assert "Gate" in summary
    assert "- auto_dispatch_requested: 2" in summary
    assert "- auto_dispatch_candidates: 3" in summary
    assert "- auto_dispatch_eligible: 2" in summary
    assert "- auto_dispatch_selected: 2" in summary
    assert "- auto_dispatch_selected_unique_checks: 2" in summary
    assert "- auto_dispatch_min_priority_tier: medium" in summary
    assert "- auto_dispatch_selection_mode: priority_diversified" in summary
    assert "- auto_dispatch_max_per_check: 1" in summary
    assert "- auto_dispatch_dispatched: 1" in summary
    assert "- auto_dispatch_skipped_pending: 1" in summary
    assert "- auto_dispatch_skipped_limit: 1" in summary
    assert "- auto_dispatch_skipped_low_priority: 1" in summary
    assert "- auto_dispatch_skipped_check_quota: 1" in summary
    assert "- auto_dispatch_command: python scripts/eval/release_soak.py --profile extended --include-accuracy" in summary
    assert "- auto_dispatch_skipped_pending_command: python scripts/eval/release_gate.py" in summary
    assert "- auto_dispatch_skipped_limit_command: python scripts/eval/phase3_accuracy_suite.py" in summary
    assert "- auto_dispatch_skipped_low_priority_command: python scripts/eval/future_state_consistency.py" in summary
    assert "- auto_dispatch_skipped_check_quota_command: python scripts/eval/release_soak.py --profile release --include-accuracy" in summary
    assert "retry_queue_entry: python scripts/eval/release_gate.py (reason=failed, attempt=2/2, priority=" in summary
