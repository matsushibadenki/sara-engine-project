from sara_engine.learning.synaptic_tag import SynapticTagConfig, evaluate_synaptic_tags


def test_synaptic_tags_rank_replay_worthy_connection_above_pruning_candidate():
    trace = [
        {"step": 1, "pre_id": "task", "post_id": "answer", "pre_step": 1, "post_step": 2, "weight": 0.70, "replayed": True, "replay_useful": True},
        {"step": 3, "pre_id": "task", "post_id": "answer", "pre_step": 3, "post_step": 4, "weight": 0.75, "replayed": True, "replay_useful": True},
        {"step": 5, "pre_id": "task", "post_id": "answer", "pre_step": 5, "post_step": 6, "weight": 0.82, "replayed": True, "replay_useful": True},
        {"step": 2, "pre_id": "noise", "post_id": "drift", "pre_step": 2, "post_step": 8, "weight": 0.08, "replayed": True, "replay_useful": False},
        {"step": 9, "pre_id": "noise", "post_id": "drift", "pre_step": 9, "post_step": 14, "weight": 0.06, "replayed": True, "replay_useful": False},
    ]

    report = evaluate_synaptic_tags(trace, SynapticTagConfig(state_budget=4))

    assert report["observed_only"] is True
    assert report["metrics"]["synaptic_tag_integrity"] == 1.0
    assert report["metrics"]["synaptic_tag_state_budget_observed"] == 1.0
    assert report["tags"][0]["pre_id"] == "task"
    assert report["tags"][0]["post_id"] == "answer"
    assert report["tags"][0]["tag"] == "consolidate"
    assert report["tags"][-1]["pruning_candidate"] is True


def test_synaptic_tags_report_budget_pressure_without_mutating_tags():
    trace = [
        {"step": index, "pre_id": f"pre-{index}", "post_id": f"post-{index}", "weight": 0.4}
        for index in range(3)
    ]

    report = evaluate_synaptic_tags(trace, SynapticTagConfig(state_budget=2))

    assert len(report["tags"]) == 3
    assert report["metrics"]["synaptic_tag_state_budget_observed"] == 0.0
