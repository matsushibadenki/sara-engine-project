from __future__ import annotations

from sara_engine.evaluation.r1_temporal_learning_benchmark import (
    CONTROL_FAMILY,
    ExactSparseSpikingLearner,
    SpikingLocalLearner,
    destroy_timing,
    generate_episode,
    generate_stream,
    holm_adjust,
    paired_bootstrap_interval,
    run_arm,
)


def test_same_multiset_order_pair_changes_label():
    zero = generate_episode(seed=7, index=0, family="same_multiset_reversed_order", generator_family="development_base", split="training")
    one = next(
        generate_episode(seed=7, index=index, family="same_multiset_reversed_order", generator_family="development_base", split="training")
        for index in range(1, 100)
        if generate_episode(seed=7, index=index, family="same_multiset_reversed_order", generator_family="development_base", split="training").label != zero.label
    )
    assert sorted(event.symbol for event in zero.events if event.symbol < 2) == [0, 1]
    assert sorted(event.symbol for event in one.events if event.symbol < 2) == [0, 1]
    assert zero.label != one.label


def test_timing_destruction_preserves_symbols_and_removes_intervals():
    episode = generate_episode(seed=11, index=2, family="interval_coded_cue", generator_family="development_base", split="training")
    destroyed = destroy_timing(episode)
    assert [event.symbol for event in destroyed.events] == [event.symbol for event in episode.events]
    assert [event.time for event in destroyed.events] == list(range(1, len(episode.events) + 1))


def test_spiking_learner_uses_bounded_fixed_topology():
    learner = SpikingLocalLearner()
    stream = generate_stream(seeds=[3], count_per_seed=30, generator_family="development_base", split="training")
    for episode in stream:
        learner.predict(episode)
        learner.learn(episode, episode.label)
    assert len(learner.units) == 24
    assert len(learner.weights) <= 96
    assert learner.manager.trace_count <= 256


def test_exact_sparse_candidate_preserves_order_and_interval_addresses():
    short = ExactSparseSpikingLearner._feature_id(0, 1, 2)
    long = ExactSparseSpikingLearner._feature_id(0, 1, 10)
    reversed_order = ExactSparseSpikingLearner._feature_id(1, 0, 2)
    assert len({short, long, reversed_order}) == 3
    assert 0 <= short < 432
    assert 0 <= long < 432
    assert 0 <= reversed_order < 432


def test_outcome_is_applied_after_prediction():
    learner = SpikingLocalLearner()
    episode = generate_episode(seed=5, index=4, family="ordered_cue_short_gap", generator_family="development_base", split="training")
    prediction, _ = learner.predict(episode)
    assert prediction in (0, 1)
    assert learner.weights == {}
    learner.learn(episode, episode.label)
    assert learner.weights


def test_arm_reports_resource_contracts():
    stream = generate_stream(seeds=[13], count_per_seed=20, generator_family="development_base", split="training")
    budgets = {"max_state_bytes": 65536, "max_event_work_per_episode": 8192, "max_cpu_ms_per_episode": 25}
    result = run_arm("intact_spiking_local", stream[:10], stream[10:], budgets, feedback_seed=19)
    assert result["resources"]["contracts_passed"] is True
    assert result["metrics"]["answered_coverage"] == 1.0


def test_paired_bootstrap_uses_primary_episode_pairs_only():
    candidate = [
        {"family": "ordered_cue_short_gap", "correct": 1},
        {"family": CONTROL_FAMILY, "correct": 0},
    ]
    baseline = [
        {"family": "ordered_cue_short_gap", "correct": 0},
        {"family": CONTROL_FAMILY, "correct": 1},
    ]
    interval = paired_bootstrap_interval(candidate, baseline, seed=23, resamples=100)
    assert interval["gain"] == 1.0
    assert interval["lower"] == 1.0


def test_holm_adjustment_stops_after_first_failed_comparison():
    adjusted = holm_adjust({
        "first": {"one_sided_p": 0.001},
        "second": {"one_sided_p": 0.03},
        "third": {"one_sided_p": 0.04},
    })
    assert adjusted["first"]["rejected_at_0_05"] is True
    assert adjusted["second"]["rejected_at_0_05"] is False
    assert adjusted["third"]["rejected_at_0_05"] is False
