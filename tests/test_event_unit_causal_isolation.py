import pytest

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitLearner, generate_episodes, run_development_arm


def test_generator_is_deterministic_and_split_scoped():
    left = generate_episodes(seeds=(1,), count_per_family=2, split="training")
    right = generate_episodes(seeds=(1,), count_per_family=2, split="training")
    development = generate_episodes(seeds=(1,), count_per_family=2, split="development")
    assert left == right
    assert {row.identity for row in left}.isdisjoint(row.identity for row in development)


def test_scalar_and_compact_event_arms_are_exactly_equivalent():
    training = generate_episodes(seeds=(1,), count_per_family=4, split="training")
    development = generate_episodes(seeds=(1,), count_per_family=2, split="development")
    scalar = run_development_arm("A_scalar_local", training, development)
    compact = run_development_arm("B_compact_event", training, development)
    assert scalar == compact
    assert scalar["neuron_count"] == 0


def test_stateful_and_dendritic_arms_retain_explicit_neurons():
    episodes = generate_episodes(seeds=(2,), count_per_family=2, split="training")
    for arm in ("C_stateful_spiking", "D_dendritic_structural"):
        learner = EventUnitLearner(arm)
        prediction = learner.predict(episodes[0])
        learner.observe(prediction, episodes[0].label)
        assert learner._neurons
        assert learner.state_bytes() > 0


def test_stateful_episode_boundary_resets_dynamic_neuron_state():
    episode = generate_episodes(seeds=(4,), count_per_family=1, split="training")[0]
    learner = EventUnitLearner("C_stateful_spiking")
    first = learner.predict(episode)
    learner.observe(first, episode.label)
    for neuron in learner._neurons.values():
        neuron.v = 99.0
        neuron.refractory_time = 2
    second = learner.predict(episode)
    assert second.features == first.features
    learner.observe(second, episode.label)


def test_receipt_and_pending_contracts_fail_closed():
    episode = generate_episodes(seeds=(3,), count_per_family=1, split="training")[0]
    left = EventUnitLearner("A_scalar_local")
    right = EventUnitLearner("A_scalar_local")
    receipt = left.predict(episode)
    with pytest.raises(ValueError):
        left.predict(episode)
    with pytest.raises(ValueError):
        right.observe(receipt, episode.label)
    left.observe(receipt, episode.label)


def test_all_frozen_interventions_are_deterministic_and_bounded():
    training = generate_episodes(seeds=(5,), count_per_family=4, split="training")
    development = generate_episodes(seeds=(5,), count_per_family=2, split="development")
    configurations = (
        ("C_stateful_spiking", {"intervention": "time_shuffle"}),
        ("C_stateful_spiking", {"intervention": "state_reset"}),
        ("C_stateful_spiking", {"intervention": "spike_count_preserving_shuffle"}),
        ("C_stateful_spiking", {"intervention": "refractory_disable"}),
        ("D_dendritic_structural", {"intervention": "branch_assignment_shuffle"}),
        ("D_dendritic_structural", {"outcome_shuffle_seed": 7}),
        ("A_scalar_local", {"capacity_reserve_bytes": 26_240}),
    )
    for arm, options in configurations:
        first = run_development_arm(arm, training, development, **options)
        second = run_development_arm(arm, training, development, **options)
        assert first["prediction_trace_sha256"] == second["prediction_trace_sha256"]
        assert first["maximum_event_work"] <= 512
        assert first["state_bytes"] <= 4_194_304


def test_refractory_disable_is_currently_an_exact_negative_control():
    training = generate_episodes(seeds=(6,), count_per_family=6, split="training")
    development = generate_episodes(seeds=(6,), count_per_family=3, split="development")
    intact = run_development_arm("C_stateful_spiking", training, development)
    disabled = run_development_arm(
        "C_stateful_spiking", training, development, intervention="refractory_disable"
    )
    assert intact["prediction_trace_sha256"] == disabled["prediction_trace_sha256"]
