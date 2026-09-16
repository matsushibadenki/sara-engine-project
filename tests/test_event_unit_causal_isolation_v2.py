from sara_engine.evaluation.event_unit_causal_isolation import EventUnitV2Learner, generate_episodes, run_v2_development_arm


def test_v2_temporal_arm_has_no_neuron_or_refractory_state():
    episodes = generate_episodes(
        seeds=(11,), count_per_family=2, split="training", namespace="event-unit-v2"
    )
    learner = EventUnitV2Learner("C_temporal_state")
    receipt = learner.predict(episodes[0])
    learner.observe(receipt, episodes[0].label)
    assert learner._neurons == {}


def test_v2_refractory_arm_uses_neurons_and_dendritic_arm_does_not():
    episode = generate_episodes(
        seeds=(12,), count_per_family=1, split="training", namespace="event-unit-v2"
    )[0]
    refractory = EventUnitV2Learner("R_temporal_refractory")
    receipt = refractory.predict(episode); refractory.observe(receipt, episode.label)
    dendritic = EventUnitV2Learner("D_temporal_dendritic")
    receipt = dendritic.predict(episode); dendritic.observe(receipt, episode.label)
    assert refractory._neurons
    assert dendritic._neurons == {}


def test_v2_runs_are_deterministic():
    training = generate_episodes(
        seeds=(13,), count_per_family=4, split="training", namespace="event-unit-v2"
    )
    development = generate_episodes(
        seeds=(13,), count_per_family=2, split="development", namespace="event-unit-v2"
    )
    for arm in EventUnitV2Learner.ARM_NAMES:
        first = run_v2_development_arm(arm, training, development)
        second = run_v2_development_arm(arm, training, development)
        assert first["prediction_trace_sha256"] == second["prediction_trace_sha256"]
