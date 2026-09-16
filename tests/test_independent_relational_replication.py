from sara_engine.evaluation.independent_relational_replication import (
    CONTEXTS,
    generate_independent_episodes,
    run_independent_arm,
)


def _label(context, values):
    if context == "ascending_or_equal":
        return int(values[1] >= values[0])
    if context == "same_parity":
        return int(values[0] % 2 == values[1] % 2)
    if context == "bounded_jump":
        return int(abs(values[1] - values[0]) <= 2)
    return int(values[2] - values[1] > values[1] - values[0])


def test_independent_generator_is_balanced_unique_and_oracle_aligned():
    rows = generate_independent_episodes(
        seeds=[933019],
        values=[250, 251, 254, 255, 258, 259],
        count_per_context=16,
        split="development",
        namespace="test-independent",
    )
    assert len(rows) == 64
    for context in CONTEXTS:
        context_rows = [row for row in rows if row.context == context]
        assert len({row.values for row in context_rows}) == 16
        assert sum(row.label for row in context_rows) == 8
        assert all(row.label == _label(row.context, row.values) for row in context_rows)


def test_relational_features_transfer_without_development_updates():
    common = {"seeds": [933019], "namespace": "test-independent"}
    training = generate_independent_episodes(
        values=[130, 131, 134, 135, 138, 139],
        count_per_context=32,
        split="training",
        **common,
    )
    development = generate_independent_episodes(
        values=[250, 251, 254, 255, 258, 259],
        count_per_context=16,
        split="development",
        **common,
    )
    categorical = run_independent_arm("categorical_zero_shot", training, development)
    relational = run_independent_arm(
        "contextual_relational_zero_shot", training, development
    )
    assert categorical["accuracy"] == 0.5
    assert relational["accuracy"] == 1.0
    assert relational["development_updates"] == 0
    assert relational["maximum_event_work"] <= 4
