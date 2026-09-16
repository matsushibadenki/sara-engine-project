from copy import deepcopy

from sara_engine.evaluation.structural_revision_transfer import (
    decide,
    load_protocol,
    make_streams,
    run_arm,
    shared_routes,
)


def development_protocol():
    protocol = deepcopy(load_protocol())
    protocol["training_episodes"] = 288
    protocol["seen_pre_revision_evaluation_episodes"] = 144
    protocol["held_out_pre_revision_evaluation_episodes"] = 96
    protocol["held_out_post_revision_evaluation_episodes"] = 192
    protocol["early_post_revision_episodes"] = 96
    protocol["adaptation_horizon"] = 96
    return protocol


def test_split_holds_out_compositions_but_shares_every_feature_value():
    protocol = development_protocol()
    training, _, held_pre, held_post = make_streams(protocol, 91)
    seen_addresses = {(row.left, row.right, row.interval) for row in training}
    held_addresses = {(row.left, row.right, row.interval) for row in held_pre}
    assert seen_addresses.isdisjoint(held_addresses)
    assert len(seen_addresses) == 144
    assert len(held_addresses) == 48
    seen_features = {route for row in training for route in shared_routes(row, protocol["symbols"])}
    held_features = {route for row in held_pre for route in shared_routes(row, protocol["symbols"])}
    assert seen_features == held_features == set(range(19))
    assert sum(row.revision_notice for row in held_post) == 1
    assert held_post[0].revision_notice and held_post[0].revision == 2


def test_revision_flips_the_same_held_out_composition_labels():
    protocol = development_protocol()
    _, _, held_pre, held_post = make_streams(protocol, 91)
    before = {(row.left, row.right, row.interval): row.label for row in held_pre}
    assert all(before[(row.left, row.right, row.interval)] == 1 - row.label for row in held_post)


def test_shared_snn_and_scalar_predictions_match_exactly():
    protocol = development_protocol()
    streams = make_streams(protocol, 91)
    snn = run_arm(protocol, "revision_gain_shared_snn", streams, 91)
    scalar = run_arm(protocol, "revision_gain_shared_scalar", streams, 91)
    assert snn["prediction_trace_sha256"] == scalar["prediction_trace_sha256"]
    assert snn["resources"]["units"] == 19
    assert scalar["resources"]["units"] == 0


def test_atomic_routes_cannot_reuse_training_state_for_held_out_compositions():
    protocol = development_protocol()
    protocol["held_out_pre_revision_evaluation_episodes"] = 48
    streams = make_streams(protocol, 91)
    atomic = run_arm(protocol, "revision_gain_atomic_snn", streams, 91)
    assert atomic["held_out_pre_revision"]["brier"] == 0.25


def test_decision_requires_every_registered_gate():
    protocol = development_protocol()

    def result(held=0.95, early=0.95, post=0.01, seen=0.96, latency=100, digest="same"):
        metric = lambda accuracy, brier: {"accuracy": accuracy, "brier": brier, "count": 1}
        return {
            "seen_pre_revision": metric(seen, 0.01),
            "held_out_pre_revision": metric(held, 0.01),
            "early_held_out_post_revision": metric(early, post),
            "late_held_out_post_revision": metric(1.0, post),
            "full_held_out_post_revision": metric(early, post),
            "adaptation_latency_episodes": latency,
            "prediction_trace_sha256": digest,
            "resources": {"contracts_passed": True},
        }

    arms = {
        "revision_gain_shared_snn": result(),
        "previous_policy_shared_snn": result(early=0.80, post=0.05),
        "always_three_factor_shared_snn": result(early=0.70, post=0.08),
        "always_residual_shared_snn": result(early=0.80, post=0.05),
        "revision_gain_shared_scalar": result(),
        "revision_gain_atomic_snn": result(held=0.50, early=0.60, post=0.20),
        "shuffled_feedback_shared_snn": result(held=0.50, early=0.50, post=0.25),
    }
    rows = [{"seed": 1, "arms": arms}]
    assert decide(protocol, rows)["passed"]
    arms["revision_gain_shared_scalar"]["prediction_trace_sha256"] = "different"
    assert not decide(protocol, rows)["passed"]
