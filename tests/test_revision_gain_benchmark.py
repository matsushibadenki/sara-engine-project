from copy import deepcopy

from sara_engine.evaluation.revision_gain_benchmark import decide, load_protocol, make_streams, run_arm


def development_protocol():
    protocol = deepcopy(load_protocol())
    protocol["training_episodes"] = 192
    protocol["pre_revision_evaluation_episodes"] = 192
    protocol["post_revision_evaluation_episodes"] = 384
    protocol["early_post_revision_episodes"] = 192
    protocol["adaptation_horizon"] = 192
    return protocol


def test_stream_has_one_visible_revision_and_hidden_mapping_change_only_in_true_scenario():
    protocol = development_protocol()
    _, true_evaluation = make_streams(protocol, 71, "true_revision")
    _, false_evaluation = make_streams(protocol, 71, "false_revision")
    notices = [index for index, episode in enumerate(true_evaluation) if episode.revision_notice]
    assert notices == [192]
    assert true_evaluation[192].revision == false_evaluation[192].revision == 2
    assert [episode.label for episode in true_evaluation[:192]] == [episode.label for episode in false_evaluation[:192]]
    assert all(left.label == 1 - right.label for left, right in zip(true_evaluation[192:], false_evaluation[192:]))


def test_observable_snn_and_matched_scalar_have_identical_predictions():
    protocol = development_protocol()
    training, evaluation = make_streams(protocol, 71, "true_revision")
    snn = run_arm(protocol, "revision_gain_snn", training, evaluation, 81)
    scalar = run_arm(protocol, "revision_gain_scalar", training, evaluation, 81)
    assert snn["prediction_trace_sha256"] == scalar["prediction_trace_sha256"]
    assert snn["resources"]["units"] == 192
    assert scalar["resources"]["units"] == 0


def test_decision_requires_all_gates():
    protocol = development_protocol()
    def result(pre, early, post, latency=100, digest="same"):
        segment = lambda accuracy, brier: {"accuracy": accuracy, "brier": brier, "count": 1}
        return {"pre_revision": segment(1.0, pre), "early_post_revision": segment(early, post),
                "late_post_revision": segment(1.0, post), "full_post_revision": segment(early, post),
                "adaptation_latency_episodes": latency, "prediction_trace_sha256": digest,
                "resources": {"contracts_passed": True}}
    def row(scenario):
        arms = {
            "revision_gain_snn": result(0.0, 0.95, 0.01),
            "previous_policy_snn": result(0.0, 0.8, 0.05),
            "always_three_factor_snn": result(0.0, 0.7, 0.08),
            "always_residual_snn": result(0.01, 0.8, 0.05),
            "ignored_revision_snn": result(0.0, 0.7, 0.08),
            "revision_gain_scalar": result(0.0, 0.95, 0.01),
            "shuffled_feedback_snn": result(0.3, 0.5, 0.25),
        }
        if scenario == "false_revision":
            arms["always_three_factor_snn"] = result(0.0, 1.0, 0.01)
        return {"scenario": scenario, "seed": 1, "arms": arms}
    rows = [row("true_revision"), row("false_revision")]
    assert decide(protocol, rows)["passed"]
    rows[0]["arms"]["revision_gain_scalar"]["prediction_trace_sha256"] = "different"
    assert not decide(protocol, rows)["passed"]
