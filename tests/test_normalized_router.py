import math
from sara_engine.learning.normalized_router import calibrate_override_threshold, normalized_override_score, route_normalized

def test_threshold_uses_top_fraction_of_all_predictions():
    assert calibrate_override_threshold([None, 1., 2., 3., 4., None, None, None, None, None], .2) == 3.
    assert math.isinf(calibrate_override_threshold([None] * 9 + [1.], .2))

def test_normalized_router_scores_and_swaps_on_override():
    probabilities = {"a": .5, "b": .3, "c": .2}; scores = {"a": .1, "b": .5, "c": .2}
    score = normalized_override_score(base_predicted="a", base_probabilities=probabilities, base_support=20,
        local_predicted="b", local_scores=scores)
    assert score == .6
    decision = route_normalized(labels=("a", "b", "c"), base_predicted="a", base_probabilities=probabilities,
        base_support=20, local_predicted="b", local_scores=scores, threshold=.5)
    assert decision.overridden and decision.predicted == "b" and dict(decision.probabilities)["b"] == .5
