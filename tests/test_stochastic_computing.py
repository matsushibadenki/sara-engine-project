import json
import os

from sara_engine.edge.runtime import SaraEdgeRuntime
from sara_engine.utils.project_paths import workspace_path
from sara_engine.utils.stochastic_computing import StochasticAccumulator, clamp_probability


def test_stochastic_accumulator_preserves_strongest_score():
    accumulator = StochasticAccumulator(bit_count=64, seed=17)
    scores = {"primary": 0.95, "alternative": 0.70, "secondary": 0.35}

    approximated = accumulator.approximate_scores(scores, confidence_weight=0.9)

    assert clamp_probability(-1.0) == 0.0
    assert clamp_probability(2.0) == 1.0
    assert accumulator.argmax(scores, confidence_weight=0.9) == "primary"
    assert approximated["primary"] >= approximated["alternative"]
    assert accumulator.state_units() >= 1


def test_edge_runtime_supports_opt_in_stochastic_readout():
    model_path = workspace_path("tests", "edge_runtime_stochastic.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "context_length": 4,
        "embed_dim": 1,
        "total_readout_size": 4,
        "readout_synapses": [
            {"65": 1.0, "66": 0.3},
            {"65": 0.9, "66": 0.2},
            {},
            {},
        ],
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    runtime = SaraEdgeRuntime(
        model_path,
        use_stochastic_readout=True,
        stochastic_bit_count=64,
        stochastic_seed=19,
    )
    runtime._get_sdr = lambda _delay, _tok: [0, 1]

    deterministic = runtime.forward_step(97, use_stochastic_readout=False)
    runtime.reset_state()
    stochastic = runtime.forward_step(97, use_stochastic_readout=True)

    assert deterministic == 65
    assert stochastic == 65
