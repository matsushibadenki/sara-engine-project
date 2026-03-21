import os
import sys

import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.encoders.time_series import TimeSeriesCurrentEncoder
from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.learning.force_io import load_force_artifact
from sara_engine.learning.force_workflow import load_series, split_series
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.models.readout_layer import SpikeReadoutLayer
from sara_engine.models.unified_snn import UnifiedSNNModel
from sara_engine.utils.project_paths import model_path, workspace_path


def test_load_series_accepts_json_dict_payload():
    series_path = workspace_path("tests", "force_series_dict.json")
    os.makedirs(os.path.dirname(series_path), exist_ok=True)
    with open(series_path, "w", encoding="utf-8") as handle:
        handle.write('{"values": [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]}')

    values = load_series(series_path)

    assert values[3] == 0.75
    assert len(values) == 8


def test_load_series_rejects_non_finite_values():
    series_path = workspace_path("tests", "force_series_invalid.txt")
    os.makedirs(os.path.dirname(series_path), exist_ok=True)
    with open(series_path, "w", encoding="utf-8") as handle:
        handle.write("0.0\n1.0\nnan\n2.0\n")

    with pytest.raises(ValueError, match="non-finite"):
        load_series(series_path)


def test_split_series_rejects_too_short_signal():
    with pytest.raises(ValueError, match="at least 8 values"):
        split_series([0.0, 1.0, 0.5, -0.5], test_ratio=0.25)


def test_unified_snn_resets_dynamic_state_before_sequence_run():
    reservoir = LiquidReservoir(
        n_neurons=6,
        p_connect=0.0,
        enable_force_readout=False,
    )
    readout = SpikeReadoutLayer(d_model=6, vocab_size=3)
    model = UnifiedSNNModel(reservoir=reservoir, spike_readout=readout)

    priming_input = [12.0] + [0.0] * 5
    model.step(priming_input, learning=False)

    outputs = model.run_sequence(
        [([0.0] * 6), ([0.0] * 6)],
        learning=False,
        reset_state_before_run=True,
    )

    assert outputs[0]["spikes"] == []
    assert outputs[1]["spikes"] == []


def test_unified_snn_validates_auxiliary_sequence_lengths():
    reservoir = LiquidReservoir(
        n_neurons=4,
        p_connect=0.0,
        enable_force_readout=False,
    )
    model = UnifiedSNNModel(reservoir=reservoir)

    with pytest.raises(ValueError, match="target_future_spikes length"):
        model.run_sequence(
            [[0.0] * 4, [0.0] * 4],
            target_future_spikes=[[1]],
            learning=False,
        )


def test_agent_session_restores_topic_tracker_state():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    session_path = workspace_path("tests", "agent_session.pkl")

    agent.chat(
        "Pythonのリスト内包表記は、既存のリストから条件付きで新しいリストを作る構文です。",
        teaching_mode=True,
    )
    agent.chat("その構文の利点は何ですか？", teaching_mode=False)

    before_terms = agent.topic_tracker.active_terms(limit=4)
    assert before_terms

    restored = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    agent.save_session(session_path)
    restored.load_session(session_path)

    after_terms = restored.topic_tracker.active_terms(limit=4)
    assert after_terms == before_terms


def test_agent_session_restores_runtime_issues():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    session_path = workspace_path("tests", "agent_session_with_issues.pkl")
    os.makedirs(os.path.dirname(session_path), exist_ok=True)

    def failing_tool(_: str) -> str:
        raise RuntimeError("persistent tool failure")

    agent.register_tool("<FAIL>", failing_tool)
    agent.chat("確認 <FAIL>", teaching_mode=False)
    assert agent.get_recent_issues()

    restored = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    agent.save_session(session_path)
    restored.load_session(session_path)

    restored_issues = restored.get_recent_issues()
    assert restored_issues
    assert any(issue["stage"] == "tool_execution" for issue in restored_issues)


def test_agent_formats_recent_issues():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    assert agent.format_recent_issues() == "No runtime issues recorded."

    def failing_tool(_: str) -> str:
        raise RuntimeError("format test failure")

    agent.register_tool("<FAIL>", failing_tool)
    agent.chat("確認 <FAIL>", teaching_mode=False)

    diagnostics = agent.format_recent_issues(limit=2)
    assert "Recent runtime issues:" in diagnostics
    assert "[tool_execution]" in diagnostics


def test_unified_snn_force_readout_matches_single_reservoir_step():
    reservoir = LiquidReservoir(
        n_neurons=8,
        p_connect=0.0,
        enable_force_readout=True,
        force_output_dim=1,
    )
    model = UnifiedSNNModel(reservoir=reservoir)
    currents = [0.3] * 8

    before_time = reservoir.current_time
    result = model.step(currents, learning=False)

    assert reservoir.current_time == before_time + reservoir.dt
    assert isinstance(result["readout"], list)
    assert len(result["readout"]) == 1


def test_load_force_artifact_rejects_mismatched_synapse_rows():
    artifact_path = model_path("tests", "invalid_force_artifact.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    encoder = TimeSeriesCurrentEncoder()
    payload = {
        "metadata": {"series_name": "invalid"},
        "encoder": {
            "amplitude": encoder.amplitude,
            "delta_scale": encoder.delta_scale,
            "quadratic_scale": encoder.quadratic_scale,
            "magnitude_scale": encoder.magnitude_scale,
            "band_growth": encoder.band_growth,
        },
        "reservoir": {
            "n_neurons": 4,
            "dt": 1.0,
            "max_weight": 2.0,
            "max_delay_limit": 50,
            "readout_decay": 0.9,
            "synapses": [{}, {}, {}],
            "is_inhibitory": [False, False, False, False],
        },
        "force_readout": {
            "weights": [[0.0] * 8],
            "bias": [0.0],
            "inverse_correlation": [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)],
            "alpha": 1.0,
            "forgetting_factor": 1.0,
            "weight_clip": 10.0,
        },
    }
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="synapse row count"):
        load_force_artifact(artifact_path)


def test_load_force_artifact_rejects_invalid_readout_shapes():
    artifact_path = model_path("tests", "invalid_force_readout.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    encoder = TimeSeriesCurrentEncoder()
    payload = {
        "metadata": {"series_name": "invalid_readout"},
        "encoder": {
            "amplitude": encoder.amplitude,
            "delta_scale": encoder.delta_scale,
            "quadratic_scale": encoder.quadratic_scale,
            "magnitude_scale": encoder.magnitude_scale,
            "band_growth": encoder.band_growth,
        },
        "reservoir": {
            "n_neurons": 4,
            "dt": 1.0,
            "max_weight": 2.0,
            "max_delay_limit": 50,
            "readout_decay": 0.9,
            "synapses": [{}, {}, {}, {}],
            "is_inhibitory": [False, False, False, False],
        },
        "force_readout": {
            "weights": [[0.0] * 3],
            "bias": [0.0],
            "inverse_correlation": [[1.0, 0.0], [0.0, 1.0]],
            "alpha": 1.0,
            "forgetting_factor": 1.0,
            "weight_clip": 10.0,
        },
    }
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="weights must contain rows of length"):
        load_force_artifact(artifact_path)
