import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.encoders.time_series import TimeSeriesCurrentEncoder
from sara_engine.learning.force_io import export_force_artifact, load_force_artifact
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_script_module(relative_parts):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", *relative_parts)
    )
    spec = importlib.util.spec_from_file_location("_dynamic_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_force_artifact_roundtrip_preserves_readout_and_synapses():
    encoder = TimeSeriesCurrentEncoder()
    reservoir = LiquidReservoir(
        n_neurons=12,
        p_connect=0.15,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
    )

    drive = [0.5] + [0.0] * 11
    for _ in range(8):
        reservoir.train_force(drive, [0.3])

    artifact_path = model_path("test_force_roundtrip", "force_run.json")
    report_path = workspace_path("force_roundtrip", "report.json")
    export_force_artifact(
        artifact_path=artifact_path,
        report_path=report_path,
        reservoir=reservoir,
        encoder=encoder,
        metadata={"series_name": "roundtrip_test"},
    )

    restored_reservoir, restored_encoder, metadata = load_force_artifact(artifact_path)

    assert metadata["series_name"] == "roundtrip_test"
    assert restored_encoder.amplitude == encoder.amplitude
    assert restored_reservoir.synapses == reservoir.synapses
    assert restored_reservoir.force_readout is not None
    assert reservoir.force_readout is not None
    assert restored_reservoir.force_readout.weights == reservoir.force_readout.weights

    artifact_dir = os.path.dirname(artifact_path)
    assert os.path.exists(os.path.join(artifact_dir, "latest_force_run.json"))
    assert os.path.exists(os.path.join(artifact_dir, "best_force_run.json"))


def test_force_artifact_best_file_updates_only_for_better_score():
    encoder = TimeSeriesCurrentEncoder()
    artifact_dir = model_path("test_force_best")
    artifact_path = os.path.join(artifact_dir, "force_run.json")
    report_path = workspace_path("test_force_best", "report.json")

    first_reservoir = LiquidReservoir(
        n_neurons=10,
        p_connect=0.1,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
    )
    second_reservoir = LiquidReservoir(
        n_neurons=10,
        p_connect=0.1,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
    )

    export_force_artifact(
        artifact_path=artifact_path,
        report_path=report_path,
        reservoir=first_reservoir,
        encoder=encoder,
        metadata={"series_name": "best_test", "final_test_mae": 0.4},
    )
    best_path = os.path.join(artifact_dir, "best_force_run.json")
    with open(best_path, "r", encoding="utf-8") as handle:
        first_best = json.load(handle)

    export_force_artifact(
        artifact_path=artifact_path,
        report_path=report_path,
        reservoir=second_reservoir,
        encoder=encoder,
        metadata={"series_name": "best_test", "final_test_mae": 0.6},
    )
    with open(best_path, "r", encoding="utf-8") as handle:
        second_best = json.load(handle)

    assert first_best["metadata"]["final_test_mae"] == second_best["metadata"]["final_test_mae"]


def test_force_infer_cli_runs_on_exported_artifact():
    train_module = _load_script_module(["scripts", "train", "train_force_cli.py"])
    infer_module = _load_script_module(["scripts", "eval", "force_infer_cli.py"])

    series = train_module.build_sine_series(length=80, frequency=0.09)
    train_signal, test_signal = train_module.split_series(series, test_ratio=0.25)
    encoder = train_module.TimeSeriesCurrentEncoder()
    reservoir = train_module.LiquidReservoir(
        n_neurons=24,
        p_connect=0.12,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
        force_alpha=1.0,
        force_forgetting_factor=0.999,
    )
    train_module.train_sequence(
        reservoir,
        encoder,
        train_signal,
        epochs=10,
        report_every=5,
    )

    artifact_path = model_path("test_force_infer", "force_run.json")
    report_path = workspace_path("test_force_infer", "report.json")
    export_force_artifact(
        artifact_path=artifact_path,
        report_path=report_path,
        reservoir=reservoir,
        encoder=encoder,
        metadata={"series_name": "infer_test"},
    )

    results = infer_module.run_inference(test_signal, artifact_path=artifact_path)

    assert results["metadata"]["series_name"] == "infer_test"
    assert results["predictions"]
    assert results["mae"] < 0.5


def test_force_infer_cli_resolves_best_and_latest_variants():
    infer_module = _load_script_module(["scripts", "eval", "force_infer_cli.py"])
    model_dir = model_path("force_generation_demo")

    auto_path = infer_module.resolve_artifact_path(
        artifact_path="",
        model_dir=model_dir,
        artifact_variant="auto",
    )
    best_path = infer_module.resolve_artifact_path(
        artifact_path="",
        model_dir=model_dir,
        artifact_variant="best",
    )
    latest_path = infer_module.resolve_artifact_path(
        artifact_path="",
        model_dir=model_dir,
        artifact_variant="latest",
    )
    explicit_path = infer_module.resolve_artifact_path(
        artifact_path="/tmp/custom_force_run.json",
        model_dir=model_dir,
        artifact_variant="best",
    )

    assert auto_path.endswith("best_force_run.json")
    assert best_path.endswith("best_force_run.json")
    assert latest_path.endswith("latest_force_run.json")
    assert explicit_path == "/tmp/custom_force_run.json"
