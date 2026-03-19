import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.learning.force_io import export_force_artifact, load_force_artifact
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_script_module(relative_parts):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", *relative_parts)
    )
    spec = importlib.util.spec_from_file_location("_dynamic_resume_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_force_cli_continues_from_saved_artifact():
    train_module = _load_script_module(["scripts", "train", "train_force_cli.py"])
    resume_module = _load_script_module(["scripts", "train", "resume_force_cli.py"])

    series = train_module.build_sine_series(length=100, frequency=0.09)
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

    train_module.train_force_sequence(
        reservoir,
        encoder,
        train_signal,
        epochs=4,
        report_every=2,
    )
    baseline = train_module.evaluate_force_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=False,
    )

    artifact_path = model_path("test_resume_force", "force_run.json")
    report_path = workspace_path("test_resume_force", "report.json")
    export_force_artifact(
        artifact_path=artifact_path,
        report_path=report_path,
        reservoir=reservoir,
        encoder=encoder,
        metadata={"series_name": "resume_test", "epochs": 4, "history": []},
    )

    resumed_reservoir, resumed_encoder, resumed_metadata = load_force_artifact(artifact_path)
    resume_module.train_force_sequence(
        resumed_reservoir,
        resumed_encoder,
        train_signal,
        epochs=6,
        report_every=3,
    )
    resumed_metrics = resume_module.evaluate_force_sequence(
        resumed_reservoir,
        resumed_encoder,
        test_signal,
        reset_readout=False,
    )

    assert resumed_metadata["epochs"] == 4
    assert resumed_metrics["mae"] <= baseline["mae"] + 0.02
