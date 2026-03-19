import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.utils.project_paths import workspace_path


def _load_force_cli_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train", "train_force_cli.py")
    )
    spec = importlib.util.spec_from_file_location("train_force_cli", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_force_cli_loads_json_series_and_splits():
    module = _load_force_cli_module()
    series_path = workspace_path("test_force_cli_series.json")

    with open(series_path, "w", encoding="utf-8") as handle:
        handle.write("[0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0]")

    values = module.load_series(series_path)
    train_values, test_values = module.split_series(values, test_ratio=0.3)

    assert len(values) == 9
    assert len(train_values) >= 4
    assert len(test_values) >= 4
    assert abs(values[2] - 1.0) < 1e-9


def test_force_cli_training_improves_test_mae():
    module = _load_force_cli_module()

    series = module.build_sine_series(length=80, frequency=0.09)
    train_signal, test_signal = module.split_series(series, test_ratio=0.25)
    encoder = module.TimeSeriesCurrentEncoder()
    reservoir = module.LiquidReservoir(
        n_neurons=24,
        p_connect=0.12,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
        force_alpha=1.0,
        force_forgetting_factor=0.999,
    )

    baseline = module.evaluate_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=True,
    )
    history = module.train_sequence(
        reservoir,
        encoder,
        train_signal,
        epochs=12,
        report_every=6,
    )
    final_metrics = module.evaluate_sequence(reservoir, encoder, test_signal)

    assert history
    assert final_metrics["mae"] < baseline["mae"]
