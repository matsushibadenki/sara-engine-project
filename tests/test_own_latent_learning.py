import importlib.util
import json
import os
import sys

from sara_engine.learning.own_latent import SparseOwnLatentPredictor, jaccard_overlap
from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_benchmark_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "own_latent_learning_benchmark.py")
    )
    spec = importlib.util.spec_from_file_location("own_latent_learning_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixture_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "build_own_latent_rhm_fixture.py")
    )
    spec = importlib.util.spec_from_file_location("build_own_latent_rhm_fixture", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_own_latent_predictor_uses_local_sparse_state():
    predictor = SparseOwnLatentPredictor(width=128, max_events=8)
    predictor.update(
        context_text="sparrow wing glide feather sky",
        latent_terms=["animal_dynamics", "winged_agent", "flight_motion"],
        label="avian_motion",
    )
    prediction = predictor.predict("falcon wing sky")

    assert prediction.label == "avian_motion"
    assert prediction.event_cost > 0
    assert prediction.state_budget_units > 0
    assert prediction.trace["context_event_count"] > 0
    assert jaccard_overlap(prediction.predicted_signature, prediction.predicted_signature) == 1.0


def test_sparse_own_latent_prioritizes_exact_label_events():
    predictor = SparseOwnLatentPredictor(width=128, max_events=8)
    predictor.update(
        context_text="avian motion wing sky",
        latent_terms=["animal_dynamics", "winged_agent"],
        label="avian_motion",
    )
    predictor.update(
        context_text="aquatic motion fin water",
        latent_terms=["animal_dynamics", "water_agent"],
        label="aquatic_motion",
    )
    prediction = predictor.predict("aquatic motion context")

    assert prediction.label == "aquatic_motion"
    assert prediction.trace["candidate_labels"][0]["label"] == "aquatic_motion"


def test_own_latent_fixture_generator_writes_repository_safe_cases():
    fixture = _load_fixture_module()
    path = processed_data_path("benchmark_fixtures", "test_own_latent_rhm_cases.jsonl")

    exit_code = fixture.main(["--output-path", path, "--train-per-group", "2", "--eval-per-group", "1"])

    assert exit_code == 0
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 12
    assert {row["split"] for row in rows} == {"train", "eval"}
    assert all(row["expected_behavior"] == "recover_latent_group" for row in rows)


def test_own_latent_benchmark_writes_observed_only_report():
    benchmark = _load_benchmark_module()
    fixture_path = processed_data_path("benchmark_fixtures", "test_own_latent_benchmark_cases.jsonl")
    report_path = workspace_path("evaluation", "test_own_latent_learning_benchmark.json")
    summary_path = workspace_path("evaluation", "test_own_latent_learning_benchmark_summary.txt")
    history_path = workspace_path("evaluation", "test_own_latent_learning_history.json")

    exit_code = benchmark.main(
        [
            "--fixture-path",
            fixture_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--history-path",
            history_path,
            "--train-sizes",
            "4,8",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["schema"] == "sara-own-latent-learning-benchmark-v1"
    assert report["observed_only"] is True
    assert report["passed"] is True
    assert report["metrics"]["own_latent_sample_efficiency_ok"] == 1.0
    assert os.path.exists(summary_path)
    assert os.path.exists(history_path)
