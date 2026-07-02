import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import interim_data_path, processed_data_path, workspace_path


def _load_benchmark_module():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "synesthetic_multimodal_binding_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("synesthetic_multimodal_binding_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synesthetic_multimodal_binding_benchmark_writes_managed_outputs():
    benchmark = _load_benchmark_module()
    fixture_path = processed_data_path("benchmark_fixtures", "test_synesthetic_multimodal_cases.jsonl")
    cross_link_path = interim_data_path("autobot", "test_synesthetic_cross_links.jsonl")
    binding_manifest_path = processed_data_path("autobot", "test_synesthetic_binding_manifest.jsonl")
    trace_path = workspace_path("evaluation", "test_synesthetic_multimodal_traces.jsonl")
    plug_swap_path = workspace_path("evaluation", "test_sparse_cortical_column_plug_swap.json")
    report_path = workspace_path("evaluation", "test_synesthetic_multimodal_binding.json")
    summary_path = workspace_path("evaluation", "test_synesthetic_multimodal_binding.txt")

    exit_code = benchmark.main(
        [
            "--fixture-path",
            fixture_path,
            "--cross-link-path",
            cross_link_path,
            "--binding-manifest-path",
            binding_manifest_path,
            "--latent-manifest-path",
            processed_data_path("autobot", "latent_manifest.jsonl"),
            "--trace-path",
            trace_path,
            "--plug-swap-path",
            plug_swap_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["observed_only"] is True
    assert report["metrics"]["temporal_alignment_quality"] == 1.0
    assert report["metrics"]["plug_swap_integrity"] == 1.0
    assert report["metrics"]["cross_modal_link_precision"] == 1.0
    assert report["metrics"]["missing_modality_abstention_integrity"] == 1.0
    assert report["metrics"]["non_language_route_usefulness"] == 1.0
    assert report["metrics"]["route_traceability"] == 1.0
    assert report["metrics"]["bundle_integrity"] == 1.0
    assert report["metrics"]["binding_audit_coverage"] == 1.0
    assert report["metrics"]["bundle_event_state_promotion"] == 1.0
    assert report["metrics"]["bundle_event_state_cache_integrity"] == 1.0
    assert report["metrics"]["adapter_ir_integrity"] == 1.0
    assert report["metrics"]["own_latent_integration"] == 1.0
    assert report["metrics"]["dendritic_route_hint_integrity"] == 1.0
    assert {row["window_ms"] for row in report["window_profiles"]} == {25.0, 32.0, 40.0}
    manifest_rows = benchmark.read_jsonl(binding_manifest_path)
    assert manifest_rows
    assert manifest_rows[0]["schema"] == "sara-synesthetic-binding-manifest-v2"
    assert manifest_rows[0]["bundles"]
