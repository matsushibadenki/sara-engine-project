import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _load_module():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "neuromorphic_capability_matrix.py",
        )
    )
    spec = importlib.util.spec_from_file_location("neuromorphic_capability_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_report_passes_default_profiles():
    module = _load_module()

    report = module.build_report(
        profiles=["lava", "spinnaker", "akida"],
        active_row_count=8,
        context_length=16,
        total_readout_size=64,
        quantization_bits=3,
    )

    assert report["passed"] is True
    assert report["capability_matrix"]["profile_count"] == 3
    assert report["capability_matrix"]["unsupported_summary"] == {}
    assert report["capability_matrix"]["profiles"]["akida"]["state_trace_adapter_policy"] == (
        "freeze_state_for_inference_profile"
    )
    routing_hints = report["capability_matrix"]["common_event_ir"]["routing_hints"]
    assert "bounded_dendritic_route_hint" in routing_hints
    assert "equal_modality_thalamic_route" in routing_hints


def test_main_writes_managed_report():
    module = _load_module()
    report_path = os.path.abspath("workspace/evaluation/test_neuromorphic_matrix.json")
    summary_path = os.path.abspath("workspace/evaluation/test_neuromorphic_matrix.txt")

    exit_code = module.main(
        [
            "--profile",
            "lava",
            "--profile",
            "akida",
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["passed"] is True
    assert payload["capability_matrix"]["profile_count"] == 2
    with open(summary_path, "r", encoding="utf-8") as handle:
        assert handle.read().startswith("Neuromorphic capability matrix: PASS")
