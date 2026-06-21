import json
import os
import pytest
import types

from sara_engine.edge.exporter import export_for_edge
from sara_engine.edge.neuromorphic import (
    build_neuromorphic_capabilities,
    build_neuromorphic_capability_matrix,
    build_neuromorphic_profile_report,
    build_spike_event_ir,
    normalize_neuromorphic_profiles,
)
from sara_engine.edge.runtime import SaraEdgeRuntime, validate_edge_model_file
from sara_engine.nn.delta_associative_memory import DeltaAssociativeSpikeMemory
from sara_engine.utils.project_paths import workspace_path


class _DummyEdgeModel:
    def __init__(self) -> None:
        self.context_length = 4
        self.total_readout_size = 4
        self.config = types.SimpleNamespace(embed_dim=1)
        self.readout_synapses = [
            {65: (1.0, 0), 66: (0.25, 1)},
            {65: (0.9, 0), 66: (0.2, 1)},
            {},
            {},
        ]


class _SparseRoutingEdgeModel:
    def __init__(self) -> None:
        self.context_length = 4
        self.total_readout_size = 8
        self.config = types.SimpleNamespace(embed_dim=1)
        self.readout_synapses = [
            {65: (1.0, 0)},
            {},
            {},
            {66: (0.9, 1)},
            {},
            {},
            {},
            {67: (0.8, 2)},
        ]


def test_export_for_edge_normalizes_tuple_weights():
    model = _DummyEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_basic.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(model, export_path)
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["edge_quantization"]["enabled"] is False
    assert payload["readout_synapses"][0]["65"] == 1.0
    assert payload["readout_synapses"][0]["66"] == 0.25


def test_export_for_edge_quantization_and_runtime_compatibility():
    model = _DummyEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_quantized.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(model, export_path, quantization_bits=3)
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["edge_quantization"]["enabled"] is True
    assert payload["edge_quantization"]["bits"] == 3
    assert isinstance(payload["readout_synapses"][0]["65"], float)

    runtime = SaraEdgeRuntime(export_path)
    runtime._get_sdr = lambda _delay, _tok: [0, 1]
    predicted = runtime.forward_step(97)
    assert predicted == 65


def test_export_for_edge_compact_quantization_and_sparse_routing_table():
    model = _DummyEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_compact_quantized.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(model, export_path, quantization_bits=3, compact_quantized=True)
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["edge_quantization"]["enabled"] is True
    assert payload["edge_quantization"]["bits"] == 3
    assert payload["edge_quantization"]["storage"] == "compact_int"
    assert payload["multilevel_weight_profile"] == {
        "enabled": True,
        "bits": 3,
        "levels": 8,
        "storage": "compact_int",
        "quantized_row_count": 2,
        "quantized_weight_count": 4,
        "flat_row_count": 0,
    }
    assert payload["sparse_routing_table"] == {
        "enabled": True,
        "active_rows": [0, 1],
        "row_count": 4,
        "stored_row_count": 4,
    }

    first_row = payload["readout_synapses"][0]
    assert first_row["keys"] == [65, 66]
    assert all(isinstance(value, int) for value in first_row["qweights"])
    assert sorted(first_row.keys()) == ["keys", "max", "min", "qweights"]

    runtime = SaraEdgeRuntime(export_path)
    assert runtime.edge_quantization["storage"] == "compact_int"
    assert runtime.multilevel_weight_profile["levels"] == 8
    assert runtime.sparse_routing_table["active_rows"] == [0, 1]
    runtime._get_sdr = lambda _delay, _tok: [0, 1]
    predicted = runtime.forward_step(97)
    assert predicted == 65


def test_export_for_edge_compresses_sparse_routing_events():
    model = _SparseRoutingEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_event_compressed.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["event_compression"] == {
        "enabled": True,
        "encoding": "delta",
        "uncompressed_event_count": 3,
        "compressed_event_count": 3,
    }
    assert payload["sparse_routing_table"]["encoding"] == "delta"
    assert payload["sparse_routing_table"]["active_row_deltas"] == [0, 3, 4]
    assert "active_rows" not in payload["sparse_routing_table"]

    runtime = SaraEdgeRuntime(export_path)
    assert runtime.event_compression["encoding"] == "delta"
    assert runtime.sparse_routing_table["active_rows"] == [0, 3, 7]
    runtime._get_sdr = lambda _delay, _tok: [7]
    predicted = runtime.forward_step(97)
    assert predicted == 67


def test_export_for_edge_sparse_readout_stores_only_active_rows():
    model = _SparseRoutingEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_sparse_readout.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["readout_storage"] == "active_rows"
    assert payload["format_version"] == 2
    assert payload["format_capabilities"] == [
        "active_row_readout_storage",
        "compact_int_weights",
        "delta_event_compression",
        "dense_row_index_space",
        "multilevel_weights",
        "sparse_routing_table",
    ]
    assert len(payload["readout_synapses"]) == 3
    assert payload["edge_manifest"]["schema"] == "sara-edge-manifest-v1"
    assert payload["edge_manifest"]["digest_algorithm"] == "sha256"
    assert payload["edge_manifest"]["capability_count"] == len(payload["format_capabilities"])
    assert payload["edge_manifest"]["stored_row_count"] == 3
    assert len(payload["edge_manifest"]["payload_digest"]) == 64
    assert payload["edge_storage_profile"] == {
        "enabled": True,
        "row_count": 8,
        "stored_row_count": 3,
        "active_row_count": 3,
        "empty_row_count": 5,
        "row_reduction_ratio": 0.625,
        "synapse_count": 3,
        "compact_weight_count": 3,
    }
    assert payload["multilevel_weight_profile"]["quantized_weight_count"] == 3
    assert payload["multilevel_weight_profile"]["levels"] == 8
    assert payload["sparse_routing_table"]["row_count"] == 8
    assert payload["sparse_routing_table"]["stored_row_count"] == 3
    assert payload["sparse_routing_table"]["active_row_deltas"] == [0, 3, 4]

    runtime = SaraEdgeRuntime(export_path)
    assert runtime.format_version == 2
    assert runtime.runtime_compatibility["supported"] is True
    assert runtime.runtime_compatibility["unsupported_capabilities"] == []
    assert runtime.manifest_integrity["present"] is True
    assert runtime.manifest_integrity["valid"] is True
    assert runtime.readout_storage == "active_rows"
    assert runtime.edge_storage_profile["row_reduction_ratio"] == 0.625
    assert runtime.multilevel_weight_profile["storage"] == "compact_int"
    assert len(runtime.readout_synapses) == 8
    assert runtime.readout_synapses[1] == {}
    assert runtime.readout_synapses[3]
    runtime._get_sdr = lambda _delay, _tok: [3]
    predicted = runtime.forward_step(97)
    assert predicted == 66

    validation = validate_edge_model_file(export_path, strict_format=True)
    assert validation["passed"] is True
    assert validation["errors"] == []
    assert validation["row_count"] == 8
    assert validation["stored_row_count"] == 3
    assert validation["active_row_count"] == 3
    assert validation["multilevel_weight_levels"] == 8


def test_export_for_edge_persists_delta_associative_state_in_manifest():
    model = _SparseRoutingEdgeModel()
    memory = DeltaAssociativeSpikeMemory(capacity=4)
    memory.update(context_events=[1, 2], predicted_events=[10], observed_events=[10, 11])
    export_path = workspace_path("tests", "edge_exporter_delta_state.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
        delta_associative_state=memory.snapshot(),
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    delta_state = payload["delta_associative_state"]
    assert "delta_associative_memory_state" in payload["format_capabilities"]
    assert delta_state["enabled"] is True
    assert delta_state["storage"] == "compact_sparse_float"
    assert delta_state["capacity"] == 4
    assert delta_state["state_units"] == 2
    assert delta_state["entry_count"] == 2
    assert delta_state["budget_ok"] is True
    assert len(payload["edge_manifest"]["payload_digest"]) == 64

    runtime = SaraEdgeRuntime(export_path, strict_format=True)
    assert runtime.delta_associative_state["entries"][0]["value_id"] == 11
    validation = validate_edge_model_file(export_path, strict_format=True)
    assert validation["passed"] is True
    assert validation["delta_state_enabled"] is True
    assert validation["delta_state_units"] == 2
    assert validation["delta_state_budget_ok"] is True


def test_export_for_edge_emits_neuromorphic_spike_event_ir_profile():
    model = _SparseRoutingEdgeModel()
    memory = DeltaAssociativeSpikeMemory(capacity=4)
    memory.update(context_events=[1, 2], predicted_events=[10], observed_events=[10, 11])
    export_path = workspace_path("tests", "edge_exporter_neuromorphic_ir.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
        delta_associative_state=memory.snapshot(),
        neuromorphic_profile=["lava", "spinnaker", "akida"],
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "neuromorphic_spike_event_ir" in payload["format_capabilities"]
    assert "neuromorphic_backend_profile" in payload["format_capabilities"]
    assert payload["spike_event_ir"]["schema"] == "sara-spike-event-ir-v1"
    assert payload["spike_event_ir"]["event_count"] == 3
    assert payload["spike_event_ir"]["event_encoding"] == "delta"
    assert payload["spike_event_ir"]["weight_storage"] == "compact_int"
    assert payload["spike_event_ir"]["budget_ok"] is True
    assert payload["neuromorphic_capabilities"]["profiles"] == ["lava", "spinnaker", "akida"]
    assert payload["neuromorphic_capabilities"]["backend_compatibility"] == {
        "akida": True,
        "lava": True,
        "spinnaker": True,
    }
    profile_report = payload["neuromorphic_profile_report"]
    assert profile_report["schema"] == "sara-neuromorphic-profile-report-v1"
    assert profile_report["profile_count"] == 3
    assert profile_report["all_profiles_compatible"] is True
    assert profile_report["profiles"]["lava"]["adapter"] == "lava_profile"
    assert profile_report["profiles"]["spinnaker"]["adapter"] == "spinnaker_profile"
    assert profile_report["profiles"]["akida"]["adapter"] == "akida_profile"
    assert (
        profile_report["profiles"]["akida"]["online_update_adapter_policy"]
        == "freeze_state_for_inference_profile"
    )

    runtime = SaraEdgeRuntime(export_path, strict_format=True)
    assert runtime.spike_event_ir["event_count"] == 3
    assert runtime.neuromorphic_capabilities["event_routing"] is True
    assert runtime.neuromorphic_profile_report["profile_count"] == 3
    validation = validate_edge_model_file(export_path, strict_format=True)
    assert validation["passed"] is True
    assert validation["spike_event_ir_enabled"] is True
    assert validation["spike_event_ir_schema"] == "sara-spike-event-ir-v1"
    assert validation["spike_event_count"] == 3
    assert validation["neuromorphic_capabilities_enabled"] is True
    assert validation["neuromorphic_profiles"] == ["lava", "spinnaker", "akida"]
    assert validation["neuromorphic_backend_compatible"] is True
    assert validation["neuromorphic_profile_report_enabled"] is True
    assert validation["neuromorphic_profile_count"] == 3
    assert validation["neuromorphic_profile_compatibility"] == {
        "akida": True,
        "lava": True,
        "spinnaker": True,
    }


def test_neuromorphic_profile_helpers_are_adapter_ready():
    profiles = normalize_neuromorphic_profiles("lava, spinnaker, lava, akida")
    spike_ir = build_spike_event_ir(
        active_rows=[3, 1, 3],
        context_length=4,
        total_readout_size=16,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        delta_state={"enabled": True, "state_units": 2},
        neuromorphic_profile=profiles,
    )
    capabilities = build_neuromorphic_capabilities(
        spike_event_ir=spike_ir,
        delta_state={"enabled": True},
        quantization_bits=3,
        neuromorphic_profile=profiles,
    )
    report = build_neuromorphic_profile_report(spike_ir, capabilities)

    assert profiles == ["lava", "spinnaker", "akida"]
    assert spike_ir["schema"] == "sara-spike-event-ir-v1"
    assert spike_ir["event_count"] == 2
    assert spike_ir["state_budget_units"] == 4
    assert capabilities["backend_compatibility"] == {
        "akida": True,
        "lava": True,
        "spinnaker": True,
    }
    assert report["all_profiles_compatible"] is True
    assert report["profiles"]["akida"]["online_update_adapter_policy"] == (
        "freeze_state_for_inference_profile"
    )


def test_neuromorphic_spike_event_ir_carries_state_trace_policy():
    profiles = normalize_neuromorphic_profiles(["lava", "akida"])
    spike_ir = build_spike_event_ir(
        active_rows=[3, 1],
        context_length=8,
        total_readout_size=32,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        delta_state={"enabled": True, "state_units": 2},
        neuromorphic_profile=profiles,
        state_traces={
            "forward_only": {
                "state_units": 1,
                "delay": 2,
                "routing_hint": "forward_only_eligibility",
                "online_update_policy": "local_credit_trace",
            },
            "multi_timescale": {
                "state_units": 6,
                "delay": 3,
                "routing_hint": "multi_timescale_state",
                "online_update_policy": "bounded_leak_state",
            },
            "phase_binding": {
                "state_units": 2,
                "delay": 1,
                "routing_hint": "phase_bucket_binding",
                "online_update_policy": "coincidence_trace",
            },
            "predictive_error": {
                "state_units": 3,
                "delay": 0,
                "routing_hint": "predictive_error_correction",
                "online_update_policy": "residual_correction_trace",
            },
        },
    )
    capabilities = build_neuromorphic_capabilities(
        spike_event_ir=spike_ir,
        delta_state={"enabled": True},
        quantization_bits=3,
        neuromorphic_profile=profiles,
    )
    report = build_neuromorphic_profile_report(spike_ir, capabilities)

    assert spike_ir["event_count"] == 6
    assert spike_ir["readout_event_count"] == 2
    assert spike_ir["state_trace_event_count"] == 4
    assert spike_ir["state_budget_units"] == 16
    assert "bounded_leak_state" in spike_ir["online_update_policies"]
    assert "residual_correction_trace" in spike_ir["online_update_policies"]
    assert "forward_only_eligibility" in spike_ir["state_trace_routing_hints"]
    assert capabilities["state_trace_routing_hints"] == [
        "forward_only_eligibility",
        "multi_timescale_state",
        "phase_bucket_binding",
        "predictive_error_correction",
    ]
    assert report["profiles"]["lava"]["state_trace_adapter_policy"] == "native_online_update"
    assert report["profiles"]["akida"]["state_trace_adapter_policy"] == (
        "freeze_state_for_inference_profile"
    )
    assert report["profiles"]["akida"]["checks"]["online_update_policy_ok"] is True


def test_neuromorphic_capability_matrix_summarizes_backend_limits():
    profiles = normalize_neuromorphic_profiles(["lava", "akida"])
    spike_ir = build_spike_event_ir(
        active_rows=[1, 2, 3],
        context_length=8,
        total_readout_size=32,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        delta_state={"enabled": True, "state_units": 2},
        neuromorphic_profile=profiles,
        state_traces={
            "predictive_error": {
                "state_units": 2,
                "routing_hint": "predictive_error_correction",
                "online_update_policy": "residual_correction_trace",
            }
        },
    )
    capabilities = build_neuromorphic_capabilities(
        spike_event_ir=spike_ir,
        delta_state={"enabled": True},
        quantization_bits=3,
        neuromorphic_profile=profiles,
    )
    profile_report = build_neuromorphic_profile_report(spike_ir, capabilities)
    matrix = build_neuromorphic_capability_matrix(spike_ir, capabilities, profile_report)

    assert matrix["schema"] == "sara-neuromorphic-capability-matrix-v1"
    assert matrix["all_profiles_compatible"] is True
    assert matrix["unsupported_summary"] == {}
    assert matrix["common_event_ir"]["event_count"] == 4
    assert matrix["profiles"]["lava"]["event_budget_headroom"] == 4092
    assert matrix["profiles"]["akida"]["state_trace_adapter_policy"] == (
        "freeze_state_for_inference_profile"
    )
    assert matrix["profiles"]["akida"]["routing_hints"] == ["predictive_error_correction"]


def test_export_for_edge_accepts_neuromorphic_state_trace_ir(tmp_path):
    model = _SparseRoutingEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_state_trace_ir.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
        neuromorphic_profile=["lava", "akida"],
        neuromorphic_state_traces={
            "multi_timescale": {
                "state_units": 4,
                "routing_hint": "multi_timescale_state",
                "online_update_policy": "bounded_leak_state",
            }
        },
    )

    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "neuromorphic_state_trace_ir" in payload["format_capabilities"]
    assert payload["spike_event_ir"]["state_trace_event_count"] == 1
    assert payload["spike_event_ir"]["state_trace_routing_hints"] == ["multi_timescale_state"]
    runtime = SaraEdgeRuntime(export_path, strict_format=True)
    validation = validate_edge_model_file(export_path, strict_format=True)

    assert runtime.runtime_compatibility["supported"] is True
    assert validation["passed"] is True


def test_edge_runtime_detects_manifest_digest_mismatch():
    model = _SparseRoutingEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_manifest_mismatch.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(
        model,
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["readout_synapses"][0]["min"] = 9.0
    with open(export_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    runtime = SaraEdgeRuntime(export_path)

    assert runtime.manifest_integrity["present"] is True
    assert runtime.manifest_integrity["valid"] is False
    assert runtime.manifest_integrity["expected_digest"] != runtime.manifest_integrity["actual_digest"]
    validation = validate_edge_model_file(export_path, strict_format=True)
    assert validation["passed"] is False
    assert validation["errors"] == ["manifest_integrity_failed"]
    with pytest.raises(ValueError, match="manifest integrity"):
        SaraEdgeRuntime(export_path, strict_format=True)


def test_edge_runtime_reports_unknown_capabilities_without_breaking_legacy_load():
    model = _DummyEdgeModel()
    export_path = workspace_path("tests", "edge_exporter_unknown_capability.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    export_for_edge(model, export_path, quantization_bits=3, compact_quantized=True)
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["format_capabilities"].append("future_neuromorphic_dma")
    with open(export_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    runtime = SaraEdgeRuntime(export_path)

    assert runtime.runtime_compatibility["supported"] is False
    assert runtime.runtime_compatibility["unsupported_capabilities"] == ["future_neuromorphic_dma"]
    validation = validate_edge_model_file(export_path, strict_format=True)
    assert validation["passed"] is False
    assert validation["errors"] == [
        "unsupported_capabilities=future_neuromorphic_dma",
        "manifest_integrity_failed",
    ]
    with pytest.raises(ValueError, match="Unsupported Sara-Edge format capabilities"):
        SaraEdgeRuntime(export_path, strict_format=True)
    runtime._get_sdr = lambda _delay, _tok: [0, 1]
    predicted = runtime.forward_step(97)
    assert predicted == 65
