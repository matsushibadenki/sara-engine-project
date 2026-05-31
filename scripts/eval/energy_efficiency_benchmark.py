# Directory Path: scripts/eval/energy_efficiency_benchmark.py
# English Title: Energy Efficiency Benchmark
# Purpose/Content: Runs a lightweight CPU-first efficiency proxy benchmark for SaraInference fast-path and session-memory responses.

import argparse
import json
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))


from sara_engine.inference import SaraInference
from sara_engine.edge.exporter import export_for_edge
from sara_engine.edge.runtime import SaraEdgeRuntime, validate_edge_model_file
from sara_engine.nn.delta_associative_memory import DeltaAssociativeSpikeMemory
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path
from sara_engine.utils.stochastic_computing import StochasticAccumulator


DEFAULT_HISTORY_PATH = workspace_path("evaluation", "energy_efficiency_history.json")
NEUROMORPHIC_PROFILE_CHECK_NAMES = (
    "known_profile",
    "event_budget_ok",
    "delay_support_ok",
    "low_precision_weight_ok",
    "online_update_policy_ok",
)
STAGE_E_ARCHITECTURE_TRACE_HINTS = (
    "micro_turn_interaction",
    "foreground_background_handoff",
    "phase_assigned_submodel_block",
    "denoising_correction_trace",
)
STAGE_E_ARCHITECTURE_UPDATE_POLICIES = (
    "micro_turn_event_budget_observed_only",
    "foreground_background_context_handoff_observed_only",
    "phase_block_local_credit_observed_only",
    "denoising_correction_observed_only",
)


def _build_engine() -> SaraInference:
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.retrieval_diagnostics = []
    engine.refractory_buffer = []
    engine.session_memory = {}
    engine.lif_network = None
    return engine


def _run_fast_identity_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.tokenizer = types.SimpleNamespace(
        __call__=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tokenizer should not run"))
    )
    # Warm-up call keeps the measured window focused on steady-state fast-path behavior.
    _ = engine.generate("You: Who are you?\nSARA:")
    start = time.perf_counter()
    response = engine.generate("You: Who are you?\nSARA:")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    diagnostic = diagnostics[0] if diagnostics else {}
    success = (
        response == "I am SARA, a CPU-first spiking neural network assistant."
        and diagnostic.get("source") == "inference_fast_path"
        and diagnostic.get("memory_hit") == "fast_path"
    )
    return {
        "success": success,
        "elapsed_ms": elapsed_ms,
        "memory_hit": diagnostic.get("memory_hit", ""),
        "state_units": len(getattr(engine, "refractory_buffer", [])) + len(diagnostics),
        "route_work_units": 1,
        "ann_reference_cost_units": 128,
        "description": "Identity responses should use the lightweight fast path without tokenizer work.",
    }


def _run_session_memory_case() -> Dict[str, Any]:
    engine = _build_engine()
    engine.session_memory = {"goal": "finish this project", "task": "the sara engine"}
    _ = engine.generate("You: What should I do next?\nSARA:")
    start = time.perf_counter()
    response = engine.generate("You: What should I do next?\nSARA:")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    diagnostics = engine.get_recent_retrieval_diagnostics(limit=1)
    diagnostic = diagnostics[0] if diagnostics else {}
    success = (
        "Step 1:" in response
        and "Step 2:" in response
        and "the sara engine" in response
        and diagnostic.get("source") == "inference_fast_path"
        and diagnostic.get("memory_hit") == "session_memory"
    )
    return {
        "success": success,
        "elapsed_ms": elapsed_ms,
        "memory_hit": diagnostic.get("memory_hit", ""),
        "state_units": (
            len(getattr(engine, "refractory_buffer", []))
            + len(diagnostics)
            + len(getattr(engine, "session_memory", {}))
        ),
        "route_work_units": 2,
        "ann_reference_cost_units": 192,
        "description": "Next-step planning should stay on a low-overhead session-memory path.",
    }


def _run_stochastic_readout_case() -> Dict[str, Any]:
    accumulator = StochasticAccumulator(bit_count=64, seed=11)
    deterministic_scores = {
        "primary": 0.95,
        "alternative": 0.72,
        "secondary": 0.40,
    }
    approximate_scores = accumulator.approximate_scores(deterministic_scores, confidence_weight=0.9)
    selected = accumulator.argmax(deterministic_scores, confidence_weight=0.9)
    success = (
        selected == "primary"
        and approximate_scores.get("primary", 0.0) >= approximate_scores.get("alternative", 0.0)
    )
    return {
        "success": success,
        "elapsed_ms": 0.0,
        "memory_hit": "stochastic_readout",
        "state_units": accumulator.state_units(),
        "route_work_units": 3,
        "ann_reference_cost_units": 128,
        "selected_label": selected,
        "bit_count": accumulator.bit_count,
        "approximate_scores": approximate_scores,
        "description": "Stochastic score aggregation should preserve the strongest branch with bounded state.",
    }


def _build_stage_e_architecture_state_traces() -> Dict[str, Dict[str, Any]]:
    return {
        "micro_turn_interaction": {
            "delay": 0,
            "routing_hint": "micro_turn_interaction",
            "online_update_policy": "micro_turn_event_budget_observed_only",
            "state_budget_units": 1,
            "timestep": 0,
        },
        "foreground_background_handoff": {
            "delay": 1,
            "routing_hint": "foreground_background_handoff",
            "online_update_policy": "foreground_background_context_handoff_observed_only",
            "state_budget_units": 1,
            "timestep": 1,
        },
        "phase_assigned_submodel_block": {
            "delay": 2,
            "routing_hint": "phase_assigned_submodel_block",
            "online_update_policy": "phase_block_local_credit_observed_only",
            "state_budget_units": 1,
            "timestep": 2,
        },
        "denoising_correction_trace": {
            "delay": 3,
            "routing_hint": "denoising_correction_trace",
            "online_update_policy": "denoising_correction_observed_only",
            "state_budget_units": 1,
            "timestep": 3,
        },
    }


def _score_stage_e_architecture_neuromorphic_compatibility(
    *,
    payload: Dict[str, Any],
    validation_report: Dict[str, Any],
    spike_event_ir: Dict[str, Any],
    neuromorphic_capabilities: Dict[str, Any],
    neuromorphic_profile_report: Dict[str, Any],
) -> Dict[str, Any]:
    event_fields = set(spike_event_ir.get("event_fields", []))
    routing_hints = {
        str(hint)
        for hint in spike_event_ir.get("state_trace_routing_hints", [])
        if str(hint).strip()
    }
    capability_hints = {
        str(hint)
        for hint in neuromorphic_capabilities.get("state_trace_routing_hints", [])
        if str(hint).strip()
    }
    online_update_policies = {
        str(policy)
        for policy in neuromorphic_capabilities.get("online_update_policies", [])
        if str(policy).strip()
    }
    profile_reports = (
        neuromorphic_profile_report.get("profiles", {})
        if isinstance(neuromorphic_profile_report.get("profiles", {}), dict)
        else {}
    )
    adapter_policies = {
        str(profile_report.get("online_update_adapter_policy", ""))
        for profile_report in profile_reports.values()
        if isinstance(profile_report, dict)
    }
    required_fields = {
        "event_id",
        "timestep",
        "channel",
        "delay",
        "routing_hint",
        "online_update_policy",
        "state_budget_units",
    }
    required_hints = set(STAGE_E_ARCHITECTURE_TRACE_HINTS)
    required_policies = set(STAGE_E_ARCHITECTURE_UPDATE_POLICIES)
    state_trace_event_count = int(spike_event_ir.get("state_trace_event_count", 0) or 0)
    state_budget_units = int(spike_event_ir.get("state_budget_units", 0) or 0)
    state_budget_limit = int(spike_event_ir.get("state_budget_limit", 0) or 0)
    profile_max_events = [
        int(profile.get("max_events", 0) or 0)
        for profile in profile_reports.values()
        if isinstance(profile, dict)
    ]
    return {
        "state_trace_ir": 1.0
        if "neuromorphic_state_trace_ir" in payload.get("format_capabilities", [])
        and state_trace_event_count >= len(required_hints)
        and required_fields.issubset(event_fields)
        and bool(validation_report.get("passed", False))
        else 0.0,
        "routing_hint_coverage": 1.0
        if required_hints.issubset(routing_hints)
        and required_hints.issubset(capability_hints)
        else 0.0,
        "online_update_policy": 1.0
        if required_policies.issubset(online_update_policies)
        and "native_online_update" in adapter_policies
        and "freeze_state_for_inference_profile" in adapter_policies
        else 0.0,
        "event_budget": 1.0
        if bool(spike_event_ir.get("budget_ok", False))
        and state_budget_units <= state_budget_limit
        and int(spike_event_ir.get("event_count", 0) or 0)
        <= min(profile_max_events or [0])
        else 0.0,
        "state_trace_event_count": state_trace_event_count,
        "state_budget_units": state_budget_units,
        "state_budget_limit": state_budget_limit,
        "routing_hints": sorted(routing_hints),
        "online_update_policies": sorted(online_update_policies),
        "adapter_policies": sorted(policy for policy in adapter_policies if policy),
    }


def _run_edge_low_precision_persistence_case() -> Dict[str, Any]:
    class _EdgeFixture:
        context_length = 4
        total_readout_size = 8
        config = types.SimpleNamespace(embed_dim=1)
        readout_synapses = [
            {65: (1.0, 0), 66: (0.25, 1)},
            {65: (0.9, 0), 66: (0.2, 1)},
            {},
            {},
        ]

    export_path = ensure_parent_directory(
        workspace_path("evaluation", f"edge_low_precision_persistence_{os.getpid()}.json")
    )
    delta_memory = DeltaAssociativeSpikeMemory(capacity=4)
    delta_memory.update(context_events=[1, 2], predicted_events=[10], observed_events=[10, 11])
    export_for_edge(
        _EdgeFixture(),
        export_path,
        quantization_bits=3,
        compact_quantized=True,
        compress_events=True,
        sparse_readout=True,
        delta_associative_state=delta_memory.snapshot(),
        neuromorphic_profile=["lava", "spinnaker", "akida"],
        neuromorphic_state_traces=_build_stage_e_architecture_state_traces(),
    )
    with open(export_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    runtime = SaraEdgeRuntime(export_path)
    strict_runtime = SaraEdgeRuntime(export_path, strict_format=True)
    validation_report = validate_edge_model_file(export_path, strict_format=True)
    runtime._get_sdr = lambda _delay, _tok: [0, 1]
    predicted = runtime.forward_step(97)

    first_row = payload.get("readout_synapses", [{}])[0]
    storage = payload.get("edge_quantization", {}).get("storage", "")
    sparse_routing = payload.get("sparse_routing_table", {})
    event_compression = payload.get("event_compression", {})
    storage_profile = payload.get("edge_storage_profile", {})
    multilevel_profile = payload.get("multilevel_weight_profile", {})
    readout_storage = payload.get("readout_storage", "")
    compatibility = runtime.runtime_compatibility
    manifest_integrity = runtime.manifest_integrity
    delta_state = runtime.delta_associative_state
    spike_event_ir = runtime.spike_event_ir
    neuromorphic_capabilities = runtime.neuromorphic_capabilities
    neuromorphic_profile_report = runtime.neuromorphic_profile_report
    stage_e_architecture_profile = _score_stage_e_architecture_neuromorphic_compatibility(
        payload=payload,
        validation_report=validation_report,
        spike_event_ir=spike_event_ir,
        neuromorphic_capabilities=neuromorphic_capabilities,
        neuromorphic_profile_report=neuromorphic_profile_report,
    )
    profile_reports = (
        neuromorphic_profile_report.get("profiles", {})
        if isinstance(neuromorphic_profile_report.get("profiles", {}), dict)
        else {}
    )
    validation_delta_state_units = int(validation_report.get("delta_state_units", 0) or 0)
    strict_validation_passed = (
        strict_runtime.runtime_compatibility.get("supported", False)
        and strict_runtime.manifest_integrity.get("valid", False)
    )
    compact_keys = first_row.get("keys", []) if isinstance(first_row, dict) else []
    compact_qweights = first_row.get("qweights", []) if isinstance(first_row, dict) else []
    decoded_active_rows = runtime.sparse_routing_table.get("active_rows", [])
    success = (
        predicted == 65
        and storage == "compact_int"
        and all(isinstance(value, int) for value in compact_qweights)
        and decoded_active_rows == [0, 1]
        and event_compression.get("encoding") == "delta"
        and readout_storage == "active_rows"
        and float(storage_profile.get("row_reduction_ratio", 0.0) or 0.0) > 0.0
        and int(multilevel_profile.get("levels", 0) or 0) == 8
        and bool(compatibility.get("supported", False))
        and bool(manifest_integrity.get("valid", False))
        and strict_validation_passed
        and bool(validation_report.get("passed", False))
        and bool(delta_state.get("enabled", False))
        and bool(delta_state.get("budget_ok", False))
        and bool(spike_event_ir.get("enabled", False))
        and spike_event_ir.get("schema") == "sara-spike-event-ir-v1"
        and bool(neuromorphic_capabilities.get("backend_compatibility", {}).get("lava", False))
        and bool(neuromorphic_profile_report.get("all_profiles_compatible", False))
    )
    return {
        "success": success,
        "elapsed_ms": 0.0,
        "memory_hit": "edge_low_precision_persistence",
        "state_units": len(compact_keys) + 1,
        "route_work_units": max(1, len(decoded_active_rows)),
        "ann_reference_cost_units": 128,
        "edge_low_precision_persistence": 1.0 if storage == "compact_int" else 0.0,
        "edge_sparse_routing_table": 1.0 if decoded_active_rows == [0, 1] else 0.0,
        "edge_event_compression": 1.0 if event_compression.get("encoding") == "delta" else 0.0,
        "edge_sparse_readout_storage": 1.0 if readout_storage == "active_rows" else 0.0,
        "edge_storage_profile_integrity": 1.0
        if float(storage_profile.get("row_reduction_ratio", 0.0) or 0.0) > 0.0
        else 0.0,
        "edge_multilevel_weight_profile": 1.0
        if int(multilevel_profile.get("levels", 0) or 0) == 8
        and int(multilevel_profile.get("quantized_weight_count", 0) or 0)
        == int(storage_profile.get("compact_weight_count", 0) or 0)
        else 0.0,
        "edge_format_compatibility": 1.0 if compatibility.get("supported", False) else 0.0,
        "edge_manifest_integrity": 1.0 if manifest_integrity.get("valid", False) else 0.0,
        "edge_strict_format_validation": 1.0 if strict_validation_passed else 0.0,
        "edge_payload_validation_report": 1.0 if validation_report.get("passed", False) else 0.0,
        "edge_delta_state_persistence": 1.0
        if bool(delta_state.get("enabled", False))
        and int(delta_state.get("entry_count", 0) or 0) > 0
        else 0.0,
        "edge_delta_state_budget": 1.0 if bool(delta_state.get("budget_ok", False)) else 0.0,
        "edge_delta_state_manifest_integrity": 1.0
        if validation_report.get("passed", False) and validation_delta_state_units > 0
        else 0.0,
        "neuromorphic_ir_schema_integrity": 1.0
        if spike_event_ir.get("schema") == "sara-spike-event-ir-v1"
        and int(spike_event_ir.get("event_count", 0) or 0) == len(spike_event_ir.get("events", []))
        else 0.0,
        "neuromorphic_capability_manifest_integrity": 1.0
        if bool(neuromorphic_capabilities.get("enabled", False))
        and "neuromorphic_spike_event_ir" in payload.get("format_capabilities", [])
        and manifest_integrity.get("valid", False)
        else 0.0,
        "neuromorphic_backend_profile_compatibility": 1.0
        if validation_report.get("neuromorphic_backend_compatible", False)
        else 0.0,
        "neuromorphic_sparse_event_budget": 1.0
        if bool(spike_event_ir.get("budget_ok", False))
        else 0.0,
        "neuromorphic_profile_report_integrity": 1.0
        if validation_report.get("neuromorphic_profile_report_enabled", False)
        and int(validation_report.get("neuromorphic_profile_count", 0) or 0) == 3
        and all(validation_report.get("neuromorphic_profile_compatibility", {}).values())
        else 0.0,
        "neuromorphic_stage_e_state_trace_ir": float(
            stage_e_architecture_profile.get("state_trace_ir", 0.0)
        ),
        "neuromorphic_stage_e_routing_hint_coverage": float(
            stage_e_architecture_profile.get("routing_hint_coverage", 0.0)
        ),
        "neuromorphic_stage_e_online_update_policy": float(
            stage_e_architecture_profile.get("online_update_policy", 0.0)
        ),
        "neuromorphic_stage_e_event_budget": float(
            stage_e_architecture_profile.get("event_budget", 0.0)
        ),
        "predicted_token": predicted,
        "compact_row_keys": compact_keys,
        "compact_row_qweight_count": len(compact_qweights),
        "active_row_deltas": sparse_routing.get("active_row_deltas", []),
        "stored_row_count": int(sparse_routing.get("stored_row_count", 0) or 0),
        "readout_row_count": int(sparse_routing.get("row_count", 0) or 0),
        "row_reduction_ratio": float(storage_profile.get("row_reduction_ratio", 0.0) or 0.0),
        "compact_weight_count": int(storage_profile.get("compact_weight_count", 0) or 0),
        "multilevel_weight_levels": int(multilevel_profile.get("levels", 0) or 0),
        "quantized_weight_count": int(multilevel_profile.get("quantized_weight_count", 0) or 0),
        "format_version": int(payload.get("format_version", 1) or 1),
        "format_capabilities": payload.get("format_capabilities", []),
        "unsupported_capabilities": compatibility.get("unsupported_capabilities", []),
        "manifest_schema": manifest_integrity.get("schema", ""),
        "manifest_digest_algorithm": manifest_integrity.get("digest_algorithm", ""),
        "delta_state_units": int(delta_state.get("state_units", 0) or 0),
        "delta_state_entry_count": int(delta_state.get("entry_count", 0) or 0),
        "delta_state_budget_ok": bool(delta_state.get("budget_ok", False)),
        "spike_event_ir_schema": spike_event_ir.get("schema", ""),
        "spike_event_ir_event_count": int(spike_event_ir.get("event_count", 0) or 0),
        "neuromorphic_profiles": neuromorphic_capabilities.get("profiles", []),
        "neuromorphic_backend_compatibility": neuromorphic_capabilities.get("backend_compatibility", {}),
        "neuromorphic_profile_count": int(neuromorphic_profile_report.get("profile_count", 0) or 0),
        "neuromorphic_stage_e_architecture_profile": stage_e_architecture_profile,
        "neuromorphic_profile_compatibility": {
            str(profile_name): bool(profile_report.get("compatible", False))
            for profile_name, profile_report in profile_reports.items()
            if isinstance(profile_report, dict)
        },
        "neuromorphic_profile_report_checks": {
            str(profile_name): {
                "adapter": str(profile_report.get("adapter", "")),
                "compatible": bool(profile_report.get("compatible", False)),
                "checks": dict(profile_report.get("checks", {}))
                if isinstance(profile_report.get("checks", {}), dict)
                else {},
                "online_update_adapter_policy": str(
                    profile_report.get("online_update_adapter_policy", "")
                ),
                "max_events": int(profile_report.get("max_events", 0) or 0),
            }
            for profile_name, profile_report in profile_reports.items()
            if isinstance(profile_report, dict)
        },
        "validation_errors": validation_report.get("errors", []),
        "description": "Edge exports should persist compact integer readout rows with sparse routing metadata.",
    }


def _edge_profile_case(report: Dict[str, Any]) -> Dict[str, Any]:
    details = report.get("details", {}) if isinstance(report.get("details"), dict) else {}
    cases = details.get("test_results", []) if isinstance(details.get("test_results"), list) else []
    for case in cases:
        if isinstance(case, dict) and case.get("memory_hit") == "edge_low_precision_persistence":
            return case
    return {}


def _neuromorphic_profile_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    case = _edge_profile_case(report)
    profiles = (
        case.get("neuromorphic_profile_report_checks", {})
        if isinstance(case.get("neuromorphic_profile_report_checks", {}), dict)
        else {}
    )
    snapshot_profiles: Dict[str, Any] = {}
    for profile_name, profile_report in profiles.items():
        if not isinstance(profile_report, dict):
            continue
        checks = (
            profile_report.get("checks", {})
            if isinstance(profile_report.get("checks", {}), dict)
            else {}
        )
        snapshot_profiles[str(profile_name)] = {
            "adapter": str(profile_report.get("adapter", "")),
            "compatible": bool(profile_report.get("compatible", False)),
            "checks": {
                check_name: bool(checks.get(check_name, False))
                for check_name in NEUROMORPHIC_PROFILE_CHECK_NAMES
            },
            "online_update_adapter_policy": str(
                profile_report.get("online_update_adapter_policy", "")
            ),
            "max_events": int(profile_report.get("max_events", 0) or 0),
        }
    return {
        "profile_count": int(case.get("neuromorphic_profile_count", 0) or 0),
        "event_count": int(case.get("spike_event_ir_event_count", 0) or 0),
        "profiles": snapshot_profiles,
    }


def build_neuromorphic_profile_trend(
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current = _neuromorphic_profile_snapshot(current_report)
    previous = _neuromorphic_profile_snapshot(previous_report or {})
    current_profiles = current.get("profiles", {}) if isinstance(current.get("profiles"), dict) else {}
    previous_profiles = (
        previous.get("profiles", {}) if isinstance(previous.get("profiles"), dict) else {}
    )
    regressions: List[Dict[str, Any]] = []
    policy_changes: List[Dict[str, Any]] = []
    new_profiles = sorted(profile for profile in current_profiles if profile not in previous_profiles)

    for profile_name, previous_profile in sorted(previous_profiles.items()):
        if profile_name not in current_profiles:
            regressions.append(
                {
                    "profile": profile_name,
                    "kind": "missing_profile",
                    "previous": True,
                    "current": False,
                }
            )
            continue

        current_profile = current_profiles[profile_name]
        if bool(previous_profile.get("compatible", False)) and not bool(
            current_profile.get("compatible", False)
        ):
            regressions.append(
                {
                    "profile": profile_name,
                    "kind": "compatibility_regression",
                    "previous": True,
                    "current": False,
                }
            )

        previous_checks = (
            previous_profile.get("checks", {})
            if isinstance(previous_profile.get("checks", {}), dict)
            else {}
        )
        current_checks = (
            current_profile.get("checks", {})
            if isinstance(current_profile.get("checks", {}), dict)
            else {}
        )
        for check_name in NEUROMORPHIC_PROFILE_CHECK_NAMES:
            if bool(previous_checks.get(check_name, False)) and not bool(
                current_checks.get(check_name, False)
            ):
                regressions.append(
                    {
                        "profile": profile_name,
                        "kind": "check_regression",
                        "check": check_name,
                        "previous": True,
                        "current": False,
                    }
                )

        previous_policy = str(previous_profile.get("online_update_adapter_policy", ""))
        current_policy = str(current_profile.get("online_update_adapter_policy", ""))
        if previous_policy and current_policy and previous_policy != current_policy:
            policy_changes.append(
                {
                    "profile": profile_name,
                    "previous": previous_policy,
                    "current": current_policy,
                }
            )

    return {
        "schema": "sara-neuromorphic-profile-trend-v1",
        "has_previous": bool(previous_profiles),
        "current_profile_count": int(current.get("profile_count", 0) or 0),
        "previous_profile_count": int(previous.get("profile_count", 0) or 0),
        "current_event_count": int(current.get("event_count", 0) or 0),
        "previous_event_count": int(previous.get("event_count", 0) or 0),
        "regression_count": len(regressions),
        "new_profiles": new_profiles,
        "missing_profiles": [
            str(item.get("profile", ""))
            for item in regressions
            if item.get("kind") == "missing_profile"
        ],
        "policy_change_count": len(policy_changes),
        "policy_changes": policy_changes,
        "regressions": regressions,
    }


def load_energy_history(history_path: str) -> List[Dict[str, Any]]:
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def append_energy_history(
    history_path: str,
    report: Dict[str, Any],
    max_entries: int = 64,
) -> List[Dict[str, Any]]:
    history = load_energy_history(history_path)
    entry = dict(report)
    entry.setdefault("recorded_at", time.time())
    history.append(entry)
    if max_entries > 0:
        history = history[-max_entries:]
    resolved_path = ensure_parent_directory(history_path)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    return history


def _attach_energy_proxy(case: Dict[str, Any]) -> Dict[str, Any]:
    state_units = max(1.0, float(case.get("state_units", 1.0) or 1.0))
    route_work_units = max(1.0, float(case.get("route_work_units", 1.0) or 1.0))
    sara_cost_units = state_units + route_work_units
    ann_reference_cost_units = max(
        sara_cost_units,
        float(case.get("ann_reference_cost_units", 1.0) or 1.0),
    )
    success = 1.0 if bool(case.get("success", False)) else 0.0
    enriched = dict(case)
    enriched.update(
        {
            "sara_energy_cost_units": sara_cost_units,
            "ann_reference_cost_units": ann_reference_cost_units,
            "success_per_energy_proxy": success / sara_cost_units,
            "ann_to_sara_cost_ratio": ann_reference_cost_units / sara_cost_units,
        }
    )
    return enriched


def run_energy_efficiency_benchmark(
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = [
        _attach_energy_proxy(_run_fast_identity_case()),
        _attach_energy_proxy(_run_session_memory_case()),
        _attach_energy_proxy(_run_edge_low_precision_persistence_case()),
        _attach_energy_proxy(_run_stochastic_readout_case()),
    ]
    success_scores = [1.0 if case["success"] else 0.0 for case in cases]
    route_scores = [
        1.0
        if case.get("memory_hit")
        in {
            "fast_path",
            "session_memory",
            "edge_low_precision_persistence",
            "stochastic_readout",
        }
        else 0.0
        for case in cases
    ]
    memory_scores = [
        1.0
        if case["success"] and float(case.get("state_units", 999.0)) <= 4.0
        else 0.0
        for case in cases
    ]
    stochastic_scores = [
        1.0 if case.get("memory_hit") != "stochastic_readout" or case["success"] else 0.0
        for case in cases
    ]
    avg_elapsed_ms = sum(float(case["elapsed_ms"]) for case in cases) / max(len(cases), 1)
    avg_state_units = sum(float(case.get("state_units", 0.0)) for case in cases) / max(len(cases), 1)
    total_successes = sum(success_scores)
    total_sara_energy_cost_units = sum(float(case["sara_energy_cost_units"]) for case in cases)
    total_ann_reference_cost_units = sum(float(case["ann_reference_cost_units"]) for case in cases)
    performance_energy_ratio_proxy = total_successes / max(total_sara_energy_cost_units, 1e-9)
    ann_cost_advantage_proxy = total_ann_reference_cost_units / max(total_sara_energy_cost_units, 1e-9)
    sparse_event_cost_scores = [
        min(1.0, 8.0 / max(float(case["sara_energy_cost_units"]), 1e-9))
        for case in cases
    ]
    sparse_event_cost_score = sum(sparse_event_cost_scores) / max(len(sparse_event_cost_scores), 1)
    brain_efficiency_alignment_proxy = min(1.0, ann_cost_advantage_proxy / 20.0) * sparse_event_cost_score
    # The benchmark treats sub-20ms average latency as ideal for the current CPU-first proxy.
    bounded_latency_score = max(0.0, min(1.0, 1.0 - (avg_elapsed_ms / 20.0)))

    metrics = {
        "energy_per_success_proxy": sum(success_scores) / max(len(success_scores), 1),
        "performance_energy_ratio_proxy": performance_energy_ratio_proxy,
        "ann_cost_advantage_proxy": ann_cost_advantage_proxy,
        "sparse_event_cost_score": sparse_event_cost_score,
        "brain_efficiency_alignment_proxy": brain_efficiency_alignment_proxy,
        "memory_per_success_proxy": sum(memory_scores) / max(len(memory_scores), 1),
        "low_overhead_route_score": sum(route_scores) / max(len(route_scores), 1),
        "bounded_latency_score": bounded_latency_score,
        "stochastic_readout_integrity": sum(stochastic_scores) / max(len(stochastic_scores), 1),
        "edge_low_precision_persistence_observed": min(
            float(case.get("edge_low_precision_persistence", 1.0))
            for case in cases
        ),
        "edge_sparse_routing_table_observed": min(
            float(case.get("edge_sparse_routing_table", 1.0))
            for case in cases
        ),
        "edge_event_compression_observed": min(
            float(case.get("edge_event_compression", 1.0))
            for case in cases
        ),
        "edge_sparse_readout_storage_observed": min(
            float(case.get("edge_sparse_readout_storage", 1.0))
            for case in cases
        ),
        "edge_storage_profile_integrity_observed": min(
            float(case.get("edge_storage_profile_integrity", 1.0))
            for case in cases
        ),
        "edge_sparse_readout_row_reduction_observed": max(
            float(case.get("row_reduction_ratio", 0.0))
            for case in cases
        ),
        "edge_multilevel_weight_profile_observed": min(
            float(case.get("edge_multilevel_weight_profile", 1.0))
            for case in cases
        ),
        "edge_format_compatibility_observed": min(
            float(case.get("edge_format_compatibility", 1.0))
            for case in cases
        ),
        "edge_manifest_integrity_observed": min(
            float(case.get("edge_manifest_integrity", 1.0))
            for case in cases
        ),
        "edge_strict_format_validation_observed": min(
            float(case.get("edge_strict_format_validation", 1.0))
            for case in cases
        ),
        "edge_payload_validation_report_observed": min(
            float(case.get("edge_payload_validation_report", 1.0))
            for case in cases
        ),
        "edge_delta_state_persistence_observed": min(
            float(case.get("edge_delta_state_persistence", 1.0))
            for case in cases
        ),
        "edge_delta_state_budget_observed": min(
            float(case.get("edge_delta_state_budget", 1.0))
            for case in cases
        ),
        "edge_delta_state_manifest_integrity_observed": min(
            float(case.get("edge_delta_state_manifest_integrity", 1.0))
            for case in cases
        ),
        "neuromorphic_ir_schema_integrity_observed": min(
            float(case.get("neuromorphic_ir_schema_integrity", 1.0))
            for case in cases
        ),
        "neuromorphic_capability_manifest_integrity_observed": min(
            float(case.get("neuromorphic_capability_manifest_integrity", 1.0))
            for case in cases
        ),
        "neuromorphic_backend_profile_compatibility_observed": min(
            float(case.get("neuromorphic_backend_profile_compatibility", 1.0))
            for case in cases
        ),
        "neuromorphic_sparse_event_budget_observed": min(
            float(case.get("neuromorphic_sparse_event_budget", 1.0))
            for case in cases
        ),
        "neuromorphic_profile_report_integrity_observed": min(
            float(case.get("neuromorphic_profile_report_integrity", 1.0))
            for case in cases
        ),
        "neuromorphic_stage_e_state_trace_ir_observed": min(
            float(case.get("neuromorphic_stage_e_state_trace_ir", 1.0))
            for case in cases
        ),
        "neuromorphic_stage_e_routing_hint_coverage_observed": min(
            float(case.get("neuromorphic_stage_e_routing_hint_coverage", 1.0))
            for case in cases
        ),
        "neuromorphic_stage_e_online_update_policy_observed": min(
            float(case.get("neuromorphic_stage_e_online_update_policy", 1.0))
            for case in cases
        ),
        "neuromorphic_stage_e_event_budget_observed": min(
            float(case.get("neuromorphic_stage_e_event_budget", 1.0))
            for case in cases
        ),
    }
    thresholds = {
        "energy_per_success_proxy": 1.0,
        "performance_energy_ratio_proxy": 0.20,
        "ann_cost_advantage_proxy": 8.0,
        "sparse_event_cost_score": 1.0,
        "brain_efficiency_alignment_proxy": 0.85,
        "memory_per_success_proxy": 1.0,
        "low_overhead_route_score": 1.0,
        "bounded_latency_score": 0.80,
        "stochastic_readout_integrity": 1.0,
    }
    threshold_results = {
        name: metrics.get(name, 0.0) >= threshold
        for name, threshold in thresholds.items()
    }
    metric_scores = {
        name: max(0.0, min(1.0, float(metrics.get(name, 0.0)) / max(float(threshold), 1e-9)))
        for name, threshold in thresholds.items()
    }

    report = {
        "evaluator_name": "EnergyEfficiencyBenchmark",
        "overall_score": sum(metric_scores.values()) / max(len(metric_scores), 1),
        "metrics": metrics,
        "metric_scores": metric_scores,
        "details": {
            "test_results": cases,
            "average_elapsed_ms": avg_elapsed_ms,
            "average_state_units": avg_state_units,
            "total_successes": total_successes,
            "total_sara_energy_cost_units": total_sara_energy_cost_units,
            "total_ann_reference_cost_units": total_ann_reference_cost_units,
            "energy_model": (
                "Proxy only: sara_energy_cost_units = bounded state units + route work units. "
                "ann_reference_cost_units models a dense tokenizer/ANN-style fallback path for the same fixture."
            ),
        },
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "passed": all(threshold_results.values()),
    }
    previous_report = history[-1] if history else None
    neuromorphic_profile_trend = build_neuromorphic_profile_trend(report, previous_report)
    report["neuromorphic_profile_trend"] = neuromorphic_profile_trend
    report["metrics"]["neuromorphic_profile_history_regression_observed"] = (
        1.0 if int(neuromorphic_profile_trend.get("regression_count", 0) or 0) == 0 else 0.0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight energy-efficiency proxy benchmark.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "energy_efficiency_benchmark.json"),
        help="Managed output path for the benchmark report.",
    )
    parser.add_argument(
        "--history-path",
        default=DEFAULT_HISTORY_PATH,
        help="Managed output path for the benchmark history.",
    )
    parser.add_argument("--no-history-update", action="store_true")
    args = parser.parse_args()

    history = load_energy_history(str(args.history_path))
    report = run_energy_efficiency_benchmark(history=history)
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    history_path = ""
    if not bool(args.no_history_update):
        append_energy_history(str(args.history_path), report)
        history_path = str(args.history_path)

    print("Energy-efficiency benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    if history_path:
        print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
