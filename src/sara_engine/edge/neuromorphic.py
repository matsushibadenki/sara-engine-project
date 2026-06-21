# Directory Path: src/sara_engine/edge/neuromorphic.py
# English Title: Neuromorphic Edge Profiles
# Purpose/Content: Builds chip-neutral spike event IR and backend compatibility reports for Sara-Edge payloads.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


NeuromorphicProfileInput = Optional[Union[str, Sequence[str]]]

NEUROMORPHIC_BACKEND_PROFILES: Dict[str, Dict[str, Any]] = {
    "lava": {
        "adapter": "lava_profile",
        "max_events": 4096,
        "requires_low_precision_weights": True,
        "requires_delay_support": True,
        "supports_online_update": True,
        "notes": "Research-friendly Python profile for Loihi 2/Lava conversion checks.",
    },
    "spinnaker": {
        "adapter": "spinnaker_profile",
        "max_events": 8192,
        "requires_low_precision_weights": False,
        "requires_delay_support": True,
        "supports_online_update": True,
        "notes": "Event packet routing profile for SpiNNaker-style sparse event delivery.",
    },
    "akida": {
        "adapter": "akida_profile",
        "max_events": 2048,
        "requires_low_precision_weights": True,
        "requires_delay_support": False,
        "supports_online_update": False,
        "notes": "Low-precision edge inference profile for Akida-style deployment checks.",
    },
}


def normalize_neuromorphic_profiles(neuromorphic_profile: NeuromorphicProfileInput) -> List[str]:
    if neuromorphic_profile is None:
        return []
    if isinstance(neuromorphic_profile, str):
        raw_profiles = [item.strip() for item in neuromorphic_profile.split(",")]
    else:
        raw_profiles = [str(item).strip() for item in neuromorphic_profile]
    profiles: List[str] = []
    for profile in raw_profiles:
        if profile and profile not in profiles:
            profiles.append(profile)
    return profiles


def _stable_event_id(name: str, offset: int = 900000) -> int:
    value = 0
    for index, char in enumerate(str(name)):
        value += (index + 1) * ord(char)
    return int(offset + (value % 90000))


def _normalize_state_trace_events(
    state_traces: Optional[Mapping[str, Mapping[str, Any]]],
    context_length: int,
) -> List[Dict[str, Any]]:
    if not isinstance(state_traces, Mapping):
        return []
    events: List[Dict[str, Any]] = []
    for index, (trace_name, trace) in enumerate(sorted(state_traces.items())):
        if not isinstance(trace, Mapping):
            continue
        state_units = int(trace.get("state_budget_units", trace.get("state_units", 0)) or 0)
        if state_units <= 0:
            continue
        delay = int(trace.get("delay", index) or 0) % max(1, int(context_length))
        routing_hint = str(trace.get("routing_hint", trace_name) or trace_name)
        online_update_policy = str(
            trace.get("online_update_policy", "observed_only_no_runtime_update")
            or "observed_only_no_runtime_update"
        )
        events.append(
            {
                "event_id": int(trace.get("event_id", _stable_event_id(str(trace_name))) or 0),
                "timestep": int(trace.get("timestep", 0) or 0),
                "channel": "state_trace",
                "weight_ref": str(trace.get("weight_ref", "state_trace") or "state_trace"),
                "delay": delay,
                "routing_hint": routing_hint,
                "online_update_policy": online_update_policy,
                "state_budget_units": state_units,
                "trace_name": str(trace_name),
            }
        )
    return events


def build_spike_event_ir(
    active_rows: List[int],
    context_length: int,
    total_readout_size: int,
    quantization_bits: Optional[int],
    compact_quantized: bool,
    compress_events: bool,
    delta_state: Mapping[str, Any],
    neuromorphic_profile: NeuromorphicProfileInput,
    state_traces: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    profiles = normalize_neuromorphic_profiles(neuromorphic_profile)
    if not profiles:
        return {
            "enabled": False,
            "schema": "none",
            "event_count": 0,
            "budget_ok": True,
            "events": [],
        }

    readout_events = [
        {
            "event_id": int(row_id),
            "timestep": 0,
            "channel": "readout_route",
            "weight_ref": "readout_synapses",
            "delay": int(index % max(1, int(context_length))),
            "routing_hint": "active_row",
            "online_update_policy": "static_readout_route",
            "state_budget_units": 1,
        }
        for index, row_id in enumerate(sorted(set(int(row) for row in active_rows)))
    ]
    trace_events = _normalize_state_trace_events(state_traces, context_length)
    events = readout_events + trace_events
    trace_state_units = sum(int(event.get("state_budget_units", 0) or 0) for event in trace_events)
    state_units = len(readout_events) + int(delta_state.get("state_units", 0) or 0) + trace_state_units
    online_update_policies = sorted(
        {
            str(event.get("online_update_policy", ""))
            for event in events
            if str(event.get("online_update_policy", "")).strip()
        }
    )
    return {
        "enabled": True,
        "schema": "sara-spike-event-ir-v1",
        "event_fields": [
            "event_id",
            "timestep",
            "channel",
            "weight_ref",
            "delay",
            "routing_hint",
            "online_update_policy",
            "state_budget_units",
        ],
        "event_count": len(events),
        "readout_event_count": len(readout_events),
        "state_trace_event_count": len(trace_events),
        "events": events,
        "event_encoding": "delta" if compress_events else "explicit",
        "max_delay": max(0, int(context_length) - 1),
        "online_update_policies": online_update_policies,
        "state_trace_routing_hints": sorted(
            {
                str(event.get("routing_hint", ""))
                for event in trace_events
                if str(event.get("routing_hint", "")).strip()
            }
        ),
        "weight_storage": "compact_int"
        if compact_quantized and quantization_bits is not None
        else "float",
        "state_budget_units": state_units,
        "state_budget_limit": max(1, int(total_readout_size)),
        "budget_ok": state_units <= max(1, int(total_readout_size)),
    }


def build_neuromorphic_capabilities(
    spike_event_ir: Mapping[str, Any],
    delta_state: Mapping[str, Any],
    quantization_bits: Optional[int],
    neuromorphic_profile: NeuromorphicProfileInput,
) -> Dict[str, Any]:
    profiles = normalize_neuromorphic_profiles(neuromorphic_profile)
    if not profiles:
        return {
            "enabled": False,
            "profiles": [],
            "backend_compatibility": {},
        }

    base_compatible = bool(
        spike_event_ir.get("enabled", False)
        and spike_event_ir.get("schema") == "sara-spike-event-ir-v1"
        and int(spike_event_ir.get("event_count", 0) or 0) > 0
        and bool(spike_event_ir.get("budget_ok", False))
    )
    compatibility = {
        profile_name: bool(base_compatible and profile_name in NEUROMORPHIC_BACKEND_PROFILES)
        for profile_name in profiles
    }
    return {
        "enabled": True,
        "profiles": profiles,
        "event_routing": True,
        "delay_support": int(spike_event_ir.get("max_delay", 0) or 0) >= 0,
        "low_precision_weights": bool(quantization_bits is not None),
        "state_persistence": bool(delta_state.get("enabled", False)),
        "online_update_support": bool(delta_state.get("enabled", False)),
        "online_update_policies": [
            str(policy)
            for policy in spike_event_ir.get("online_update_policies", [])
            if str(policy).strip()
        ]
        if isinstance(spike_event_ir.get("online_update_policies", []), list)
        else [],
        "state_trace_routing_hints": [
            str(hint)
            for hint in spike_event_ir.get("state_trace_routing_hints", [])
            if str(hint).strip()
        ]
        if isinstance(spike_event_ir.get("state_trace_routing_hints", []), list)
        else [],
        "backend_compatibility": compatibility,
    }


def build_neuromorphic_profile_report(
    spike_event_ir: Mapping[str, Any],
    neuromorphic_capabilities: Mapping[str, Any],
) -> Dict[str, Any]:
    profiles = [
        str(profile)
        for profile in neuromorphic_capabilities.get("profiles", [])
        if str(profile).strip()
    ]
    if not profiles:
        return {
            "enabled": False,
            "schema": "sara-neuromorphic-profile-report-v1",
            "profile_count": 0,
            "profiles": {},
            "all_profiles_compatible": True,
        }

    event_count = int(spike_event_ir.get("event_count", 0) or 0)
    delay_supported = bool(neuromorphic_capabilities.get("delay_support", False))
    low_precision = bool(neuromorphic_capabilities.get("low_precision_weights", False))
    online_update = bool(neuromorphic_capabilities.get("online_update_support", False))
    online_update_policies = [
        str(policy)
        for policy in neuromorphic_capabilities.get("online_update_policies", [])
        if str(policy).strip()
    ] if isinstance(neuromorphic_capabilities.get("online_update_policies", []), list) else []
    state_trace_routing_hints = [
        str(hint)
        for hint in neuromorphic_capabilities.get("state_trace_routing_hints", [])
        if str(hint).strip()
    ] if isinstance(neuromorphic_capabilities.get("state_trace_routing_hints", []), list) else []
    reports: Dict[str, Any] = {}
    for profile_name in profiles:
        spec = NEUROMORPHIC_BACKEND_PROFILES.get(profile_name, {})
        state_trace_adapter_policy = (
            "native_online_update"
            if bool(spec.get("supports_online_update", False))
            else "freeze_state_for_inference_profile"
        )
        checks = {
            "known_profile": bool(spec),
            "event_budget_ok": event_count <= int(spec.get("max_events", 0) or 0),
            "delay_support_ok": (not bool(spec.get("requires_delay_support", False))) or delay_supported,
            "low_precision_weight_ok": (
                not bool(spec.get("requires_low_precision_weights", False))
            )
            or low_precision,
            "online_update_policy_ok": bool(
                spec and (bool(spec.get("supports_online_update", False)) or state_trace_adapter_policy)
            ),
        }
        reports[profile_name] = {
            "adapter": str(spec.get("adapter", f"{profile_name}_profile")),
            "compatible": all(checks.values()),
            "checks": checks,
            "online_update_adapter_policy": "native_online_update"
            if bool(spec.get("supports_online_update", False))
            else "freeze_state_for_inference_profile",
            "state_trace_adapter_policy": state_trace_adapter_policy,
            "online_update_policies": online_update_policies,
            "state_trace_routing_hints": state_trace_routing_hints,
            "max_events": int(spec.get("max_events", 0) or 0),
            "notes": str(spec.get("notes", "Unknown neuromorphic backend profile.")),
        }
    return {
        "enabled": True,
        "schema": "sara-neuromorphic-profile-report-v1",
        "profile_count": len(profiles),
        "profiles": reports,
        "all_profiles_compatible": all(
            bool(profile_report.get("compatible", False))
            for profile_report in reports.values()
        ),
    }


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def build_neuromorphic_capability_matrix(
    spike_event_ir: Mapping[str, Any],
    neuromorphic_capabilities: Mapping[str, Any],
    neuromorphic_profile_report: Mapping[str, Any],
) -> Dict[str, Any]:
    profiles = _as_string_list(neuromorphic_capabilities.get("profiles", []))
    profile_reports = neuromorphic_profile_report.get("profiles", {})
    if not isinstance(profile_reports, Mapping):
        profile_reports = {}

    event_count = int(spike_event_ir.get("event_count", 0) or 0)
    readout_event_count = int(spike_event_ir.get("readout_event_count", 0) or 0)
    state_trace_event_count = int(spike_event_ir.get("state_trace_event_count", 0) or 0)
    state_budget_units = int(spike_event_ir.get("state_budget_units", 0) or 0)
    state_budget_limit = int(spike_event_ir.get("state_budget_limit", 0) or 0)
    routing_hints = _as_string_list(spike_event_ir.get("state_trace_routing_hints", []))
    update_policies = _as_string_list(spike_event_ir.get("online_update_policies", []))

    matrix: Dict[str, Any] = {}
    unsupported_summary: Dict[str, List[str]] = {}
    for profile_name in profiles:
        spec = NEUROMORPHIC_BACKEND_PROFILES.get(profile_name, {})
        profile_report = profile_reports.get(profile_name, {})
        if not isinstance(profile_report, Mapping):
            profile_report = {}
        checks = profile_report.get("checks", {})
        if not isinstance(checks, Mapping):
            checks = {}
        unsupported_operations = [
            str(check_name)
            for check_name, passed in sorted(checks.items())
            if not bool(passed)
        ]
        if unsupported_operations:
            unsupported_summary[profile_name] = unsupported_operations

        max_events = int(profile_report.get("max_events", spec.get("max_events", 0)) or 0)
        event_budget_headroom = max_events - event_count if max_events > 0 else 0
        matrix[profile_name] = {
            "adapter": str(profile_report.get("adapter", spec.get("adapter", "")) or ""),
            "compatible": bool(profile_report.get("compatible", False)),
            "known_profile": bool(checks.get("known_profile", profile_name in NEUROMORPHIC_BACKEND_PROFILES)),
            "event_count": event_count,
            "readout_event_count": readout_event_count,
            "state_trace_event_count": state_trace_event_count,
            "max_events": max_events,
            "event_budget_headroom": event_budget_headroom,
            "event_budget_ok": event_budget_headroom >= 0,
            "state_budget_units": state_budget_units,
            "state_budget_limit": state_budget_limit,
            "state_budget_ok": bool(spike_event_ir.get("budget_ok", False)),
            "delay_supported": bool(neuromorphic_capabilities.get("delay_support", False)),
            "requires_delay_support": bool(spec.get("requires_delay_support", False)),
            "low_precision_weights": bool(neuromorphic_capabilities.get("low_precision_weights", False)),
            "requires_low_precision_weights": bool(spec.get("requires_low_precision_weights", False)),
            "online_update_support": bool(neuromorphic_capabilities.get("online_update_support", False)),
            "supports_online_update": bool(spec.get("supports_online_update", False)),
            "online_update_adapter_policy": str(
                profile_report.get("online_update_adapter_policy", "") or ""
            ),
            "state_trace_adapter_policy": str(
                profile_report.get("state_trace_adapter_policy", "") or ""
            ),
            "routing_hints": routing_hints,
            "online_update_policies": update_policies,
            "unsupported_operations": unsupported_operations,
            "notes": str(profile_report.get("notes", spec.get("notes", "")) or ""),
        }

    return {
        "enabled": bool(profiles),
        "schema": "sara-neuromorphic-capability-matrix-v1",
        "profile_count": len(profiles),
        "profiles": matrix,
        "all_profiles_compatible": bool(
            profiles and all(bool(item.get("compatible", False)) for item in matrix.values())
        ),
        "unsupported_summary": unsupported_summary,
        "common_event_ir": {
            "schema": str(spike_event_ir.get("schema", "")),
            "event_count": event_count,
            "readout_event_count": readout_event_count,
            "state_trace_event_count": state_trace_event_count,
            "event_encoding": str(spike_event_ir.get("event_encoding", "")),
            "weight_storage": str(spike_event_ir.get("weight_storage", "")),
            "state_budget_units": state_budget_units,
            "state_budget_limit": state_budget_limit,
            "budget_ok": bool(spike_event_ir.get("budget_ok", False)),
            "routing_hints": routing_hints,
            "online_update_policies": update_policies,
        },
    }
