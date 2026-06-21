#!/usr/bin/env python3
"""Generate a managed neuromorphic backend capability matrix report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.neuromorphic import (  # noqa: E402
    build_neuromorphic_capabilities,
    build_neuromorphic_capability_matrix,
    build_neuromorphic_profile_report,
    build_spike_event_ir,
    normalize_neuromorphic_profiles,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "neuromorphic_capability_matrix.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "neuromorphic_capability_matrix_summary.txt")


def build_default_state_traces() -> Dict[str, Dict[str, Any]]:
    return {
        "forward_only": {
            "state_units": 1,
            "delay": 2,
            "routing_hint": "forward_only_eligibility",
            "online_update_policy": "local_credit_trace",
        },
        "multi_timescale": {
            "state_units": 4,
            "delay": 3,
            "routing_hint": "multi_timescale_state",
            "online_update_policy": "bounded_leak_state",
        },
        "predictive_error": {
            "state_units": 2,
            "delay": 1,
            "routing_hint": "predictive_error_correction",
            "online_update_policy": "residual_correction_trace",
        },
        "dendritic_feedback": {
            "state_units": 2,
            "delay": 1,
            "routing_hint": "bounded_dendritic_route_hint",
            "online_update_policy": "local_hebbian_homeostatic_update",
        },
        "synesthetic_thalamic_route": {
            "state_units": 1,
            "delay": 0,
            "routing_hint": "equal_modality_thalamic_route",
            "online_update_policy": "observed_only_bounded_route_selection",
        },
    }


def build_report(
    *,
    profiles: Sequence[str],
    active_row_count: int,
    context_length: int,
    total_readout_size: int,
    quantization_bits: Optional[int],
) -> Dict[str, Any]:
    normalized_profiles = normalize_neuromorphic_profiles(list(profiles))
    active_rows = list(range(max(0, int(active_row_count))))
    spike_event_ir = build_spike_event_ir(
        active_rows=active_rows,
        context_length=max(1, int(context_length)),
        total_readout_size=max(1, int(total_readout_size)),
        quantization_bits=quantization_bits,
        compact_quantized=quantization_bits is not None,
        compress_events=True,
        delta_state={"enabled": True, "state_units": 3},
        neuromorphic_profile=normalized_profiles,
        state_traces=build_default_state_traces(),
    )
    capabilities = build_neuromorphic_capabilities(
        spike_event_ir=spike_event_ir,
        delta_state={"enabled": True},
        quantization_bits=quantization_bits,
        neuromorphic_profile=normalized_profiles,
    )
    profile_report = build_neuromorphic_profile_report(spike_event_ir, capabilities)
    matrix = build_neuromorphic_capability_matrix(spike_event_ir, capabilities, profile_report)
    passed = bool(
        matrix.get("enabled")
        and matrix.get("all_profiles_compatible")
        and matrix.get("common_event_ir", {}).get("budget_ok")
        and not matrix.get("unsupported_summary")
    )
    return {
        "schema": "sara-neuromorphic-capability-matrix-report-v1",
        "passed": passed,
        "profiles": normalized_profiles,
        "active_row_count": int(active_row_count),
        "context_length": int(context_length),
        "total_readout_size": int(total_readout_size),
        "quantization_bits": quantization_bits,
        "spike_event_ir": spike_event_ir,
        "neuromorphic_capabilities": capabilities,
        "neuromorphic_profile_report": profile_report,
        "capability_matrix": matrix,
        "policy_notes": [
            "CPU behavior remains the reference path.",
            "Hardware-specific adapters are optional and represented as profile checks.",
            "The matrix records unsupported operations instead of requiring accelerator-specific execution.",
            "Dendritic feedback and synesthetic thalamic routing are optional bounded state traces.",
            "Reports are written only under workspace/evaluation.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    matrix = report.get("capability_matrix", {})
    common_ir = matrix.get("common_event_ir", {}) if isinstance(matrix, dict) else {}
    lines = [
        f"Neuromorphic capability matrix: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Profiles: {', '.join(report.get('profiles', []))}",
        f"Event count: {common_ir.get('event_count')}",
        f"State budget: {common_ir.get('state_budget_units')}/{common_ir.get('state_budget_limit')}",
        f"All profiles compatible: {matrix.get('all_profiles_compatible')}",
    ]
    profile_rows = matrix.get("profiles", {}) if isinstance(matrix, dict) else {}
    if isinstance(profile_rows, dict):
        for profile_name, row in sorted(profile_rows.items()):
            if not isinstance(row, dict):
                continue
            unsupported = row.get("unsupported_operations", [])
            lines.append(
                "- {name}: compatible={compatible}, adapter={adapter}, headroom={headroom}, unsupported={unsupported}".format(
                    name=profile_name,
                    compatible=row.get("compatible"),
                    adapter=row.get("adapter"),
                    headroom=row.get("event_budget_headroom"),
                    unsupported=", ".join(str(item) for item in unsupported) if unsupported else "none",
                )
            )
    return "\n".join(lines) + "\n"


def write_outputs(report: Dict[str, Any], report_path: str, summary_path: str) -> None:
    resolved_report_path = ensure_parent_directory(report_path)
    with open(resolved_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a neuromorphic backend capability matrix.")
    parser.add_argument("--profile", action="append", default=None, help="Backend profile name. May be repeated.")
    parser.add_argument("--active-row-count", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--total-readout-size", type=int, default=64)
    parser.add_argument("--quantization-bits", type=int, default=3)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    profiles = args.profile or ["lava", "spinnaker", "akida"]
    quantization_bits = args.quantization_bits if args.quantization_bits > 0 else None
    report = build_report(
        profiles=profiles,
        active_row_count=args.active_row_count,
        context_length=args.context_length,
        total_readout_size=args.total_readout_size,
        quantization_bits=quantization_bits,
    )
    write_outputs(report, args.report_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "profile_count": len(report["profiles"]),
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
