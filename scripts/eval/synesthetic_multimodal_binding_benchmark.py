#!/usr/bin/env python3
"""Run the observed-only sparse synesthetic multimodal binding benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.multimodal.synesthetic_binding import (  # noqa: E402
    AudioEventAdapter,
    LanguageEventAdapter,
    SparseMultimodalEvent,
    SparsePluggableCorticalColumn,
    SparseSynestheticLinker,
    SparseTemporalBinder,
    SparseThalamicGate,
    TactileEventAdapter,
    VisionEventAdapter,
)
from sara_engine.learning.dendritic_feedback import SparseDendriticFeedbackGate  # noqa: E402
from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache  # noqa: E402
from sara_engine.memory.multimodal_event_bundle_admission import (  # noqa: E402
    build_multimodal_event_state_candidate,
)  # noqa: E402
from sara_engine.multimodal.structural_verification import (  # noqa: E402
    ModalityEvidence,
    MultimodalStructuralVerifier,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    interim_data_path,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "synesthetic_multimodal_cases.jsonl")
DEFAULT_CROSS_LINK_PATH = interim_data_path("autobot", "synesthetic_cross_links.jsonl")
DEFAULT_BINDING_MANIFEST_PATH = processed_data_path("autobot", "synesthetic_binding_manifest.jsonl")
DEFAULT_LATENT_MANIFEST_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "synesthetic_multimodal_binding_traces.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark_summary.txt")
DEFAULT_PLUG_SWAP_PATH = workspace_path("evaluation", "sparse_cortical_column_plug_swap_report.json")


def default_fixture_cases() -> List[Dict[str, Any]]:
    return [
        {
            "schema": "sara-synesthetic-multimodal-case-v1",
            "case_id": "hard_surface",
            "events": [
                {"modality": "language", "timestamp_ms": 2, "source_id": "hard-language", "signature": [101, 102], "label": "hard"},
                {"modality": "vision", "timestamp_ms": 8, "source_id": "hard-vision", "signature": [201, 202], "label": "hard"},
                {"modality": "audio", "timestamp_ms": 15, "source_id": "hard-audio", "signature": [301, 302], "label": "hard"},
                {"modality": "tactile", "timestamp_ms": 21, "source_id": "hard-tactile", "signature": [401, 402], "label": "hard"},
            ],
        },
        {
            "schema": "sara-synesthetic-multimodal-case-v1",
            "case_id": "soft_surface",
            "events": [
                {"modality": "language", "timestamp_ms": 34, "source_id": "soft-language", "signature": [111, 112], "label": "soft"},
                {"modality": "vision", "timestamp_ms": 39, "source_id": "soft-vision", "signature": [211, 212], "label": "soft"},
                {"modality": "audio", "timestamp_ms": 45, "source_id": "soft-audio", "signature": [311, 312], "label": "soft"},
                {"modality": "tactile", "timestamp_ms": 55, "source_id": "soft-tactile", "signature": [411, 412], "label": "soft"},
            ],
        },
        {
            "schema": "sara-synesthetic-multimodal-case-v1",
            "case_id": "missing_tactile",
            "events": [
                {"modality": "audio", "timestamp_ms": 68, "source_id": "hard-audio-recall", "signature": [301, 302], "label": "hard"}
            ],
            "missing_modality": "tactile",
            "expected_signature": [401, 402],
        },
        {
            "schema": "sara-synesthetic-multimodal-case-v1",
            "case_id": "unknown_missing_tactile",
            "events": [
                {"modality": "audio", "timestamp_ms": 100, "source_id": "unknown-audio", "signature": [901, 902], "label": "unknown"}
            ],
            "missing_modality": "tactile",
            "expected_abstain": True,
        },
    ]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def ensure_fixture(path: str) -> str:
    rows = read_jsonl(path)
    if rows and all(row.get("schema") == "sara-synesthetic-multimodal-case-v1" for row in rows):
        return path
    return write_jsonl(path, default_fixture_cases())


def _normalize_cases(
    cases: Sequence[Dict[str, Any]],
    binder: SparseTemporalBinder,
) -> Dict[str, List[SparseMultimodalEvent]]:
    normalized: Dict[str, List[SparseMultimodalEvent]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        case_events: List[SparseMultimodalEvent] = []
        for event in case.get("events", []):
            case_events.append(
                binder.normalize_event(
                    modality=event["modality"],
                    timestamp_ms=float(event["timestamp_ms"]),
                    source_id=str(event["source_id"]),
                    sparse_signature=event["signature"],
                    confidence=float(event.get("confidence", 0.9)),
                    label=str(event.get("label", "")),
                    source_ref=str(event.get("source_ref", f"fixture://{case_id}")),
                )
            )
        normalized[case_id] = case_events
    return normalized


def build_report(
    cases: Sequence[Dict[str, Any]],
    *,
    window_ms: float,
    cross_link_path: str,
    binding_manifest_path: str,
    latent_manifest_path: str,
    trace_path: str,
    plug_swap_path: str,
) -> Dict[str, Any]:
    binder = SparseTemporalBinder(window_ms=window_ms, max_events_per_chunk=32)
    normalized = _normalize_cases(cases, binder)
    training_events = normalized.get("hard_surface", []) + normalized.get("soft_surface", [])
    latent_rows = read_jsonl(latent_manifest_path)
    latent_row = latent_rows[0] if latent_rows else {}

    adapters = {
        "language": LanguageEventAdapter(max_events=8),
        "vision": VisionEventAdapter(max_events=8),
        "audio": AudioEventAdapter(max_events=8),
        "tactile": TactileEventAdapter(max_events=8),
    }
    adapter_events = [
        adapters[event.modality].encode(
            [str(item) for item in event.sparse_signature],
            binder=binder,
            timestamp_ms=event.timestamp_ms,
            source_id=f"adapter-{event.source_id}",
            confidence=event.confidence,
            label=event.label,
            source_ref=str(latent_row.get("source_ref", "fixture://synesthetic-binding")),
            latent_cluster_id=str(latent_row.get("latent_cluster_id", "")),
            latent_signature=latent_row.get("sparse_signature", [])[:1],
            topology_terms=[f"chunk:{event.time_chunk_id}"],
            gate_history_terms=["observed_only"],
        )
        for event in training_events
    ]
    adapter_ir_integrity = float(
        len(adapter_events) == len(training_events)
        and {event.modality for event in adapter_events}
        == {"language", "vision", "audio", "tactile"}
        and all(event.specialization_factors for event in adapter_events)
    )
    own_latent_integration = float(
        not latent_rows
        or all(
            event.latent_cluster_id == str(latent_row.get("latent_cluster_id", ""))
            and "own_latent" in event.specialization_factors
            for event in adapter_events
        )
    )

    linker = SparseSynestheticLinker(max_links_per_event=4, max_total_links=256)
    linker.update(training_events)

    column_a = SparsePluggableCorticalColumn()
    column_b = SparsePluggableCorticalColumn()
    result_a = [column_a.process(event) for event in training_events]
    swapped_events = [
        binder.normalize_event(
            modality="tactile" if event.modality == "audio" else "audio",
            timestamp_ms=event.timestamp_ms,
            source_id=f"plug-swap-{event.source_id}",
            sparse_signature=event.sparse_signature,
            confidence=event.confidence,
            uncertainty=event.uncertainty,
            label=event.label,
        )
        for event in training_events
    ]
    result_b = [column_b.process(event) for event in swapped_events]
    rule_consistency = float(
        all(item["learning_rule"] == "shared_local_hebbian" for item in result_a + result_b)
    )
    activation_consistency = float(
        all(
            left["active_event_ids"] == right["active_event_ids"]
            and left["input_event_count"] == right["input_event_count"]
            for left, right in zip(result_a, result_b)
        )
    )
    plug_swap_integrity = min(rule_consistency, activation_consistency)

    window_profiles: List[Dict[str, Any]] = []
    for profile_ms in (25.0, 32.0, 40.0):
        profile_binder = SparseTemporalBinder(window_ms=profile_ms, max_events_per_chunk=32)
        profile_normalized = _normalize_cases(cases, profile_binder)
        aligned_case_count = 0
        profile_cost = 0
        for case_id in ("hard_surface", "soft_surface"):
            profile_events = profile_normalized.get(case_id, [])
            chunks = profile_binder.bind(profile_events)
            profile_cost += sum(event.event_cost for event in profile_events)
            if len(chunks) == 1 and len(next(iter(chunks.values()))) == 4:
                aligned_case_count += 1
        window_profiles.append(
            {
                "window_ms": profile_ms,
                "alignment_quality": float(aligned_case_count) / 2.0,
                "event_cost": profile_cost,
            }
        )
    best_window = sorted(
        window_profiles,
        key=lambda row: (-row["alignment_quality"], row["event_cost"], row["window_ms"]),
    )[0]
    temporal_alignment_quality = float(best_window["alignment_quality"])

    missing_rows: List[Dict[str, Any]] = []
    correct_prediction_count = 0
    abstention_correct_count = 0
    non_language_route_useful = 0.0
    for case in cases:
        missing_modality = case.get("missing_modality")
        if not missing_modality:
            continue
        event = normalized[str(case["case_id"])][0]
        prediction = linker.predict(event, target_modality=str(missing_modality))
        expected_signature = set(int(item) for item in case.get("expected_signature", []))
        predicted_signature = set(prediction["predicted_missing_modality_events"])
        if expected_signature and predicted_signature == expected_signature:
            correct_prediction_count += 1
            if event.modality != "language":
                non_language_route_useful = 1.0
        if bool(case.get("expected_abstain")) and prediction["abstained"]:
            abstention_correct_count += 1
        missing_rows.append({"case_id": case["case_id"], **prediction})

    missing_modality_accuracy = float(correct_prediction_count)
    missing_modality_abstention = float(abstention_correct_count)
    thalamic_gate = SparseThalamicGate(route_threshold=0.3)
    equal_gate = thalamic_gate.route(training_events, mode="equal")
    dendritic_gate = SparseDendriticFeedbackGate(threshold=0.8, event_budget=64)
    route_hints: Dict[str, float] = {}
    for event in training_events:
        dendritic_result = dendritic_gate.gate(
            active_event_ids=event.sparse_signature,
            local_potentials={event_id: event.confidence for event_id in event.sparse_signature},
        )
        route_hints[event.source_id] = 0.1 if dendritic_result.gated_events else -0.1
    focused_gate = thalamic_gate.route(
        training_events,
        mode="focused",
        focused_modality="tactile",
        route_hints=route_hints,
    )
    dendritic_route_hint_integrity = float(
        bool(route_hints)
        and all("route_hint" in row for row in focused_gate.trace)
        and all(abs(float(row["route_hint"])) <= 0.25 for row in focused_gate.trace)
    )
    route_traceability = float(
        bool(equal_gate.trace)
        and bool(focused_gate.trace)
        and all(
            {"modality", "source_id", "score", "focus_gain", "route_hint", "cost_penalty", "routed"}
            <= set(row)
            for row in equal_gate.trace + focused_gate.trace
        )
    )
    event_bundles = binder.bundle_events(training_events, route_trace=focused_gate.trace)
    bundle_integrity = float(
        bool(event_bundles)
        and all(bundle.audit is not None for bundle in event_bundles)
        and all(bundle.audit.payload_separable for bundle in event_bundles if bundle.audit is not None)
        and all(len(bundle.child_records) >= len(bundle.modality_ids) for bundle in event_bundles)
    )
    binding_audit_coverage = float(
        len([bundle for bundle in event_bundles if bundle.audit is not None and bundle.audit.admitted])
        / float(max(1, len(event_bundles)))
    )
    structural_verifier = MultimodalStructuralVerifier()
    bundle_admission_results = [
        build_multimodal_event_state_candidate(
            bundle,
            time_segment=bundle.time_chunk_id,
            structural_decision=structural_verifier.verify(
                (
                    ModalityEvidence(
                        modality=item.modality,
                        label=item.label,
                        claim_key=item.claim_key,
                        timestamp_ms=item.timestamp_ms,
                        source_ref=item.source_ref,
                        observed=item.observed,
                        confidence=item.confidence,
                    )
                    for item in bundle.child_records
                ),
                expected_modalities=bundle.modality_ids,
            ),
        )
        for bundle in event_bundles
    ]
    event_state_cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=8)
    cache_admissions = [
        event_state_cache.admit(result.candidate)
        for result in bundle_admission_results
        if result.promotion_allowed
    ]
    bundle_event_state_promotion = float(
        len([result for result in bundle_admission_results if result.promotion_allowed])
        / float(max(1, len(bundle_admission_results)))
    )
    bundle_event_state_cache_integrity = float(
        bool(cache_admissions)
        and all(admission.accepted for admission in cache_admissions)
        and event_state_cache.state_dict()["entry_count"] >= 1
    )

    link_rows = [
        {
            "source_modality": key[0],
            "source_event_id": key[1],
            "target_modality": key[2],
            "target_event_id": key[3],
            "count": count,
        }
        for key, count in sorted(linker.link_counts.items())
    ]
    manifest_rows = [
        {
            "schema": "sara-synesthetic-binding-manifest-v2",
            "case_id": case_id,
            "events": [event.to_dict() for event in events],
            "bundles": [bundle.to_dict() for bundle in binder.bundle_events(events)],
            "observed_only": True,
        }
        for case_id, events in sorted(normalized.items())
    ]
    trace_rows = [
        {"trace_type": "equal_gate", "trace": equal_gate.trace},
        {"trace_type": "focused_gate", "trace": focused_gate.trace},
        {"trace_type": "missing_modality", "trace": missing_rows},
        {
            "trace_type": "binding_audit",
            "trace": [bundle.audit.to_dict() for bundle in event_bundles if bundle.audit is not None],
        },
        {
            "trace_type": "bundle_event_state_admission",
            "trace": [result.to_dict() for result in bundle_admission_results],
        },
    ]
    write_jsonl(cross_link_path, link_rows)
    write_jsonl(binding_manifest_path, manifest_rows)
    write_jsonl(trace_path, trace_rows)
    plug_swap_report = {
        "schema": "sara-sparse-cortical-column-plug-swap-v1",
        "passed": plug_swap_integrity == 1.0,
        "learning_rule_consistency": rule_consistency,
        "activation_consistency": activation_consistency,
        "column_a_state_budget": column_a.state_budget_units(),
        "column_b_state_budget": column_b.state_budget_units(),
    }
    resolved_plug_swap = ensure_parent_directory(plug_swap_path)
    with open(resolved_plug_swap, "w", encoding="utf-8") as handle:
        json.dump(plug_swap_report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    max_event_cost = max(
        [item["event_cost"] for item in result_a + result_b]
        + [row["event_cost"] for row in missing_rows],
        default=0,
    )
    max_state_budget = max(
        column_a.state_budget_units(),
        column_b.state_budget_units(),
        linker.state_budget_units(),
    )
    passed = bool(
        temporal_alignment_quality == 1.0
        and plug_swap_integrity == 1.0
        and missing_modality_accuracy == 1.0
        and missing_modality_abstention == 1.0
        and non_language_route_useful == 1.0
        and route_traceability == 1.0
        and bundle_integrity == 1.0
        and binding_audit_coverage == 1.0
        and bundle_event_state_promotion == 1.0
        and bundle_event_state_cache_integrity == 1.0
        and adapter_ir_integrity == 1.0
        and own_latent_integration == 1.0
        and dendritic_route_hint_integrity == 1.0
        and max_event_cost <= 64
        and max_state_budget <= 256
    )
    return {
        "schema": "sara-synesthetic-multimodal-binding-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(cases),
        "window_ms": float(window_ms),
        "window_profiles": window_profiles,
        "selected_window_ms": best_window["window_ms"],
        "metrics": {
            "temporal_alignment_quality": temporal_alignment_quality,
            "plug_swap_integrity": plug_swap_integrity,
            "cross_modal_link_precision": missing_modality_accuracy,
            "missing_modality_abstention_integrity": missing_modality_abstention,
            "non_language_route_usefulness": non_language_route_useful,
            "route_traceability": route_traceability,
            "bundle_integrity": bundle_integrity,
            "binding_audit_coverage": binding_audit_coverage,
            "bundle_event_state_promotion": bundle_event_state_promotion,
            "bundle_event_state_cache_integrity": bundle_event_state_cache_integrity,
            "adapter_ir_integrity": adapter_ir_integrity,
            "own_latent_integration": own_latent_integration,
            "dendritic_route_hint_integrity": dendritic_route_hint_integrity,
            "max_event_cost": max_event_cost,
            "max_state_budget_units": max_state_budget,
            "equal_gate_routed_count": len(equal_gate.routed_events),
            "focused_gate_routed_count": len(focused_gate.routed_events),
        },
        "missing_modality_results": missing_rows,
        "outputs": {
            "cross_links": os.path.abspath(cross_link_path),
            "binding_manifest": os.path.abspath(binding_manifest_path),
            "latent_manifest": os.path.abspath(latent_manifest_path),
            "traces": os.path.abspath(trace_path),
            "plug_swap_report": os.path.abspath(plug_swap_path),
        },
        "policy_notes": [
            "All modality streams use sparse events rather than dense universal embeddings.",
            "The same cortical-column learning rule processes every modality.",
            "Missing-modality predictions are observed-only and uncertainty-aware.",
            "Cross-modal links and thalamic routes are bounded and auditable.",
            "Shared event bundles preserve modality-local payloads rather than collapsing them.",
            "Only verified source-backed multimodal bundles may bridge into durable Event Memory candidates.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        f"Synesthetic multimodal binding benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
        f"Window ms: {report.get('window_ms')}",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(metrics.items()))
    return "\n".join(lines) + "\n"


def write_outputs(report: Dict[str, Any], report_path: str, summary_path: str) -> None:
    resolved_report = ensure_parent_directory(report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse synesthetic multimodal binding benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--cross-link-path", default=DEFAULT_CROSS_LINK_PATH)
    parser.add_argument("--binding-manifest-path", default=DEFAULT_BINDING_MANIFEST_PATH)
    parser.add_argument("--latent-manifest-path", default=DEFAULT_LATENT_MANIFEST_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--plug-swap-path", default=DEFAULT_PLUG_SWAP_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--window-ms", type=float, default=32.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fixture_path = ensure_fixture(args.fixture_path)
    cases = read_jsonl(fixture_path)
    report = build_report(
        cases,
        window_ms=args.window_ms,
        cross_link_path=args.cross_link_path,
        binding_manifest_path=args.binding_manifest_path,
        latent_manifest_path=args.latent_manifest_path,
        trace_path=args.trace_path,
        plug_swap_path=args.plug_swap_path,
    )
    report["fixture_path"] = os.path.abspath(fixture_path)
    write_outputs(report, args.report_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "case_count": report["case_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
