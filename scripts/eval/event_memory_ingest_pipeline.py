from __future__ import annotations

import json
from typing import Any, Dict

from sara_engine.ingest import (
    EventMemoryIngestPipeline,
    FrequentSequenceMiner,
    PredictionGainEstimator,
    ProposalVerifier,
    SynchronyDetector,
    make_candidate_event,
)
from sara_engine.multimodal.synesthetic_binding import SparseTemporalBinder
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _synthetic_fixture() -> Dict[str, Any]:
    return {
        "source_ref": "synthetic_session_a",
        "source_hash": "synthetic_hash_a",
        "streams": [
            {
                "stream_id": "audio-1",
                "modality": "audio",
                "samples": [
                    {"time_ms": 0, "value": 0.0},
                    {"time_ms": 100, "value": 0.7},
                    {"time_ms": 220, "value": 0.0},
                    {"time_ms": 360, "value": 0.8},
                ],
            },
            {
                "stream_id": "text-1",
                "modality": "text",
                "samples": [
                    {"time_ms": 0, "value": 0.0},
                    {"time_ms": 135, "value": 0.85},
                    {"time_ms": 255, "value": 0.0},
                    {"time_ms": 395, "value": 0.9},
                ],
            },
        ],
        "candidate_events": [
            make_candidate_event(
                {
                    "record_id": "cand-vision-1",
                    "modality": "vision",
                    "label": "visual_cluster_018",
                    "local_time_ms": 150,
                    "confidence": 0.88,
                    "source_ref": "synthetic_session_a",
                    "source_hash": "synthetic_hash_a",
                    "extractor_name": "candidate_proposals",
                    "extractor_version": "v1",
                    "evidence_count": 3,
                    "counterexample_count": 0,
                    "prediction_gain": 0.2,
                }
            ),
        ],
    }


def _synthetic_multimodal_bundles():
    binder = SparseTemporalBinder(window_ms=32.0)
    events = [
        binder.normalize_event(
            modality="language",
            timestamp_ms=128.0,
            source_id="bundle-language",
            sparse_signature=[101, 102],
            confidence=0.9,
            label="hard",
            source_ref="fixture://bundle-hard",
        ),
        binder.normalize_event(
            modality="vision",
            timestamp_ms=130.0,
            source_id="bundle-vision",
            sparse_signature=[201, 202],
            confidence=0.9,
            label="hard",
            source_ref="fixture://bundle-hard",
        ),
        binder.normalize_event(
            modality="audio",
            timestamp_ms=136.0,
            source_id="bundle-audio",
            sparse_signature=[301, 302],
            confidence=0.9,
            label="hard",
            source_ref="fixture://bundle-hard",
        ),
        binder.normalize_event(
            modality="tactile",
            timestamp_ms=140.0,
            source_id="bundle-tactile",
            sparse_signature=[401, 402],
            confidence=0.9,
            label="hard",
            source_ref="fixture://bundle-hard",
        ),
    ]
    return binder.bundle_events(events)


def build_report() -> Dict[str, Any]:
    fixture = _synthetic_fixture()
    pipeline = EventMemoryIngestPipeline(
        synchrony_detector=SynchronyDetector(window_ms=80, cross_modal_only=True),
        prediction_gain_estimator=PredictionGainEstimator(min_support=1, min_gain=0.0, max_delay_ms=120),
        verifier=ProposalVerifier(
            min_confidence=0.1,
            min_evidence_count=1,
            min_prediction_gain=0.0,
            max_counterexample_rate=0.9,
        ),
        sequence_miner=FrequentSequenceMiner(min_support_episodes=1, max_pattern_length=3, max_span_ms=160),
    )
    result = pipeline.ingest_streams(
        fixture["streams"],
        source_ref=str(fixture["source_ref"]),
        source_hash=str(fixture["source_hash"]),
        candidate_events=fixture["candidate_events"],
        multimodal_bundles=_synthetic_multimodal_bundles(),
    )
    payload = result.to_dict()
    change_point_count = len(payload["change_points"])
    observed_event_count = len(payload["observed_events"])
    accepted_candidate_count = len(payload["accepted_candidate_events"])
    rejected_candidate_count = len(payload["rejected_candidate_events"])
    episode_count = len(payload["episodes"])
    candidate_relation_count = len(payload["candidate_relations"])
    verified_relation_count = len(payload["verified_relations"])
    lineage_count = len(payload["lineage_ledger"])
    total_candidate_count = accepted_candidate_count + rejected_candidate_count
    traces = payload["traces"]
    persistent_trace = (
        traces.get("persistent_self_state", {})
        if isinstance(traces.get("persistent_self_state", {}), dict)
        else {}
    )
    bundle_trace = (
        traces.get("multimodal_bundle_admission", {})
        if isinstance(traces.get("multimodal_bundle_admission", {}), dict)
        else {}
    )
    bundle_count = int(bundle_trace.get("bundle_count", 0) or 0)
    bundle_promotion_allowed_count = int(bundle_trace.get("promotion_allowed_count", 0) or 0)
    bundle_promotion_rate = float(bundle_promotion_allowed_count) / float(max(bundle_count, 1))
    bundle_supported_relation_yield = bundle_promotion_rate * (
        float(verified_relation_count) / float(max(candidate_relation_count, 1))
    )
    bundle_compression_contribution = bundle_promotion_rate * (
        float(observed_event_count + accepted_candidate_count) / float(max(episode_count, 1))
    )
    metrics = {
        "eventization_emission_ratio": float(observed_event_count) / float(max(change_point_count, 1)),
        "candidate_event_acceptance_rate": float(accepted_candidate_count)
        / float(max(total_candidate_count, 1)),
        "episode_compression_ratio": float(observed_event_count + accepted_candidate_count)
        / float(max(episode_count, 1)),
        "relation_verification_yield": float(verified_relation_count)
        / float(max(candidate_relation_count, 1)),
        "lineage_coverage_ratio": float(lineage_count)
        / float(
            max(
                observed_event_count
                + total_candidate_count
                + candidate_relation_count
                + verified_relation_count,
                1,
            )
        ),
        "self_state_continuity": float(persistent_trace.get("continuity_score", 0.0) or 0.0),
        "self_state_active_count": float(
            len(persistent_trace.get("current_active_ids", []) or [])
        ),
        "self_state_external_event_ratio": float(
            len(persistent_trace.get("current_active_ids", []) or [])
        )
        / float(max(int(persistent_trace.get("external_event_count", 0) or 0), 1)),
        "multimodal_bundle_promotion_rate": bundle_promotion_rate,
        "multimodal_bundle_relation_verification_yield": bundle_supported_relation_yield,
        "multimodal_bundle_compression_contribution": bundle_compression_contribution,
    }
    return {
        "schema": "sara-event-memory-ingest-pipeline-report-v1",
        "passed": bool(
            payload["observed_events"]
            and payload["episodes"]
            and payload["verified_relations"]
            and payload["traces"].get("persistent_self_state", {}).get("current_active_ids")
            and int(bundle_trace.get("promotion_allowed_count", 0) or 0) >= 1
        ),
        "counts": {
            "change_points": len(payload["change_points"]),
            "observed_events": len(payload["observed_events"]),
            "accepted_candidate_events": len(payload["accepted_candidate_events"]),
            "episodes": len(payload["episodes"]),
            "frequent_sequences": len(payload["frequent_sequences"]),
            "candidate_relations": len(payload["candidate_relations"]),
            "verified_relations": len(payload["verified_relations"]),
            "lineage_ledger_entries": len(payload["lineage_ledger"]),
        },
        "metrics": metrics,
        "traces": traces,
        "result": payload,
    }


def build_summary(report: Dict[str, Any]) -> str:
    counts = report.get("counts", {})
    metrics = report.get("metrics", {})
    traces = report.get("traces", {})
    lines = [
        "SARA Event Memory ingest pipeline",
        f"- passed: {bool(report.get('passed', False))}",
        f"- observed_events: {int(counts.get('observed_events', 0) or 0)}",
        f"- accepted_candidate_events: {int(counts.get('accepted_candidate_events', 0) or 0)}",
        f"- episodes: {int(counts.get('episodes', 0) or 0)}",
        f"- frequent_sequences: {int(counts.get('frequent_sequences', 0) or 0)}",
        f"- candidate_relations: {int(counts.get('candidate_relations', 0) or 0)}",
        f"- verified_relations: {int(counts.get('verified_relations', 0) or 0)}",
        f"- eventization_emission_ratio: {float(metrics.get('eventization_emission_ratio', 0.0) or 0.0):.3f}",
        f"- candidate_event_acceptance_rate: {float(metrics.get('candidate_event_acceptance_rate', 0.0) or 0.0):.3f}",
        f"- episode_compression_ratio: {float(metrics.get('episode_compression_ratio', 0.0) or 0.0):.3f}",
        f"- relation_verification_yield: {float(metrics.get('relation_verification_yield', 0.0) or 0.0):.3f}",
        f"- lineage_coverage_ratio: {float(metrics.get('lineage_coverage_ratio', 0.0) or 0.0):.3f}",
        f"- eventization_emitted: {int(traces.get('eventization', {}).get('emitted_count', 0) or 0)}",
        f"- sequence_patterns: {int(traces.get('frequent_sequence', {}).get('accepted_sequences', 0) or 0)}",
        f"- self_state_active: {len(traces.get('persistent_self_state', {}).get('current_active_ids', []) or [])}",
        f"- self_state_continuity: {float(traces.get('persistent_self_state', {}).get('continuity_score', 0.0) or 0.0):.3f}",
        f"- multimodal_bundle_promotion_rate: {float(metrics.get('multimodal_bundle_promotion_rate', 0.0) or 0.0):.3f}",
        f"- multimodal_bundle_relation_verification_yield: {float(metrics.get('multimodal_bundle_relation_verification_yield', 0.0) or 0.0):.3f}",
        f"- multimodal_bundle_compression_contribution: {float(metrics.get('multimodal_bundle_compression_contribution', 0.0) or 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


def main(
    *,
    report_path: str = workspace_path("evaluation", "event_memory_ingest_pipeline.json"),
    summary_path: str = workspace_path("evaluation", "event_memory_ingest_pipeline_summary.txt"),
) -> Dict[str, Any]:
    report = build_report()
    summary = build_summary(report)
    ensure_parent_directory(report_path)
    ensure_parent_directory(summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary)
    return report


if __name__ == "__main__":
    main()
