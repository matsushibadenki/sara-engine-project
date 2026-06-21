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
    )
    payload = result.to_dict()
    return {
        "schema": "sara-event-memory-ingest-pipeline-report-v1",
        "passed": bool(
            payload["observed_events"]
            and payload["episodes"]
            and payload["verified_relations"]
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
        "traces": payload["traces"],
        "result": payload,
    }


def build_summary(report: Dict[str, Any]) -> str:
    counts = report.get("counts", {})
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
        f"- eventization_emitted: {int(traces.get('eventization', {}).get('emitted_count', 0) or 0)}",
        f"- sequence_patterns: {int(traces.get('frequent_sequence', {}).get('accepted_sequences', 0) or 0)}",
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
