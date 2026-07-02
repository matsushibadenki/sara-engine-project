from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.ingest import (
    EventMemoryIngestPipeline,
    FrequentSequenceMiner,
    PredictionGainEstimator,
    ProposalVerifier,
    SynchronyDetector,
    TemporalEventizer,
    make_candidate_event,
)
from sara_engine.ingest.episode_segmentation import EpisodeSegmenter
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _fixture_series() -> List[Dict[str, Any]]:
    return [
        {
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
        },
        {
            "source_ref": "synthetic_session_b",
            "source_hash": "synthetic_hash_b",
            "streams": [
                {
                    "stream_id": "audio-1",
                    "modality": "audio",
                    "samples": [
                        {"time_ms": 0, "value": 0.0},
                        {"time_ms": 90, "value": 0.65},
                        {"time_ms": 160, "value": 0.68},
                        {"time_ms": 310, "value": 0.0},
                        {"time_ms": 430, "value": 0.78},
                    ],
                },
                {
                    "stream_id": "text-1",
                    "modality": "text",
                    "samples": [
                        {"time_ms": 0, "value": 0.0},
                        {"time_ms": 118, "value": 0.82},
                        {"time_ms": 248, "value": 0.0},
                        {"time_ms": 455, "value": 0.88},
                    ],
                },
            ],
            "candidate_events": [
                make_candidate_event(
                    {
                        "record_id": "cand-vision-2",
                        "modality": "vision",
                        "label": "visual_cluster_018",
                        "local_time_ms": 128,
                        "confidence": 0.81,
                        "source_ref": "synthetic_session_b",
                        "source_hash": "synthetic_hash_b",
                        "extractor_name": "candidate_proposals",
                        "extractor_version": "v1",
                        "evidence_count": 2,
                        "counterexample_count": 0,
                        "prediction_gain": 0.12,
                    }
                ),
                make_candidate_event(
                    {
                        "record_id": "cand-touch-2",
                        "modality": "touch",
                        "label": "touch_cluster_004",
                        "local_time_ms": 462,
                        "confidence": 0.73,
                        "source_ref": "synthetic_session_b",
                        "source_hash": "synthetic_hash_b",
                        "extractor_name": "candidate_proposals",
                        "extractor_version": "v1",
                        "evidence_count": 1,
                        "counterexample_count": 1,
                        "prediction_gain": 0.08,
                    }
                ),
            ],
        },
        {
            "source_ref": "synthetic_session_c",
            "source_hash": "synthetic_hash_c",
            "streams": [
                {
                    "stream_id": "audio-1",
                    "modality": "audio",
                    "samples": [
                        {"time_ms": 0, "value": 0.0},
                        {"time_ms": 105, "value": 0.76},
                        {"time_ms": 240, "value": 0.0},
                        {"time_ms": 345, "value": 0.79},
                        {"time_ms": 520, "value": 0.0},
                    ],
                },
                {
                    "stream_id": "text-1",
                    "modality": "text",
                    "samples": [
                        {"time_ms": 0, "value": 0.0},
                        {"time_ms": 142, "value": 0.84},
                        {"time_ms": 265, "value": 0.0},
                        {"time_ms": 378, "value": 0.86},
                        {"time_ms": 545, "value": 0.0},
                    ],
                },
            ],
            "candidate_events": [
                make_candidate_event(
                    {
                        "record_id": "cand-vision-3",
                        "modality": "vision",
                        "label": "visual_cluster_021",
                        "local_time_ms": 150,
                        "confidence": 0.77,
                        "source_ref": "synthetic_session_c",
                        "source_hash": "synthetic_hash_c",
                        "extractor_name": "candidate_proposals",
                        "extractor_version": "v1",
                        "evidence_count": 2,
                        "counterexample_count": 0,
                        "prediction_gain": 0.11,
                    }
                ),
            ],
        },
    ]


def _profile_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "profile_id": "tight",
            "merge_window_ms": 40,
            "max_gap_ms": 120,
            "max_events_per_episode": 4,
            "synchrony_window_ms": 40,
            "prediction_max_delay_ms": 80,
            "sequence_max_span_ms": 90,
        },
        {
            "profile_id": "balanced",
            "merge_window_ms": 120,
            "max_gap_ms": 250,
            "max_events_per_episode": 12,
            "synchrony_window_ms": 80,
            "prediction_max_delay_ms": 120,
            "sequence_max_span_ms": 160,
        },
        {
            "profile_id": "wide",
            "merge_window_ms": 220,
            "max_gap_ms": 420,
            "max_events_per_episode": 20,
            "synchrony_window_ms": 140,
            "prediction_max_delay_ms": 220,
            "sequence_max_span_ms": 260,
        },
    ]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    return numerator / ((x_var ** 0.5) * (y_var ** 0.5))


def _build_controller() -> PersistentSelfStateController:
    core_ids = (
        stable_self_state_id("event_memory"),
        stable_self_state_id("maintenance"),
        stable_self_state_id("compression"),
        stable_self_state_id("continuity"),
    )
    return PersistentSelfStateController(core_event_ids=core_ids)


def _run_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    controller = _build_controller()
    pipeline = EventMemoryIngestPipeline(
        temporal_eventizer=TemporalEventizer(
            merge_window_ms=int(profile["merge_window_ms"])
        ),
        episode_segmenter=EpisodeSegmenter(
            max_gap_ms=int(profile["max_gap_ms"]),
            max_events_per_episode=int(profile["max_events_per_episode"]),
        ),
        sequence_miner=FrequentSequenceMiner(
            min_support_episodes=1,
            max_pattern_length=3,
            max_span_ms=int(profile["sequence_max_span_ms"]),
        ),
        synchrony_detector=SynchronyDetector(
            window_ms=int(profile["synchrony_window_ms"]),
            cross_modal_only=True,
        ),
        prediction_gain_estimator=PredictionGainEstimator(
            min_support=1,
            min_gain=0.0,
            max_delay_ms=int(profile["prediction_max_delay_ms"]),
        ),
        verifier=ProposalVerifier(
            min_confidence=0.1,
            min_evidence_count=1,
            min_prediction_gain=0.0,
            max_counterexample_rate=0.9,
        ),
        persistent_self_state=controller,
    )
    fixture_runs: List[Dict[str, Any]] = []
    for fixture in _fixture_series():
        result = pipeline.ingest_streams(
            fixture["streams"],
            source_ref=str(fixture["source_ref"]),
            source_hash=str(fixture["source_hash"]),
            candidate_events=fixture["candidate_events"],
        )
        payload = result.to_dict()
        traces = payload["traces"]
        persistent = traces.get("persistent_self_state", {})
        bundle_trace = (
            traces.get("multimodal_bundle_admission", {})
            if isinstance(traces.get("multimodal_bundle_admission", {}), dict)
            else {}
        )
        observed_event_count = len(payload["observed_events"])
        accepted_candidate_count = len(payload["accepted_candidate_events"])
        episode_count = len(payload["episodes"])
        candidate_relation_count = len(payload["candidate_relations"])
        verified_relation_count = len(payload["verified_relations"])
        lineage_count = len(payload["lineage_ledger"])
        total_candidate_count = accepted_candidate_count + len(
            payload["rejected_candidate_events"]
        )
        bundle_count = int(bundle_trace.get("bundle_count", 0) or 0)
        bundle_promotion_allowed_count = int(bundle_trace.get("promotion_allowed_count", 0) or 0)
        bundle_promotion_rate = float(bundle_promotion_allowed_count) / float(max(bundle_count, 1))
        fixture_runs.append(
            {
                "source_ref": fixture["source_ref"],
                "observed_event_count": observed_event_count,
                "accepted_candidate_event_count": accepted_candidate_count,
                "episode_count": episode_count,
                "candidate_relation_count": candidate_relation_count,
                "verified_relation_count": verified_relation_count,
                "lineage_ledger_count": lineage_count,
                "eventization_emission_ratio": float(observed_event_count)
                / float(max(len(payload["change_points"]), 1)),
                "candidate_event_acceptance_rate": float(accepted_candidate_count)
                / float(max(total_candidate_count, 1)),
                "episode_compression_ratio": float(
                    observed_event_count + accepted_candidate_count
                )
                / float(max(episode_count, 1)),
                "relation_verification_yield": float(verified_relation_count)
                / float(max(candidate_relation_count, 1)),
                "multimodal_bundle_promotion_rate": bundle_promotion_rate,
                "multimodal_bundle_relation_verification_yield": bundle_promotion_rate
                * float(verified_relation_count)
                / float(max(candidate_relation_count, 1)),
                "multimodal_bundle_compression_contribution": bundle_promotion_rate
                * float(observed_event_count + accepted_candidate_count)
                / float(max(episode_count, 1)),
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
                "self_state_continuity": float(
                    persistent.get("continuity_score", 0.0) or 0.0
                ),
                "self_state_active_count": len(
                    persistent.get("current_active_ids", []) or []
                ),
            }
        )
    compression_scores = [
        float(run["episode_compression_ratio"])
        * float(run["relation_verification_yield"])
        * float(run["lineage_coverage_ratio"])
        for run in fixture_runs
    ]
    maintenance_loads = [
        (
            float(run["self_state_active_count"]) + float(run["verified_relation_count"])
        )
        / max(float(run["self_state_continuity"]), 0.05)
        for run in fixture_runs
    ]
    avg_compression_score = mean(compression_scores) if compression_scores else 0.0
    avg_maintenance_load = mean(maintenance_loads) if maintenance_loads else 0.0
    avg_continuity = mean(
        float(run["self_state_continuity"]) for run in fixture_runs
    ) if fixture_runs else 0.0
    profile_report = {
        "profile_id": str(profile["profile_id"]),
        "config": dict(profile),
        "fixture_run_count": len(fixture_runs),
        "metrics": {
            "avg_eventization_emission_ratio": mean(
                float(run["eventization_emission_ratio"]) for run in fixture_runs
            ),
            "avg_candidate_event_acceptance_rate": mean(
                float(run["candidate_event_acceptance_rate"]) for run in fixture_runs
            ),
            "avg_episode_compression_ratio": mean(
                float(run["episode_compression_ratio"]) for run in fixture_runs
            ),
            "avg_relation_verification_yield": mean(
                float(run["relation_verification_yield"]) for run in fixture_runs
            ),
            "avg_multimodal_bundle_promotion_rate": mean(
                float(run["multimodal_bundle_promotion_rate"]) for run in fixture_runs
            ),
            "avg_multimodal_bundle_relation_verification_yield": mean(
                float(run["multimodal_bundle_relation_verification_yield"]) for run in fixture_runs
            ),
            "avg_multimodal_bundle_compression_contribution": mean(
                float(run["multimodal_bundle_compression_contribution"]) for run in fixture_runs
            ),
            "avg_lineage_coverage_ratio": mean(
                float(run["lineage_coverage_ratio"]) for run in fixture_runs
            ),
            "avg_self_state_continuity": avg_continuity,
            "avg_self_state_active_count": mean(
                float(run["self_state_active_count"]) for run in fixture_runs
            ),
            "avg_verified_relation_count": mean(
                float(run["verified_relation_count"]) for run in fixture_runs
            ),
            "compression_quality_score": avg_compression_score,
            "maintenance_load_proxy": avg_maintenance_load,
            "compression_efficiency_per_maintenance": (
                avg_compression_score / max(avg_maintenance_load, 1e-9)
            ),
        },
        "fixture_runs": fixture_runs,
    }
    return profile_report


def build_report() -> Dict[str, Any]:
    profiles = [_run_profile(profile) for profile in _profile_definitions()]
    compression_scores = [
        float(profile["metrics"]["compression_quality_score"]) for profile in profiles
    ]
    maintenance_loads = [
        float(profile["metrics"]["maintenance_load_proxy"]) for profile in profiles
    ]
    correlation = _pearson(compression_scores, maintenance_loads)
    best_profile = max(
        profiles,
        key=lambda item: float(
            item["metrics"]["compression_efficiency_per_maintenance"]
        ),
    ) if profiles else {}
    passed = bool(
        profiles
        and all(
            float(profile["metrics"]["avg_relation_verification_yield"]) >= 0.5
            and float(profile["metrics"]["avg_episode_compression_ratio"]) >= 1.0
            for profile in profiles
        )
    )
    return {
        "schema": "sara-event-memory-maintenance-coupling-benchmark-v1",
        "observed_only": True,
        "passed": passed,
        "profile_count": len(profiles),
        "profiles": profiles,
        "metrics": {
            "compression_to_maintenance_correlation": correlation,
            "best_profile_compression_efficiency_per_maintenance": float(
                best_profile.get("metrics", {}).get(
                    "compression_efficiency_per_maintenance", 0.0
                )
                or 0.0
            ),
            "best_profile_self_state_continuity": float(
                best_profile.get("metrics", {}).get("avg_self_state_continuity", 0.0)
                or 0.0
            ),
            "best_profile_episode_compression_ratio": float(
                best_profile.get("metrics", {}).get(
                    "avg_episode_compression_ratio", 0.0
                )
                or 0.0
            ),
            "best_profile_multimodal_bundle_compression_contribution": float(
                best_profile.get("metrics", {}).get(
                    "avg_multimodal_bundle_compression_contribution", 0.0
                )
                or 0.0
            ),
        },
        "best_profile": {
            "profile_id": str(best_profile.get("profile_id", "") or ""),
            "config": dict(best_profile.get("config", {}))
            if isinstance(best_profile.get("config", {}), dict)
            else {},
        },
    }


def build_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    best = report.get("best_profile", {})
    lines = [
        "SARA Event Memory maintenance coupling benchmark",
        f"- passed: {bool(report.get('passed', False))}",
        f"- observed_only: {bool(report.get('observed_only', False))}",
        f"- profile_count: {int(report.get('profile_count', 0) or 0)}",
        f"- compression_to_maintenance_correlation: {float(metrics.get('compression_to_maintenance_correlation', 0.0) or 0.0):.3f}",
        f"- best_profile: {best.get('profile_id', '')}",
        f"- best_profile_episode_compression_ratio: {float(metrics.get('best_profile_episode_compression_ratio', 0.0) or 0.0):.3f}",
        f"- best_profile_self_state_continuity: {float(metrics.get('best_profile_self_state_continuity', 0.0) or 0.0):.3f}",
        f"- best_profile_compression_efficiency_per_maintenance: {float(metrics.get('best_profile_compression_efficiency_per_maintenance', 0.0) or 0.0):.3f}",
        f"- best_profile_multimodal_bundle_compression_contribution: {float(metrics.get('best_profile_multimodal_bundle_compression_contribution', 0.0) or 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


DEFAULT_REPORT_PATH = workspace_path(
    "evaluation", "event_memory_maintenance_coupling_benchmark.json"
)
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "event_memory_maintenance_coupling_benchmark_summary.txt"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Event Memory compression-maintenance coupling benchmark."
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def run_benchmark(
    *,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(report_path=args.report_path, summary_path=args.summary_path)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
