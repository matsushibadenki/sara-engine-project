from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.learning.idle_replay import IdleReplayConfig
from sara_engine.learning.sleep_consolidation import SleepConsolidationConfig
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.idle_consolidation_loop import IdleConsolidationLoop
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _build_fixture() -> tuple[
    VerifiedHierarchicalEventStateCache,
    tuple[ConceptRevalidationEntry, ...],
    PersistentSelfStateController,
]:
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=8)
    queue_entries = []
    core_ids = []
    for index, (query, answer, doc_id) in enumerate(
        (
            ("cat food", "fish", "doc-cat"),
            ("dog sound", "bark", "doc-dog"),
            ("bird move", "fly", "doc-bird"),
        ),
        start=1,
    ):
        source_ref = f"fixture:{doc_id}"
        concept_key = f"predicts:text:{query}->doc:{doc_id}"
        cache.admit(
            EventStateCandidate.from_verified_evidence(
                verifier_id="internal-maintenance-efficiency-benchmark",
                evidence={"query": query, "answer": answer, "doc_id": doc_id},
                entry_id=f"maintenance-{index}",
                signature=(
                    stable_self_state_id(query, modulus=1024),
                    stable_self_state_id(answer, modulus=1024),
                    stable_self_state_id(doc_id, modulus=1024),
                ),
                source_ref=source_ref,
                time_segment=index,
                own_latent_id=concept_key,
                confidence=0.93,
                uncertainty=0.07,
                source_reliability=0.92,
                resonance_score=0.91,
                sequence_support_score=0.5,
                sequence_support_count=2,
                metabolic_headroom=0.85,
                observed=True,
                source_backed=True,
                verified=True,
                event_cost=3,
            )
        )
        queue_entries.append(
            ConceptRevalidationEntry(
                concept_key=concept_key,
                decision="quarantine_source_revision_conflict",
                supporting_relation_ids=(concept_key,),
                source_refs=(source_ref,),
                source_hashes=(f"hash-{index}",),
                revision_conflict_count=1,
                contradiction_score=0.1,
                next_action="wait",
                attempt_count=0,
                blocked_at_segment=index,
                last_review_segment=index,
                retry_after_segment=index + 1,
            )
        )
        core_ids.extend(
            (
                stable_self_state_id(f"text:{query}"),
                stable_self_state_id(f"doc:{doc_id}"),
            )
        )
    controller = PersistentSelfStateController(core_event_ids=tuple(core_ids[:6]))
    return cache, tuple(queue_entries), controller


def build_report() -> Dict[str, Any]:
    cache, queue_entries, controller = _build_fixture()
    loop = IdleConsolidationLoop()
    phase_counts = 0
    selected_counts = 0
    refresh_counts = 0
    maintenance_event_cost = 0.0
    idle_self_state_ok_count = 0
    spontaneous_count = 0
    predicted_count = 0
    prioritized_concept_hits = 0
    traces = []

    for segment in range(1, 4):
        result = loop.run(
            cache,
            queue_entries,
            [],
            current_segment=segment,
            persistent_self_state=controller,
            replay_config=IdleReplayConfig(max_candidates=2, event_budget=12, min_replay_score=0.2),
            sleep_config=SleepConsolidationConfig(event_budget=12.0),
        )
        payload = result.to_dict()
        idle_replay = payload["idle_replay_report"]
        self_state_trace = idle_replay.get("self_state_trace", {})
        selected = idle_replay.get("selected", [])
        phase_tracks = payload["memory_phase_report"].get("phase_tracks", [])
        cache_refresh = payload.get("cache_refresh", [])
        sleep_traces = payload["sleep_consolidation_report"].get("traces", [])

        selected_counts += len(selected)
        phase_counts += len(phase_tracks)
        refresh_counts += len(cache_refresh)
        maintenance_event_cost += sum(float(item.get("event_cost", 0.0) or 0.0) for item in sleep_traces)
        idle_self_state_ok_count += 1 if bool(self_state_trace.get("idle_self_state_ok", False)) else 0
        spontaneous_count += len(self_state_trace.get("spontaneous_event_ids", []))
        predicted_count += len(self_state_trace.get("predicted_event_ids", []))
        prioritized_concept_hits += len(payload.get("prioritized_concept_keys", []))
        traces.append(payload)

    selected_denominator = max(1, selected_counts)
    refresh_denominator = max(1, refresh_counts)
    metrics = {
        "maintenance_replay_selection_observed": 1.0 if selected_counts >= 1 else 0.0,
        "maintenance_self_state_continuity_observed": 1.0 if idle_self_state_ok_count >= 1 else 0.0,
        "maintenance_prediction_support_observed": 1.0 if predicted_count >= 1 else 0.0,
        "maintenance_cache_refresh_observed": 1.0 if refresh_counts >= 1 else 0.0,
        "maintenance_concept_priority_observed": 1.0 if prioritized_concept_hits >= 1 else 0.0,
        "maintenance_event_cost_efficiency_observed": 1.0
        if (maintenance_event_cost / float(selected_denominator)) <= 6.0
        else 0.0,
    }
    return {
        "schema": "sara-internal-maintenance-efficiency-benchmark-v1",
        "observed_only": True,
        "passed": all(value >= 1.0 for value in metrics.values()),
        "metrics": metrics,
        "counts": {
            "maintenance_selected_count": int(selected_counts),
            "maintenance_phase_count": int(phase_counts),
            "maintenance_refresh_count": int(refresh_counts),
            "maintenance_idle_self_state_ok_count": int(idle_self_state_ok_count),
            "maintenance_spontaneous_event_count": int(spontaneous_count),
            "maintenance_predicted_event_count": int(predicted_count),
            "prioritized_concept_key_count": int(prioritized_concept_hits),
        },
        "normalized_metrics": {
            "maintenance_event_cost": float(maintenance_event_cost),
            "maintenance_event_cost_per_selected": float(maintenance_event_cost) / float(selected_denominator),
            "maintenance_event_cost_per_refresh": float(maintenance_event_cost) / float(refresh_denominator),
        },
        "traces": {
            "segments": traces,
        },
    }


def build_summary(report: Dict[str, Any]) -> str:
    counts = report.get("counts", {})
    normalized = report.get("normalized_metrics", {})
    metrics = report.get("metrics", {})
    lines = [
        "SARA internal maintenance efficiency benchmark",
        f"- passed: {bool(report.get('passed', False))}",
        f"- maintenance_selected_count: {int(counts.get('maintenance_selected_count', 0) or 0)}",
        f"- maintenance_phase_count: {int(counts.get('maintenance_phase_count', 0) or 0)}",
        f"- maintenance_refresh_count: {int(counts.get('maintenance_refresh_count', 0) or 0)}",
        f"- maintenance_idle_self_state_ok_count: {int(counts.get('maintenance_idle_self_state_ok_count', 0) or 0)}",
        f"- maintenance_spontaneous_event_count: {int(counts.get('maintenance_spontaneous_event_count', 0) or 0)}",
        f"- maintenance_predicted_event_count: {int(counts.get('maintenance_predicted_event_count', 0) or 0)}",
        f"- maintenance_event_cost: {float(normalized.get('maintenance_event_cost', 0.0) or 0.0):.3f}",
        f"- maintenance_event_cost_per_selected: {float(normalized.get('maintenance_event_cost_per_selected', 0.0) or 0.0):.3f}",
        f"- maintenance_event_cost_per_refresh: {float(normalized.get('maintenance_event_cost_per_refresh', 0.0) or 0.0):.3f}",
        f"- maintenance_self_state_continuity_observed: {float(metrics.get('maintenance_self_state_continuity_observed', 0.0) or 0.0):.3f}",
        f"- maintenance_event_cost_efficiency_observed: {float(metrics.get('maintenance_event_cost_efficiency_observed', 0.0) or 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


DEFAULT_REPORT_PATH = workspace_path("evaluation", "internal_maintenance_efficiency_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation",
    "internal_maintenance_efficiency_benchmark_summary.txt",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the internal maintenance efficiency benchmark.")
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
