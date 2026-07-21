from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.learning import IdleReplayConfig, plan_idle_replay
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _candidate(entry_id: str, **overrides: Any) -> EventStateCandidate:
    values = {
        "entry_id": entry_id,
        "signature": (1, 3, 5),
        "source_ref": f"source:{entry_id}",
        "time_segment": 1,
        "own_latent_id": f"latent:{entry_id}",
        "confidence": 0.9,
        "uncertainty": 0.1,
        "source_reliability": 0.9,
        "resonance_score": 0.9,
        "sequence_support_score": 0.2,
        "sequence_support_count": 1,
        "metabolic_headroom": 0.8,
        "observed": True,
        "source_backed": True,
        "verified": True,
    }
    values.update(overrides)
    return EventStateCandidate.from_verified_evidence(
        verifier_id="idle-replay-benchmark",
        evidence={"entry_id": entry_id, "signature": list(values["signature"])},
        **values,
    )


def _build_cache() -> VerifiedHierarchicalEventStateCache:
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "plain",
            signature=(11, 13, 17),
            own_latent_id="latent:plain",
            source_ref="source:plain",
            sequence_support_score=0.2,
        )
    )
    cache.admit(
        _candidate(
            "aligned",
            signature=(21, 23, 27),
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            source_ref="source:aligned",
            sequence_support_score=0.45,
            sequence_support_count=3,
        )
    )
    cache.admit(
        _candidate(
            "anchor",
            signature=tuple(range(10)),
            source_ref="source:anchor",
            own_latent_id="latent:anchor",
            sequence_support_score=0.5,
            sequence_support_count=2,
        )
    )
    return cache


def build_report() -> Dict[str, Any]:
    cache = _build_cache()
    controller = PersistentSelfStateController(core_event_ids=(101, 202))
    controller.step(external_event_ids=(stable_self_state_id("vision:visual_cluster_018"),))
    controller.step(external_event_ids=(stable_self_state_id("audio:audio_cluster_044"),))

    aligned = plan_idle_replay(
        cache,
        persistent_self_state=controller,
    )
    budgeted = plan_idle_replay(
        cache,
        persistent_self_state=controller,
        reactivation_hints=(
            {
                "entry_id": "anchor",
                "activation": 0.95,
                "mutates_durable_state": False,
            },
        ),
        config=IdleReplayConfig(max_candidates=2, event_budget=4, min_replay_score=0.3),
    )
    stressed_modulator = AstroReplayModulator()
    stressed_modulator.update(interference_ratio=0.9, replay_recovery_signal=0.0)
    stressed = plan_idle_replay(cache, astro_modulator=stressed_modulator)
    calm = plan_idle_replay(cache, astro_modulator=AstroReplayModulator())

    aligned_entry = aligned.get("selected", [{}])[0] if aligned.get("selected") else {}
    stressed_score = 0.0
    calm_score = 0.0
    if stressed.get("candidates"):
        stressed_score = float(stressed["candidates"][0].get("replay_score", 0.0) or 0.0)
    if calm.get("candidates"):
        calm_score = float(calm["candidates"][0].get("replay_score", 0.0) or 0.0)

    metrics = {
        "idle_replay_candidate_selection_observed": float(
            aligned.get("metrics", {}).get("idle_replay_candidate_selection_observed", 0.0) or 0.0
        ),
        "idle_replay_budget_observed": float(
            budgeted.get("metrics", {}).get("idle_replay_budget_observed", 0.0) or 0.0
        ),
        "idle_replay_self_state_alignment_observed": 1.0
        if str(aligned_entry.get("entry_id", "")) == "aligned"
        else 0.0,
        "idle_replay_memory_reactivation_observed": float(
            budgeted.get("metrics", {}).get("idle_replay_memory_reactivation_observed", 0.0) or 0.0
        ),
        "idle_replay_state_continuity_observed": float(
            aligned.get("metrics", {}).get("idle_replay_state_continuity_observed", 0.0) or 0.0
        ),
        "idle_replay_astro_modulation_observed": 1.0 if stressed_score < calm_score else 0.0,
    }
    return {
        "schema": "sara-idle-replay-benchmark-v1",
        "observed_only": True,
        "passed": all(value >= 1.0 for value in metrics.values()),
        "metrics": metrics,
        "traces": {
            "aligned": aligned,
            "budgeted": budgeted,
            "stressed": stressed,
            "calm": calm,
            "stressed_top_score": stressed_score,
            "calm_top_score": calm_score,
        },
    }


def build_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "SARA idle replay benchmark",
        f"- passed: {bool(report.get('passed', False))}",
        f"- candidate_selection: {float(metrics.get('idle_replay_candidate_selection_observed', 0.0) or 0.0):.3f}",
        f"- budget: {float(metrics.get('idle_replay_budget_observed', 0.0) or 0.0):.3f}",
        f"- self_state_alignment: {float(metrics.get('idle_replay_self_state_alignment_observed', 0.0) or 0.0):.3f}",
        f"- memory_reactivation: {float(metrics.get('idle_replay_memory_reactivation_observed', 0.0) or 0.0):.3f}",
        f"- state_continuity: {float(metrics.get('idle_replay_state_continuity_observed', 0.0) or 0.0):.3f}",
        f"- astro_modulation: {float(metrics.get('idle_replay_astro_modulation_observed', 0.0) or 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


DEFAULT_REPORT_PATH = workspace_path("evaluation", "idle_replay_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "idle_replay_benchmark_summary.txt")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the idle replay benchmark.")
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
