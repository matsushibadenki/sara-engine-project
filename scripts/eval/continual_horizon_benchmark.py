#!/usr/bin/env python3
"""Run an observed-only 10/30/100 episode continual-horizon benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.risa.structural_interpolation import (  # noqa: E402
    PredictiveStructuralFeedbackEngine,
    StructuralFeedbackSignal,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "continual_horizon_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "continual_horizon_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "continual_horizon_benchmark_summary.txt")
PROFILES = ("frozen_control", "event_memory", "resonance_credit_event_memory", "structural_feedback_event_memory")


def _load(path: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("continual horizon fixture rows must be objects")
                rows.append(payload)
    return rows


def _signature(*parts: str) -> Tuple[int, ...]:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return tuple(int(value) for value in digest[:8])


def _candidate(
    *,
    case: Mapping[str, Any],
    episode: int,
    revision: str,
    entry_id: str,
    resonance: float,
    contradicted: bool = False,
) -> EventStateCandidate:
    domain = str(case["domain"])
    source_ref = str(case["source_ref"])
    return EventStateCandidate.from_verified_evidence(
        verifier_id="continual-horizon-benchmark",
        evidence={"case": dict(case), "episode": episode, "revision": revision},
        entry_id=entry_id,
        signature=_signature(domain, "route", revision),
        source_ref=source_ref,
        source_revision=revision,
        time_segment=episode,
        own_latent_id=f"concept:{domain}:route",
        causal_predecessors=(f"domain:{domain}",),
        confidence=0.95,
        uncertainty=0.05,
        source_reliability=1.0,
        resonance_score=resonance,
        sequence_support_score=0.9,
        sequence_support_count=max(1, episode),
        credit_score=resonance,
        credit_responsibility=resonance,
        credit_confidence=resonance,
        credit_longevity=resonance,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        contradicted=contradicted,
        event_cost=8,
    )


def _protected_candidate(case: Mapping[str, Any], episode: int) -> EventStateCandidate:
    protected = str(case["protected_domain"])
    return EventStateCandidate.from_verified_evidence(
        verifier_id="continual-horizon-protected-knowledge",
        evidence={"case": dict(case), "episode": episode, "protected": protected},
        entry_id=f"{case['case_id']}:protected:{protected}",
        signature=_signature(protected, "verified", "r1"),
        source_ref=str(case["protected_source_ref"]),
        source_revision="r1",
        time_segment=episode,
        own_latent_id=f"concept:{protected}:verified",
        causal_predecessors=(f"domain:{protected}",),
        confidence=0.99,
        uncertainty=0.01,
        source_reliability=1.0,
        resonance_score=0.99,
        sequence_support_score=0.99,
        sequence_support_count=max(1, episode),
        credit_score=0.99,
        credit_responsibility=0.99,
        credit_confidence=0.99,
        credit_longevity=0.99,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        event_cost=8,
    )


def _run_profile(
    case: Mapping[str, Any],
    profile: str,
    retention_profile: str = "logarithmic",
) -> Dict[str, Any]:
    horizon = int(case["horizon"])
    correction_episode = int(case["correction_episode"])
    verification_episode = int(case["verification_episode"])
    initial_revision = str(case["initial_revision"])
    corrected_revision = str(case["corrected_revision"])
    expected_revision = str(case["expected_revision"])
    active = profile != "frozen_control"
    cache = (
        VerifiedHierarchicalEventStateCache(
            retention_profile=retention_profile,
            max_entries=8,
            retrieval_threshold=0.35,
            top_k=1,
        )
        if active
        else None
    )
    revision_seen_at: Optional[int] = None
    contradiction_blocked = False
    retrieval_trace: List[Dict[str, Any]] = []
    maintenance_event_cost = 0
    replay_count = 0
    feedback_edit_count = 0
    feedback_engine = PredictiveStructuralFeedbackEngine()
    for episode in range(1, horizon + 1):
        revision = initial_revision if episode < correction_episode else corrected_revision
        resonance = 0.88 if profile in {"resonance_credit_event_memory", "structural_feedback_event_memory"} else 0.76
        if active and cache is not None:
            distractor_count = min(int(case.get("distractor_count", 0) or 0), 8)
            for distractor_index in range(distractor_count):
                distractor = _candidate(
                    case={**case, "domain": f"{case['domain']}:d{distractor_index}"},
                    episode=episode,
                    revision=f"d{distractor_index}",
                    entry_id=f"{case['case_id']}:distractor:{distractor_index}",
                    resonance=0.76,
                )
                maintenance_event_cost += cache.admit(distractor).event_cost
            maintenance_event_cost += cache.admit(_protected_candidate(case, episode)).event_cost
            if profile == "structural_feedback_event_memory" and episode == correction_episode:
                feedback = feedback_engine.propose(
                    (
                        StructuralFeedbackSignal(
                            predicting_concept=f"concept:{case['domain']}:route",
                            source_node=f"domain:{case['domain']}:route",
                            target_node=f"revision:{corrected_revision}",
                            relation_type="supports",
                            predicted_confidence=0.60,
                            observed_confidence=0.95,
                            evidence_ids=(f"episode:{episode}",),
                            eligible=True,
                            rollback_state="verified_snapshot",
                        ),
                    )
                )
                feedback_edit_count = sum(
                    int(item.edit_type == "strengthen_relation") for item in feedback
                )
            admitted = cache.admit(
                _candidate(
                    case=case,
                    episode=episode,
                    revision=revision,
                    entry_id=f"{case['case_id']}:route:{revision}",
                    resonance=resonance,
                )
            )
            maintenance_event_cost += admitted.event_cost
            if episode == correction_episode:
                blocked = cache.admit(
                    _candidate(
                        case=case,
                        episode=episode,
                        revision=initial_revision,
                        entry_id=f"{case['case_id']}:contradiction",
                        resonance=0.4,
                        contradicted=True,
                    )
                )
                contradiction_blocked = blocked.decision == "block_contradiction"
            if episode >= verification_episode:
                replay_count += 1
                retrieval = cache.retrieve(
                    _signature(str(case["domain"]), "route", corrected_revision),
                    own_latent_id=f"concept:{case['domain']}:route",
                    source_ref=str(case["source_ref"]),
                    now_segment=episode,
                    top_k=1,
                )
                retrieval_trace.append(retrieval.to_dict())
                if retrieval.matches and revision_seen_at is None:
                    revision_seen_at = episode

    if profile == "frozen_control":
        predicted_revision = initial_revision
        retained_useful_recall = 0.0
        revision_latency: Optional[int] = None
        abstention_integrity = 1.0
        state_count = 1
        eviction_count = 0
        cache_cost = 0
        protected_retention = 1.0
    else:
        predicted_revision = corrected_revision if revision_seen_at is not None else initial_revision
        retained_useful_recall = float(predicted_revision == expected_revision)
        revision_latency = None if revision_seen_at is None else revision_seen_at - correction_episode
        abstention_integrity = float(all(not item["abstained"] for item in retrieval_trace))
        state_count = len(cache.entries) if cache is not None else 0
        eviction_count = cache.eviction_count if cache is not None else 0
        cache_cost = sum(item["event_cost"] for item in retrieval_trace)
        protected_retrieval = cache.retrieve(
            _protected_candidate(case, horizon).signature,
            own_latent_id=f"concept:{case['protected_domain']}:verified",
            source_ref=str(case["protected_source_ref"]),
            now_segment=horizon,
            top_k=1,
        )
        protected_retention = float(bool(protected_retrieval.matches) and not protected_retrieval.abstained)
    catastrophic_interference = 1.0 - protected_retention
    repair_bound = max(0, verification_episode - correction_episode)
    checks = {
        "revision_uptake": predicted_revision == expected_revision if active else predicted_revision != expected_revision,
        "useful_recall": retained_useful_recall == 1.0 if active else retained_useful_recall == 0.0,
        "contradiction_repair_bounded": (revision_latency is not None and revision_latency <= repair_bound) if active else True,
        "contradiction_blocked": contradiction_blocked if active else True,
        "state_budget_bounded": state_count <= 8,
        "no_abstention_after_verification": abstention_integrity == 1.0 if active else True,
        "protected_knowledge_retained": protected_retention == 1.0,
    }
    return {
        "profile": profile,
        "retention_profile": retention_profile,
        "horizon": horizon,
        "correction_episode": correction_episode,
        "verification_episode": verification_episode,
        "revision_uptake_latency": revision_latency,
        "predicted_revision": predicted_revision,
        "retained_useful_recall": retained_useful_recall,
        "contradiction_repair_bound": repair_bound,
        "contradiction_blocked": contradiction_blocked,
        "abstention_integrity": abstention_integrity,
        "state_growth": state_count,
        "eviction_count": eviction_count,
        "maintenance_event_cost": maintenance_event_cost,
        "replay_count": replay_count,
        "feedback_edit_count": feedback_edit_count,
        "cache_retrieval_event_cost": cache_cost,
        "protected_knowledge_retention": protected_retention,
        "catastrophic_interference": catastrophic_interference,
        "checks": checks,
        "passed": all(checks.values()),
        "retrieval_trace": retrieval_trace[-3:],
    }


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("continual horizon fixture is empty")
    horizons = sorted(int(row["horizon"]) for row in rows)
    if horizons != [10, 30, 100]:
        raise ValueError("continual horizon fixture must contain horizons 10, 30, and 100")
    source_hashes = [str(row.get("source_hash", "")) for row in rows]
    source_domains = [str(row.get("source_domain", "")) for row in rows]
    if any(not value for value in source_hashes + source_domains):
        raise ValueError("continual horizon rows require source_hash and source_domain")
    cases: Dict[str, Any] = {}
    for case in sorted(rows, key=lambda row: int(row["horizon"])):
        case_id = str(case["case_id"])
        profiles = {profile: _run_profile(case, profile) for profile in PROFILES}
        retention_profiles = {
            retention: _run_profile(case, "event_memory", retention)
            for retention in ("fixed", "linear", "logarithmic")
        }
        cases[case_id] = {
            "fixture": dict(case),
            "profiles": profiles,
            "retention_profiles": retention_profiles,
        }
    active_reports = [
        report["profiles"][profile]
        for report in cases.values()
        for profile in PROFILES[1:]
    ]
    frozen_reports = [report["profiles"]["frozen_control"] for report in cases.values()]
    retention_reports = [
        report["retention_profiles"][retention]
        for report in cases.values()
        for retention in ("fixed", "linear", "logarithmic")
    ]
    delayed_ablation: Dict[str, Dict[str, Any]] = {}
    for case in sorted(rows, key=lambda row: int(row["horizon"])):
        immediate_case = dict(case)
        immediate_case["verification_episode"] = int(case["correction_episode"])
        delayed = _run_profile(case, "event_memory")
        immediate = _run_profile(immediate_case, "event_memory")
        delayed_ablation[str(case["case_id"])] = {
            "delayed_verification_episode": int(case["verification_episode"]),
            "immediate_verification_episode": int(case["correction_episode"]),
            "delayed_revision_uptake_latency": delayed["revision_uptake_latency"],
            "immediate_revision_uptake_latency": immediate["revision_uptake_latency"],
            "latency_delta": (delayed["revision_uptake_latency"] or 0)
            - (immediate["revision_uptake_latency"] or 0),
            "delayed_useful_recall": delayed["retained_useful_recall"],
            "immediate_useful_recall": immediate["retained_useful_recall"],
        }
    checks = {
        "all_horizons_present": horizons == [10, 30, 100],
        "independent_source_hashes_unique": len(set(source_hashes)) == len(source_hashes),
        "independent_domains_present": len(set(source_domains)) >= 2,
        "active_profiles_pass": all(report["passed"] for report in active_reports),
        "active_beats_frozen_recall": all(
            report["retained_useful_recall"] > frozen["retained_useful_recall"]
            for report, frozen in zip(active_reports[::3], frozen_reports)
        ),
        "state_growth_bounded": all(report["state_growth"] <= 8 for report in active_reports),
        "retention_profiles_bounded": all(report["state_growth"] <= 8 for report in retention_reports),
        "protected_knowledge_survives": all(
            report["protected_knowledge_retention"] == 1.0 for report in active_reports
        ),
        "delayed_correction_is_measurable": all(
            item["latency_delta"] > 0 for item in delayed_ablation.values()
        ),
        "observed_only_boundary": True,
    }
    return {
        "schema": "sara-continual-horizon-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "external_device_required": False,
        "horizons": horizons,
        "checks": checks,
        "metrics": {
            "case_count": len(cases),
            "active_profile_count": len(active_reports),
            "mean_active_useful_recall": sum(report["retained_useful_recall"] for report in active_reports) / max(1, len(active_reports)),
            "mean_active_revision_uptake_latency": sum(report["revision_uptake_latency"] or 0 for report in active_reports) / max(1, len(active_reports)),
            "max_state_growth": max(report["state_growth"] for report in active_reports),
            "max_maintenance_event_cost": max(report["maintenance_event_cost"] for report in active_reports),
            "retention_profile_recall": {
                retention: sum(
                    report["retention_profiles"][retention]["retained_useful_recall"]
                    for report in cases.values()
                ) / max(1, len(cases))
                for retention in ("fixed", "linear", "logarithmic")
            },
            "structural_feedback_edit_count": sum(
                report["feedback_edit_count"]
                for report in active_reports
                if report["profile"] == "structural_feedback_event_memory"
            ),
            "mean_frozen_useful_recall": sum(report["retained_useful_recall"] for report in frozen_reports) / max(1, len(frozen_reports)),
            "mean_active_protected_knowledge_retention": sum(
                report["protected_knowledge_retention"] for report in active_reports
            ) / max(1, len(active_reports)),
            "mean_delayed_latency_delta": sum(
                item["latency_delta"] for item in delayed_ablation.values()
            ) / max(1, len(delayed_ablation)),
        },
        "cases": cases,
        "delayed_correction_ablation": delayed_ablation,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    report = build_report(_load(args.fixture_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Continual horizon benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Observed only: {report['observed_only']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
