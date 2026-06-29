#!/usr/bin/env python3
"""Evaluate source-aware event-state caching with live managed evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.metabolic_budget import (  # noqa: E402
    MetabolicBudgetConfig,
    evaluate_structural_metabolic_budget,
)
from sara_engine.memory.event_state_cache import (  # noqa: E402
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.concept_admission import (  # noqa: E402
    ConceptRevalidationEntry,
)
from sara_engine.memory.concept_revalidation_fixture import (  # noqa: E402
    build_concept_revalidation_cases,
)
from sara_engine.memory.concept_queue_store import (  # noqa: E402
    load_revalidation_queue,
    run_persisted_concept_review_cycle,
    save_revalidation_queue,
)
from sara_engine.memory.event_state_evidence import (  # noqa: E402
    build_event_state_candidate,
)
from sara_engine.ingest import make_candidate_relation  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MANIFEST_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path(
    "evaluation",
    "event_state_cache_integration_benchmark.json",
)
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation",
    "event_state_cache_integration_benchmark_summary.txt",
)
DEFAULT_TRACE_PATH = workspace_path(
    "evaluation",
    "event_state_cache_integration_traces.jsonl",
)
DEFAULT_ROUND_TRIP_STATE_PATH = workspace_path(
    "evaluation",
    "event_state_cache_round_trip_state.json",
)
DEFAULT_CONCEPT_QUEUE_PATH = workspace_path(
    "evaluation",
    "event_state_cache_concept_revalidation_queue.json",
)
DEFAULT_CONCEPT_REVIEW_REPORT_PATH = workspace_path(
    "evaluation",
    "event_state_cache_concept_review_report.json",
)
DEFAULT_CONCEPT_FIXTURE_PATH = processed_data_path(
    "benchmark_fixtures",
    "concept_revalidation_cases.jsonl",
)
DEFAULT_SOURCE_PATHS = {
    "reasoning_prior": workspace_path(
        "evaluation",
        "sparse_reasoning_prior_benchmark.json",
    ),
    "plan_verifier": workspace_path(
        "evaluation",
        "sparse_plan_trace_verifier.json",
    ),
    "multimodal_binding": workspace_path(
        "evaluation",
        "synesthetic_multimodal_binding_benchmark.json",
    ),
    "dendritic_feedback": workspace_path(
        "evaluation",
        "dendritic_feedback_gate_benchmark.json",
    ),
    "own_latent": workspace_path(
        "evaluation",
        "own_latent_learning_benchmark.json",
    ),
}


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def build_metabolic_report() -> Dict[str, Any]:
    report = evaluate_structural_metabolic_budget(
        [
            {
                "kind": "grow",
                "synapse_delta": 1,
                "event_cost": 0.5,
                "reserve_cost": 0.1,
                "importance": 0.9,
            }
        ],
        MetabolicBudgetConfig(
            max_synapses=4,
            event_budget=4.0,
            plasticity_reserve=1.0,
        ),
    )
    report["schema"] = "sara-structural-metabolic-budget-v1"
    return report


def load_source_reports(
    source_paths: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    reports = {
        name: load_json(path)
        for name, path in source_paths.items()
    }
    reports["metabolic_budget"] = build_metabolic_report()
    return reports


def _eligible_materials(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if row.get("schema") == "sara-own-latent-manifest-row-v1"
            and row.get("material_type") != "negative_query"
            and row.get("source_ref")
            and row.get("material_hash")
            and isinstance(row.get("sparse_signature"), list)
        ],
        key=lambda row: (
            -float(row.get("quality_score", 0.0) or 0.0),
            str(row.get("manifest_id", "")),
        ),
    )


def _run_profile(
    profile: str,
    materials: Sequence[Dict[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    cache = VerifiedHierarchicalEventStateCache(
        retention_profile=profile,
        max_entries=12,
        top_k=1,
    )
    traces: List[Dict[str, Any]] = []
    targets = list(materials[:2])
    ordered = targets + list(materials[2:])
    for index, material in enumerate(ordered):
        evidence = build_event_state_candidate(
            material,
            reports,
            time_segment=index,
        )
        admission = cache.admit(evidence.candidate)
        traces.append(
            {
                "profile": profile,
                "kind": "admission",
                "material_id": material.get("manifest_id"),
                "evidence": evidence.to_dict(),
                "admission": admission.to_dict(),
            }
        )

    recall_success = 0
    max_event_cost = 0
    hint_integrity = 0
    for target in targets:
        result = cache.retrieve(
            target.get("sparse_signature", []),
            own_latent_id=str(target.get("latent_cluster_id", "")),
            source_ref=str(target.get("source_ref", "")),
            now_segment=len(ordered) + 20,
        )
        expected_id = str(target.get("manifest_id", ""))
        matched = bool(result.matches) and result.matches[0]["entry_id"] == expected_id
        recall_success += int(matched)
        hint_integrity += int(
            bool(result.reactivation_hints)
            and result.reactivation_hints[0]["mutates_durable_state"] is False
        )
        max_event_cost = max(max_event_cost, result.event_cost)
        traces.append(
            {
                "profile": profile,
                "kind": "delayed_retrieval",
                "material_id": expected_id,
                "matched": matched,
                "retrieval": result.to_dict(),
            }
        )
    return {
        "profile": profile,
        "target_count": len(targets),
        "recall_success": recall_success,
        "reactivation_hint_integrity": float(hint_integrity)
        / float(max(1, len(targets))),
        "max_retrieval_event_cost": max_event_cost,
        "state": cache.state_dict(),
        "traces": traces,
    }


def _build_concept_revalidation_fixture(
    materials: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    distinct = []
    seen_refs = set()
    seen_hashes = set()
    for row in materials:
        source_ref = str(row.get("source_ref", ""))
        material_hash = str(row.get("material_hash", ""))
        if not source_ref or not material_hash:
            continue
        if source_ref in seen_refs or material_hash in seen_hashes:
            continue
        distinct.append(row)
        seen_refs.add(source_ref)
        seen_hashes.add(material_hash)
    if len(distinct) < 2:
        return {
            "queue_entries": (),
            "relations": (),
            "concept_keys": (),
        }
    queue_entries = []
    relations = []
    concept_keys = []
    pair_count = min(3, len(distinct) // 2)
    for index in range(pair_count):
        first = distinct[index * 2]
        second = distinct[(index * 2) + 1]
        visual_id = f"vision:visual_cluster_{18 + index:03d}"
        audio_id = f"audio:audio_cluster_{44 + index:03d}"
        concept_key = f"predicts:{visual_id}->{audio_id}"
        concept_keys.append(concept_key)
        queue_entries.append(
            ConceptRevalidationEntry(
                concept_key=concept_key,
                decision="quarantine_source_revision_conflict",
                supporting_relation_ids=(concept_key,),
                source_refs=(str(first.get("source_ref", "")),),
                source_hashes=(str(first.get("material_hash", "")),),
                revision_conflict_count=1,
                contradiction_score=0.2,
                next_action="wait_for_source_revision_resolution",
                attempt_count=0,
                blocked_at_segment=1,
                last_review_segment=1,
                retry_after_segment=2,
            )
        )
        relations.extend(
            [
                make_candidate_relation(
                    {
                        "record_id": f"concept-rel-{index}-a",
                        "relation": "predicts",
                        "source_event_id": visual_id,
                        "target_event_id": audio_id,
                        "delay_lower_ms": 60,
                        "delay_upper_ms": 140,
                        "confidence": 0.88,
                        "source_ref": str(first.get("source_ref", "")),
                        "source_hash": str(first.get("material_hash", "")),
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                        "evidence_count": 5,
                        "counterexample_count": 0,
                        "prediction_gain": 0.18,
                    }
                ),
                make_candidate_relation(
                    {
                        "record_id": f"concept-rel-{index}-b",
                        "relation": "predicts",
                        "source_event_id": visual_id,
                        "target_event_id": audio_id,
                        "delay_lower_ms": 60,
                        "delay_upper_ms": 140,
                        "confidence": 0.88,
                        "source_ref": str(second.get("source_ref", "")),
                        "source_hash": str(second.get("material_hash", "")),
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                        "evidence_count": 5,
                        "counterexample_count": 0,
                        "prediction_gain": 0.18,
                    }
                ),
            ]
        )
    return {
        "queue_entries": tuple(queue_entries),
        "relations": tuple(relations),
        "concept_keys": tuple(concept_keys),
    }


def _load_or_build_concept_revalidation_fixture(
    materials: Sequence[Dict[str, Any]],
    *,
    fixture_path: str,
) -> Dict[str, Any]:
    rows = read_jsonl(fixture_path)
    if rows:
        for row in rows:
            pass
        cases = []
        for row in rows:
            case = _parse_concept_fixture_case(row)
            if case is not None:
                cases.append(case)
        if cases:
            return {
                "cases": tuple(cases),
                "queue_entries": tuple(item["queue_entry"] for item in cases),
                "relations": tuple(
                    relation
                    for item in cases
                    for relation in item["relations"]
                ),
                "concept_keys": tuple(item["concept_key"] for item in cases),
                "fixture_mode": "external_fixture",
                "fixture_case_count": len(cases),
            }
    built_rows = build_concept_revalidation_cases(materials)
    if built_rows:
        temp_path = ensure_parent_directory(fixture_path)
        with open(temp_path, "w", encoding="utf-8") as handle:
            for row in built_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    built = _build_concept_revalidation_fixture(materials)
    built_rows = build_concept_revalidation_cases(materials)
    built["cases"] = tuple(
        case
        for row in built_rows
        for case in ([_parse_concept_fixture_case(row)] if _parse_concept_fixture_case(row) is not None else [])
    )
    built["fixture_mode"] = "synthetic_fallback"
    built["fixture_case_count"] = len(built.get("queue_entries", ()))
    return built


def _parse_concept_fixture_case(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    queue_payload = row.get("queue_entry", {})
    if not isinstance(queue_payload, Mapping):
        return None
    concept_key = str(row.get("concept_key", "") or queue_payload.get("concept_key", "") or "")
    if not concept_key:
        return None
    queue_entry = ConceptRevalidationEntry(
        concept_key=concept_key,
        decision=str(queue_payload.get("decision", "")),
        supporting_relation_ids=tuple(
            str(item) for item in queue_payload.get("supporting_relation_ids", ()) if str(item)
        ),
        source_refs=tuple(
            str(item) for item in queue_payload.get("source_refs", ()) if str(item)
        ),
        source_hashes=tuple(
            str(item) for item in queue_payload.get("source_hashes", ()) if str(item)
        ),
        revision_conflict_count=int(queue_payload.get("revision_conflict_count", 0) or 0),
        contradiction_score=float(queue_payload.get("contradiction_score", 0.0) or 0.0),
        next_action=str(queue_payload.get("next_action", "")),
        attempt_count=int(queue_payload.get("attempt_count", 0) or 0),
        blocked_at_segment=int(queue_payload.get("blocked_at_segment", 0) or 0),
        last_review_segment=int(queue_payload.get("last_review_segment", 0) or 0),
        retry_after_segment=int(queue_payload.get("retry_after_segment", 0) or 0),
    )
    relations = []
    relation_rows = row.get("relations", [])
    if isinstance(relation_rows, list):
        for relation_payload in relation_rows:
            if not isinstance(relation_payload, Mapping):
                continue
            relations.append(
                make_candidate_relation(
                    {
                        "record_id": str(relation_payload.get("record_id", "")),
                        "relation": str(relation_payload.get("relation", "predicts")),
                        "source_event_id": str(relation_payload.get("source_event_id", "")),
                        "target_event_id": str(relation_payload.get("target_event_id", "")),
                        "delay_lower_ms": int(relation_payload.get("delay_lower_ms", 0) or 0),
                        "delay_upper_ms": int(relation_payload.get("delay_upper_ms", 0) or 0),
                        "confidence": float(relation_payload.get("confidence", 0.0) or 0.0),
                        "source_ref": str(
                            relation_payload.get("lineage", {}).get("source_ref", "")
                            if isinstance(relation_payload.get("lineage"), Mapping)
                            else relation_payload.get("source_ref", "")
                        ),
                        "source_hash": str(
                            relation_payload.get("lineage", {}).get("source_hash", "")
                            if isinstance(relation_payload.get("lineage"), Mapping)
                            else relation_payload.get("source_hash", "")
                        ),
                        "extractor_name": str(
                            relation_payload.get("lineage", {}).get("extractor_name", "prediction_gain")
                            if isinstance(relation_payload.get("lineage"), Mapping)
                            else relation_payload.get("extractor_name", "prediction_gain")
                        ),
                        "extractor_version": str(
                            relation_payload.get("lineage", {}).get("extractor_version", "v1")
                            if isinstance(relation_payload.get("lineage"), Mapping)
                            else relation_payload.get("extractor_version", "v1")
                        ),
                        "evidence_count": int(relation_payload.get("evidence_count", 0) or 0),
                        "counterexample_count": int(relation_payload.get("counterexample_count", 0) or 0),
                        "prediction_gain": float(relation_payload.get("prediction_gain", 0.0) or 0.0),
                    }
                )
            )
    return {
        "case_id": str(row.get("case_id", concept_key)),
        "case_type": str(row.get("case_type", "unknown")),
        "expected_outcome": str(row.get("expected_outcome", "unknown")),
        "concept_key": concept_key,
        "queue_entry": queue_entry,
        "relations": tuple(relations),
    }


def _case_path(base_path: str, case_id: str) -> str:
    root, ext = os.path.splitext(base_path)
    return f"{root}_{case_id}{ext or '.json'}"


def _evaluate_concept_revalidation_cases(
    concept_fixture: Mapping[str, Any],
    *,
    concept_queue_path: str,
    concept_review_report_path: str,
) -> Dict[str, Any]:
    cases = concept_fixture.get("cases", ())
    if not isinstance(cases, Sequence):
        cases = ()
    evaluations = []
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            continue
        queue_entry = raw_case.get("queue_entry")
        relations = raw_case.get("relations", ())
        if not isinstance(queue_entry, ConceptRevalidationEntry):
            continue
        case_id = str(raw_case.get("case_id", queue_entry.concept_key))
        queue_path = _case_path(concept_queue_path, case_id)
        report_path = _case_path(concept_review_report_path, case_id)
        save_revalidation_queue((queue_entry,), queue_path)
        review = run_persisted_concept_review_cycle(
            relations,
            current_segment=6,
            queue_path=queue_path,
            report_path=report_path,
        )
        persisted_queue = load_revalidation_queue(queue_path)
        evaluations.append(
            {
                "case_id": case_id,
                "case_type": str(raw_case.get("case_type", "unknown")),
                "expected_outcome": str(raw_case.get("expected_outcome", "unknown")),
                "review": review,
                "remaining_queue_count": len(persisted_queue),
                "blocked_reason_counts": _blocked_reason_counts(review.to_dict()),
            }
        )
    return {"evaluations": evaluations}


def _blocked_reason_counts(concept_review: Mapping[str, Any]) -> Dict[str, int]:
    schedule = concept_review.get("schedule", {})
    blocked_queue = schedule.get("blocked_queue", [])
    if not isinstance(blocked_queue, list):
        blocked_queue = []
    counts = {
        "source_diversity": 0,
        "revision_conflict": 0,
        "counterexample_pressure": 0,
        "attempt_budget": 0,
        "cooldown": 0,
        "other": 0,
    }
    for item in blocked_queue:
        if not isinstance(item, Mapping):
            counts["other"] += 1
            continue
        decision = str(item.get("decision", "") or "")
        next_action = str(item.get("next_action", "") or "")
        if decision == "blocked_attempt_budget":
            counts["attempt_budget"] += 1
        elif decision == "blocked_cooldown":
            counts["cooldown"] += 1
        elif next_action == "collect_more_distinct_sources":
            counts["source_diversity"] += 1
        elif next_action == "wait_for_source_revision_resolution":
            counts["revision_conflict"] += 1
        elif next_action == "collect_counterexamples_and_retest":
            counts["counterexample_pressure"] += 1
        else:
            counts["other"] += 1
    return counts


def _concept_followup_actions(
    *,
    blocked_reason_counts: Mapping[str, int],
    concept_case_count: int,
    concept_admitted_count: int,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if int(blocked_reason_counts.get("source_diversity", 0) or 0) > 0:
        actions.append(
            {
                "priority": 1,
                "reason": "source_diversity",
                "action": "collect_additional_distinct_sources",
                "detail": "Increase independent source coverage for concept candidates blocked by insufficient source diversity.",
            }
        )
    if int(blocked_reason_counts.get("revision_conflict", 0) or 0) > 0:
        actions.append(
            {
                "priority": 2,
                "reason": "revision_conflict",
                "action": "resolve_source_revision_conflicts",
                "detail": "Wait for or reconcile conflicting source revisions before re-auditing concept candidates.",
            }
        )
    if int(blocked_reason_counts.get("counterexample_pressure", 0) or 0) > 0:
        actions.append(
            {
                "priority": 3,
                "reason": "counterexample_pressure",
                "action": "add_negative_and_contrastive_materials",
                "detail": "Collect negative or contrastive materials to lower counterexample pressure before another review cycle.",
            }
        )
    if int(blocked_reason_counts.get("attempt_budget", 0) or 0) > 0:
        actions.append(
            {
                "priority": 4,
                "reason": "attempt_budget",
                "action": "manual_review_high_stall_candidates",
                "detail": "Escalate repeatedly stalled concept candidates to manual review or tighter heuristics.",
            }
        )
    if int(blocked_reason_counts.get("cooldown", 0) or 0) > 0:
        actions.append(
            {
                "priority": 5,
                "reason": "cooldown",
                "action": "wait_for_next_review_window",
                "detail": "Cooldown-blocked candidates should be revisited after additional segments or evidence arrive.",
            }
        )
    if int(blocked_reason_counts.get("other", 0) or 0) > 0:
        actions.append(
            {
                "priority": 6,
                "reason": "other",
                "action": "inspect_unclassified_revalidation_blocks",
                "detail": "Review unclassified blocking reasons and extend deterministic routing if needed.",
            }
        )
    if not actions and int(concept_case_count) > 0 and int(concept_admitted_count) == int(concept_case_count):
        actions.append(
            {
                "priority": 1,
                "reason": "none",
                "action": "scale_revalidation_case_coverage",
                "detail": "All current concept revalidation cases recovered; expand case coverage with harder source-aware materials.",
            }
        )
    return actions


def build_report(
    materials: Sequence[Dict[str, Any]],
    reports: Mapping[str, Dict[str, Any]],
    *,
    source_paths: Mapping[str, str],
    manifest_path: str,
    trace_path: str,
    round_trip_state_path: str,
    concept_queue_path: str,
    concept_review_report_path: str,
    concept_fixture_path: str,
) -> Dict[str, Any]:
    eligible = _eligible_materials(materials)
    selected = eligible[:10]
    logarithmic = _run_profile("logarithmic", selected, reports)
    fixed = _run_profile("fixed", selected, reports)

    resolved_state = ensure_parent_directory(round_trip_state_path)
    with open(resolved_state, "w", encoding="utf-8") as handle:
        json.dump(logarithmic["state"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    restored = VerifiedHierarchicalEventStateCache.from_state_dict(
        load_json(resolved_state)
    )
    first_target = selected[0] if selected else {}
    round_trip_result = restored.retrieve(
        first_target.get("sparse_signature", []),
        own_latent_id=str(first_target.get("latent_cluster_id", "")),
        source_ref=str(first_target.get("source_ref", "")),
    )
    round_trip_integrity = float(
        bool(round_trip_result.matches)
        and round_trip_result.matches[0]["entry_id"]
        == str(first_target.get("manifest_id", ""))
    )

    corrupted = copy.deepcopy(logarithmic["state"])
    corrupted["schema"] = "corrupted-schema"
    corruption_rejected = False
    try:
        VerifiedHierarchicalEventStateCache.from_state_dict(corrupted)
    except ValueError:
        corruption_rejected = True

    missing_reports = copy.deepcopy(dict(reports))
    missing_reports["own_latent"] = {}
    blocked_probe = (
        build_event_state_candidate(
            selected[0],
            missing_reports,
            time_segment=0,
        )
        if selected
        else None
    )
    missing_report_freeze = float(
        blocked_probe is not None
        and not blocked_probe.promotion_allowed
        and blocked_probe.promotion_decision == "freeze_unverified_source"
    )

    concept_fixture = _load_or_build_concept_revalidation_fixture(
        selected,
        fixture_path=concept_fixture_path,
    )
    concept_review_bundle = _evaluate_concept_revalidation_cases(
        concept_fixture,
        concept_queue_path=concept_queue_path,
        concept_review_report_path=concept_review_report_path,
    )
    concept_evaluations = concept_review_bundle["evaluations"]
    concept_case_count = len(concept_evaluations)
    concept_expected_recoverable_count = sum(
        1 for item in concept_evaluations if item["expected_outcome"] == "admit"
    )
    concept_expected_blocked_count = sum(
        1 for item in concept_evaluations if item["expected_outcome"] == "blocked"
    )
    concept_ready_count = sum(
        len(item["review"].schedule.ready_queue) for item in concept_evaluations
    )
    concept_blocked_count = sum(
        len(item["review"].schedule.blocked_queue) for item in concept_evaluations
    )
    concept_attempt_budget_blocked_count = sum(
        sum(
            1
            for queued in item["review"].schedule.blocked_queue
            if queued.decision == "blocked_attempt_budget"
        )
        for item in concept_evaluations
    )
    blocked_reason_counts = {
        "source_diversity": sum(item["blocked_reason_counts"]["source_diversity"] for item in concept_evaluations),
        "revision_conflict": sum(item["blocked_reason_counts"]["revision_conflict"] for item in concept_evaluations),
        "counterexample_pressure": sum(item["blocked_reason_counts"]["counterexample_pressure"] for item in concept_evaluations),
        "attempt_budget": sum(item["blocked_reason_counts"]["attempt_budget"] for item in concept_evaluations),
        "cooldown": sum(item["blocked_reason_counts"]["cooldown"] for item in concept_evaluations),
        "other": sum(item["blocked_reason_counts"]["other"] for item in concept_evaluations),
    }
    concept_admitted_count = sum(
        len(item["review"].admission_plan.admitted_candidates)
        for item in concept_evaluations
    )
    concept_ready_credit_scores = [
        float(queued.credit_score)
        for item in concept_evaluations
        for queued in item["review"].schedule.ready_queue
    ]
    concept_ready_credit_confidences = [
        float(queued.credit_confidence)
        for item in concept_evaluations
        for queued in item["review"].schedule.ready_queue
    ]
    concept_blocked_credit_scores = [
        float(queued.credit_score)
        for item in concept_evaluations
        for queued in item["review"].schedule.blocked_queue
    ]
    concept_recovered_count = sum(
        1
        for item in concept_evaluations
        if item["expected_outcome"] == "admit"
        and len(item["review"].admission_plan.admitted_candidates) >= 1
        and item["remaining_queue_count"] == 0
    )
    concept_blocked_integrity_count = sum(
        1
        for item in concept_evaluations
        if item["expected_outcome"] == "blocked"
        and len(item["review"].admission_plan.admitted_candidates) == 0
        and item["remaining_queue_count"] >= 1
    )
    concept_queue_drained = float(
        concept_expected_recoverable_count > 0
        and concept_recovered_count == concept_expected_recoverable_count
    ) if concept_expected_recoverable_count else 0.0
    concept_review_recovered = float(
        concept_expected_recoverable_count > 0
        and concept_recovered_count == concept_expected_recoverable_count
    ) if concept_expected_recoverable_count else 0.0
    concept_blocked_integrity = float(
        concept_expected_blocked_count > 0
        and concept_blocked_integrity_count == concept_expected_blocked_count
    ) if concept_expected_blocked_count else 1.0
    concept_recovery_rate = float(concept_recovered_count) / float(max(1, concept_expected_recoverable_count))
    concept_followup_actions = _concept_followup_actions(
        blocked_reason_counts=blocked_reason_counts,
        concept_case_count=concept_case_count,
        concept_admitted_count=concept_admitted_count,
    )

    traces = logarithmic["traces"] + fixed["traces"]
    if blocked_probe is not None:
        traces.append(
            {
                "profile": "integration_fault",
                "kind": "missing_report",
                "evidence": blocked_probe.to_dict(),
            }
        )
    traces.append(
        {
            "profile": "concept_review",
            "kind": "persisted_revalidation_cycle",
            "fixture_mode": concept_fixture.get("fixture_mode", "unknown"),
            "case_count": concept_case_count,
            "ready_count": concept_ready_count,
            "admitted_count": concept_admitted_count,
            "expected_recoverable_count": concept_expected_recoverable_count,
            "expected_blocked_count": concept_expected_blocked_count,
            "case_results": [
                {
                    "case_id": item["case_id"],
                    "case_type": item["case_type"],
                    "expected_outcome": item["expected_outcome"],
                    "remaining_queue_count": item["remaining_queue_count"],
                    "ready_credit_scores": [
                        float(queued.credit_score)
                        for queued in item["review"].schedule.ready_queue
                    ],
                    "blocked_credit_scores": [
                        float(queued.credit_score)
                        for queued in item["review"].schedule.blocked_queue
                    ],
                    "result": item["review"].to_dict(),
                }
                for item in concept_evaluations
            ],
        }
    )
    resolved_trace = ensure_parent_directory(trace_path)
    with open(resolved_trace, "w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

    target_count = max(1, int(logarithmic["target_count"]))
    logarithmic_recall = float(logarithmic["recall_success"]) / target_count
    fixed_recall = float(fixed["recall_success"]) / target_count
    source_revision_integrity = float(
        bool(logarithmic["state"]["entries"])
        and all(
            entry.get("source_ref") and entry.get("source_revision")
            for entry in logarithmic["state"]["entries"]
        )
    )
    passed = bool(
        len(selected) >= 6
        and logarithmic_recall == 1.0
        and logarithmic_recall > fixed_recall
        and logarithmic["reactivation_hint_integrity"] == 1.0
        and round_trip_integrity == 1.0
        and corruption_rejected
        and missing_report_freeze == 1.0
        and source_revision_integrity == 1.0
        and concept_review_recovered == 1.0
        and concept_blocked_integrity == 1.0
        and logarithmic["max_retrieval_event_cost"] <= 256
    )
    return {
        "schema": "sara-event-state-cache-integration-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "material_count": len(selected),
        "next_actions": concept_followup_actions,
        "metrics": {
            "source_aware_logarithmic_delayed_recall": logarithmic_recall,
            "source_aware_fixed_delayed_recall": fixed_recall,
            "reactivation_hint_integrity": logarithmic[
                "reactivation_hint_integrity"
            ],
            "round_trip_integrity": round_trip_integrity,
            "corrupted_state_rejection": float(corruption_rejected),
            "missing_report_freeze_integrity": missing_report_freeze,
            "source_revision_integrity": source_revision_integrity,
            "concept_revalidation_case_count": concept_case_count,
            "concept_revalidation_fixture_case_count": int(concept_fixture.get("fixture_case_count", concept_case_count) or 0),
            "concept_revalidation_expected_recoverable_case_count": concept_expected_recoverable_count,
            "concept_revalidation_expected_blocked_case_count": concept_expected_blocked_count,
            "concept_revalidation_ready_count": concept_ready_count,
            "concept_revalidation_blocked_count": concept_blocked_count,
            "concept_revalidation_attempt_budget_blocked_count": concept_attempt_budget_blocked_count,
            "concept_revalidation_source_diversity_blocked_count": blocked_reason_counts["source_diversity"],
            "concept_revalidation_revision_conflict_blocked_count": blocked_reason_counts["revision_conflict"],
            "concept_revalidation_counterexample_blocked_count": blocked_reason_counts["counterexample_pressure"],
            "concept_revalidation_cooldown_blocked_count": blocked_reason_counts["cooldown"],
            "concept_revalidation_other_blocked_count": blocked_reason_counts["other"],
            "concept_revalidation_admitted_count": concept_admitted_count,
            "concept_revalidation_recovery_rate": concept_recovery_rate,
            "concept_revalidation_blocked_integrity": concept_blocked_integrity,
            "concept_revalidation_queue_drained": concept_queue_drained,
            "concept_revalidation_recovered_integrity": concept_review_recovered,
            "concept_revalidation_ready_mean_credit_score": round(
                sum(concept_ready_credit_scores) / float(max(1, len(concept_ready_credit_scores))),
                6,
            ),
            "concept_revalidation_ready_mean_credit_confidence": round(
                sum(concept_ready_credit_confidences) / float(max(1, len(concept_ready_credit_confidences))),
                6,
            ),
            "concept_revalidation_blocked_mean_credit_score": round(
                sum(concept_blocked_credit_scores) / float(max(1, len(concept_blocked_credit_scores))),
                6,
            ),
            "logarithmic_entry_count": logarithmic["state"]["entry_count"],
            "fixed_entry_count": fixed["state"]["entry_count"],
            "max_retrieval_event_cost": logarithmic[
                "max_retrieval_event_cost"
            ],
        },
        "source_paths": {
            key: os.path.abspath(value)
            for key, value in source_paths.items()
        },
        "manifest_path": os.path.abspath(manifest_path),
        "round_trip_state_path": os.path.abspath(round_trip_state_path),
        "concept_fixture_path": os.path.abspath(concept_fixture_path),
        "concept_queue_path": os.path.abspath(concept_queue_path),
        "concept_review_report_path": os.path.abspath(concept_review_report_path),
        "trace_path": os.path.abspath(trace_path),
        "concept_fixture_mode": str(concept_fixture.get("fixture_mode", "unknown")),
        "policy_notes": [
            "Cache candidates are derived from managed Phase 17 evidence and source-aware latent manifest rows.",
            "Reactivation hints are bounded read-only routing signals.",
            "Persistent state is restored only after strict schema and budget validation.",
            "The benchmark remains observed-only and does not alter production memory.",
        ],
    }


def summarize(report: Mapping[str, Any]) -> str:
    lines = [
        f"Event-state cache integration: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Materials: {report.get('material_count')}",
    ]
    lines.extend(
        f"- {key}: {value}"
        for key, value in sorted(report.get("metrics", {}).items())
    )
    next_actions = report.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.append("- concept_revalidation_followups:")
        for item in next_actions:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  - "
                f"{item.get('action', '')} "
                f"(reason={item.get('reason', '')}, priority={item.get('priority', '')})"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source-aware event-state cache integration."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    for name, default_path in DEFAULT_SOURCE_PATHS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}-path",
            default=default_path,
        )
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument(
        "--round-trip-state-path",
        default=DEFAULT_ROUND_TRIP_STATE_PATH,
    )
    parser.add_argument("--concept-queue-path", default=DEFAULT_CONCEPT_QUEUE_PATH)
    parser.add_argument(
        "--concept-review-report-path",
        default=DEFAULT_CONCEPT_REVIEW_REPORT_PATH,
    )
    parser.add_argument(
        "--concept-fixture-path",
        default=DEFAULT_CONCEPT_FIXTURE_PATH,
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_paths = {
        name: str(getattr(args, f"{name}_path"))
        for name in DEFAULT_SOURCE_PATHS
    }
    report = build_report(
        read_jsonl(args.manifest_path),
        load_source_reports(source_paths),
        source_paths=source_paths,
        manifest_path=args.manifest_path,
        trace_path=args.trace_path,
        round_trip_state_path=args.round_trip_state_path,
        concept_queue_path=args.concept_queue_path,
        concept_review_report_path=args.concept_review_report_path,
        concept_fixture_path=args.concept_fixture_path,
    )
    resolved_report = ensure_parent_directory(args.report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(args.summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize(report))
    print(summarize(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
