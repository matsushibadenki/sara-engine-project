#!/usr/bin/env python3
"""Run one frozen retrieval workload for paired physical energy measurement."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id  # noqa: E402
from sara_engine.learning.idle_replay import IdleReplayConfig  # noqa: E402
from sara_engine.learning.sleep_consolidation import SleepConsolidationConfig  # noqa: E402
from sara_engine.memory.concept_admission import ConceptRevalidationEntry  # noqa: E402
from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.idle_consolidation_loop import IdleConsolidationLoop  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_CORPUS_PATH = processed_data_path("corpus.txt")
DEFAULT_OUTPUT_PATH = workspace_path("evaluation", "energy_pair_workload_result.json")


def _build_maintenance_seed(
    docs: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[
    VerifiedHierarchicalEventStateCache,
    tuple[ConceptRevalidationEntry, ...],
    PersistentSelfStateController,
]:
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=8)
    queue_entries = []
    core_ids = []
    for index, task in enumerate(tasks[:3]):
        query = str(task.get("query", "") or f"query-{index}")
        answer = str(task.get("answer", "") or f"answer-{index}")
        source_ref = str(task.get("source_ref", "") or f"task:{index}")
        doc_id = str(task.get("document_id", "") or f"doc-{index}")
        concept_key = f"predicts:text:{query}->doc:{doc_id}"
        signature = (
            stable_self_state_id(query, modulus=1024),
            stable_self_state_id(answer, modulus=1024),
            stable_self_state_id(doc_id, modulus=1024),
        )
        cache.admit(
            EventStateCandidate(
                entry_id=f"maintenance-{index}",
                signature=signature,
                source_ref=source_ref,
                time_segment=index + 1,
                own_latent_id=concept_key,
                confidence=0.92,
                uncertainty=0.08,
                source_reliability=0.9,
                resonance_score=0.9,
                sequence_support_score=0.45,
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
                source_hashes=(f"seed-{index}",),
                revision_conflict_count=1,
                contradiction_score=0.15,
                next_action="wait",
                attempt_count=0,
                blocked_at_segment=index + 1,
                last_review_segment=index + 1,
                retry_after_segment=index + 2,
            )
        )
        core_ids.extend(
            (
                stable_self_state_id(f"text:{query}"),
                stable_self_state_id(f"doc:{doc_id}"),
            )
        )
    if not queue_entries and docs:
        first_doc = docs[0]
        doc_text = str(first_doc.get("text", "") or "")
        doc_key = str(first_doc.get("id", "") or "doc-0")
        cache.admit(
            EventStateCandidate(
                entry_id="maintenance-fallback",
                signature=(stable_self_state_id(doc_key, modulus=1024),),
                source_ref=f"doc:{doc_key}",
                time_segment=1,
                own_latent_id=f"latent:{doc_key}",
                confidence=0.9,
                uncertainty=0.1,
                source_reliability=0.9,
                resonance_score=0.88,
                sequence_support_score=0.2,
                sequence_support_count=1,
                metabolic_headroom=0.9,
                observed=True,
                source_backed=True,
                verified=True,
                event_cost=1,
            )
        )
        core_ids.append(stable_self_state_id(doc_text[:32] or doc_key))
    controller = PersistentSelfStateController(core_event_ids=tuple(core_ids[:6]))
    return cache, tuple(queue_entries), controller


def _run_sparse_maintenance_cycle(
    *,
    cache: VerifiedHierarchicalEventStateCache,
    queue_entries: Sequence[ConceptRevalidationEntry],
    controller: PersistentSelfStateController,
    segment: int,
) -> Dict[str, Any]:
    result = IdleConsolidationLoop().run(
        cache,
        queue_entries,
        [],
        current_segment=segment,
        persistent_self_state=controller,
        replay_config=IdleReplayConfig(max_candidates=2, event_budget=12, min_replay_score=0.2),
        sleep_config=SleepConsolidationConfig(event_budget=12.0),
    )
    idle_report = result.idle_replay_report
    sleep_report = result.sleep_consolidation_report
    memory_phase_report = result.memory_phase_report
    self_state_trace = (
        idle_report.get("self_state_trace", {})
        if isinstance(idle_report.get("self_state_trace", {}), dict)
        else {}
    )
    return {
        "maintenance_selected_count": len(idle_report.get("selected", ())),
        "maintenance_phase_count": len(memory_phase_report.get("phase_tracks", ())),
        "maintenance_refresh_count": len(result.cache_refresh),
        "maintenance_event_cost": float(
            sum(float(item.get("event_cost", 0.0) or 0.0) for item in sleep_report.get("traces", ()))
        ),
        "maintenance_idle_self_state_ok_count": 1 if bool(self_state_trace.get("idle_self_state_ok", False)) else 0,
        "maintenance_spontaneous_event_count": len(self_state_trace.get("spontaneous_event_ids", ())),
        "maintenance_predicted_event_count": len(self_state_trace.get("predicted_event_ids", ())),
    }


def _load_external_validity_module():
    path = os.path.join(PROJECT_ROOT, "scripts", "eval", "real_data_external_validity.py")
    spec = importlib.util.spec_from_file_location("energy_pair_external_validity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load real-data external-validity module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_retrieval_workload(
    *,
    system: str,
    corpus_path: str,
    max_docs: int,
    max_cases: int,
    repetitions: int,
    warmup_count: int,
) -> Dict[str, Any]:
    module = _load_external_validity_module()
    docs = module._load_corpus(corpus_path, limit=max_docs)
    tasks = module.build_real_data_tasks(docs, max_cases=max_cases)
    if not docs or not tasks:
        raise ValueError("Frozen retrieval workload requires non-empty docs and tasks.")
    retriever_type = (
        module.MetabolicSparseEventRetriever
        if system == "sara"
        else module.BM25OfflineProxyRetriever
    )
    cache: VerifiedHierarchicalEventStateCache | None = None
    queue_entries: tuple[ConceptRevalidationEntry, ...] = ()
    controller: PersistentSelfStateController | None = None
    if system == "sara":
        cache, queue_entries, controller = _build_maintenance_seed(docs, tasks)

    for _ in range(max(0, int(warmup_count))):
        warmup_retriever = retriever_type(docs)
        module._score_retriever(warmup_retriever, tasks, docs)

    started = time.perf_counter()
    success_count = 0
    trial_count = 0
    event_cost = 0.0
    maintenance_selected_count = 0
    maintenance_phase_count = 0
    maintenance_refresh_count = 0
    maintenance_event_cost = 0.0
    maintenance_idle_self_state_ok_count = 0
    maintenance_spontaneous_event_count = 0
    maintenance_predicted_event_count = 0
    for _ in range(max(1, int(repetitions))):
        retriever = retriever_type(docs)
        score = module._score_retriever(retriever, tasks, docs)
        case_results = score.get("case_results", [])
        success_count += sum(
            1
            for row in case_results
            if isinstance(row, dict) and bool(row.get("correct", False))
        )
        trial_count += len(case_results)
        event_cost += float(score.get("avg_event_cost_proxy", 0.0) or 0.0)
        if cache is not None and controller is not None:
            maintenance = _run_sparse_maintenance_cycle(
                cache=cache,
                queue_entries=queue_entries,
                controller=controller,
                segment=max(1, int(trial_count)),
            )
            maintenance_selected_count += int(maintenance["maintenance_selected_count"])
            maintenance_phase_count += int(maintenance["maintenance_phase_count"])
            maintenance_refresh_count += int(maintenance["maintenance_refresh_count"])
            maintenance_event_cost += float(maintenance["maintenance_event_cost"])
            maintenance_idle_self_state_ok_count += int(
                maintenance["maintenance_idle_self_state_ok_count"]
            )
            maintenance_spontaneous_event_count += int(
                maintenance["maintenance_spontaneous_event_count"]
            )
            maintenance_predicted_event_count += int(
                maintenance["maintenance_predicted_event_count"]
            )
    duration_seconds = time.perf_counter() - started
    success_rate = float(success_count) / float(max(1, trial_count))
    return {
        "schema": "sara-energy-pair-workload-result-v1",
        "task": "paired_retrieval",
        "system": system,
        "success_criterion_id": "retrieval-exact-document-index-v1",
        "success_count": success_count,
        "trial_count": trial_count,
        "success_rate": success_rate,
        "passed": success_rate >= 0.80,
        "duration_seconds": duration_seconds,
        "warmup_count": int(warmup_count),
        "measured_repetitions": int(repetitions),
        "doc_count": len(docs),
        "case_count": len(tasks),
        "avg_event_cost_proxy_across_repetitions": event_cost
        / float(max(1, int(repetitions))),
        "maintenance_selected_count": int(maintenance_selected_count),
        "maintenance_phase_count": int(maintenance_phase_count),
        "maintenance_refresh_count": int(maintenance_refresh_count),
        "maintenance_event_cost": float(maintenance_event_cost),
        "maintenance_idle_self_state_ok_count": int(maintenance_idle_self_state_ok_count),
        "maintenance_spontaneous_event_count": int(maintenance_spontaneous_event_count),
        "maintenance_predicted_event_count": int(maintenance_predicted_event_count),
        "retriever": retriever_type.__name__,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen paired energy workload.")
    parser.add_argument("--system", choices=["sara", "ann"], required=True)
    parser.add_argument("--task", choices=["paired_retrieval"], default="paired_retrieval")
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_retrieval_workload(
        system=args.system,
        corpus_path=args.corpus_path,
        max_docs=args.max_docs,
        max_cases=args.max_cases,
        repetitions=args.repetitions,
        warmup_count=args.warmup_count,
    )
    report["corpus_path"] = os.path.abspath(args.corpus_path)
    output_path = ensure_parent_directory(args.output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
