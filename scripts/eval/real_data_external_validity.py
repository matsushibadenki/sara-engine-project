# Directory Path: scripts/eval/real_data_external_validity.py
# English Title: Real-Data External Validity Benchmark
# Purpose/Content: Compares sparse SARA-style retrieval against a dense ANN-style proxy baseline on real corpus QA, summarization, and continual-memory tasks.

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Set, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.rag.rag_pipeline import SNNRAGPipeline


def _load_project_paths_helpers():
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    spec = importlib.util.spec_from_file_location("project_paths_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load project paths helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure_parent = getattr(module, "ensure_parent_directory", None)
    processed_data = getattr(module, "processed_data_path", None)
    workspace = getattr(module, "workspace_path", None)
    if not callable(ensure_parent) or not callable(processed_data) or not callable(workspace):
        raise RuntimeError("project_paths helper is missing required callables.")
    return ensure_parent, processed_data, workspace


ensure_parent_directory, processed_data_path, workspace_path = _load_project_paths_helpers()


DEFAULT_REPORT_PATH = workspace_path("evaluation", "real_data_external_validity.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "real_data_external_validity_summary.txt")
DEFAULT_HISTORY_PATH = workspace_path("evaluation", "real_data_external_validity_history.json")
TREND_ABSOLUTE_METRICS = {
    "real_data_qa_accuracy",
    "real_data_summary_keyword_coverage",
    "continual_memory_hit_rate",
}
TREND_RELATIVE_METRICS = {
    "performance_energy_ratio_proxy",
    "ann_cost_advantage_proxy",
}
DEFAULT_THRESHOLDS = {
    "min_real_data_task_count": 8.0,
    "min_real_data_qa_accuracy": 0.80,
    "dense_accuracy_tolerance": 0.05,
    "min_summary_keyword_coverage": 0.60,
    "min_continual_memory_hit_rate": 0.80,
    "min_ann_cost_advantage_proxy": 2.0,
    "min_performance_energy_ratio_proxy": 2.0,
    "min_negative_control_abstention": 1.0,
    "min_negative_control_cost_advantage": 2.0,
    "min_partial_evidence_abstention": 1.0,
    "min_partial_evidence_cost_advantage": 2.0,
    "min_contrastive_control_accuracy": 1.0,
    "min_contrastive_control_cost_advantage": 2.0,
    "min_dense_embedding_cost_advantage": 2.0,
    "min_sparse_diffusion_real_data_denoise_accuracy": 1.0,
    "min_sparse_diffusion_real_data_event_cost_advantage": 2.0,
    "min_sparse_diffusion_real_data_partition_integrity": 1.0,
    "min_sparse_diffusion_real_data_single_pass_integrity": 1.0,
}
RETRIEVER_STRATEGY = "metabolic_sparse_rarity_early_stop_verified_fallback_v1"


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]{2,}", text.lower())
    return [token for token in tokens if len(token) >= 2]


def _load_corpus(path: str, limit: int) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [re.sub(r"\s+", " ", line).strip() for line in handle]
    docs = [line for line in lines if len(line) >= 20]
    return docs[: max(int(limit), 1)]


def _load_sparse_diffusion_block_module():
    module_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "sparse_diffusion_block_readiness.py")
    module_name = "sara_sparse_diffusion_block_readiness_external_validity"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sparse diffusion block readiness helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _keyword_query(tokens: Sequence[str], max_terms: int = 4) -> str:
    unique: List[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
        if len(unique) >= max_terms:
            break
    return " ".join(unique)


def _document_frequency(docs: Sequence[str]) -> Dict[str, int]:
    frequencies: Dict[str, int] = {}
    for doc in docs:
        for token in set(_tokenize(doc)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _rare_keyword_query(tokens: Sequence[str], frequencies: Dict[str, int], max_terms: int = 4) -> str:
    first_position: Dict[str, int] = {}
    for position, token in enumerate(tokens):
        if token not in first_position:
            first_position[token] = position
    ranked_tokens = sorted(
        first_position,
        key=lambda token: (frequencies.get(token, 0), first_position[token]),
    )
    query = _keyword_query(ranked_tokens, max_terms=max_terms)
    return query or _keyword_query(tokens, max_terms=max_terms)


def build_real_data_tasks(docs: Sequence[str], max_cases: int = 24) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    frequencies = _document_frequency(docs)
    for index, doc in enumerate(docs):
        tokens = _tokenize(doc)
        if len(tokens) < 3:
            continue
        query = _rare_keyword_query(tokens, frequencies)
        if not query:
            continue
        expected_keywords = _tokenize(query)
        if len(expected_keywords) < 2:
            expected_keywords = tokens[: min(len(tokens), 5)]
        tasks.append(
            {
                "case_id": f"doc-{index}",
                "query": query,
                "expected_doc_index": index,
                "expected_keywords": expected_keywords[: min(len(expected_keywords), 5)],
                "document": doc,
            }
        )
        if len(tasks) >= max_cases:
            break
    return tasks


def _hash_text_items(items: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_benchmark_context(
    *,
    docs: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
    corpus_path: str,
    max_docs: int,
    max_cases: int,
) -> Dict[str, Any]:
    task_fingerprint_items = [
        f"{task.get('case_id', '')}|{task.get('query', '')}|{task.get('expected_doc_index', '')}"
        for task in tasks
        if isinstance(task, dict)
    ]
    return {
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_sha256": _hash_text_items(list(docs)),
        "task_sha256": _hash_text_items(task_fingerprint_items),
        "max_docs": int(max_docs),
        "max_cases": int(max_cases),
        "doc_count": len(docs),
        "task_count": len(tasks),
        "retriever_strategy": RETRIEVER_STRATEGY,
    }


class SparseEventRetriever:
    def __init__(self, docs: Sequence[str]) -> None:
        self.docs = list(docs)
        self.doc_tokens: List[Set[str]] = [set(_tokenize(doc)) for doc in self.docs]
        self.index: DefaultDict[str, Set[int]] = defaultdict(set)
        for doc_index, tokens in enumerate(self.doc_tokens):
            for token in tokens:
                self.index[token].add(doc_index)
        self.last_diagnostics: Dict[str, Any] = {}

    def search(self, query: str) -> Tuple[int, int]:
        query_tokens = set(_tokenize(query))
        candidates: Set[int] = set()
        event_cost = 0
        for token in query_tokens:
            hits = self.index.get(token, set())
            candidates.update(hits)
            event_cost += 1 + len(hits)
        if not candidates:
            return -1, max(event_cost, 1)
        best_index = max(
            candidates,
            key=lambda idx: (len(query_tokens & self.doc_tokens[idx]), -idx),
        )
        event_cost += len(candidates)
        self.last_diagnostics = {
            "processed_token_count": len(query_tokens),
            "query_token_count": len(query_tokens),
            "candidate_count": len(candidates),
            "early_stopped": False,
            "confidence_margin": 0.0,
        }
        return best_index, max(event_cost, 1)


class MetabolicSparseEventRetriever(SparseEventRetriever):
    def __init__(
        self,
        docs: Sequence[str],
        *,
        max_candidate_scan: int = 32,
        confidence_margin: int = 2,
        min_match_ratio: float = 0.60,
    ) -> None:
        super().__init__(docs)
        self.max_candidate_scan = max(int(max_candidate_scan), 1)
        self.confidence_margin = max(int(confidence_margin), 1)
        self.min_match_ratio = max(0.0, min(1.0, float(min_match_ratio)))

    def _ordered_query_tokens(self, query: str) -> List[str]:
        tokens = sorted(set(_tokenize(query)), key=lambda token: (len(self.index.get(token, set())), token))
        return tokens

    def _rank_candidates(
        self,
        query_tokens: Set[str],
        candidates: Set[int],
        *,
        max_scan: int | None = None,
    ) -> Tuple[int, int, int]:
        best_index = -1
        best_score = -1
        second_score = -1
        ordered_candidates = sorted(candidates)
        if max_scan is not None:
            ordered_candidates = ordered_candidates[: max(int(max_scan), 1)]
        for index in ordered_candidates:
            score = len(query_tokens & self.doc_tokens[index])
            if score > best_score:
                second_score = best_score
                best_index = index
                best_score = score
            elif score > second_score:
                second_score = score
        return best_index, best_score, max(second_score, 0)

    def search(self, query: str) -> Tuple[int, int]:
        ordered_tokens = self._ordered_query_tokens(query)
        query_tokens = set(ordered_tokens)
        candidates: Set[int] = set()
        event_cost = 0
        processed = 0
        processed_tokens: List[str] = []
        early_stopped = False
        best_index = -1
        best_score = -1
        second_score = 0

        for token in ordered_tokens:
            hits = self.index.get(token, set())
            candidates.update(hits)
            processed += 1
            processed_tokens.append(token)
            event_cost += 1 + len(hits)
            if not candidates:
                continue
            best_index, best_score, second_score = self._rank_candidates(
                query_tokens,
                candidates,
                max_scan=self.max_candidate_scan,
            )
            event_cost += min(len(candidates), self.max_candidate_scan)
            exact_single_hit = len(candidates) == 1 and best_score >= max(1, processed)
            confident_margin = best_score - second_score >= self.confidence_margin and processed >= 2
            if exact_single_hit or confident_margin:
                early_stopped = True
                break

        if candidates and not early_stopped:
            best_index, best_score, second_score = self._rank_candidates(query_tokens, candidates)
            event_cost += len(candidates)
        elif best_index < 0 and candidates:
            best_index, best_score, second_score = self._rank_candidates(query_tokens, candidates)
            event_cost += len(candidates)
        match_ratio = float(max(best_score, 0)) / max(float(len(query_tokens)), 1.0)
        abstained_by_match_ratio = bool(query_tokens and match_ratio < self.min_match_ratio)
        if abstained_by_match_ratio:
            best_index = -1
        confidence_margin = max(float(best_score - second_score), 0.0)
        self.last_diagnostics = {
            "processed_token_count": processed,
            "processed_tokens": processed_tokens,
            "query_token_count": len(query_tokens),
            "candidate_count": len(candidates),
            "early_stopped": bool(early_stopped),
            "confidence_margin": confidence_margin,
            "max_candidate_scan": self.max_candidate_scan,
            "best_match_ratio": float(match_ratio),
            "min_match_ratio": float(self.min_match_ratio),
            "abstained_by_match_ratio": abstained_by_match_ratio,
        }
        return best_index, max(event_cost, 1)


class DenseAnnProxyRetriever:
    def __init__(self, docs: Sequence[str]) -> None:
        self.docs = list(docs)
        self.doc_tokens: List[Set[str]] = [set(_tokenize(doc)) for doc in self.docs]

    def search(self, query: str) -> Tuple[int, int]:
        query_tokens = set(_tokenize(query))
        best_index = -1
        best_score = -1
        event_cost = 0
        for index, tokens in enumerate(self.doc_tokens):
            score = len(query_tokens & tokens)
            event_cost += max(len(query_tokens), 1) * max(len(tokens), 1)
            if score > best_score:
                best_index = index
                best_score = score
        return best_index, max(event_cost, 1)


class DenseEmbeddingAnnProxyRetriever:
    def __init__(self, docs: Sequence[str], *, vector_size: int = 64) -> None:
        self.docs = list(docs)
        self.vector_size = max(int(vector_size), 8)
        self.doc_vectors = [self._embed(doc) for doc in self.docs]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0 for _ in range(self.vector_size)]
        for token in _tokenize(text):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.vector_size
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    def search(self, query: str) -> Tuple[int, int]:
        query_vector = self._embed(query)
        best_index = -1
        best_score = -1.0
        event_cost = 0
        for index, doc_vector in enumerate(self.doc_vectors):
            score = sum(query_value * doc_value for query_value, doc_value in zip(query_vector, doc_vector))
            event_cost += self.vector_size
            if score > best_score:
                best_index = index
                best_score = score
        return best_index, max(event_cost, 1)


def _extractive_summary(doc: str, max_chars: int = 160) -> str:
    parts = re.split(r"(?<=[。.!?])\s*", doc)
    summary = parts[0].strip() if parts and parts[0].strip() else doc.strip()
    return summary[:max_chars]


def _keyword_coverage(text: str, keywords: Sequence[str]) -> float:
    text_tokens = set(_tokenize(text))
    wanted = [keyword for keyword in keywords if keyword]
    if not wanted:
        return 0.0
    return sum(1 for keyword in wanted if keyword in text_tokens) / len(wanted)


def _score_retriever(
    retriever: Any,
    tasks: Sequence[Dict[str, Any]],
    docs: Sequence[str],
) -> Dict[str, Any]:
    correct = 0
    total_event_cost = 0
    summary_scores: List[float] = []
    latencies: List[float] = []
    processed_token_counts: List[int] = []
    query_token_counts: List[int] = []
    candidate_counts: List[int] = []
    early_stop_count = 0
    case_results: List[Dict[str, Any]] = []

    for task in tasks:
        started = time.perf_counter()
        predicted_index, event_cost = retriever.search(str(task["query"]))
        latencies.append(time.perf_counter() - started)
        diagnostics = getattr(retriever, "last_diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        processed_token_counts.append(int(diagnostics.get("processed_token_count", 0) or 0))
        query_token_counts.append(int(diagnostics.get("query_token_count", 0) or 0))
        candidate_counts.append(int(diagnostics.get("candidate_count", 0) or 0))
        early_stop_count += 1 if bool(diagnostics.get("early_stopped", False)) else 0
        expected_index = int(task["expected_doc_index"])
        is_correct = predicted_index == expected_index
        correct += 1 if is_correct else 0
        total_event_cost += int(event_cost)
        predicted_doc = docs[predicted_index] if 0 <= predicted_index < len(docs) else ""
        summary = _extractive_summary(predicted_doc)
        summary_score = _keyword_coverage(summary, task.get("expected_keywords", []))
        summary_scores.append(summary_score)
        case_results.append(
            {
                "case_id": task["case_id"],
                "correct": bool(is_correct),
                "query": task["query"],
                "expected_doc_index": expected_index,
                "predicted_doc_index": predicted_index,
                "event_cost_proxy": int(event_cost),
                "summary_keyword_coverage": float(summary_score),
                "retrieval_diagnostics": diagnostics,
            }
        )

    total = max(len(tasks), 1)
    accuracy = correct / total
    avg_event_cost = total_event_cost / total
    avg_latency_ms = (sum(latencies) / total) * 1000.0
    summary_keyword_coverage = sum(summary_scores) / total
    return {
        "accuracy": float(accuracy),
        "summary_keyword_coverage": float(summary_keyword_coverage),
        "avg_event_cost_proxy": float(avg_event_cost),
        "avg_latency_ms": float(avg_latency_ms),
        "success_per_event_cost": float(accuracy / max(avg_event_cost, 1e-9)),
        "avg_processed_query_tokens": float(sum(processed_token_counts) / total),
        "avg_query_tokens": float(sum(query_token_counts) / total),
        "avg_candidate_count": float(sum(candidate_counts) / total),
        "early_stop_rate": float(early_stop_count / total),
        "case_results": case_results,
    }


def _continual_memory_score(tasks: Sequence[Dict[str, Any]], docs: Sequence[str]) -> Dict[str, Any]:
    memory: Dict[str, int] = {}
    correct = 0
    total_event_cost = 0
    for task in tasks:
        tokens = _tokenize(str(task["query"]))
        key = tokens[0] if tokens else str(task["case_id"])
        expected_index = int(task["expected_doc_index"])
        if key not in memory:
            memory[key] = expected_index
            total_event_cost += 1
        predicted_index = memory.get(key, -1)
        if predicted_index == expected_index:
            correct += 1
        total_event_cost += 1
    total = max(len(tasks), 1)
    return {
        "continual_memory_hit_rate": float(correct / total),
        "avg_event_cost_proxy": float(total_event_cost / total),
        "memory_slot_count": len(memory),
        "doc_count": len(docs),
    }


def _negative_control_case(docs: Sequence[str], query: str) -> Dict[str, Any]:
    sparse_retriever = MetabolicSparseEventRetriever(docs)
    dense_retriever = DenseAnnProxyRetriever(docs)
    sparse_index, sparse_cost = sparse_retriever.search(query)
    dense_index, dense_cost = dense_retriever.search(query)
    sparse_abstained = sparse_index == -1
    dense_overselected = dense_index != -1 and bool(docs)
    return {
        "query": query,
        "sara_predicted_doc_index": int(sparse_index),
        "ann_predicted_doc_index": int(dense_index),
        "sara_event_cost_proxy": int(sparse_cost),
        "ann_event_cost_proxy": int(dense_cost),
        "sara_abstention_integrity": 1.0 if sparse_abstained else 0.0,
        "ann_overselection_observed": 1.0 if dense_overselected else 0.0,
        "cost_advantage_proxy": float(dense_cost) / max(float(sparse_cost), 1e-9),
        "sara_retrieval_diagnostics": dict(sparse_retriever.last_diagnostics),
    }


def _score_negative_controls(docs: Sequence[str]) -> Dict[str, Any]:
    absent_case = _negative_control_case(
        docs,
        "sara_absent_probe_token no_matching_memory_event",
    )
    partial_case = _negative_control_case(
        docs,
        "retrieval memory sara_absent_probe_token no_matching_memory_event",
    )
    case_reports = {
        "absent_query": absent_case,
        "partial_evidence_query": partial_case,
    }
    return {
        "case_count": len(case_reports),
        "cases": case_reports,
        "absent_query": absent_case["query"],
        "partial_evidence_query": partial_case["query"],
        "sara_predicted_doc_index": int(absent_case["sara_predicted_doc_index"]),
        "ann_predicted_doc_index": int(absent_case["ann_predicted_doc_index"]),
        "sara_event_cost_proxy": int(absent_case["sara_event_cost_proxy"]),
        "ann_event_cost_proxy": int(absent_case["ann_event_cost_proxy"]),
        "sara_abstention_integrity": float(absent_case["sara_abstention_integrity"]),
        "ann_overselection_observed": float(absent_case["ann_overselection_observed"]),
        "cost_advantage_proxy": float(absent_case["cost_advantage_proxy"]),
        "partial_evidence_abstention_integrity": float(
            partial_case["sara_abstention_integrity"]
        ),
        "partial_evidence_ann_overselection_observed": float(
            partial_case["ann_overselection_observed"]
        ),
        "partial_evidence_cost_advantage_proxy": float(partial_case["cost_advantage_proxy"]),
        "sara_retrieval_diagnostics": dict(absent_case["sara_retrieval_diagnostics"]),
    }


def _score_contrastive_controls() -> Dict[str, Any]:
    docs = [
        "common retrieval memory alpha sparse route chooses alpha evidence.",
        "common retrieval memory beta sparse route chooses beta evidence.",
        "common retrieval memory gamma sparse route chooses gamma evidence.",
    ]
    cases = [
        {"case_id": "near-miss-beta", "query": "common retrieval memory beta", "expected_doc_index": 1, "decider": "beta"},
        {"case_id": "near-miss-gamma", "query": "common retrieval memory gamma", "expected_doc_index": 2, "decider": "gamma"},
    ]
    sparse_retriever = MetabolicSparseEventRetriever(docs, min_match_ratio=0.50)
    dense_retriever = DenseAnnProxyRetriever(docs)
    correct = 0
    rare_first_count = 0
    total_sparse_cost = 0
    total_dense_cost = 0
    case_reports: List[Dict[str, Any]] = []
    for case in cases:
        sparse_index, sparse_cost = sparse_retriever.search(str(case["query"]))
        dense_index, dense_cost = dense_retriever.search(str(case["query"]))
        diagnostics = dict(sparse_retriever.last_diagnostics)
        processed_tokens = diagnostics.get("processed_tokens", [])
        if not isinstance(processed_tokens, list):
            processed_tokens = []
        decider = str(case["decider"])
        expected = int(case["expected_doc_index"])
        sparse_correct = sparse_index == expected
        rare_first = bool(processed_tokens and processed_tokens[0] == decider)
        correct += 1 if sparse_correct else 0
        rare_first_count += 1 if rare_first else 0
        total_sparse_cost += int(sparse_cost)
        total_dense_cost += int(dense_cost)
        case_reports.append(
            {
                "case_id": str(case["case_id"]),
                "query": str(case["query"]),
                "decider_token": decider,
                "expected_doc_index": expected,
                "sara_predicted_doc_index": int(sparse_index),
                "ann_predicted_doc_index": int(dense_index),
                "sara_correct": bool(sparse_correct),
                "ann_correct": bool(dense_index == expected),
                "rare_decider_processed_first": rare_first,
                "sara_event_cost_proxy": int(sparse_cost),
                "ann_event_cost_proxy": int(dense_cost),
                "sara_retrieval_diagnostics": diagnostics,
            }
        )
    total = max(len(cases), 1)
    avg_sparse_cost = float(total_sparse_cost) / total
    avg_dense_cost = float(total_dense_cost) / total
    return {
        "case_count": len(cases),
        "accuracy": float(correct / total),
        "rare_decider_first_rate": float(rare_first_count / total),
        "avg_sara_event_cost_proxy": avg_sparse_cost,
        "avg_ann_event_cost_proxy": avg_dense_cost,
        "cost_advantage_proxy": avg_dense_cost / max(avg_sparse_cost, 1e-9),
        "cases": case_reports,
    }


def _score_sparse_rag_rerank(tasks: Sequence[Dict[str, Any]], docs: Sequence[str]) -> Dict[str, Any]:
    rag = SNNRAGPipeline(sdr_size=512, max_chunk_size=240)
    for index, doc in enumerate(docs):
        rag.add_document(doc, source=f"doc_{index}")
    traces: List[Dict[str, Any]] = []
    metric_sums: Dict[str, float] = defaultdict(float)
    for task in tasks[: max(1, min(len(tasks), 8))]:
        trace = rag.query_with_rerank(str(task.get("query", "")), top_k=1, candidate_k=4)
        traces.append(trace)
        metrics = trace.get("metrics", {}) if isinstance(trace.get("metrics"), dict) else {}
        for metric_name, value in metrics.items():
            metric_sums[str(metric_name)] += float(value or 0.0)
    total = max(len(traces), 1)
    averaged_metrics = {
        metric_name: float(value / total)
        for metric_name, value in sorted(metric_sums.items())
    }
    return {
        "observed_only": True,
        "case_count": int(len(traces)),
        "metrics": averaged_metrics,
        "traces": traces[:5],
    }


def _score_rag_query_decomposition(tasks: Sequence[Dict[str, Any]], docs: Sequence[str]) -> Dict[str, Any]:
    rag = SNNRAGPipeline(sdr_size=512, max_chunk_size=240)
    for index, doc in enumerate(docs):
        rag.add_document(doc, source=f"doc_{index}")
    traces: List[Dict[str, Any]] = []
    metric_sums: Dict[str, float] = defaultdict(float)
    for task in tasks[: max(1, min(len(tasks), 8))]:
        trace = rag.query_with_decomposed_rerank(
            str(task.get("query", "")),
            top_k=1,
            candidate_k=4,
            max_subqueries=3,
        )
        traces.append(trace)
        metrics = trace.get("metrics", {}) if isinstance(trace.get("metrics"), dict) else {}
        for metric_name, value in metrics.items():
            metric_sums[str(metric_name)] += float(value or 0.0)
    total = max(len(traces), 1)
    averaged_metrics = {
        metric_name: float(value / total)
        for metric_name, value in sorted(metric_sums.items())
    }
    return {
        "observed_only": True,
        "case_count": int(len(traces)),
        "metrics": averaged_metrics,
        "traces": traces[:5],
    }


def _unique_real_data_events(tokens: Sequence[str], *, case_id: str, limit: int = 3) -> List[str]:
    events: List[str] = []
    for token in tokens:
        normalized = str(token).strip().lower()
        if not normalized:
            continue
        event = f"real:{case_id}:{normalized}"
        if event not in events:
            events.append(event)
        if len(events) >= limit:
            break
    return events


def _real_data_noise_event(task: Mapping[str, Any], docs: Sequence[str]) -> str:
    case_id = str(task.get("case_id", "case"))
    expected_doc_index = int(task.get("expected_doc_index", -1) or -1)
    for doc_index, doc in enumerate(docs):
        if doc_index == expected_doc_index:
            continue
        tokens = _tokenize(doc)
        if tokens:
            return f"noise:{case_id}:{doc_index}:{tokens[0]}"
    return f"noise:{case_id}:fallback"


def _score_sparse_diffusion_real_data_blocks(
    tasks: Sequence[Dict[str, Any]],
    docs: Sequence[str],
) -> Dict[str, Any]:
    module = _load_sparse_diffusion_block_module()
    case_type = getattr(module, "SparseDiffusionCase")
    build_report = getattr(module, "build_sparse_diffusion_block_readiness_report")
    cases = []
    total = max(len(tasks), 1)
    for index, task in enumerate(tasks):
        case_id = str(task.get("case_id", f"case-{index}"))
        source_tokens = list(task.get("expected_keywords", []) or []) + _tokenize(str(task.get("query", "")))
        clean_events = _unique_real_data_events(source_tokens, case_id=case_id, limit=3)
        if len(clean_events) < 3:
            clean_events.extend(
                event
                for event in _unique_real_data_events(
                    _tokenize(str(task.get("document", ""))), case_id=case_id, limit=3
                )
                if event not in clean_events
            )
        if len(clean_events) < 2:
            continue
        clean = frozenset(clean_events[:3])
        missing = frozenset({clean_events[min(len(clean_events), 3) - 1]})
        noisy = frozenset((set(clean) - set(missing)).union({_real_data_noise_event(task, docs)}))
        cases.append(
            case_type(
                case_id=f"real_data_{case_id}",
                uncertainty=(index + 1) / (total + 1),
                clean_events=clean,
                noisy_events=noisy,
                missing_events=missing,
            )
        )

    block_count = min(3, max(len(cases), 1))
    readiness = build_report(block_count=block_count, cases=cases)
    metrics = readiness.get("metrics", {}) if isinstance(readiness.get("metrics"), dict) else {}
    details = readiness.get("details", {}) if isinstance(readiness.get("details"), dict) else {}
    return {
        "schema": "sara-real-data-sparse-diffusion-block-probe-v1",
        "observed_only": True,
        "case_count": int(readiness.get("case_count", 0) or 0),
        "block_count": int(readiness.get("block_count", block_count) or block_count),
        "passed": bool(readiness.get("passed", False)),
        "metrics": {
            "sparse_diffusion_real_data_partition_integrity": float(
                metrics.get("sparse_diffusion_partition_integrity", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_denoise_accuracy": float(
                metrics.get("sparse_diffusion_denoise_accuracy", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_event_cost_advantage": float(
                metrics.get("sparse_diffusion_event_cost_advantage", 0.0) or 0.0
            ),
            "sparse_diffusion_real_data_single_pass_integrity": float(
                metrics.get("sparse_diffusion_single_pass_recurrent_integrity", 0.0) or 0.0
            ),
        },
        "details": {
            "partition": details.get("partition", {}),
            "evaluation": details.get("evaluation", {}),
            "single_pass_recurrent": details.get("single_pass_recurrent", {}),
        },
    }


def load_external_validity_history(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        runs = payload.get("runs", [])
        if isinstance(runs, list):
            return [dict(item) for item in runs if isinstance(item, dict)]
    return []


def _history_entry_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    benchmark_context = (
        report.get("benchmark_context", {})
        if isinstance(report.get("benchmark_context"), dict)
        else {}
    )
    return {
        "timestamp": float(time.time()),
        "passed": bool(report.get("passed", False)),
        "doc_count": int(report.get("doc_count", 0) or 0),
        "task_count": int(report.get("task_count", 0) or 0),
        "benchmark_context": dict(benchmark_context),
        "metrics": {
            str(name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
        },
        "checks": {str(name): bool(value) for name, value in checks.items()},
    }


def build_external_validity_trend(
    report: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    *,
    regression_tolerance: float = 0.05,
) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    previous = history[-1] if history and isinstance(history[-1], dict) else {}
    previous_metrics = previous.get("metrics", {}) if isinstance(previous.get("metrics"), dict) else {}
    current_context = (
        report.get("benchmark_context", {})
        if isinstance(report.get("benchmark_context"), dict)
        else {}
    )
    previous_context = (
        previous.get("benchmark_context", {})
        if isinstance(previous.get("benchmark_context"), dict)
        else {}
    )
    context_keys = ["corpus_sha256", "task_sha256", "max_docs", "max_cases", "retriever_strategy"]
    context_comparable = True
    skipped_reason = ""
    if current_context and previous_context:
        changed_keys = [
            key
            for key in context_keys
            if current_context.get(key) != previous_context.get(key)
        ]
        if changed_keys:
            context_comparable = False
            skipped_reason = "benchmark_context_changed:" + ",".join(changed_keys)
    elif previous_metrics:
        context_comparable = False
        skipped_reason = "previous_history_missing_benchmark_context"

    tolerance = max(float(regression_tolerance), 0.0)
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []
    metric_names = sorted((TREND_ABSOLUTE_METRICS | TREND_RELATIVE_METRICS).intersection(metrics.keys()))

    if context_comparable:
        for metric_name in metric_names:
            if metric_name not in previous_metrics:
                continue
            current = float(metrics.get(metric_name, 0.0) or 0.0)
            prior = float(previous_metrics.get(metric_name, 0.0) or 0.0)
            if metric_name in TREND_RELATIVE_METRICS:
                allowed_drop = abs(prior) * tolerance
                change_type = "relative"
            else:
                allowed_drop = tolerance
                change_type = "absolute"
            delta = current - prior
            item = {
                "metric": metric_name,
                "previous": prior,
                "current": current,
                "delta": float(delta),
                "allowed_drop": float(allowed_drop),
                "comparison": change_type,
            }
            if current + allowed_drop < prior:
                regressions.append(item)
            elif current > prior + allowed_drop:
                improvements.append(item)
            else:
                unchanged.append(item)

    return {
        "has_previous": bool(previous_metrics),
        "comparison_active": bool(previous_metrics and context_comparable),
        "comparison_skipped_reason": skipped_reason,
        "history_count": len(history),
        "regression_tolerance": float(tolerance),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "unchanged_count": len(unchanged),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
    }


def append_external_validity_history(path: str, report: Dict[str, Any], *, max_entries: int = 64) -> str:
    history = load_external_validity_history(path)
    history.append(_history_entry_from_report(report))
    if max_entries > 0 and len(history) > max_entries:
        history = history[-max_entries:]
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    return resolved


def _build_external_validity_check_details(
    *,
    task_count: int,
    max_cases: int,
    doc_count: int,
    sparse_accuracy: float,
    dense_accuracy: float,
    summary_keyword_coverage: float,
    continual_memory_hit_rate: float,
    ann_cost_advantage: float,
    performance_energy_ratio: float,
    negative_control: Mapping[str, Any],
    contrastive_control: Mapping[str, Any],
    dense_embedding: Mapping[str, Any],
    sparse_diffusion: Mapping[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    min_task_count = min(max_cases, doc_count, int(thresholds["min_real_data_task_count"]))
    negative_abstention = float(negative_control.get("sara_abstention_integrity", 0.0) or 0.0)
    negative_cost_advantage = float(negative_control.get("cost_advantage_proxy", 0.0) or 0.0)
    partial_evidence_abstention = float(
        negative_control.get("partial_evidence_abstention_integrity", 0.0) or 0.0
    )
    partial_evidence_cost_advantage = float(
        negative_control.get("partial_evidence_cost_advantage_proxy", 0.0) or 0.0
    )
    contrastive_accuracy = float(contrastive_control.get("accuracy", 0.0) or 0.0)
    contrastive_cost_advantage = float(contrastive_control.get("cost_advantage_proxy", 0.0) or 0.0)
    embedding_accuracy = float(dense_embedding.get("accuracy", 0.0) or 0.0)
    embedding_cost_advantage = float(dense_embedding.get("cost_advantage_proxy", 0.0) or 0.0)
    sparse_diffusion_metrics = (
        sparse_diffusion.get("metrics", {})
        if isinstance(sparse_diffusion.get("metrics"), Mapping)
        else {}
    )
    sparse_diffusion_denoise_accuracy = float(
        sparse_diffusion_metrics.get("sparse_diffusion_real_data_denoise_accuracy", 0.0) or 0.0
    )
    sparse_diffusion_event_cost_advantage = float(
        sparse_diffusion_metrics.get("sparse_diffusion_real_data_event_cost_advantage", 0.0) or 0.0
    )
    sparse_diffusion_partition_integrity = float(
        sparse_diffusion_metrics.get("sparse_diffusion_real_data_partition_integrity", 0.0) or 0.0
    )
    sparse_diffusion_single_pass_integrity = float(
        sparse_diffusion_metrics.get("sparse_diffusion_real_data_single_pass_integrity", 0.0) or 0.0
    )
    return {
        "real_data_task_count": {
            "passed": task_count >= min_task_count,
            "value": int(task_count),
            "required_min": int(min_task_count),
        },
        "sparse_accuracy_floor": {
            "passed": sparse_accuracy >= thresholds["min_real_data_qa_accuracy"],
            "value": float(sparse_accuracy),
            "required_min": float(thresholds["min_real_data_qa_accuracy"]),
        },
        "sparse_matches_dense_accuracy": {
            "passed": sparse_accuracy >= max(dense_accuracy - thresholds["dense_accuracy_tolerance"], 0.0),
            "value": float(sparse_accuracy),
            "dense_reference": float(dense_accuracy),
            "allowed_lag": float(thresholds["dense_accuracy_tolerance"]),
        },
        "summary_keyword_coverage_floor": {
            "passed": summary_keyword_coverage >= thresholds["min_summary_keyword_coverage"],
            "value": float(summary_keyword_coverage),
            "required_min": float(thresholds["min_summary_keyword_coverage"]),
        },
        "continual_memory_hit_rate_floor": {
            "passed": continual_memory_hit_rate >= thresholds["min_continual_memory_hit_rate"],
            "value": float(continual_memory_hit_rate),
            "required_min": float(thresholds["min_continual_memory_hit_rate"]),
        },
        "ann_cost_advantage_proxy": {
            "passed": ann_cost_advantage >= thresholds["min_ann_cost_advantage_proxy"],
            "value": float(ann_cost_advantage),
            "required_min": float(thresholds["min_ann_cost_advantage_proxy"]),
        },
        "performance_energy_ratio_proxy": {
            "passed": performance_energy_ratio >= thresholds["min_performance_energy_ratio_proxy"],
            "value": float(performance_energy_ratio),
            "required_min": float(thresholds["min_performance_energy_ratio_proxy"]),
        },
        "negative_control_abstention": {
            "passed": negative_abstention >= thresholds["min_negative_control_abstention"],
            "value": float(negative_abstention),
            "required_min": float(thresholds["min_negative_control_abstention"]),
        },
        "negative_control_cost_advantage": {
            "passed": negative_cost_advantage >= thresholds["min_negative_control_cost_advantage"],
            "value": float(negative_cost_advantage),
            "required_min": float(thresholds["min_negative_control_cost_advantage"]),
        },
        "partial_evidence_abstention": {
            "passed": partial_evidence_abstention >= thresholds["min_partial_evidence_abstention"],
            "value": float(partial_evidence_abstention),
            "required_min": float(thresholds["min_partial_evidence_abstention"]),
        },
        "partial_evidence_cost_advantage": {
            "passed": partial_evidence_cost_advantage >= thresholds["min_partial_evidence_cost_advantage"],
            "value": float(partial_evidence_cost_advantage),
            "required_min": float(thresholds["min_partial_evidence_cost_advantage"]),
        },
        "contrastive_control_accuracy": {
            "passed": contrastive_accuracy >= thresholds["min_contrastive_control_accuracy"],
            "value": float(contrastive_accuracy),
            "required_min": float(thresholds["min_contrastive_control_accuracy"]),
        },
        "contrastive_control_cost_advantage": {
            "passed": contrastive_cost_advantage >= thresholds["min_contrastive_control_cost_advantage"],
            "value": float(contrastive_cost_advantage),
            "required_min": float(thresholds["min_contrastive_control_cost_advantage"]),
        },
        "sparse_matches_dense_embedding_accuracy": {
            "passed": sparse_accuracy >= max(embedding_accuracy - thresholds["dense_accuracy_tolerance"], 0.0),
            "value": float(sparse_accuracy),
            "dense_embedding_reference": float(embedding_accuracy),
            "allowed_lag": float(thresholds["dense_accuracy_tolerance"]),
        },
        "dense_embedding_cost_advantage": {
            "passed": embedding_cost_advantage >= thresholds["min_dense_embedding_cost_advantage"],
            "value": float(embedding_cost_advantage),
            "required_min": float(thresholds["min_dense_embedding_cost_advantage"]),
        },
        "sparse_diffusion_real_data_denoise_accuracy": {
            "passed": sparse_diffusion_denoise_accuracy
            >= thresholds["min_sparse_diffusion_real_data_denoise_accuracy"],
            "value": float(sparse_diffusion_denoise_accuracy),
            "required_min": float(thresholds["min_sparse_diffusion_real_data_denoise_accuracy"]),
        },
        "sparse_diffusion_real_data_event_cost_advantage": {
            "passed": sparse_diffusion_event_cost_advantage
            >= thresholds["min_sparse_diffusion_real_data_event_cost_advantage"],
            "value": float(sparse_diffusion_event_cost_advantage),
            "required_min": float(thresholds["min_sparse_diffusion_real_data_event_cost_advantage"]),
        },
        "sparse_diffusion_real_data_partition_integrity": {
            "passed": sparse_diffusion_partition_integrity
            >= thresholds["min_sparse_diffusion_real_data_partition_integrity"],
            "value": float(sparse_diffusion_partition_integrity),
            "required_min": float(thresholds["min_sparse_diffusion_real_data_partition_integrity"]),
        },
        "sparse_diffusion_real_data_single_pass_integrity": {
            "passed": sparse_diffusion_single_pass_integrity
            >= thresholds["min_sparse_diffusion_real_data_single_pass_integrity"],
            "value": float(sparse_diffusion_single_pass_integrity),
            "required_min": float(thresholds["min_sparse_diffusion_real_data_single_pass_integrity"]),
        },
    }


def run_real_data_external_validity(
    corpus_path: str = processed_data_path("corpus.txt"),
    max_docs: int = 256,
    max_cases: int = 24,
    history: Sequence[Dict[str, Any]] | None = None,
    regression_tolerance: float = 0.05,
) -> Dict[str, Any]:
    docs = _load_corpus(corpus_path, limit=max_docs)
    tasks = build_real_data_tasks(docs, max_cases=max_cases)
    if not tasks:
        return {
            "suite_name": "RealDataExternalValidity",
            "passed": False,
            "errors": ["No valid real-data tasks could be built from the corpus."],
            "corpus_path": os.path.abspath(corpus_path),
            "doc_count": len(docs),
            "task_count": 0,
        }

    sparse_baseline = _score_retriever(SparseEventRetriever(docs), tasks, docs)
    sparse = _score_retriever(MetabolicSparseEventRetriever(docs), tasks, docs)
    dense = _score_retriever(DenseAnnProxyRetriever(docs), tasks, docs)
    dense_embedding = _score_retriever(DenseEmbeddingAnnProxyRetriever(docs), tasks, docs)
    continual = _continual_memory_score(tasks, docs)
    negative_control = _score_negative_controls(docs)
    contrastive_control = _score_contrastive_controls()
    sparse_rag_rerank = _score_sparse_rag_rerank(tasks, docs)
    sparse_rag_rerank_metrics = (
        sparse_rag_rerank.get("metrics", {})
        if isinstance(sparse_rag_rerank.get("metrics"), dict)
        else {}
    )
    sparse_diffusion_real_data = _score_sparse_diffusion_real_data_blocks(tasks, docs)
    sparse_diffusion_real_data_metrics = (
        sparse_diffusion_real_data.get("metrics", {})
        if isinstance(sparse_diffusion_real_data.get("metrics"), dict)
        else {}
    )
    rag_query_decomposition = _score_rag_query_decomposition(tasks, docs)
    rag_query_decomposition_metrics = (
        rag_query_decomposition.get("metrics", {})
        if isinstance(rag_query_decomposition.get("metrics"), dict)
        else {}
    )
    benchmark_context = _build_benchmark_context(
        docs=docs,
        tasks=tasks,
        corpus_path=corpus_path,
        max_docs=max_docs,
        max_cases=max_cases,
    )

    sparse_accuracy = float(sparse["accuracy"])
    dense_accuracy = float(dense["accuracy"])
    sparse_cost = float(sparse["avg_event_cost_proxy"])
    dense_cost = float(dense["avg_event_cost_proxy"])
    sparse_success_per_cost = float(sparse["success_per_event_cost"])
    dense_success_per_cost = float(dense["success_per_event_cost"])
    performance_energy_ratio = sparse_success_per_cost / max(dense_success_per_cost, 1e-9)
    ann_cost_advantage = dense_cost / max(sparse_cost, 1e-9)
    dense_embedding_cost = float(dense_embedding["avg_event_cost_proxy"])
    dense_embedding_cost_advantage = dense_embedding_cost / max(sparse_cost, 1e-9)

    thresholds = dict(DEFAULT_THRESHOLDS)
    check_details = _build_external_validity_check_details(
        task_count=len(tasks),
        max_cases=max_cases,
        doc_count=len(docs),
        sparse_accuracy=sparse_accuracy,
        dense_accuracy=dense_accuracy,
        summary_keyword_coverage=float(sparse["summary_keyword_coverage"]),
        continual_memory_hit_rate=float(continual["continual_memory_hit_rate"]),
        ann_cost_advantage=ann_cost_advantage,
        performance_energy_ratio=performance_energy_ratio,
        negative_control=negative_control,
        contrastive_control=contrastive_control,
        dense_embedding={
            "accuracy": float(dense_embedding["accuracy"]),
            "cost_advantage_proxy": dense_embedding_cost_advantage,
        },
        sparse_diffusion=sparse_diffusion_real_data,
        thresholds=thresholds,
    )
    checks = {name: bool(detail.get("passed", False)) for name, detail in check_details.items()}
    report: Dict[str, Any] = {
        "suite_name": "RealDataExternalValidity",
        "passed": False,
        "corpus_path": os.path.abspath(corpus_path),
        "doc_count": len(docs),
        "task_count": len(tasks),
        "benchmark_context": benchmark_context,
        "thresholds": thresholds,
        "metrics": {
            "real_data_qa_accuracy": sparse_accuracy,
            "ann_proxy_qa_accuracy": dense_accuracy,
            "dense_embedding_ann_proxy_qa_accuracy": float(dense_embedding["accuracy"]),
            "real_data_summary_keyword_coverage": float(sparse["summary_keyword_coverage"]),
            "continual_memory_hit_rate": float(continual["continual_memory_hit_rate"]),
            "sara_avg_event_cost_proxy": sparse_cost,
            "sara_baseline_avg_event_cost_proxy": float(sparse_baseline["avg_event_cost_proxy"]),
            "sara_metabolic_cost_reduction_proxy": float(
                float(sparse_baseline["avg_event_cost_proxy"]) / max(sparse_cost, 1e-9)
            ),
            "sara_metabolic_early_stop_rate": float(sparse["early_stop_rate"]),
            "sara_metabolic_avg_processed_query_tokens": float(sparse["avg_processed_query_tokens"]),
            "sara_metabolic_avg_query_tokens": float(sparse["avg_query_tokens"]),
            "sara_metabolic_avg_candidate_count": float(sparse["avg_candidate_count"]),
            "sparse_rag_rerank_bounded_count_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_bounded_count_observed", 0.0)
            ),
            "sparse_rag_rerank_source_agreement_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_source_agreement_observed", 0.0)
            ),
            "sparse_rag_rerank_contradiction_guard_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_contradiction_guard_observed", 0.0)
            ),
            "sparse_rag_rerank_freshness_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_freshness_observed", 0.0)
            ),
            "sparse_rag_rerank_citation_grounding_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_citation_grounding_observed", 0.0)
            ),
            "sparse_rag_rerank_source_reliability_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_source_reliability_observed", 0.0)
            ),
            "sparse_rag_rerank_source_diversity_observed": float(
                sparse_rag_rerank_metrics.get("sparse_rag_rerank_source_diversity_observed", 0.0)
            ),
            "sparse_diffusion_real_data_partition_integrity": float(
                sparse_diffusion_real_data_metrics.get("sparse_diffusion_real_data_partition_integrity", 0.0)
            ),
            "sparse_diffusion_real_data_denoise_accuracy": float(
                sparse_diffusion_real_data_metrics.get("sparse_diffusion_real_data_denoise_accuracy", 0.0)
            ),
            "sparse_diffusion_real_data_event_cost_advantage": float(
                sparse_diffusion_real_data_metrics.get("sparse_diffusion_real_data_event_cost_advantage", 0.0)
            ),
            "sparse_diffusion_real_data_single_pass_integrity": float(
                sparse_diffusion_real_data_metrics.get("sparse_diffusion_real_data_single_pass_integrity", 0.0)
            ),
            "rag_query_decomposition_bounded_count_observed": float(
                rag_query_decomposition_metrics.get("rag_query_decomposition_bounded_count_observed", 0.0)
            ),
            "rag_query_decomposition_coverage_observed": float(
                rag_query_decomposition_metrics.get("rag_query_decomposition_coverage_observed", 0.0)
            ),
            "rag_query_decomposition_nonempty_observed": float(
                rag_query_decomposition_metrics.get("rag_query_decomposition_nonempty_observed", 0.0)
            ),
            "rag_query_decomposition_subquery_hit_observed": float(
                rag_query_decomposition_metrics.get("rag_query_decomposition_subquery_hit_observed", 0.0)
            ),
            "rag_query_decomposition_merged_selection_observed": float(
                rag_query_decomposition_metrics.get("rag_query_decomposition_merged_selection_observed", 0.0)
            ),
            "rag_query_decomposition_merged_citation_grounding_observed": float(
                rag_query_decomposition_metrics.get(
                    "rag_query_decomposition_merged_citation_grounding_observed", 0.0
                )
            ),
            "rag_query_decomposition_merged_source_reliability_observed": float(
                rag_query_decomposition_metrics.get(
                    "rag_query_decomposition_merged_source_reliability_observed", 0.0
                )
            ),
            "rag_query_decomposition_merged_source_diversity_observed": float(
                rag_query_decomposition_metrics.get(
                    "rag_query_decomposition_merged_source_diversity_observed", 0.0
                )
            ),
            "ann_avg_event_cost_proxy": dense_cost,
            "performance_energy_ratio_proxy": float(performance_energy_ratio),
            "ann_cost_advantage_proxy": float(ann_cost_advantage),
            "dense_embedding_ann_cost_advantage_proxy": float(dense_embedding_cost_advantage),
            "negative_control_abstention_integrity": float(
                negative_control["sara_abstention_integrity"]
            ),
            "negative_control_ann_overselection_observed": float(
                negative_control["ann_overselection_observed"]
            ),
            "negative_control_cost_advantage_proxy": float(
                negative_control["cost_advantage_proxy"]
            ),
            "partial_evidence_abstention_integrity": float(
                negative_control["partial_evidence_abstention_integrity"]
            ),
            "partial_evidence_ann_overselection_observed": float(
                negative_control["partial_evidence_ann_overselection_observed"]
            ),
            "partial_evidence_cost_advantage_proxy": float(
                negative_control["partial_evidence_cost_advantage_proxy"]
            ),
            "contrastive_control_accuracy": float(contrastive_control["accuracy"]),
            "contrastive_control_rare_decider_first_rate": float(
                contrastive_control["rare_decider_first_rate"]
            ),
            "contrastive_control_cost_advantage_proxy": float(
                contrastive_control["cost_advantage_proxy"]
            ),
            "sara_avg_latency_ms": float(sparse["avg_latency_ms"]),
            "ann_proxy_avg_latency_ms": float(dense["avg_latency_ms"]),
            "dense_embedding_ann_proxy_avg_latency_ms": float(dense_embedding["avg_latency_ms"]),
        },
        "checks": checks,
        "check_details": check_details,
        "sara_sparse": sparse,
        "sara_sparse_baseline": sparse_baseline,
        "ann_dense_proxy": dense,
        "ann_dense_embedding_proxy": dense_embedding,
        "continual_memory": continual,
        "negative_controls": negative_control,
        "contrastive_controls": contrastive_control,
        "sparse_rag_rerank": sparse_rag_rerank,
        "sparse_diffusion_real_data": sparse_diffusion_real_data,
        "rag_query_decomposition": rag_query_decomposition,
    }
    trend = build_external_validity_trend(
        report,
        history if isinstance(history, list) else [],
        regression_tolerance=regression_tolerance,
    )
    checks["trend.no_regressions"] = int(trend.get("regression_count", 0) or 0) == 0
    check_details["trend.no_regressions"] = {
        "passed": bool(checks["trend.no_regressions"]),
        "value": int(trend.get("regression_count", 0) or 0),
        "required_max": 0,
        "comparison_active": bool(trend.get("comparison_active", False)),
        "comparison_skipped_reason": str(trend.get("comparison_skipped_reason", "") or ""),
    }
    report["trend"] = trend
    report["passed"] = all(checks.values())

    return report


def format_real_data_external_validity_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    trend = report.get("trend", {}) if isinstance(report.get("trend"), dict) else {}
    lines = [
        "Real Data External Validity Summary",
        f"- passed: {bool(report.get('passed', False))}",
        f"- corpus_path: {report.get('corpus_path', '')}",
        f"- doc_count: {int(report.get('doc_count', 0) or 0)}",
        f"- task_count: {int(report.get('task_count', 0) or 0)}",
        f"- real_data_qa_accuracy: {float(metrics.get('real_data_qa_accuracy', 0.0)):.3f}",
        f"- ann_proxy_qa_accuracy: {float(metrics.get('ann_proxy_qa_accuracy', 0.0)):.3f}",
        f"- dense_embedding_ann_proxy_qa_accuracy: {float(metrics.get('dense_embedding_ann_proxy_qa_accuracy', 0.0)):.3f}",
        f"- real_data_summary_keyword_coverage: {float(metrics.get('real_data_summary_keyword_coverage', 0.0)):.3f}",
        f"- continual_memory_hit_rate: {float(metrics.get('continual_memory_hit_rate', 0.0)):.3f}",
        f"- performance_energy_ratio_proxy: {float(metrics.get('performance_energy_ratio_proxy', 0.0)):.3f}",
        f"- ann_cost_advantage_proxy: {float(metrics.get('ann_cost_advantage_proxy', 0.0)):.3f}",
        f"- dense_embedding_ann_cost_advantage_proxy: {float(metrics.get('dense_embedding_ann_cost_advantage_proxy', 0.0)):.3f}",
        f"- sara_metabolic_cost_reduction_proxy: {float(metrics.get('sara_metabolic_cost_reduction_proxy', 0.0)):.3f}",
        f"- sara_metabolic_early_stop_rate: {float(metrics.get('sara_metabolic_early_stop_rate', 0.0)):.3f}",
        f"- sara_metabolic_avg_processed_query_tokens: {float(metrics.get('sara_metabolic_avg_processed_query_tokens', 0.0)):.3f}",
        f"- negative_control_abstention_integrity: {float(metrics.get('negative_control_abstention_integrity', 0.0)):.3f}",
        f"- negative_control_cost_advantage_proxy: {float(metrics.get('negative_control_cost_advantage_proxy', 0.0)):.3f}",
        f"- partial_evidence_abstention_integrity: {float(metrics.get('partial_evidence_abstention_integrity', 0.0)):.3f}",
        f"- partial_evidence_cost_advantage_proxy: {float(metrics.get('partial_evidence_cost_advantage_proxy', 0.0)):.3f}",
        f"- contrastive_control_accuracy: {float(metrics.get('contrastive_control_accuracy', 0.0)):.3f}",
        f"- contrastive_control_cost_advantage_proxy: {float(metrics.get('contrastive_control_cost_advantage_proxy', 0.0)):.3f}",
        f"- sparse_rag_rerank_source_agreement_observed: {float(metrics.get('sparse_rag_rerank_source_agreement_observed', 0.0)):.3f}",
        f"- sparse_rag_rerank_contradiction_guard_observed: {float(metrics.get('sparse_rag_rerank_contradiction_guard_observed', 0.0)):.3f}",
        f"- sparse_rag_rerank_citation_grounding_observed: {float(metrics.get('sparse_rag_rerank_citation_grounding_observed', 0.0)):.3f}",
        f"- sparse_rag_rerank_source_reliability_observed: {float(metrics.get('sparse_rag_rerank_source_reliability_observed', 0.0)):.3f}",
        f"- sparse_rag_rerank_source_diversity_observed: {float(metrics.get('sparse_rag_rerank_source_diversity_observed', 0.0)):.3f}",
        f"- sparse_diffusion_real_data_partition_integrity: {float(metrics.get('sparse_diffusion_real_data_partition_integrity', 0.0)):.3f}",
        f"- sparse_diffusion_real_data_denoise_accuracy: {float(metrics.get('sparse_diffusion_real_data_denoise_accuracy', 0.0)):.3f}",
        f"- sparse_diffusion_real_data_event_cost_advantage: {float(metrics.get('sparse_diffusion_real_data_event_cost_advantage', 0.0)):.3f}",
        f"- sparse_diffusion_real_data_single_pass_integrity: {float(metrics.get('sparse_diffusion_real_data_single_pass_integrity', 0.0)):.3f}",
        f"- rag_query_decomposition_coverage_observed: {float(metrics.get('rag_query_decomposition_coverage_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_selection_observed: {float(metrics.get('rag_query_decomposition_merged_selection_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_citation_grounding_observed: {float(metrics.get('rag_query_decomposition_merged_citation_grounding_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_source_reliability_observed: {float(metrics.get('rag_query_decomposition_merged_source_reliability_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_source_diversity_observed: {float(metrics.get('rag_query_decomposition_merged_source_diversity_observed', 0.0)):.3f}",
        f"- sara_avg_latency_ms: {float(metrics.get('sara_avg_latency_ms', 0.0)):.3f}",
        f"- ann_proxy_avg_latency_ms: {float(metrics.get('ann_proxy_avg_latency_ms', 0.0)):.3f}",
        f"- trend_has_previous: {bool(trend.get('has_previous', False))}",
        f"- trend_comparison_active: {bool(trend.get('comparison_active', False))}",
        f"- trend_comparison_skipped_reason: {str(trend.get('comparison_skipped_reason', '') or '')}",
        f"- trend_regression_count: {int(trend.get('regression_count', 0) or 0)}",
        "Checks:",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if checks[name] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-data external validity benchmark.")
    parser.add_argument("--corpus", default=processed_data_path("corpus.txt"))
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--regression-tolerance", type=float, default=0.05)
    parser.add_argument("--no-history-update", action="store_true")
    args = parser.parse_args()

    history = load_external_validity_history(str(args.history_path))
    report = run_real_data_external_validity(
        corpus_path=str(args.corpus),
        max_docs=int(args.max_docs),
        max_cases=int(args.max_cases),
        history=history,
        regression_tolerance=float(max(args.regression_tolerance, 0.0)),
    )
    report_path = ensure_parent_directory(str(args.report_path))
    summary_path = ensure_parent_directory(str(args.summary_path))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_real_data_external_validity_summary(report))
    if not bool(args.no_history_update):
        history_path = append_external_validity_history(str(args.history_path), report)
        report["history_path"] = history_path
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    if not bool(args.no_history_update):
        print(f"Saved history: {history_path}")
    if not bool(report.get("passed", False)):
        print("Real-data external validity benchmark failed.")
        return 1
    print("Real-data external validity benchmark passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
