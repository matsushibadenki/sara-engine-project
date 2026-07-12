# Directory Path: scripts/eval/real_data_external_validity.py
# English Title: Real-Data External Validity Benchmark
# Purpose/Content: Compares sparse SARA-style retrieval against a dense ANN-style proxy baseline on real corpus QA, summarization, and continual-memory tasks.

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Set, Tuple


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
DEFAULT_RESEARCH_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "external_validity_cases.jsonl")
DEFAULT_PRETRAINED_EMBEDDING_MODEL_PATH = os.environ.get("SARA_EXTERNAL_EMBEDDING_MODEL", "").strip()
DEFAULT_CROSS_ENCODER_MODEL_PATH = os.environ.get("SARA_EXTERNAL_CROSS_ENCODER_MODEL", "").strip()
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


def _load_transformers_embedding_runtime() -> Tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed for the optional embedding baseline.") from exc
    return AutoTokenizer, AutoModel


def _load_transformers_cross_encoder_runtime() -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed for the optional cross-encoder baseline.") from exc
    return AutoTokenizer, AutoModelForSequenceClassification


def _load_faiss_runtime() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is not installed for the optional FAISS baseline.") from exc
    return faiss


def _load_onnx_runtime() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("onnxruntime, tokenizers, and numpy are required for the optional ONNX embedding baseline.") from exc
    return onnxruntime, Tokenizer, np


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]{2,}", text.lower())
    return [token for token in tokens if len(token) >= 2]


def _load_corpus(path: str, limit: int) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [re.sub(r"\s+", " ", line).strip() for line in handle]
    docs = [line for line in lines if len(line) >= 20]
    return docs[: max(int(limit), 1)]


def _load_jsonl_objects(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    objects: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                objects.append(payload)
    return objects


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
        max_scan: Optional[int] = None,
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


class LocalPretrainedEmbeddingRetriever:
    def __init__(self, docs: Sequence[str], model_path: str) -> None:
        normalized_model_path = str(model_path or "").strip()
        if not normalized_model_path:
            raise ValueError("model_path is required for the local pretrained embedding retriever.")
        self.docs = list(docs)
        self.model_path = normalized_model_path
        self.runtime = "transformers"
        self.tokenizer = None
        self.model = None
        self.onnx_session = None
        self.onnx_tokenizer = None
        self.onnx_np = None
        onnx_model_path = os.path.join(self.model_path, "model.onnx")
        if os.path.isfile(onnx_model_path):
            onnxruntime, Tokenizer, np = _load_onnx_runtime()
            tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
            if not os.path.isfile(tokenizer_path):
                raise RuntimeError("ONNX embedding directory is missing tokenizer.json.")
            self.runtime = "onnx"
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_path,
                providers=["CPUExecutionProvider"],
            )
            self.onnx_tokenizer = Tokenizer.from_file(tokenizer_path)
            self.onnx_tokenizer.enable_truncation(max_length=256)
            self.onnx_np = np
        else:
            AutoTokenizer, AutoModel = _load_transformers_embedding_runtime()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            if hasattr(self.model, "eval"):
                self.model.eval()
        self.doc_vectors = [self._embed(doc) for doc in self.docs]

    def _embed(self, text: str) -> List[float]:
        if self.runtime == "onnx":
            encoding = self.onnx_tokenizer.encode(str(text))
            np = self.onnx_np
            input_names = {item.name for item in self.onnx_session.get_inputs()}
            inputs = {
                "input_ids": np.asarray([encoding.ids], dtype="int64"),
                "attention_mask": np.asarray([encoding.attention_mask], dtype="int64"),
            }
            if "token_type_ids" in input_names:
                inputs["token_type_ids"] = np.asarray([encoding.type_ids], dtype="int64")
            output = self.onnx_session.run(None, inputs)[0]
            hidden = output[0]
            mask = np.asarray(encoding.attention_mask, dtype="float32")
            if getattr(hidden, "ndim", 0) == 1:
                vector = hidden.tolist()
            else:
                vector = (hidden * mask[: hidden.shape[0], None]).sum(axis=0) / max(float(mask[: hidden.shape[0]].sum()), 1.0)
                vector = vector.tolist()
            norm = sum(value * value for value in vector) ** 0.5
            return [float(value / norm) for value in vector] if norm > 0.0 else [float(value) for value in vector]
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed for the optional embedding baseline.") from exc

        encoded = self.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            model_output = self.model(**encoded)
        hidden_state = getattr(model_output, "last_hidden_state", None)
        if hidden_state is None:
            raise RuntimeError("Optional embedding baseline model did not return last_hidden_state.")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pooled = hidden_state.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            masked = hidden_state * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        vector = pooled[0].detach().cpu().tolist()
        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0.0:
            vector = [float(value / norm) for value in vector]
        return [float(value) for value in vector]

    def search(self, query: str) -> Tuple[int, int]:
        query_vector = self._embed(query)
        best_index = -1
        best_score = -1.0
        event_cost = 0
        for index, doc_vector in enumerate(self.doc_vectors):
            score = sum(query_value * doc_value for query_value, doc_value in zip(query_vector, doc_vector))
            event_cost += len(doc_vector)
            if score > best_score:
                best_index = index
                best_score = score
        return best_index, max(event_cost, 1)


class LocalFaissPretrainedEmbeddingRetriever(LocalPretrainedEmbeddingRetriever):
    def __init__(
        self,
        docs: Sequence[str],
        model_path: str,
        *,
        hnsw_m: int = 16,
        ef_construction: int = 40,
        ef_search: int = 16,
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is not installed for the optional FAISS baseline.") from exc
        self._faiss = _load_faiss_runtime()
        self._np = np
        self._hnsw_m = max(int(hnsw_m), 4)
        self._ef_construction = max(int(ef_construction), self._hnsw_m)
        self._ef_search = max(int(ef_search), 4)
        super().__init__(docs, model_path)
        if not self.doc_vectors:
            raise RuntimeError("No document vectors were created for the optional FAISS baseline.")
        vector_dim = len(self.doc_vectors[0])
        self.index = self._faiss.IndexHNSWFlat(vector_dim, self._hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self._ef_construction
        self.index.hnsw.efSearch = self._ef_search
        doc_matrix = self._np.asarray(self.doc_vectors, dtype="float32")
        self.index.add(doc_matrix)

    def search(self, query: str) -> Tuple[int, int]:
        query_vector = self._embed(query)
        query_matrix = self._np.asarray([query_vector], dtype="float32")
        distances, indices = self.index.search(query_matrix, 1)
        best_index = int(indices[0][0]) if len(indices) and len(indices[0]) else -1
        if best_index < 0:
            return -1, max(len(query_vector) * self._ef_search, 1)
        return best_index, max(len(query_vector) * self._ef_search, 1)


class LocalCrossEncoderRetriever:
    def __init__(self, docs: Sequence[str], model_path: str) -> None:
        normalized_model_path = str(model_path or "").strip()
        if not normalized_model_path:
            raise ValueError("model_path is required for the local cross-encoder retriever.")
        AutoTokenizer, AutoModelForSequenceClassification = _load_transformers_cross_encoder_runtime()
        self.docs = list(docs)
        self.model_path = normalized_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _score_pair(self, query: str, doc: str) -> float:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed for the optional cross-encoder baseline.") from exc
        encoded = self.tokenizer(
            str(query),
            str(doc),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            output = self.model(**encoded)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("Optional cross-encoder baseline model did not return logits.")
        if logits.ndim == 2 and logits.shape[1] > 1:
            return float(logits[0][1].detach().cpu().item())
        return float(logits.reshape(-1)[0].detach().cpu().item())

    def search(self, query: str) -> Tuple[int, int]:
        best_index = -1
        best_score = float("-inf")
        event_cost = 0
        for index, doc in enumerate(self.docs):
            score = self._score_pair(query, doc)
            event_cost += 1
            if score > best_score:
                best_index = index
                best_score = score
        return best_index, max(event_cost, 1)


class BM25OfflineProxyRetriever:
    def __init__(self, docs: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_token_counts: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        document_frequency: DefaultDict[str, int] = defaultdict(int)
        for doc in self.docs:
            counts: Dict[str, int] = {}
            for token in _tokenize(doc):
                counts[token] = counts.get(token, 0) + 1
            self.doc_token_counts.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            for token in counts:
                document_frequency[token] += 1
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = {
            token: math.log(1.0 + (len(self.docs) - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def search(self, query: str) -> Tuple[int, int]:
        query_tokens = _tokenize(query)
        best_index = -1
        best_score = -1.0
        event_cost = 0
        for index, counts in enumerate(self.doc_token_counts):
            doc_length = max(float(self.doc_lengths[index]), 1.0)
            score = 0.0
            for token in query_tokens:
                frequency = float(counts.get(token, 0))
                event_cost += 1
                if frequency <= 0.0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1.0)) / denominator
            if score > best_score:
                best_score = score
                best_index = index
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


def _score_optional_local_pretrained_embedding(
    docs: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
    *,
    model_path: str,
) -> Dict[str, Any]:
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {
            "available": False,
            "model_path": "",
            "reason": "not_configured",
        }
    if not os.path.isdir(normalized_model_path):
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": "missing_directory",
        }
    try:
        retriever = LocalPretrainedEmbeddingRetriever(docs, normalized_model_path)
        score = _score_retriever(retriever, tasks, docs)
        score["runtime"] = str(getattr(retriever, "runtime", "transformers"))
    except Exception as exc:
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": exc.__class__.__name__,
            "error": str(exc),
        }
    score["available"] = True
    score["model_path"] = normalized_model_path
    return score


def _score_optional_local_pretrained_embedding_faiss(
    docs: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
    *,
    model_path: str,
) -> Dict[str, Any]:
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {
            "available": False,
            "model_path": "",
            "reason": "not_configured",
        }
    if not os.path.isdir(normalized_model_path):
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": "missing_directory",
        }
    try:
        score = _score_retriever(
            LocalFaissPretrainedEmbeddingRetriever(docs, normalized_model_path),
            tasks,
            docs,
        )
    except Exception as exc:
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": exc.__class__.__name__,
            "error": str(exc),
        }
    score["available"] = True
    score["model_path"] = normalized_model_path
    score["index_type"] = "faiss_hnsw_flat_ip"
    return score


def _score_optional_local_cross_encoder(
    docs: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
    *,
    model_path: str,
) -> Dict[str, Any]:
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        return {
            "available": False,
            "model_path": "",
            "reason": "not_configured",
        }
    if not os.path.isdir(normalized_model_path):
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": "missing_directory",
        }
    try:
        score = _score_retriever(
            LocalCrossEncoderRetriever(docs, normalized_model_path),
            tasks,
            docs,
        )
    except Exception as exc:
        return {
            "available": False,
            "model_path": normalized_model_path,
            "reason": exc.__class__.__name__,
            "error": str(exc),
        }
    score["available"] = True
    score["model_path"] = normalized_model_path
    score["model_type"] = "local_cross_encoder_sequence_classification"
    return score


def build_reference_readiness(
    *,
    pretrained_embedding_model_path: str,
    cross_encoder_model_path: str,
    pretrained_embedding_score: Mapping[str, Any],
    pretrained_embedding_faiss_score: Mapping[str, Any],
    cross_encoder_score: Mapping[str, Any],
) -> Dict[str, Any]:
    embedding_runtime = (
        "onnxruntime+tokenizers+numpy"
        if str(pretrained_embedding_score.get("runtime", "")) == "onnx"
        or os.path.isfile(os.path.join(str(pretrained_embedding_model_path or ""), "model.onnx"))
        else "transformers+torch"
    )
    references = [
        {
            "reference_id": "ann_pretrained_embedding_reference",
            "label": "Local Pretrained Embedding Reference",
            "configured_path": str(pretrained_embedding_model_path or ""),
            "available": bool(pretrained_embedding_score.get("available", False)),
            "reason": str(pretrained_embedding_score.get("reason", "") or ""),
            "expected_runtime": embedding_runtime,
        },
        {
            "reference_id": "ann_pretrained_embedding_faiss_reference",
            "label": "Local Pretrained Embedding FAISS Reference",
            "configured_path": str(pretrained_embedding_model_path or ""),
            "available": bool(pretrained_embedding_faiss_score.get("available", False)),
            "reason": str(pretrained_embedding_faiss_score.get("reason", "") or ""),
            "expected_runtime": embedding_runtime + "+faiss" if embedding_runtime.startswith("onnx") else "transformers+torch+faiss+numpy",
        },
        {
            "reference_id": "ann_cross_encoder_reference",
            "label": "Local Cross-Encoder Reference",
            "configured_path": str(cross_encoder_model_path or ""),
            "available": bool(cross_encoder_score.get("available", False)),
            "reason": str(cross_encoder_score.get("reason", "") or ""),
            "expected_runtime": "transformers+torch",
        },
    ]
    ready_count = sum(1 for item in references if bool(item["available"]))
    configured_count = sum(1 for item in references if str(item["configured_path"]).strip())
    missing_directory_count = sum(1 for item in references if item["reason"] == "missing_directory")
    not_configured_count = sum(1 for item in references if item["reason"] == "not_configured")
    dependency_error_count = sum(
        1
        for item in references
        if item["reason"] in {"RuntimeError", "ImportError", "ModuleNotFoundError"}
    )
    next_actions: List[Dict[str, Any]] = []
    if not configured_count:
        next_actions.append(
            {
                "priority": "high",
                "category": "configure_local_reference_path",
                "action": "Provide --pretrained-embedding-model and/or --cross-encoder-model with local model directories.",
            }
        )
    if missing_directory_count:
        next_actions.append(
            {
                "priority": "high",
                "category": "missing_local_reference_directory",
                "action": "Create or point to the configured local model directory before rerunning eval-external-validity.",
            }
        )
    if dependency_error_count:
        next_actions.append(
            {
                "priority": "medium",
                "category": "missing_local_reference_dependency",
                "action": "Install the missing optional CPU-only dependencies required by the configured reference models.",
            }
        )
    if ready_count == 0:
        status = "proxy_only"
    elif ready_count < len(references):
        status = "partial_reference_ready"
    else:
        status = "all_reference_paths_ready"
    return {
        "schema": "sara-external-reference-readiness-v1",
        "status": status,
        "configured_reference_count": int(configured_count),
        "ready_reference_count": int(ready_count),
        "missing_directory_count": int(missing_directory_count),
        "not_configured_count": int(not_configured_count),
        "dependency_error_count": int(dependency_error_count),
        "references": references,
        "next_actions": next_actions,
    }


def _case_results_by_id(score: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    results = score.get("case_results", [])
    if not isinstance(results, list):
        return {}
    indexed: Dict[str, Mapping[str, Any]] = {}
    for item in results:
        if isinstance(item, Mapping):
            case_id = str(item.get("case_id", "") or "")
            if case_id:
                indexed[case_id] = item
    return indexed


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _failure_type(case: Mapping[str, Any]) -> str:
    if bool(case.get("correct", False)):
        return "none"
    predicted = _safe_int(case.get("predicted_doc_index"), -1)
    if predicted < 0:
        return "abstained"
    if float(case.get("summary_keyword_coverage", 0.0) or 0.0) < 0.5:
        return "weak_summary"
    return "wrong_document"


def build_per_task_external_validity_summary(
    tasks: Sequence[Mapping[str, Any]],
    sparse_score: Mapping[str, Any],
    dense_score: Mapping[str, Any],
    dense_embedding_score: Mapping[str, Any],
) -> Dict[str, Any]:
    sparse_by_id = _case_results_by_id(sparse_score)
    dense_by_id = _case_results_by_id(dense_score)
    dense_embedding_by_id = _case_results_by_id(dense_embedding_score)
    cases: List[Dict[str, Any]] = []
    failure_counts: DefaultDict[str, int] = defaultdict(int)
    total_sparse_cost = 0.0
    total_dense_cost = 0.0
    total_embedding_cost = 0.0
    abstention_count = 0

    for task in tasks:
        case_id = str(task.get("case_id", "") or "")
        sparse_case = sparse_by_id.get(case_id, {})
        dense_case = dense_by_id.get(case_id, {})
        embedding_case = dense_embedding_by_id.get(case_id, {})
        sparse_cost = float(sparse_case.get("event_cost_proxy", 0.0) or 0.0)
        dense_cost = float(dense_case.get("event_cost_proxy", 0.0) or 0.0)
        embedding_cost = float(embedding_case.get("event_cost_proxy", 0.0) or 0.0)
        predicted_index = _safe_int(sparse_case.get("predicted_doc_index"), -1)
        abstained = predicted_index < 0
        failure_type = _failure_type(sparse_case)
        failure_counts[failure_type] += 1
        abstention_count += 1 if abstained else 0
        total_sparse_cost += sparse_cost
        total_dense_cost += dense_cost
        total_embedding_cost += embedding_cost
        cases.append(
            {
                "case_id": case_id,
                "query": str(task.get("query", "") or ""),
                "quality": {
                    "sara_correct": bool(sparse_case.get("correct", False)),
                    "ann_dense_correct": bool(dense_case.get("correct", False)),
                    "ann_dense_embedding_correct": bool(embedding_case.get("correct", False)),
                    "summary_keyword_coverage": float(
                        sparse_case.get("summary_keyword_coverage", 0.0) or 0.0
                    ),
                },
                "cost": {
                    "sara_event_cost_proxy": sparse_cost,
                    "ann_dense_event_cost_proxy": dense_cost,
                    "ann_dense_embedding_event_cost_proxy": embedding_cost,
                    "dense_cost_advantage_proxy": dense_cost / max(sparse_cost, 1e-9),
                    "dense_embedding_cost_advantage_proxy": embedding_cost / max(sparse_cost, 1e-9),
                },
                "abstention": {
                    "sara_abstained": bool(abstained),
                    "expected_behavior": "retrieve",
                },
                "failure_type": failure_type,
            }
        )

    total = max(len(cases), 1)
    return {
        "schema": "sara-per-task-external-validity-summary-v1",
        "case_count": len(cases),
        "cases": cases,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "abstention_rate": float(abstention_count / total),
        "avg_sara_event_cost_proxy": float(total_sparse_cost / total),
        "avg_ann_dense_event_cost_proxy": float(total_dense_cost / total),
        "avg_ann_dense_embedding_event_cost_proxy": float(total_embedding_cost / total),
        "avg_dense_cost_advantage_proxy": float(total_dense_cost / max(total_sparse_cost, 1e-9)),
        "avg_dense_embedding_cost_advantage_proxy": float(
            total_embedding_cost / max(total_sparse_cost, 1e-9)
        ),
    }


def _accuracy_by_type(cases: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    grouped: DefaultDict[str, List[float]] = defaultdict(list)
    for case in cases:
        task_type = str(case.get("task_type", "") or "unknown")
        grouped[task_type].append(1.0 if bool(case.get("sara_correct", False)) else 0.0)
    return {
        task_type: sum(values) / max(len(values), 1)
        for task_type, values in sorted(grouped.items())
    }


def _score_repository_fixture_cases(path: str = DEFAULT_RESEARCH_FIXTURE_PATH) -> Dict[str, Any]:
    fixture_cases = _load_jsonl_objects(path)
    if not fixture_cases:
        return {
            "schema": "sara-repository-fixture-retrieval-probe-v1",
            "observed_only": True,
            "passed": False,
            "case_count": 0,
            "fixture_path": os.path.abspath(path),
            "errors": ["No repository fixture cases were loaded."],
            "metrics": {},
            "cases": [],
        }

    docs = [str(case.get("document", "") or "") for case in fixture_cases]
    sparse_retriever = MetabolicSparseEventRetriever(docs)
    dense_retriever = DenseAnnProxyRetriever(docs)
    dense_embedding_retriever = DenseEmbeddingAnnProxyRetriever(docs)
    case_reports: List[Dict[str, Any]] = []
    retrieve_correct = 0
    retrieve_total = 0
    abstain_correct = 0
    abstain_total = 0
    total_sparse_cost = 0.0
    total_dense_cost = 0.0
    total_embedding_cost = 0.0

    for expected_index, case in enumerate(fixture_cases):
        query = str(case.get("query", "") or "")
        expected_behavior = str(case.get("expected_behavior", "") or "retrieve")
        task_type = str(case.get("task_type", "") or "unknown")
        sparse_index, sparse_cost = sparse_retriever.search(query)
        dense_index, dense_cost = dense_retriever.search(query)
        embedding_index, embedding_cost = dense_embedding_retriever.search(query)
        if expected_behavior == "abstain":
            sparse_correct = sparse_index == -1
            abstain_total += 1
            abstain_correct += 1 if sparse_correct else 0
        else:
            sparse_correct = sparse_index == expected_index
            retrieve_total += 1
            retrieve_correct += 1 if sparse_correct else 0
        total_sparse_cost += float(sparse_cost)
        total_dense_cost += float(dense_cost)
        total_embedding_cost += float(embedding_cost)
        case_reports.append(
            {
                "case_id": str(case.get("case_id", f"fixture-{expected_index}") or f"fixture-{expected_index}"),
                "task_type": task_type,
                "expected_behavior": expected_behavior,
                "query": query,
                "expected_doc_index": expected_index if expected_behavior != "abstain" else -1,
                "sara_predicted_doc_index": int(sparse_index),
                "ann_dense_predicted_doc_index": int(dense_index),
                "ann_dense_embedding_predicted_doc_index": int(embedding_index),
                "sara_correct": bool(sparse_correct),
                "sara_event_cost_proxy": int(sparse_cost),
                "ann_dense_event_cost_proxy": int(dense_cost),
                "ann_dense_embedding_event_cost_proxy": int(embedding_cost),
                "dense_cost_advantage_proxy": float(dense_cost) / max(float(sparse_cost), 1e-9),
                "dense_embedding_cost_advantage_proxy": float(embedding_cost) / max(float(sparse_cost), 1e-9),
                "sara_retrieval_diagnostics": dict(sparse_retriever.last_diagnostics),
            }
        )

    total = max(len(case_reports), 1)
    retrieve_accuracy = retrieve_correct / max(retrieve_total, 1)
    abstention_integrity = abstain_correct / max(abstain_total, 1)
    passed = bool(case_reports) and retrieve_accuracy >= 1.0 and abstention_integrity >= 1.0
    return {
        "schema": "sara-repository-fixture-retrieval-probe-v1",
        "observed_only": True,
        "passed": passed,
        "fixture_path": os.path.abspath(path),
        "case_count": len(case_reports),
        "retrieve_case_count": retrieve_total,
        "abstain_case_count": abstain_total,
        "metrics": {
            "repository_fixture_retrieval_accuracy": float(retrieve_accuracy),
            "repository_fixture_abstention_integrity": float(abstention_integrity),
            "repository_fixture_overall_accuracy": float(
                (retrieve_correct + abstain_correct) / total
            ),
            "repository_fixture_avg_sara_event_cost_proxy": float(total_sparse_cost / total),
            "repository_fixture_avg_dense_cost_advantage_proxy": float(
                total_dense_cost / max(total_sparse_cost, 1e-9)
            ),
            "repository_fixture_avg_dense_embedding_cost_advantage_proxy": float(
                total_embedding_cost / max(total_sparse_cost, 1e-9)
            ),
        },
        "accuracy_by_task_type": _accuracy_by_type(case_reports),
        "cases": case_reports,
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
    history: Optional[Sequence[Dict[str, Any]]] = None,
    regression_tolerance: float = 0.05,
    pretrained_embedding_model_path: str = DEFAULT_PRETRAINED_EMBEDDING_MODEL_PATH,
    cross_encoder_model_path: str = DEFAULT_CROSS_ENCODER_MODEL_PATH,
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
    bm25_offline = _score_retriever(BM25OfflineProxyRetriever(docs), tasks, docs)
    real_pretrained_embedding = _score_optional_local_pretrained_embedding(
        docs,
        tasks,
        model_path=pretrained_embedding_model_path,
    )
    real_pretrained_embedding_faiss = _score_optional_local_pretrained_embedding_faiss(
        docs,
        tasks,
        model_path=pretrained_embedding_model_path,
    )
    real_cross_encoder = _score_optional_local_cross_encoder(
        docs,
        tasks,
        model_path=cross_encoder_model_path,
    )
    reference_readiness = build_reference_readiness(
        pretrained_embedding_model_path=pretrained_embedding_model_path,
        cross_encoder_model_path=cross_encoder_model_path,
        pretrained_embedding_score=real_pretrained_embedding,
        pretrained_embedding_faiss_score=real_pretrained_embedding_faiss,
        cross_encoder_score=real_cross_encoder,
    )
    per_task_summary = build_per_task_external_validity_summary(tasks, sparse, dense, dense_embedding)
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
    repository_fixture_probe = _score_repository_fixture_cases()
    repository_fixture_metrics = (
        repository_fixture_probe.get("metrics", {})
        if isinstance(repository_fixture_probe.get("metrics"), dict)
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
    benchmark_context["pretrained_embedding_model_path"] = str(pretrained_embedding_model_path or "")
    benchmark_context["pretrained_embedding_reference_available"] = bool(
        real_pretrained_embedding.get("available", False)
    )
    benchmark_context["pretrained_embedding_reference_reason"] = str(
        real_pretrained_embedding.get("reason", "") or ""
    )
    benchmark_context["pretrained_embedding_faiss_reference_available"] = bool(
        real_pretrained_embedding_faiss.get("available", False)
    )
    benchmark_context["pretrained_embedding_faiss_reference_reason"] = str(
        real_pretrained_embedding_faiss.get("reason", "") or ""
    )
    benchmark_context["cross_encoder_model_path"] = str(cross_encoder_model_path or "")
    benchmark_context["cross_encoder_reference_available"] = bool(
        real_cross_encoder.get("available", False)
    )
    benchmark_context["cross_encoder_reference_reason"] = str(
        real_cross_encoder.get("reason", "") or ""
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
    bm25_cost = float(bm25_offline["avg_event_cost_proxy"])
    bm25_cost_advantage = bm25_cost / max(sparse_cost, 1e-9)
    real_pretrained_embedding_available = bool(real_pretrained_embedding.get("available", False))
    real_pretrained_embedding_accuracy = float(
        real_pretrained_embedding.get("accuracy", 0.0) if real_pretrained_embedding_available else 0.0
    )
    real_pretrained_embedding_cost_advantage = (
        float(real_pretrained_embedding.get("avg_event_cost_proxy", 0.0)) / max(sparse_cost, 1e-9)
        if real_pretrained_embedding_available
        else 0.0
    )
    real_pretrained_embedding_latency_ms = float(
        real_pretrained_embedding.get("avg_latency_ms", 0.0) if real_pretrained_embedding_available else 0.0
    )
    real_pretrained_embedding_faiss_available = bool(real_pretrained_embedding_faiss.get("available", False))
    real_pretrained_embedding_faiss_accuracy = float(
        real_pretrained_embedding_faiss.get("accuracy", 0.0)
        if real_pretrained_embedding_faiss_available
        else 0.0
    )
    real_pretrained_embedding_faiss_cost_advantage = (
        float(real_pretrained_embedding_faiss.get("avg_event_cost_proxy", 0.0)) / max(sparse_cost, 1e-9)
        if real_pretrained_embedding_faiss_available
        else 0.0
    )
    real_pretrained_embedding_faiss_latency_ms = float(
        real_pretrained_embedding_faiss.get("avg_latency_ms", 0.0)
        if real_pretrained_embedding_faiss_available
        else 0.0
    )
    real_cross_encoder_available = bool(real_cross_encoder.get("available", False))
    real_cross_encoder_accuracy = float(
        real_cross_encoder.get("accuracy", 0.0) if real_cross_encoder_available else 0.0
    )
    real_cross_encoder_cost_advantage = (
        float(real_cross_encoder.get("avg_event_cost_proxy", 0.0)) / max(sparse_cost, 1e-9)
        if real_cross_encoder_available
        else 0.0
    )
    real_cross_encoder_latency_ms = float(
        real_cross_encoder.get("avg_latency_ms", 0.0) if real_cross_encoder_available else 0.0
    )

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
            "bm25_offline_proxy_qa_accuracy": float(bm25_offline["accuracy"]),
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
            "repository_fixture_retrieval_accuracy": float(
                repository_fixture_metrics.get("repository_fixture_retrieval_accuracy", 0.0)
            ),
            "repository_fixture_abstention_integrity": float(
                repository_fixture_metrics.get("repository_fixture_abstention_integrity", 0.0)
            ),
            "repository_fixture_overall_accuracy": float(
                repository_fixture_metrics.get("repository_fixture_overall_accuracy", 0.0)
            ),
            "repository_fixture_avg_dense_cost_advantage_proxy": float(
                repository_fixture_metrics.get("repository_fixture_avg_dense_cost_advantage_proxy", 0.0)
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
            "bm25_offline_cost_advantage_proxy": float(bm25_cost_advantage),
            "bm25_offline_avg_event_cost_proxy": bm25_cost,
            "real_pretrained_embedding_reference_available": (
                1.0 if real_pretrained_embedding_available else 0.0
            ),
            "real_pretrained_embedding_reference_qa_accuracy": float(real_pretrained_embedding_accuracy),
            "real_pretrained_embedding_reference_cost_advantage_proxy": float(
                real_pretrained_embedding_cost_advantage
            ),
            "real_pretrained_embedding_reference_avg_latency_ms": float(
                real_pretrained_embedding_latency_ms
            ),
            "real_pretrained_embedding_faiss_reference_available": (
                1.0 if real_pretrained_embedding_faiss_available else 0.0
            ),
            "real_pretrained_embedding_faiss_reference_qa_accuracy": float(
                real_pretrained_embedding_faiss_accuracy
            ),
            "real_pretrained_embedding_faiss_reference_cost_advantage_proxy": float(
                real_pretrained_embedding_faiss_cost_advantage
            ),
            "real_pretrained_embedding_faiss_reference_avg_latency_ms": float(
                real_pretrained_embedding_faiss_latency_ms
            ),
            "real_cross_encoder_reference_available": 1.0 if real_cross_encoder_available else 0.0,
            "real_cross_encoder_reference_qa_accuracy": float(real_cross_encoder_accuracy),
            "real_cross_encoder_reference_cost_advantage_proxy": float(real_cross_encoder_cost_advantage),
            "real_cross_encoder_reference_avg_latency_ms": float(real_cross_encoder_latency_ms),
            "reference_ready_count": float(reference_readiness["ready_reference_count"]),
            "reference_configured_count": float(reference_readiness["configured_reference_count"]),
            "reference_dependency_error_count": float(reference_readiness["dependency_error_count"]),
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
            "per_task_external_validity_summary_available": 1.0,
            "per_task_external_validity_case_count": float(per_task_summary["case_count"]),
            "per_task_external_validity_abstention_rate": float(per_task_summary["abstention_rate"]),
            "per_task_external_validity_avg_dense_cost_advantage_proxy": float(
                per_task_summary["avg_dense_cost_advantage_proxy"]
            ),
        },
        "checks": checks,
        "check_details": check_details,
        "sara_sparse": sparse,
        "sara_sparse_baseline": sparse_baseline,
        "ann_dense_proxy": dense,
        "ann_dense_embedding_proxy": dense_embedding,
        "ann_pretrained_embedding_reference": real_pretrained_embedding,
        "ann_pretrained_embedding_faiss_reference": real_pretrained_embedding_faiss,
        "ann_cross_encoder_reference": real_cross_encoder,
        "reference_readiness": reference_readiness,
        "bm25_offline_proxy": bm25_offline,
        "per_task_external_validity_summary": per_task_summary,
        "continual_memory": continual,
        "negative_controls": negative_control,
        "contrastive_controls": contrastive_control,
        "sparse_rag_rerank": sparse_rag_rerank,
        "sparse_diffusion_real_data": sparse_diffusion_real_data,
        "repository_fixture_probe": repository_fixture_probe,
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
        f"- bm25_offline_proxy_qa_accuracy: {float(metrics.get('bm25_offline_proxy_qa_accuracy', 0.0)):.3f}",
        f"- real_pretrained_embedding_reference_available: {float(metrics.get('real_pretrained_embedding_reference_available', 0.0)):.3f}",
        f"- real_pretrained_embedding_reference_qa_accuracy: {float(metrics.get('real_pretrained_embedding_reference_qa_accuracy', 0.0)):.3f}",
        f"- real_pretrained_embedding_faiss_reference_available: {float(metrics.get('real_pretrained_embedding_faiss_reference_available', 0.0)):.3f}",
        f"- real_pretrained_embedding_faiss_reference_qa_accuracy: {float(metrics.get('real_pretrained_embedding_faiss_reference_qa_accuracy', 0.0)):.3f}",
        f"- real_cross_encoder_reference_available: {float(metrics.get('real_cross_encoder_reference_available', 0.0)):.3f}",
        f"- real_cross_encoder_reference_qa_accuracy: {float(metrics.get('real_cross_encoder_reference_qa_accuracy', 0.0)):.3f}",
        f"- real_data_summary_keyword_coverage: {float(metrics.get('real_data_summary_keyword_coverage', 0.0)):.3f}",
        f"- continual_memory_hit_rate: {float(metrics.get('continual_memory_hit_rate', 0.0)):.3f}",
        f"- performance_energy_ratio_proxy: {float(metrics.get('performance_energy_ratio_proxy', 0.0)):.3f}",
        f"- ann_cost_advantage_proxy: {float(metrics.get('ann_cost_advantage_proxy', 0.0)):.3f}",
        f"- dense_embedding_ann_cost_advantage_proxy: {float(metrics.get('dense_embedding_ann_cost_advantage_proxy', 0.0)):.3f}",
        f"- bm25_offline_cost_advantage_proxy: {float(metrics.get('bm25_offline_cost_advantage_proxy', 0.0)):.3f}",
        f"- real_pretrained_embedding_reference_cost_advantage_proxy: {float(metrics.get('real_pretrained_embedding_reference_cost_advantage_proxy', 0.0)):.3f}",
        f"- real_pretrained_embedding_faiss_reference_cost_advantage_proxy: {float(metrics.get('real_pretrained_embedding_faiss_reference_cost_advantage_proxy', 0.0)):.3f}",
        f"- real_cross_encoder_reference_cost_advantage_proxy: {float(metrics.get('real_cross_encoder_reference_cost_advantage_proxy', 0.0)):.3f}",
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
        f"- repository_fixture_retrieval_accuracy: {float(metrics.get('repository_fixture_retrieval_accuracy', 0.0)):.3f}",
        f"- repository_fixture_abstention_integrity: {float(metrics.get('repository_fixture_abstention_integrity', 0.0)):.3f}",
        f"- repository_fixture_avg_dense_cost_advantage_proxy: {float(metrics.get('repository_fixture_avg_dense_cost_advantage_proxy', 0.0)):.3f}",
        f"- rag_query_decomposition_coverage_observed: {float(metrics.get('rag_query_decomposition_coverage_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_selection_observed: {float(metrics.get('rag_query_decomposition_merged_selection_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_citation_grounding_observed: {float(metrics.get('rag_query_decomposition_merged_citation_grounding_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_source_reliability_observed: {float(metrics.get('rag_query_decomposition_merged_source_reliability_observed', 0.0)):.3f}",
        f"- rag_query_decomposition_merged_source_diversity_observed: {float(metrics.get('rag_query_decomposition_merged_source_diversity_observed', 0.0)):.3f}",
        f"- sara_avg_latency_ms: {float(metrics.get('sara_avg_latency_ms', 0.0)):.3f}",
        f"- ann_proxy_avg_latency_ms: {float(metrics.get('ann_proxy_avg_latency_ms', 0.0)):.3f}",
        f"- real_pretrained_embedding_reference_avg_latency_ms: {float(metrics.get('real_pretrained_embedding_reference_avg_latency_ms', 0.0)):.3f}",
        f"- real_pretrained_embedding_faiss_reference_avg_latency_ms: {float(metrics.get('real_pretrained_embedding_faiss_reference_avg_latency_ms', 0.0)):.3f}",
        f"- real_cross_encoder_reference_avg_latency_ms: {float(metrics.get('real_cross_encoder_reference_avg_latency_ms', 0.0)):.3f}",
        f"- reference_ready_count: {float(metrics.get('reference_ready_count', 0.0)):.3f}",
        f"- reference_configured_count: {float(metrics.get('reference_configured_count', 0.0)):.3f}",
        f"- reference_dependency_error_count: {float(metrics.get('reference_dependency_error_count', 0.0)):.3f}",
        f"- per_task_external_validity_case_count: {float(metrics.get('per_task_external_validity_case_count', 0.0)):.3f}",
        f"- per_task_external_validity_abstention_rate: {float(metrics.get('per_task_external_validity_abstention_rate', 0.0)):.3f}",
        f"- per_task_external_validity_avg_dense_cost_advantage_proxy: {float(metrics.get('per_task_external_validity_avg_dense_cost_advantage_proxy', 0.0)):.3f}",
        f"- trend_has_previous: {bool(trend.get('has_previous', False))}",
        f"- trend_comparison_active: {bool(trend.get('comparison_active', False))}",
        f"- trend_comparison_skipped_reason: {str(trend.get('comparison_skipped_reason', '') or '')}",
        f"- trend_regression_count: {int(trend.get('regression_count', 0) or 0)}",
        f"- reference_readiness_status: {str(report.get('reference_readiness', {}).get('status', '') or '')}",
        "Checks:",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if checks[name] else 'FAIL'}")
    reference_readiness = (
        report.get("reference_readiness", {})
        if isinstance(report.get("reference_readiness"), dict)
        else {}
    )
    references = (
        reference_readiness.get("references", [])
        if isinstance(reference_readiness.get("references"), list)
        else []
    )
    lines.append("Reference Readiness:")
    if references:
        for item in references:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"id={item.get('reference_id', '')}, "
                f"available={bool(item.get('available', False))}, "
                f"reason={item.get('reason', '')}, "
                f"path={item.get('configured_path', '')}"
            )
    next_actions = (
        reference_readiness.get("next_actions", [])
        if isinstance(reference_readiness.get("next_actions"), list)
        else []
    )
    if next_actions:
        lines.append("Reference Next Actions:")
        for item in next_actions:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"priority={item.get('priority', '')}, "
                f"category={item.get('category', '')}, "
                f"action={item.get('action', '')}"
            )
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
    parser.add_argument("--pretrained-embedding-model", default=DEFAULT_PRETRAINED_EMBEDDING_MODEL_PATH)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL_PATH)
    parser.add_argument("--no-history-update", action="store_true")
    args = parser.parse_args()

    history = load_external_validity_history(str(args.history_path))
    report = run_real_data_external_validity(
        corpus_path=str(args.corpus),
        max_docs=int(args.max_docs),
        max_cases=int(args.max_cases),
        history=history,
        regression_tolerance=float(max(args.regression_tolerance, 0.0)),
        pretrained_embedding_model_path=str(args.pretrained_embedding_model),
        cross_encoder_model_path=str(args.cross_encoder_model),
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
