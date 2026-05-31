# Directory Path: src/sara_engine/rag/rag_pipeline.py
# English Title: SNN RAG Pipeline
# Purpose/Content: Provides chunking, SDR encoding, sparse retrieval, and lightweight reranking for SNN-based RAG.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..memory.sdr import SDREncoder
from ..memory.snn_vector_store import SNNVectorStore


@dataclass
class ChunkMetadata:
    """チャンクに付与されるメタデータ。"""

    source: str = ""
    chunk_index: int = 0
    total_chunks: int = 0


@dataclass
class DocumentChunk:
    """分割されたドキュメントチャンク。"""

    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


def _tokenize(text: str) -> List[str]:
    cleaned = str(text or "").lower()
    cleaned = re.sub(r"[^0-9a-zA-Z一-龥ぁ-んァ-ンー]+", " ", cleaned)
    return [token for token in cleaned.split() if token]


def _has_contradiction(query_tokens: Sequence[str], text_tokens: Sequence[str]) -> bool:
    negation_tokens = {"not", "no", "never", "without", "false", "fail", "failed", "avoid", "禁止", "失敗"}
    query_set = set(query_tokens)
    text_set = set(text_tokens)
    return bool(query_set.intersection(text_set) and text_set.intersection(negation_tokens))


def _dedupe_preserving_order(items: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        normalized = " ".join(str(item or "").split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


class DocumentChunker:
    """テキストドキュメントを意味的に適切なチャンクに分割する。

    文単位での分割を基本とし、最大チャンクサイズとオーバーラップを制御可能。
    """

    def __init__(
        self,
        max_chunk_size: int = 200,
        overlap_size: int = 30,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.separators = separators or ["。", "！", "？", "!", "?", "\n\n", "\n"]

    def _split_sentences(self, text: str) -> List[str]:
        """テキストを文単位に分割する。"""
        # 日本語・英語の句読点で分割
        pattern = r"(?<=[。！？!?\n])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, source: str = "") -> List[DocumentChunk]:
        """テキストをチャンクに分割する。

        Args:
            text: 分割対象のテキスト。
            source: ドキュメントのソース情報。

        Returns:
            分割されたDocumentChunkのリスト。
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [
                DocumentChunk(
                    text=text.strip(),
                    metadata=ChunkMetadata(
                        source=source, chunk_index=0, total_chunks=1
                    ),
                )
            ]

        chunks: List[DocumentChunk] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # 現在のチャンクに追加すると最大サイズを超える場合
            if current_length + sentence_len > self.max_chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(DocumentChunk(text=chunk_text))

                # オーバーラップ処理: 末尾の文をいくつか次のチャンクに引き継ぐ
                overlap_text = ""
                overlap_sentences: List[str] = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_text = "".join(overlap_sentences)
                    else:
                        break

                current_chunk = overlap_sentences.copy()
                current_length = len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_len

        # 残りのチャンクを追加
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(DocumentChunk(text=chunk_text))

        # メタデータの設定
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata = ChunkMetadata(
                source=source, chunk_index=i, total_chunks=total
            )

        return chunks


class SNNRAGPipeline:
    """SNN (Spiking Neural Network) ベースのRAGパイプライン。

    既存のSDREncoderとSNNVectorStoreを活用して、
    ドキュメントの追加・検索・コンテキスト生成を行う。

    Example:
        >>> rag = SNNRAGPipeline(sdr_size=2048)
        >>> rag.add_document("SNNは脳の神経回路を模倣した計算モデルです。")
        >>> results = rag.query("SNNとは何ですか？", top_k=3)
    """

    def __init__(
        self,
        sdr_size: int = 2048,
        density: float = 0.02,
        max_chunk_size: int = 200,
        overlap_size: int = 30,
    ) -> None:
        self.encoder = SDREncoder(
            input_size=sdr_size,
            density=density,
            use_tokenizer=True,
            apply_vsa=False,  # RAGでは意味的類似性のためVSAを無効化
        )
        self.vector_store = SNNVectorStore()
        self.chunker = DocumentChunker(
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
        )
        self._document_count = 0
        self._chunk_count = 0
        self._chunk_metadata: List[ChunkMetadata] = []

    def _metadata_for_text(self, text: str) -> ChunkMetadata:
        for index, document in enumerate(self.vector_store.documents):
            if document == text and index < len(self._chunk_metadata):
                return self._chunk_metadata[index]
        return ChunkMetadata()

    def add_document(self, text: str, source: str = "") -> int:
        """ドキュメントをRAGパイプラインに追加する。

        テキストをチャンク分割し、各チャンクをSDRエンコードして
        ベクトルストアに格納する。

        Args:
            text: 追加するドキュメントのテキスト。
            source: ドキュメントのソース識別子。

        Returns:
            追加されたチャンク数。
        """
        if not text.strip():
            return 0

        if not source:
            self._document_count += 1
            source = f"doc_{self._document_count}"

        chunks = self.chunker.chunk(text, source=source)
        added = 0
        for chunk in chunks:
            embedding = self.encoder.encode(chunk.text)
            # float に変換してベクトルストアの cosine similarity と互換性を持たせる
            float_embedding = [float(v) for v in embedding]
            self.vector_store.add_document(chunk.text, float_embedding)
            self._chunk_metadata.append(chunk.metadata)
            self._chunk_count += 1
            added += 1

        return added

    def add_documents(self, texts: List[str], sources: Optional[List[str]] = None) -> int:
        """複数のドキュメントを一括追加する。

        Args:
            texts: 追加するドキュメントテキストのリスト。
            sources: 各ドキュメントのソース識別子のリスト。

        Returns:
            追加されたチャンクの合計数。
        """
        total_added = 0
        if sources is None:
            sources = [""] * len(texts)
        for text, src in zip(texts, sources):
            total_added += self.add_document(text, source=src)
        return total_added

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """クエリテキストに対して関連するドキュメントチャンクを検索する。

        Args:
            query_text: 検索クエリのテキスト。
            top_k: 返すチャンク数の上限。
            min_score: 最低類似度スコア（これ未満の結果は除外）。

        Returns:
            (チャンクテキスト, 類似度スコア) のタプルリスト。
        """
        if not query_text.strip():
            return []

        query_embedding = self.encoder.encode(query_text)
        float_embedding = [float(v) for v in query_embedding]
        results = self.vector_store.search(float_embedding, top_k=top_k)

        # 最低スコアでフィルタリング
        if min_score > 0.0:
            results = [(text, score)
                       for text, score in results if score >= min_score]

        return results

    def query_with_rerank(
        self,
        query_text: str,
        top_k: int = 3,
        candidate_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Searches and reranks sparse RAG candidates with auditable local signals."""

        if not query_text.strip():
            return {
                "schema": "sara-sparse-rag-rerank-v1",
                "query": query_text,
                "candidate_count": 0,
                "selected": [],
                "ranked": [],
                "metrics": {
                    "sparse_rag_rerank_bounded_count_observed": 1.0,
                    "sparse_rag_rerank_source_agreement_observed": 0.0,
                    "sparse_rag_rerank_contradiction_guard_observed": 1.0,
                    "sparse_rag_rerank_freshness_observed": 0.0,
                    "sparse_rag_rerank_citation_grounding_observed": 0.0,
                    "sparse_rag_rerank_source_reliability_observed": 0.0,
                    "sparse_rag_rerank_source_diversity_observed": 1.0,
                },
                "observed_only": True,
            }

        requested_candidate_k = max(int(candidate_k or max(top_k * 3, top_k)), top_k)
        raw_results = self.query(query_text, top_k=requested_candidate_k, min_score=min_score)
        query_tokens = _tokenize(query_text)
        query_token_set = set(query_tokens)
        ranked: List[Dict[str, Any]] = []
        for text, score in raw_results:
            metadata = self._metadata_for_text(text)
            text_tokens = _tokenize(text)
            text_token_set = set(text_tokens)
            overlap = len(query_token_set.intersection(text_token_set))
            source_reliability = 1.0 if metadata.source else 0.5
            freshness_score = 1.0 / float(1 + max(metadata.chunk_index, 0))
            contradiction = _has_contradiction(query_tokens, text_tokens)
            agreement_score = overlap / float(max(len(query_token_set), 1))
            citation_id = f"{metadata.source or 'unknown'}#{int(metadata.chunk_index)}"
            rerank_score = (
                float(score) * 0.55
                + agreement_score * 0.25
                + source_reliability * 0.10
                + freshness_score * 0.10
                - (0.50 if contradiction else 0.0)
            )
            ranked.append(
                {
                    "text": text,
                    "score": float(score),
                    "rerank_score": float(rerank_score),
                    "source": metadata.source,
                    "chunk_index": int(metadata.chunk_index),
                    "citation_id": citation_id,
                    "source_agreement": float(agreement_score),
                    "source_reliability": float(source_reliability),
                    "freshness_score": float(freshness_score),
                    "contradiction_flag": bool(contradiction),
                    "overlap_token_count": int(overlap),
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item["rerank_score"]),
                float(item["source_agreement"]),
                float(item["freshness_score"]),
            ),
            reverse=True,
        )
        selected = ranked[: max(int(top_k), 0)]
        selected_sources = {str(item.get("source", "")) for item in selected if str(item.get("source", ""))}
        source_agreement_ready = bool(selected and any(float(item.get("source_agreement", 0.0)) > 0.0 for item in selected))
        contradiction_guard_ready = all(not bool(item.get("contradiction_flag", False)) for item in selected)
        freshness_ready = bool(selected and max(float(item.get("freshness_score", 0.0)) for item in selected) > 0.0)
        citation_grounding_ready = bool(
            selected
            and all(str(item.get("citation_id", "") or "") for item in selected)
            and all(str(item.get("source", "") or "") for item in selected)
        )
        source_reliability_ready = bool(
            selected
            and all(float(item.get("source_reliability", 0.0)) >= 1.0 for item in selected)
        )
        source_diversity_ready = bool(
            len(selected) <= 1
            or len(selected_sources) >= min(len(selected), 2)
        )
        bounded_ready = bool(len(ranked) <= requested_candidate_k)
        return {
            "schema": "sara-sparse-rag-rerank-v1",
            "query": query_text,
            "candidate_count": int(len(ranked)),
            "candidate_limit": int(requested_candidate_k),
            "selected_source_count": int(len(selected_sources)),
            "selected": selected,
            "ranked": ranked,
            "metrics": {
                "sparse_rag_rerank_bounded_count_observed": 1.0 if bounded_ready else 0.0,
                "sparse_rag_rerank_source_agreement_observed": 1.0 if source_agreement_ready else 0.0,
                "sparse_rag_rerank_contradiction_guard_observed": 1.0 if contradiction_guard_ready else 0.0,
                "sparse_rag_rerank_freshness_observed": 1.0 if freshness_ready else 0.0,
                "sparse_rag_rerank_citation_grounding_observed": 1.0 if citation_grounding_ready else 0.0,
                "sparse_rag_rerank_source_reliability_observed": 1.0 if source_reliability_ready else 0.0,
                "sparse_rag_rerank_source_diversity_observed": 1.0 if source_diversity_ready else 0.0,
            },
            "observed_only": True,
        }

    def decompose_query(
        self,
        query_text: str,
        max_subqueries: int = 3,
        min_tokens_per_subquery: int = 2,
    ) -> Dict[str, Any]:
        """Splits a query into bounded sparse subqueries for local retrieval."""

        tokens = _tokenize(query_text)
        subquery_limit = max(1, int(max_subqueries))
        min_tokens = max(1, int(min_tokens_per_subquery))
        if not tokens:
            return {
                "schema": "sara-rag-query-decomposition-v1",
                "query": query_text,
                "subqueries": [],
                "subquery_count": 0,
                "max_subqueries": int(subquery_limit),
                "observed_only": True,
                "metrics": {
                    "rag_query_decomposition_bounded_count_observed": 1.0,
                    "rag_query_decomposition_coverage_observed": 0.0,
                    "rag_query_decomposition_nonempty_observed": 0.0,
                },
            }

        separators = re.split(r"\b(?:and|or|with|plus|then|かつ|または|と|や)\b|[,;、。]", str(query_text))
        candidates = [
            part.strip()
            for part in separators
            if len(_tokenize(part.strip())) >= min_tokens
        ]
        if not candidates:
            window = max(min_tokens, max(2, len(tokens) // subquery_limit))
            candidates = [
                " ".join(tokens[index : index + window])
                for index in range(0, len(tokens), window)
            ]
        subqueries = _dedupe_preserving_order(candidates)[:subquery_limit]
        subquery_tokens = set()
        for subquery in subqueries:
            subquery_tokens.update(_tokenize(subquery))
        query_tokens = set(tokens)
        coverage = len(query_tokens.intersection(subquery_tokens)) / float(max(len(query_tokens), 1))
        bounded_ready = len(subqueries) <= subquery_limit
        nonempty_ready = bool(subqueries)
        return {
            "schema": "sara-rag-query-decomposition-v1",
            "query": query_text,
            "subqueries": subqueries,
            "subquery_count": int(len(subqueries)),
            "max_subqueries": int(subquery_limit),
            "token_coverage": float(coverage),
            "observed_only": True,
            "metrics": {
                "rag_query_decomposition_bounded_count_observed": 1.0 if bounded_ready else 0.0,
                "rag_query_decomposition_coverage_observed": 1.0 if coverage >= 0.75 else 0.0,
                "rag_query_decomposition_nonempty_observed": 1.0 if nonempty_ready else 0.0,
            },
        }

    def query_with_decomposed_rerank(
        self,
        query_text: str,
        top_k: int = 3,
        candidate_k: Optional[int] = None,
        max_subqueries: int = 3,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs bounded query decomposition before sparse RAG reranking."""

        decomposition = self.decompose_query(query_text, max_subqueries=max_subqueries)
        subqueries = list(decomposition.get("subqueries", []))
        if not subqueries:
            subqueries = [query_text] if query_text.strip() else []
        traces = [
            self.query_with_rerank(
                str(subquery),
                top_k=top_k,
                candidate_k=candidate_k,
                min_score=min_score,
            )
            for subquery in subqueries
        ]
        merged_by_text: Dict[str, Dict[str, Any]] = {}
        for trace in traces:
            for item in trace.get("selected", []) if isinstance(trace.get("selected", []), list) else []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "")
                if not text:
                    continue
                existing = merged_by_text.get(text)
                if existing is None or float(item.get("rerank_score", 0.0)) > float(existing.get("rerank_score", 0.0)):
                    merged_by_text[text] = dict(item)
        merged = sorted(
            merged_by_text.values(),
            key=lambda item: (
                float(item.get("rerank_score", 0.0)),
                float(item.get("source_agreement", 0.0)),
                float(item.get("freshness_score", 0.0)),
            ),
            reverse=True,
        )
        selected = merged[: max(int(top_k), 0)]
        decomposition_metrics = (
            decomposition.get("metrics", {})
            if isinstance(decomposition.get("metrics"), dict)
            else {}
        )
        subquery_hit_ready = bool(traces and all(bool(trace.get("selected", [])) for trace in traces))
        merged_ready = bool(selected)
        merged_citation_ready = bool(
            selected
            and all(str(item.get("citation_id", "") or "") for item in selected)
            and all(str(item.get("source", "") or "") for item in selected)
        )
        merged_source_reliability_ready = bool(
            selected
            and all(float(item.get("source_reliability", 0.0)) >= 1.0 for item in selected)
        )
        merged_sources = {str(item.get("source", "") or "") for item in selected if str(item.get("source", "") or "")}
        merged_source_diversity_ready = bool(
            len(selected) <= 1
            or len(merged_sources) >= min(len(selected), 2)
        )
        return {
            "schema": "sara-rag-decomposed-rerank-v1",
            "query": query_text,
            "decomposition": decomposition,
            "subquery_traces": traces,
            "selected": selected,
            "observed_only": True,
            "metrics": {
                **decomposition_metrics,
                "rag_query_decomposition_subquery_hit_observed": 1.0 if subquery_hit_ready else 0.0,
                "rag_query_decomposition_merged_selection_observed": 1.0 if merged_ready else 0.0,
                "rag_query_decomposition_merged_citation_grounding_observed": (
                    1.0 if merged_citation_ready else 0.0
                ),
                "rag_query_decomposition_merged_source_reliability_observed": (
                    1.0 if merged_source_reliability_ready else 0.0
                ),
                "rag_query_decomposition_merged_source_diversity_observed": (
                    1.0 if merged_source_diversity_ready else 0.0
                ),
            },
        }

    def query_with_context(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.0,
        context_separator: str = "\n---\n",
    ) -> str:
        """クエリに対して検索結果をコンテキスト文字列として返す。

        LLMのプロンプトに直接使えるフォーマットで返す。

        Args:
            query_text: 検索クエリのテキスト。
            top_k: 返すチャンク数の上限。
            min_score: 最低類似度スコア。
            context_separator: チャンク間の区切り文字列。

        Returns:
            検索結果を結合したコンテキスト文字列。
        """
        results = self.query(query_text, top_k=top_k, min_score=min_score)
        if not results:
            return ""

        context_parts = [text for text, _score in results]
        return context_separator.join(context_parts)

    def save(self, directory: str) -> None:
        """RAGパイプラインの状態を保存する。

        Args:
            directory: 保存先ディレクトリパス。
        """
        self.vector_store.save_pretrained(directory)

    def load(self, directory: str) -> None:
        """RAGパイプラインの状態を読み込む。

        Args:
            directory: 読み込み元ディレクトリパス。
        """
        self.vector_store = SNNVectorStore.from_pretrained(directory)
        self._chunk_count = len(self.vector_store.documents)
        self._chunk_metadata = [ChunkMetadata(source=f"loaded_{index}", chunk_index=index, total_chunks=self._chunk_count) for index in range(self._chunk_count)]

    @property
    def chunk_count(self) -> int:
        """格納されているチャンク数を返す。"""
        return self._chunk_count
