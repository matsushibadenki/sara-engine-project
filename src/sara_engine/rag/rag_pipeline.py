# ディレクトリパス: src/sara_engine/rag/rag_pipeline.py
# ファイル名: rag_pipeline.py
# ファイルの目的や内容: SNN (Spiking Neural Network) ベースのRAGパイプライン。
#   ドキュメントのチャンク分割、SDRエンコード、ベクトルストアへの格納、
#   クエリ検索、コンテキスト生成を一貫して行う。

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

    @property
    def chunk_count(self) -> int:
        """格納されているチャンク数を返す。"""
        return self._chunk_count
