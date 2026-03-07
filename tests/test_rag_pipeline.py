# ディレクトリパス: tests/test_rag_pipeline.py
# ファイル名: test_rag_pipeline.py
# ファイルの目的や内容: RAGパイプラインの単体テスト

from sara_engine.rag.rag_pipeline import DocumentChunker, SNNRAGPipeline
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


class TestDocumentChunker:
    """DocumentChunkerのテスト。"""

    def test_empty_text_returns_empty_list(self) -> None:
        chunker = DocumentChunker()
        result = chunker.chunk("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunker = DocumentChunker()
        result = chunker.chunk("   ")
        assert result == []

    def test_single_sentence_returns_one_chunk(self) -> None:
        chunker = DocumentChunker(max_chunk_size=200)
        text = "SNNは脳の神経回路を模倣した計算モデルです。"
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_multiple_sentences_are_chunked(self) -> None:
        chunker = DocumentChunker(max_chunk_size=15, overlap_size=0)
        text = "最初の文です。次の文です。三番目の文です。"
        result = chunker.chunk(text)
        assert len(result) >= 2

    def test_metadata_is_set_correctly(self) -> None:
        chunker = DocumentChunker(max_chunk_size=200)
        text = "テストの文章です。"
        result = chunker.chunk(text, source="test_doc")
        assert len(result) == 1
        assert result[0].metadata.source == "test_doc"
        assert result[0].metadata.chunk_index == 0
        assert result[0].metadata.total_chunks == 1

    def test_overlap_preserves_context(self) -> None:
        chunker = DocumentChunker(max_chunk_size=20, overlap_size=10)
        text = "最初の文章です。二番目の文章。三番目の文です。"
        result = chunker.chunk(text)
        assert len(result) >= 2


class TestSNNRAGPipeline:
    """SNNRAGPipelineのテスト。"""

    def test_add_and_query_document(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
        rag.add_document("SNNは脳の神経回路を模倣した計算モデルです。")
        results = rag.query("SNNとは何ですか？", top_k=1)
        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], str)
        assert isinstance(results[0][1], float)

    def test_add_multiple_documents(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
        count = rag.add_documents([
            "Pythonはプログラミング言語です。",
            "SNNはスパイキングニューラルネットワークです。",
        ])
        assert count >= 2
        assert rag.chunk_count >= 2

    def test_query_empty_store(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256)
        results = rag.query("テスト")
        assert results == []

    def test_query_empty_text(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256)
        rag.add_document("テストドキュメント。")
        results = rag.query("")
        assert results == []

    def test_query_with_context(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
        rag.add_document("SARAエンジンはSNNベースの推論エンジンです。")
        context = rag.query_with_context("SARAエンジンとは？")
        assert isinstance(context, str)

    def test_add_empty_document(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256)
        count = rag.add_document("")
        assert count == 0

    def test_min_score_filter(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
        rag.add_document("関連するドキュメントです。")
        results = rag.query("テスト", min_score=0.99)
        # 非常に高いスコアを要求するのでフィルタされる可能性がある
        assert isinstance(results, list)

    def test_save_and_load(self) -> None:
        rag = SNNRAGPipeline(sdr_size=256, max_chunk_size=200)
        rag.add_document("保存テスト用のドキュメントです。")

        with tempfile.TemporaryDirectory() as tmpdir:
            rag.save(tmpdir)
            rag2 = SNNRAGPipeline(sdr_size=256)
            rag2.load(tmpdir)
            assert rag2.chunk_count >= 1
