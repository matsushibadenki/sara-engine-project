import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.utils.corpus import clean_corpus_lines, merge_wrapped_lines


def test_merge_wrapped_lines_joins_sentence_continuations():
    lines = [
        "1979年、福島邦彦は、畳み込み層とダウンサンプリング層、および重み複製を備えた",
        "（CNN）のディープラーニングアーキテクチャを導入した",
    ]

    merged = merge_wrapped_lines(lines)
    assert merged == [
        "1979年、福島邦彦は、畳み込み層とダウンサンプリング層、および重み複製を備えた（CNN）のディープラーニングアーキテクチャを導入した"
    ]


def test_clean_corpus_lines_keeps_real_sentences_and_removes_noise():
    lines = [
        "https://example.com/file.pdf",
        "ハードウェア上にCNNを実装した",
        "デジタル化された小切手の手書き数字を認識するために適用された",
    ]

    cleaned = clean_corpus_lines(lines, merge_wrapped=True)
    assert cleaned == [
        "ハードウェア上にCNNを実装した",
        "デジタル化された小切手の手書き数字を認識するために適用された",
    ]
