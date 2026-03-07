import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.utils.chat import ChatSessionHelper


def test_chat_helper_uses_recent_history_for_short_followup():
    helper = ChatSessionHelper(max_turns=3)
    helper.add_turn("user", "福島邦彦は何を導入しましたか")
    helper.add_turn("assistant", "畳み込みニューラルネットワークに関する説明です。")

    prompt = helper.build_prompt_text("および重み複製を備えた")
    assert "福島邦彦" in prompt
    assert "および重み複製を備えた" in prompt


def test_chat_helper_penalizes_repeated_response():
    helper = ChatSessionHelper(max_turns=3)
    helper.add_turn("assistant", "人工ニューラルネットワークです。")

    same_score = helper.rerank_score("人工ニューラルネットワークとは", "人工ニューラルネットワークです。")
    better_score = helper.rerank_score("人工ニューラルネットワークとは", "人工ニューラルネットワークは、重み付き結合を持つ計算モデルです。")
    assert better_score > same_score


def test_chat_helper_fallback_response_is_nonempty():
    helper = ChatSessionHelper(max_turns=3)
    assert helper.fallback_response("および重み複製を備えた")
    assert helper.fallback_response("これは何ですか？")
