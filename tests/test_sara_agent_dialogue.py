from sara_engine.agent.sara_agent import SaraAgent


def test_agent_can_resolve_demonstrative_followup_after_teaching():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    agent.chat(
        "Pythonのリスト内包表記とは、既存のリストから新しいリストを短く簡潔に生成するための構文のことです。",
        teaching_mode=True,
    )
    agent.chat("Pythonのリスト内包表記とは？", teaching_mode=False)
    agent.chat(
        "そのメリットは、コードの行数が減って可読性が上がり、処理速度も速くなることです。",
        teaching_mode=True,
    )

    response = agent.chat("それを書くメリットは何ですか？", teaching_mode=False)

    assert "メリット" in response or "可読性" in response or "処理速度" in response
    assert "Fallback" not in response


def test_agent_records_tool_failures_without_breaking_chat():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    def failing_tool(_: str) -> str:
        raise RuntimeError("tool exploded")

    def working_tool(_: str) -> str:
        return "42"

    agent.register_tool("<FAIL>", failing_tool)
    agent.register_tool("<OK>", working_tool)

    response = agent.chat("計算して <FAIL> と <OK>", teaching_mode=False)
    issues = agent.get_recent_issues()

    assert "42" in response
    assert "Tool warnings" in response
    assert issues
    assert any(issue["stage"] == "tool_execution" for issue in issues)
    assert any("<FAIL>" in issue["message"] for issue in issues)


def test_agent_can_clear_runtime_issues():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    def failing_tool(_: str) -> str:
        raise RuntimeError("transient failure")

    agent.register_tool("<FAIL>", failing_tool)
    agent.chat("実行して <FAIL>", teaching_mode=False)

    assert agent.get_recent_issues()
    agent.clear_runtime_issues()
    assert agent.get_recent_issues() == []


def test_agent_retrieval_prefers_higher_query_keyword_coverage():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    candidate_partial = {
        "content": "【文脈: python 関数】 Pythonの関数は処理をまとめる単位です。",
        "score": 0.9,
    }
    candidate_precise = {
        "content": "【文脈: python 関数 引数】 Pythonの関数の引数は入力値を受け取るために使います。",
        "score": 0.7,
    }

    partial_scored = agent._score_retrieval_candidate(
        candidate_partial,
        current_keywords={"python", "関数", "引数"},
        context_keywords={"python", "関数"},
        has_demonstrative=False,
    )
    precise_scored = agent._score_retrieval_candidate(
        candidate_precise,
        current_keywords={"python", "関数", "引数"},
        context_keywords={"python", "関数"},
        has_demonstrative=False,
    )

    assert partial_scored is not None
    assert precise_scored is not None
    assert precise_scored["keyword_score"] > partial_scored["keyword_score"]
    assert precise_scored["current_keyword_coverage"] > partial_scored["current_keyword_coverage"]


def test_agent_retrieval_allows_demonstrative_followup_with_context_match():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    scored = agent._score_retrieval_candidate(
        {
            "content": "【文脈: リスト内包表記 メリット】 可読性が上がりコード量を減らせます。",
            "score": 0.4,
        },
        current_keywords={"メリット"},
        context_keywords={"リスト内包表記", "可読性"},
        has_demonstrative=True,
    )

    assert scored is not None
    assert scored["context_keyword_coverage"] > 0.0


def test_agent_routing_prefers_domain_expert_for_matching_keywords():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert", "biology"],
    )

    routing_sdr = set(agent.prefrontal.context_anchors["python_expert"])
    scored = agent._score_expert_routing(
        routing_sdr=routing_sdr,
        current_keywords={"python", "コード", "関数"},
        context_keywords={"python", "可読性"},
    )

    assert scored[0][0] == "python_expert"
    assert scored[0][1] > scored[-1][1]


def test_agent_routing_can_still_favor_general_when_no_domain_signal_exists():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert", "biology"],
    )

    scored = agent._score_expert_routing(
        routing_sdr=set(),
        current_keywords={"こんにちは"},
        context_keywords=set(),
    )

    assert scored[0][0] == "general"


def test_agent_memory_blending_keeps_supportive_memory_and_skips_unrelated_memory():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    memory_context, blended_memory = agent._blend_retrieval_memories(
        [
            {
                "clean_content": "Pythonの関数は再利用可能な処理のまとまりです。",
                "keyword_score": 9.0,
                "score": 0.8,
            },
            {
                "clean_content": "関数の引数は入力値を受け取るために使います。",
                "keyword_score": 8.0,
                "score": 0.7,
            },
            {
                "clean_content": "ミトコンドリアは細胞のエネルギーを作ります。",
                "keyword_score": 7.0,
                "score": 0.9,
            },
        ],
        current_keywords={"python", "関数", "引数"},
        context_keywords={"python", "コード"},
    )

    assert "Pythonの関数" in memory_context
    assert "補足" in memory_context
    assert "引数" in blended_memory
    assert "ミトコンドリア" not in blended_memory


def test_agent_prepare_teaching_memory_includes_bounded_metadata_keywords():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    content, metadata = agent._prepare_teaching_memory(
        user_text="Pythonの関数は入力値を受け取り結果を返します。",
        context="python_expert",
        context_keywords={"python", "関数", "入力値", "返す"},
        current_keywords={"python", "関数", "結果"},
    )

    assert content.startswith("【文脈:")
    assert metadata["context"] == "python_expert"
    assert "python" in metadata["keywords"]
    assert len(metadata["keywords"]) <= 6


def test_agent_topic_aware_fallback_uses_context_for_demonstratives():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    agent.topic_tracker.update(["リスト内包表記", "可読性"])

    message = agent._build_topic_aware_fallback(
        user_text="それはなぜですか？",
        current_keywords={"なぜ"},
        context_keywords={"リスト内包表記", "可読性"},
        active_experts=["python_expert"],
        mode="question",
        has_demonstrative=True,
    )

    assert "リスト内包表記" in message or "可読性" in message
    assert "主語" in message or "どの点" in message


def test_agent_topic_aware_fallback_requests_more_specific_query_terms():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    message = agent._build_topic_aware_fallback(
        user_text="Pythonの関数は？",
        current_keywords={"python", "関数"},
        context_keywords=set(),
        active_experts=["python_expert"],
        mode="question",
        has_demonstrative=False,
    )

    assert "python" in message.lower() or "関数" in message
    assert "具体的" in message


def test_agent_retrieval_diagnostics_capture_score_breakdown():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    agent._capture_retrieval_diagnostics(
        [
            {
                "clean_content": "Pythonの関数は再利用できます。",
                "keyword_score": 11.5,
                "current_keyword_coverage": 1.0,
                "context_keyword_coverage": 0.5,
                "metadata_keyword_coverage": 0.5,
                "retrieval_score_base": 0.8,
            }
        ]
    )

    diagnostics = agent.get_recent_retrieval_diagnostics()
    formatted = agent.format_recent_retrieval_diagnostics()

    assert diagnostics
    assert diagnostics[0]["keyword_score"] == 11.5
    assert "Recent retrieval diagnostics:" in formatted
    assert "metadata=0.50" in formatted


def test_agent_extract_keywords_preserves_followup_focus_terms():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    keywords = agent._extract_keywords("それを書くメリットとデメリットは何ですか？")

    assert "メリット" in keywords
    assert "デメリット" in keywords
