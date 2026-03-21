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
