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
