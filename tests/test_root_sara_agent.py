from agent.sara_codex_agent import (
    ApprovalMode,
    SaraCodexAgent,
    SaraCodexAgentConfig,
    TrainingDataIndex,
    TrainingRecord,
)


def _agent_with_records(records):
    index = TrainingDataIndex(records)
    config = SaraCodexAgentConfig(data_paths=(), top_k=2, approval_mode=ApprovalMode.AUTO)
    return SaraCodexAgent(config=config, index=index)


def test_agent_retrieves_project_training_data():
    agent = _agent_with_records(
        [
            TrainingRecord(
                prompt="Pythonとは何ですか？",
                response="Pythonは読みやすさを重視したプログラミング言語です。",
                source="data/interim/chat_data.jsonl",
                kind="chat_jsonl",
            ),
            TrainingRecord(
                prompt="ミトコンドリアとは？",
                response="ミトコンドリアは細胞のエネルギー産生に関わります。",
                source="data/raw/chat_data.jsonl",
                kind="chat_jsonl",
            ),
        ]
    )

    result = agent.run("Pythonの特徴を教えて")

    assert result.hits
    assert "Pythonは読みやすさ" in result.answer
    assert result.steps[0].name == "understand"


def test_agent_runs_safe_calculator_tool_in_auto_mode():
    agent = _agent_with_records([])

    result = agent.run("2 + 3 * 4 を計算して")

    assert result.tool_outputs["calculator"] == "14"
    assert "calculator: 14" in result.answer


def test_agent_suggest_mode_does_not_execute_tools():
    index = TrainingDataIndex([])
    config = SaraCodexAgentConfig(data_paths=(), approval_mode=ApprovalMode.SUGGEST)
    agent = SaraCodexAgent(config=config, index=index)

    result = agent.run("2 + 3 * 4 を計算して")

    assert result.tool_outputs == {}
    assert any(step.status == "suggested" for step in result.steps)
