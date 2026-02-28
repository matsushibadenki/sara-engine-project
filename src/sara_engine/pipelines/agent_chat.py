_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/agent_chat.py",
    "//": "ファイルの日本語タイトル: エージェント・チャットパイプライン",
    "//": "ファイルの目的や内容: ユーザーがモデルの詳細を気にせず、テキストを入力するだけで自律的なツール実行とテキスト生成を行えるようにSaraAgentをラップする。"
}

from typing import Any, Callable

class AgentChatPipeline:
    """Pipeline for autonomous agent interaction using SNN Action Spikes."""
    def __init__(self, agent: Any):
        self.agent = agent

    def __call__(self, text: str, teaching_mode: bool = False, **kwargs: Any) -> str:
        """
        Processes the input text through the Spiking Agent.
        If tools are registered, the agent will autonomously use them during generation.
        """
        return str(self.agent.chat(user_text=text, teaching_mode=teaching_mode))

    def register_tool(self, trigger_spike: str, tool_func: Callable[[str], str]) -> None:
        """Allows dynamic attachment of external tools to the agent's nervous system."""
        if hasattr(self.agent, "register_tool"):
            self.agent.register_tool(trigger_spike, tool_func)
        else:
            raise AttributeError("The provided agent does not support tool registration.")