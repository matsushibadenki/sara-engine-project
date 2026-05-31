import re
from dataclasses import dataclass
from typing import List


@dataclass
class ChatTurn:
    role: str
    text: str


class ChatSessionHelper:
    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self.history: List[ChatTurn] = []

    def add_turn(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.history.append(ChatTurn(role=role, text=cleaned))
        max_items = self.max_turns * 2
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

    def build_prompt_text(self, user_input: str) -> str:
        trimmed = user_input.strip()
        if not trimmed:
            return trimmed

        if len(trimmed) >= 18 or not self.history:
            return trimmed

        recent_user = self._last_turn("user")
        recent_assistant = self._last_turn("assistant")
        context_parts: List[str] = []
        if recent_user and recent_user != trimmed:
            context_parts.append(recent_user)
        if recent_assistant:
            context_parts.append(recent_assistant[:80])
        context_parts.append(trimmed)
        return "\n".join(context_parts[-3:])

    def rerank_score(self, user_input: str, response: str) -> float:
        text = response.strip()
        if not text:
            return -1e9

        score = 0.0
        user_terms = self._extract_terms(user_input)
        response_terms = self._extract_terms(text)
        overlap = len(user_terms & response_terms)
        score += overlap * 3.0

        if text.endswith(("。", "！", "？")):
            score += 2.0
        if 8 <= len(text) <= 140:
            score += 1.0
        if re.search(r"(です|である|した|された|している)", text):
            score += 1.0

        last_assistant = self._last_turn("assistant")
        if last_assistant and text == last_assistant:
            score -= 8.0
        if last_assistant and last_assistant and text in last_assistant:
            score -= 4.0

        if user_input.strip() and text == user_input.strip():
            score -= 8.0

        repeated = self._has_repeated_clause(text)
        if repeated:
            score -= 8.0

        return score

    def fallback_response(self, user_input: str) -> str:
        stripped = user_input.strip()
        if not stripped:
            return "..."
        if len(stripped) <= 12:
            return "続きが曖昧です。前後をもう少し長く入力してください。"
        if re.search(r"(何|なに|とは|どれ|なぜ|どうして|\?)", stripped):
            return "学習済み知識の中で近い表現を十分に想起できません。言い換えるか、キーワードを増やしてください。"
        return "学習済み知識から十分な続きや説明を想起できません。前後の文脈を少し足してください。"

    def _last_turn(self, role: str) -> str:
        for item in reversed(self.history):
            if item.role == role:
                return item.text
        return ""

    def _extract_terms(self, text: str) -> set[str]:
        candidates = re.findall(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        return {item for item in candidates if len(item) >= 2}

    def _has_repeated_clause(self, text: str) -> bool:
        for span in range(8, min(32, len(text) // 2 + 1)):
            seen: dict[str, int] = {}
            for i in range(0, len(text) - span + 1):
                frag = text[i:i + span]
                if len(frag.strip()) < max(4, span // 2):
                    continue
                seen[frag] = seen.get(frag, 0) + 1
                if seen[frag] >= 3:
                    return True
        return False
