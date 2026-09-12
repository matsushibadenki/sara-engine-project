"""Conservative routing into the closed verified-evidence question scope."""

import re
from dataclasses import dataclass

from sara_engine.memory.topic_question import parse_topic_question


@dataclass(frozen=True)
class VerifiedChatRoute:
    decision: str
    parse_decision: str


def _contains(text, value, language):
    if language == "en":
        pattern = rf"(?<![0-9a-z]){re.escape(value.casefold())}(?![0-9a-z])"
        return re.search(pattern, text.casefold()) is not None
    return value in text


def route_verified_chat(text, *, language, aliases, markers=()):
    """Classify one bounded input as verified, protected abstention or ordinary.

    Declared markers widen only the protected scope. They never authorize an
    answer. Unknown domains that have no declared marker remain undetectable.
    """
    if not isinstance(text, str) or not 0 < len(text) <= 256:
        return VerifiedChatRoute("verified_abstention", "invalid_input")
    if language not in ("en", "ja", "zh-CN") or not isinstance(aliases, tuple):
        return VerifiedChatRoute("verified_abstention", "invalid_config")
    parsed = parse_topic_question(text, language=language, aliases=aliases)
    if parsed.decision == "parsed":
        return VerifiedChatRoute("verified", parsed.decision)
    if parsed.decision in ("invalid_input", "invalid_catalog"):
        return VerifiedChatRoute("verified_abstention", "invalid_config")
    if not isinstance(markers, tuple) or len(markers) > 16:
        return VerifiedChatRoute("verified_abstention", "invalid_config")
    normalized_markers = []
    forbidden = ("\n", "\r", ";", ".", "?") if language == "en" else ("\n", "\r", "。", "，", "、")
    for marker in markers:
        if (
            not isinstance(marker, str) or not 0 < len(marker) <= 64
            or marker != marker.strip() or any(value in marker for value in forbidden)
        ):
            return VerifiedChatRoute("verified_abstention", "invalid_config")
        normalized_markers.append(marker.casefold() if language == "en" else marker)
    if len(set(normalized_markers)) != len(normalized_markers):
        return VerifiedChatRoute("verified_abstention", "invalid_config")
    protected = tuple(pair[0] for pair in aliases if isinstance(pair, tuple) and len(pair) == 2) + markers
    if any(_contains(text, value, language) for value in protected):
        return VerifiedChatRoute("verified_abstention", parsed.decision)
    return VerifiedChatRoute("ordinary", parsed.decision)
