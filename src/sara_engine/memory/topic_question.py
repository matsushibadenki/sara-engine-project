"""Closed multilingual question grammar with externally declared topic aliases.

This is a deterministic adapter, not learned semantics or a general chat parser.
Unsupported wording returns no query instead of extracting a matching substring.
"""

import re
from dataclasses import dataclass

from sara_engine.memory.structured_query import TopicQuery


@dataclass(frozen=True)
class QuestionParse:
    decision: str
    query: TopicQuery | None = None


GRAMMARS = {
    "en": (r"exclude (.+); report (.+)\.", (r"what is (.+)\?", r"report (.+)\."), " and ", (";", ".", "?", "\n", "\r")),
    "ja": (r"(.+)を除いて、(.+)を教えてください。", (r"(.+)を教えてください。",), "と", ("を除いて", "を教えてください", "、", "。", "\n", "\r")),
    "zh-CN": (r"排除(.+)，请告诉我(.+)。", (r"请告诉我(.+)。",), "和", ("排除", "请告诉我", "，", "。", "\n", "\r")),
}


def parse_topic_question(text: str, *, language: str, aliases: tuple[tuple[str, str], ...]) -> QuestionParse:
    """Bound all inputs before normalization; match complete supported wording."""
    if not isinstance(language, str) or language not in GRAMMARS:
        return QuestionParse("invalid_input")
    if not isinstance(text, str) or not 0 < len(text) <= 256:
        return QuestionParse("invalid_input")
    if not isinstance(aliases, tuple) or not 0 < len(aliases) <= 32:
        return QuestionParse("invalid_catalog")
    exclusion, frames, separator, forbidden = GRAMMARS[language]
    catalog = {}
    for pair in aliases:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return QuestionParse("invalid_catalog")
        name, topic = pair
        if any(not isinstance(value, str) or not 0 < len(value) <= 128 or value != value.strip() for value in pair):
            return QuestionParse("invalid_catalog")
        name = name.casefold() if language == "en" else name
        if separator in name or any(marker in name for marker in forbidden) or name in catalog:
            return QuestionParse("invalid_catalog")
        catalog[name] = topic
    normalized = text.strip()
    if language == "en":
        normalized = normalized.casefold()
    matched = re.fullmatch(exclusion, normalized)
    if matched:
        excluded_text, requested_text = matched.groups()
    else:
        matched = next((match for frame in frames if (match := re.fullmatch(frame, normalized))), None)
        if matched is None:
            return QuestionParse("unsupported_question")
        requested_text, excluded_text = matched.group(1), None

    def resolve_list(value):
        if value is None:
            return ()
        names = value.split(separator)
        if len(names) > 4 or any(name not in catalog for name in names):
            return None
        topics = tuple(catalog[name] for name in names)
        return topics if len(set(topics)) == len(topics) else None

    requested, excluded = resolve_list(requested_text), resolve_list(excluded_text)
    if requested is None or excluded is None or set(requested).intersection(excluded):
        return QuestionParse("unsupported_question")
    return QuestionParse("parsed", TopicQuery(requested, excluded, language))
