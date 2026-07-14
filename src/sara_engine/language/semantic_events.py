"""Convert raw text into bounded, source-labelled sparse language events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class LanguageEvent:
    """One observed or proposed sparse language event."""

    time: int
    axis: str
    feature: str
    source: str = "surface"
    evidence_type: str = "observed"
    confidence: float = 1.0
    role: str = ""


class SparseLanguageEventAdapter:
    """Tokenize text without a parser and emit bounded surface events."""

    _TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
    _NEGATION_TOKENS = {"not", "never", "no", "ない", "無い", "ません"}

    def __init__(self, *, max_events: int = 128) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = int(max_events)

    def encode(self, text: str) -> Tuple[LanguageEvent, ...]:
        tokens = self._TOKEN_RE.findall(text)[: self.max_events]
        events = []
        for index, token in enumerate(tokens):
            if len(events) >= self.max_events:
                break
            if token in ".!?。！？":
                axis, feature = "boundary", token
            else:
                axis, feature = "orthographic", token.casefold()
            events.append(LanguageEvent(time=index, axis=axis, feature=feature))
            normalized = token.casefold()
            if len(events) < self.max_events and (
                normalized in self._NEGATION_TOKENS
                or normalized.endswith("ない")
            ):
                events.append(
                    LanguageEvent(
                        time=index,
                        axis="semantic",
                        feature="negation",
                        source="bounded_lexicon",
                        evidence_type="dictionary_assisted",
                        confidence=0.8,
                        role="scope",
                    )
                )
        return tuple(events)

    def from_events(self, events: Iterable[LanguageEvent]) -> Tuple[LanguageEvent, ...]:
        return tuple(events)[: self.max_events]
