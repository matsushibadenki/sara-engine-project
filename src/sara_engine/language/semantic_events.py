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

    def __init__(self, *, max_events: int = 128) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = int(max_events)

    def encode(self, text: str) -> Tuple[LanguageEvent, ...]:
        tokens = self._TOKEN_RE.findall(text)[: self.max_events]
        events = []
        for index, token in enumerate(tokens):
            if token in ".!?。！？":
                axis, feature = "boundary", token
            else:
                axis, feature = "orthographic", token.casefold()
            events.append(LanguageEvent(time=index, axis=axis, feature=feature))
        return tuple(events)

    def from_events(self, events: Iterable[LanguageEvent]) -> Tuple[LanguageEvent, ...]:
        return tuple(events)[: self.max_events]
