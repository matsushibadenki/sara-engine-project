from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityDecision:
    accepted: bool
    score: float
    reason: str


class QualityGate:
    """Lightweight quality and safety filter for ingestion records."""

    def __init__(self, extra_block_patterns: list[str] | None = None) -> None:
        self.secret_pattern = re.compile(r"(?i)(password|api[_-]?key|secret|private key)")
        self.pii_patterns = [
            re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
            re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b"),
            re.compile(r"\b(?:\d[ -]?){13,19}\b"),
            re.compile(r"(?i)\b(?:ssn|social security number|passport number|driver[']?s license)\b"),
        ]
        self.extra_block_patterns: list[re.Pattern[str]] = []
        for raw in extra_block_patterns or []:
            pattern = str(raw).strip()
            if not pattern:
                continue
            try:
                self.extra_block_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue

    def evaluate(self, text: str) -> QualityDecision:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) < 12:
            return QualityDecision(False, 0.0, "too_short")
        if cleaned.count("\x00") > 0:
            return QualityDecision(False, 0.0, "binary_noise")
        if self.secret_pattern.search(cleaned):
            return QualityDecision(False, 0.0, "possible_secret")
        for pat in self.pii_patterns:
            if pat.search(cleaned):
                return QualityDecision(False, 0.0, "possible_pii")
        for pat in self.extra_block_patterns:
            if pat.search(cleaned):
                return QualityDecision(False, 0.0, "blocked_pattern")

        jp = len(re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]", cleaned))
        en = len(re.findall(r"[A-Za-z]", cleaned))
        alnum = len(re.findall(r"[A-Za-z0-9]", cleaned))
        diversity = min(1.0, (jp + en) / max(1, len(cleaned)))
        density = min(1.0, alnum / max(1, len(cleaned)))
        score = round((diversity * 0.6 + density * 0.4), 3)
        if score < 0.08:
            return QualityDecision(False, score, "low_information")
        return QualityDecision(True, score, "accepted")
