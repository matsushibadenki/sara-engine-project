from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


SECRET_PATTERN = re.compile(r"(?i)(password|api[_-]?key|secret|private key|token=)")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class MaterialGateDecision:
    accepted: bool
    reason: str
    material_hash: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def material_hash(material: Dict[str, object]) -> str:
    basis = "|".join(
        [
            str(material.get("material_type", "")),
            normalize_text(str(material.get("prompt", ""))).lower(),
            normalize_text(str(material.get("answer", ""))).lower(),
            normalize_text(str(material.get("content", ""))).lower(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _contains_unsafe_text(text_values: Iterable[str]) -> bool:
    joined = "\n".join(text_values)
    return bool(
        SECRET_PATTERN.search(joined)
        or EMAIL_PATTERN.search(joined)
        or PHONE_PATTERN.search(joined)
        or CARD_PATTERN.search(joined)
    )


def _source_supports(material: Dict[str, object], source_text: str) -> bool:
    source = normalize_text(source_text).lower()
    material_type = str(material.get("material_type", ""))
    if material_type == "negative_query":
        prompt = normalize_text(str(material.get("prompt", ""))).lower()
        tokens = [token for token in re.findall(r"[a-zA-Z0-9_]{4,}", prompt) if token]
        return not any(token in source for token in tokens[-3:])

    support_text = normalize_text(
        str(material.get("answer", "")) or str(material.get("content", ""))
    ).lower()
    if not support_text:
        return False
    if support_text in source:
        return True
    keywords = [token for token in re.findall(r"[a-zA-Z0-9_]{5,}", support_text) if token]
    if not keywords:
        return len(support_text) >= 12 and support_text[:24] in source
    hits = sum(1 for token in set(keywords) if token in source)
    return hits >= max(1, min(3, len(set(keywords))))


class LearningMaterialGate:
    """Filters generated learning materials before they enter processed datasets."""

    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()

    def evaluate(self, material: Dict[str, object]) -> MaterialGateDecision:
        m_hash = material_hash(material)
        if m_hash in self.seen_hashes:
            return MaterialGateDecision(False, "duplicate_material", m_hash)

        source_text = normalize_text(str(material.get("source_text", "")))
        prompt = normalize_text(str(material.get("prompt", "")))
        answer = normalize_text(str(material.get("answer", "")))
        content = normalize_text(str(material.get("content", "")))
        if len(source_text) < 24:
            return MaterialGateDecision(False, "source_too_short", m_hash)
        if len(prompt or content) < 8:
            return MaterialGateDecision(False, "material_too_short", m_hash)
        if _contains_unsafe_text([source_text, prompt, answer, content]):
            return MaterialGateDecision(False, "possible_secret_or_pii", m_hash)
        if not _source_supports(material, source_text):
            return MaterialGateDecision(False, "unsupported_by_source", m_hash)

        self.seen_hashes.add(m_hash)
        return MaterialGateDecision(True, "accepted", m_hash)


def split_accepted_rejected(materials: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    gate = LearningMaterialGate()
    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for material in materials:
        decision = gate.evaluate(material)
        item = dict(material)
        item["material_hash"] = decision.material_hash
        item["gate_reason"] = decision.reason
        item["accepted"] = decision.accepted
        if decision.accepted:
            accepted.append(item)
        else:
            rejected.append(item)
    return {"accepted": accepted, "rejected": rejected}
