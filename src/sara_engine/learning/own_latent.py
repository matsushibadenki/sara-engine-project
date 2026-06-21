from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{2,}")


def stable_event_id(value: str, width: int = 4096) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(width))


def tokenize_sparse_text(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text or ""))]


def build_sparse_signature(
    values: Iterable[str],
    *,
    width: int = 4096,
    max_events: int = 32,
) -> List[int]:
    counts = Counter(stable_event_id(value, width=width) for value in values if str(value).strip())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [event_id for event_id, _count in ordered[: max(1, int(max_events))]]


def jaccard_overlap(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set.intersection(right_set))) / float(len(left_set.union(right_set)))


@dataclass
class OwnLatentPrediction:
    label: str
    score: float
    predicted_signature: List[int]
    event_cost: int
    state_budget_units: int
    trace: Dict[str, object]


class SparseOwnLatentPredictor:
    """Sparse own-latent predictor using bounded local co-occurrence state."""

    def __init__(self, *, width: int = 4096, max_events: int = 32, top_k_labels: int = 3) -> None:
        self.width = max(16, int(width))
        self.max_events = max(1, int(max_events))
        self.top_k_labels = max(1, int(top_k_labels))
        self.label_event_counts: Dict[str, Counter[int]] = defaultdict(Counter)
        self.context_to_label_counts: Dict[int, Counter[str]] = defaultdict(Counter)
        self.label_counts: Counter[str] = Counter()
        self.update_count = 0

    def text_signature(self, text: str) -> List[int]:
        return build_sparse_signature(
            tokenize_sparse_text(text),
            width=self.width,
            max_events=self.max_events,
        )

    def latent_signature(self, latent_terms: Iterable[str]) -> List[int]:
        return build_sparse_signature(
            latent_terms,
            width=self.width,
            max_events=self.max_events,
        )

    def update(self, *, context_text: str, latent_terms: Iterable[str], label: str) -> None:
        clean_label = str(label).strip()
        if not clean_label:
            raise ValueError("label must be non-empty")
        context_signature = self.text_signature(context_text)
        target_signature = self.latent_signature(latent_terms)
        if not context_signature or not target_signature:
            raise ValueError("context and latent signatures must be non-empty")

        self.label_counts[clean_label] += 1
        for event_id in target_signature:
            self.label_event_counts[clean_label][int(event_id)] += 1
        for event_id in context_signature:
            self.context_to_label_counts[int(event_id)][clean_label] += 1
        self.update_count += 1

    def _label_signature(self, label: str) -> List[int]:
        counts = self.label_event_counts.get(label, Counter())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [event_id for event_id, _count in ordered[: self.max_events]]

    def predict(self, context_text: str) -> OwnLatentPrediction:
        context_signature = self.text_signature(context_text)
        label_scores: Counter[str] = Counter()
        event_cost = 0
        for event_id in context_signature:
            label_counts = self.context_to_label_counts.get(int(event_id), Counter())
            event_cost += len(label_counts)
            for label, count in label_counts.items():
                label_scores[label] += count

        if not label_scores:
            for label, count in self.label_counts.items():
                label_scores[label] += count
            event_cost += len(self.label_counts)

        if not label_scores:
            return OwnLatentPrediction(
                label="",
                score=0.0,
                predicted_signature=[],
                event_cost=event_cost,
                state_budget_units=0,
                trace={"reason": "empty_model"},
            )

        best_label, best_count = sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))[0]
        total = sum(label_scores.values())
        score = float(best_count) / float(max(1, total))
        label_signature = self._label_signature(best_label)
        return OwnLatentPrediction(
            label=best_label,
            score=round(score, 6),
            predicted_signature=label_signature,
            event_cost=event_cost + len(context_signature),
            state_budget_units=self.state_budget_units(),
            trace={
                "context_event_count": len(context_signature),
                "candidate_labels": [
                    {"label": label, "score": int(count)}
                    for label, count in sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))[
                        : self.top_k_labels
                    ]
                ],
            },
        )

    def state_budget_units(self) -> int:
        context_edges = sum(len(labels) for labels in self.context_to_label_counts.values())
        latent_events = sum(len(events) for events in self.label_event_counts.values())
        return int(context_edges + latent_events + len(self.label_counts))


class TokenOverlapBaseline:
    """Token-overlap reference baseline kept outside the production runtime path."""

    def __init__(self) -> None:
        self.examples: List[Tuple[Set[str], str]] = []

    def update(self, *, context_text: str, label: str) -> None:
        tokens = set(tokenize_sparse_text(context_text))
        if tokens and str(label).strip():
            self.examples.append((tokens, str(label).strip()))

    def predict(self, context_text: str) -> Tuple[str, float, int]:
        query = set(tokenize_sparse_text(context_text))
        if not self.examples:
            return "", 0.0, 0
        best_label = ""
        best_score = -1.0
        cost = 0
        for tokens, label in self.examples:
            cost += len(tokens)
            score = 0.0
            if query or tokens:
                score = float(len(query.intersection(tokens))) / float(len(query.union(tokens)))
            if score > best_score:
                best_label = label
                best_score = score
        return best_label, round(max(0.0, best_score), 6), cost


def train_predictor_from_cases(
    cases: Sequence[Mapping[str, object]],
    *,
    train_size: int,
) -> Tuple[SparseOwnLatentPredictor, TokenOverlapBaseline]:
    predictor = SparseOwnLatentPredictor()
    baseline = TokenOverlapBaseline()
    train_cases = [case for case in cases if str(case.get("split", "train")) == "train"][: max(0, int(train_size))]
    for case in train_cases:
        context_text = str(case.get("surface_text", ""))
        label = str(case.get("latent_group", ""))
        latent_terms = case.get("latent_terms", [])
        if not isinstance(latent_terms, list):
            latent_terms = []
        predictor.update(context_text=context_text, latent_terms=[str(item) for item in latent_terms], label=label)
        baseline.update(context_text=context_text, label=label)
    return predictor, baseline
