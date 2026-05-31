# ディレクトリパス: src/sara_engine/utils/dialogue.py
# ファイルの日本語タイトル: ダイアログ・トピック・トラッカー
# ファイルの目的や内容: 対話中の重要な概念（トピック）を追跡し、適応的閾値ホメオスタシスを用いてサリエンス（重要度）を動的に管理する。

from typing import Dict, Iterable, List

from ..learning.homeostasis import AdaptiveThresholdHomeostasis


class AdaptiveTopicTracker:
    """Track salient dialogue concepts with fatigue/recovery dynamics."""

    def __init__(self) -> None:
        self._term_to_id: Dict[str, int] = {}
        self._id_to_term: Dict[int, str] = {}
        self._next_id = 0
        self._salience: Dict[int, float] = {}
        self._homeostasis = AdaptiveThresholdHomeostasis(
            target_rate=0.15,
            adaptation_rate=0.22,
            decay=0.86,
            min_threshold=0.0,
            max_threshold=2.5,
            global_weight=0.15,
        )

    def update(self, terms: Iterable[str]) -> None:
        normalized = [term.strip().lower() for term in terms if term and term.strip()]
        active_ids: List[int] = []

        for salience_id in list(self._salience.keys()):
            self._salience[salience_id] *= 0.88
            if self._salience[salience_id] < 0.02:
                del self._salience[salience_id]

        for term in normalized:
            term_id = self._ensure_term(term)
            active_ids.append(term_id)
            threshold = self._homeostasis.get_threshold(term_id, 0.0)
            gain = max(0.25, 1.0 - threshold * 0.35)
            self._salience[term_id] = min(3.0, self._salience.get(term_id, 0.0) + gain)

        self._homeostasis.update(active_ids, population_size=max(1, len(self._term_to_id)))

    def active_terms(self, limit: int = 6) -> List[str]:
        ranked = sorted(
            self._salience.items(),
            key=lambda item: item[1] / (1.0 + self._homeostasis.get_threshold(item[0], 0.0)),
            reverse=True,
        )
        return [self._id_to_term[term_id] for term_id, _ in ranked[:limit]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "term_to_id": dict(self._term_to_id),
            "id_to_term": dict(self._id_to_term),
            "next_id": self._next_id,
            "salience": dict(self._salience),
            "homeostasis": self._homeostasis.state_dict(),
        }

    def load_state(self, payload: Dict[str, object]) -> None:
        term_to_id = payload.get("term_to_id", {})
        id_to_term = payload.get("id_to_term", {})
        salience = payload.get("salience", {})

        if isinstance(term_to_id, dict) and isinstance(id_to_term, dict):
            self._term_to_id = {str(k): int(v) for k, v in term_to_id.items() if isinstance(v, (int, str))}
            self._id_to_term = {int(k): str(v) for k, v in id_to_term.items() if isinstance(k, (int, str))}
        
        next_id = payload.get("next_id")
        if isinstance(next_id, int):
            self._next_id = next_id

        if isinstance(salience, dict):
            self._salience = {int(k): float(v) for k, v in salience.items() if isinstance(k, (int, str)) and isinstance(v, (float, int, str))}

        homeostasis = payload.get("homeostasis")
        if isinstance(homeostasis, dict):
            self._homeostasis.load_state_dict(homeostasis)

    def _ensure_term(self, term: str) -> int:
        if term in self._term_to_id:
            return self._term_to_id[term]
        term_id = self._next_id
        self._next_id += 1
        self._term_to_id[term] = term_id
        self._id_to_term[term_id] = term
        return term_id
