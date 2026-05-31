from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapabilityGapSignal:
    text_ratio: float
    image_ratio: float
    audio_ratio: float
    video_ratio: float
    binary_ratio: float
    jp_ratio: float
    en_ratio: float


class CollectionPlanner:
    """Decides crawl focus from observed modality balance."""

    def __init__(self) -> None:
        self.default_seeds = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD",
            "https://arxiv.org/list/cs.AI/recent",
        ]
        self.image_seeds = [
            "https://commons.wikimedia.org/wiki/Main_Page",
            "https://www.pexels.com/search/technology/",
        ]
        self.audio_seeds = [
            "https://librivox.org/",
            "https://freemusicarchive.org/",
        ]
        self.jp_text_seeds = [
            "https://ja.wikipedia.org/wiki/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92",
            "https://ja.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86",
        ]
        self.en_text_seeds = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
        ]

    def next_seeds(self, gap: CapabilityGapSignal, blocked_domains: set[str] | None = None) -> list[str]:
        blocked_domains = blocked_domains or set()
        seeds = list(self.default_seeds)
        if gap.image_ratio < 0.1:
            seeds.extend(self.image_seeds)
        if gap.audio_ratio < 0.05:
            seeds.extend(self.audio_seeds)
        if gap.jp_ratio < 0.35:
            seeds.extend(self.jp_text_seeds)
        if gap.en_ratio < 0.35:
            seeds.extend(self.en_text_seeds)
        filtered: list[str] = []
        for seed in seeds:
            host = seed.split("//", 1)[-1].split("/", 1)[0].lower()
            if host in blocked_domains:
                continue
            filtered.append(seed)
        return filtered
