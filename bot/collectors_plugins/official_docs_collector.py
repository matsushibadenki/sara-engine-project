"""Official documentation collector plugin.

Reads allowlisted documentation URLs from workspace/autobot/official_docs_urls.txt.
The collector is opt-in: no default URLs are fetched.
"""

from __future__ import annotations

import html
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from typing import List


REQUIRES_NETWORK = True


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if len(cleaned) >= 3:
            self.parts.append(cleaned)


def extract_visible_text(document_html: str, max_chars: int = 5000) -> str:
    parser = _VisibleTextParser()
    parser.feed(document_html or "")
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:max_chars]


def _config_file(bot) -> str:
    base_dir = os.path.dirname(getattr(bot, "state_path", "")) or "workspace/autobot"
    return os.path.join(base_dir, "official_docs_urls.txt")


def read_official_doc_urls(bot) -> List[str]:
    path = _config_file(bot)
    if not os.path.exists(path):
        return []
    urls: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("https://"):
                urls.append(raw)
    return urls


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "SARA-Autobot-OfficialDocs/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


def build_doc_record(url: str, document_html: str) -> dict[str, object]:
    text = extract_visible_text(document_html)
    return {
        "source": "official_docs",
        "record_text": text,
        "meta": {
            "url": url,
            "source_url": url,
            "source_type": "official_docs",
            "collector": "official_docs",
            "source_domain": urllib.parse.urlparse(url).netloc.lower(),
            "license_hint": "official_documentation_reference",
            "compliance_level": "warn",
            "collected_at": datetime.utcnow().isoformat(),
        },
    }


def _ingest_record(bot, record: dict[str, object]) -> bool:
    text = str(record.get("record_text", ""))
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    url = str(meta.get("url", ""))
    domain = str(meta.get("source_domain", ""))
    compliance = bot.compliance.decide_for_source(domain, "web")
    if not compliance.allowed:
        bot._append_dead_letter("collector_plugin:official_docs", url, "compliance_denied", compliance.reason)
        return False
    decision = bot.quality_gate.evaluate(text)
    if not decision.accepted:
        bot._append_dead_letter("collector_plugin:official_docs", url, "quality_rejected", decision.reason)
        return False
    if bot._is_duplicate_content(text) or bot._is_semantic_duplicate(text):
        bot._append_dead_letter("collector_plugin:official_docs", url, "duplicate_content", "official_docs_duplicate")
        return False
    meta["quality"] = decision.score
    meta["compliance_level"] = compliance.level
    meta["compliance_reason"] = compliance.reason
    bot._append_record("official_docs", text, meta)
    bot._update_language_stats(text)
    bot._count_modality("text")
    priority = bot._compute_training_priority(modality="text", quality=decision.score, source="web")
    bot.training_queue.enqueue(
        {
            "source": "official_docs",
            "url": url,
            "modality": "text",
            "quality": decision.score,
            "priority": priority,
            "collector": "official_docs",
        }
    )
    return True


def collect(bot) -> int:
    urls = read_official_doc_urls(bot)
    if not urls:
        return 0
    visited = set(bot.state.visited_urls)
    added = 0
    for url in urls:
        if url in visited:
            continue
        if not bot.policy.is_allowed_url(url):
            continue
        domain = urllib.parse.urlparse(url).netloc.lower()
        if bot._is_domain_low_reputation(domain):
            continue
        try:
            document_html = fetch_text(url, timeout=int(bot.config.request_timeout_sec))
            record = build_doc_record(url, document_html)
            if _ingest_record(bot, record):
                added += 1
                visited.add(url)
                bot._adjust_domain_score(domain, +0.05)
        except Exception as exc:
            bot._append_dead_letter("collector_plugin:official_docs", url, "fetch_failed", str(exc))
            bot._adjust_domain_score(domain, -0.1)
    bot.state.visited_urls = sorted(visited)[-200_000:]
    return added
