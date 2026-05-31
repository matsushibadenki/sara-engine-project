"""RSS/Atom collector plugin.

Plugin contract:
- export collect(bot) -> int
- return number of newly collected samples
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

_DEFAULT_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
]
REQUIRES_NETWORK = True


def _read_feeds_file(bot) -> list[str]:
    # Resolve from runtime state path: .../workspace/autobot/state.json -> rss_feeds.txt
    base_dir = os.path.dirname(getattr(bot, "state_path", "")) or ""
    path = os.path.join(base_dir, "rss_feeds.txt")
    if not os.path.exists(path):
        return []
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("http://") or s.startswith("https://"):
                    out.append(s)
    except Exception:
        return []
    return out


def _fetch_text(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "SARA-Autobot-RSS/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        data = res.read()
    return data.decode("utf-8", errors="ignore")


def _extract_links(feed_xml: str, base_url: str) -> list[str]:
    links: list[str] = []
    try:
        root = ET.fromstring(feed_xml)
    except Exception:
        # Fallback for malformed feeds: regex link sniffing.
        rough = re.findall(r"<link>(.*?)</link>", feed_xml, flags=re.IGNORECASE | re.DOTALL)
        for raw in rough:
            s = raw.strip()
            if s.startswith("http://") or s.startswith("https://"):
                links.append(s)
        return links[:100]

    for node in root.findall(".//item/link"):
        if node.text:
            s = node.text.strip()
            if s.startswith("http://") or s.startswith("https://"):
                links.append(s)

    atom_ns = "{http://www.w3.org/2005/Atom}"
    for node in root.findall(f".//{atom_ns}entry/{atom_ns}link"):
        href = (node.attrib.get("href") or "").strip()
        if href.startswith("http://") or href.startswith("https://"):
            links.append(href)

    # Relative link fallback.
    out: list[str] = []
    for s in links:
        out.append(urllib.parse.urljoin(base_url, s).split("#")[0])
    uniq = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq[:100]


def collect(bot) -> int:
    feeds = _DEFAULT_FEEDS + _read_feeds_file(bot)
    if not feeds:
        return 0

    visited = set(bot.state.visited_urls)
    added = 0

    for feed in feeds:
        try:
            xml_text = _fetch_text(feed, timeout=int(bot.config.request_timeout_sec))
            article_links = _extract_links(xml_text, feed)
        except Exception as exc:
            bot._append_dead_letter("collector_plugin:rss", feed, "feed_fetch_failed", str(exc))
            continue

        for url in article_links:
            if added >= max(1, int(bot.policy.max_pages_per_cycle // 2)):
                break
            if url in visited:
                continue
            if not bot.policy.is_allowed_url(url):
                continue
            if not bot.policy.is_allowed_by_robots(url):
                continue

            domain = bot._domain_from_url(url)
            if bot._is_domain_low_reputation(domain):
                continue
            compliance = bot.compliance.decide_for_source(domain, "web")
            if not compliance.allowed:
                bot._append_dead_letter("collector_plugin:rss", url, "compliance_denied", compliance.reason)
                continue

            try:
                local_file = bot._download_url(url)
                if not local_file:
                    continue
                from sara_engine.utils.multimodal_ingest import ingest_file

                rec = ingest_file(local_file)
                decision = bot.quality_gate.evaluate(rec.summary_text)
                if not decision.accepted:
                    bot._append_dead_letter("collector_plugin:rss", url, "quality_rejected", decision.reason)
                    bot._adjust_domain_score(domain, -0.03)
                    visited.add(url)
                    continue
                if bot._is_duplicate_content(rec.summary_text) or bot._is_semantic_duplicate(rec.summary_text):
                    bot._append_dead_letter("collector_plugin:rss", url, "duplicate_content", "rss_duplicate")
                    bot._adjust_domain_score(domain, -0.01)
                    visited.add(url)
                    continue

                bot._append_record(
                    "web",
                    rec.summary_text,
                    {
                        "url": url,
                        "quality": decision.score,
                        "collector": "rss",
                        "collected_at": datetime.utcnow().isoformat(),
                        "compliance_level": compliance.level,
                        "compliance_reason": compliance.reason,
                        **rec.metadata,
                    },
                )
                bot._update_language_stats(rec.summary_text)
                bot._count_modality(rec.modality)
                priority = bot._compute_training_priority(modality=rec.modality, quality=decision.score, source="web")
                bot.training_queue.enqueue(
                    {
                        "source": "web",
                        "url": url,
                        "path": local_file,
                        "modality": rec.modality,
                        "quality": decision.score,
                        "priority": priority,
                        "collector": "rss",
                    }
                )
                visited.add(url)
                bot._adjust_domain_score(domain, +0.06)
                bot.state.failed_attempts.pop(url, None)
                added += 1
            except Exception as exc:
                bot._append_dead_letter("collector_plugin:rss", url, "article_process_failed", str(exc))
                bot._adjust_domain_score(domain, -0.2)

    bot.state.visited_urls = sorted(visited)[-200_000:]
    return added
