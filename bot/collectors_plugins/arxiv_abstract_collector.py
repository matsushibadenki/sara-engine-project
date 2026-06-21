"""arXiv abstract collector plugin.

Reads topic queries from workspace/autobot/arxiv_queries.txt and ingests only
paper metadata plus abstracts. Full paper text is not fetched or treated as
unrestricted training material.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List


REQUIRES_NETWORK = True
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _query_file(bot) -> str:
    base_dir = os.path.dirname(getattr(bot, "state_path", "")) or "workspace/autobot"
    return os.path.join(base_dir, "arxiv_queries.txt")


def read_arxiv_queries(bot) -> List[str]:
    path = _query_file(bot)
    if not os.path.exists(path):
        return []
    queries: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            queries.append(raw)
    return queries


def build_query_url(query: str, max_results: int = 5) -> str:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max(1, min(25, int(max_results)))),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    return f"{ARXIV_API}?{params}"


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "SARA-Autobot-ArxivAbstract/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


def _node_text(node, child_name: str) -> str:
    child = node.find(f"{ATOM_NS}{child_name}")
    return re.sub(r"\s+", " ", child.text or "").strip() if child is not None else ""


def parse_arxiv_entries(atom_xml: str) -> List[dict[str, object]]:
    try:
        root = ET.fromstring(atom_xml or "")
    except ET.ParseError:
        return []
    entries: List[dict[str, object]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = _node_text(entry, "title")
        summary = _node_text(entry, "summary")
        published = _node_text(entry, "published")
        paper_id = _node_text(entry, "id")
        authors = [
            _node_text(author, "name")
            for author in entry.findall(f"{ATOM_NS}author")
        ]
        authors = [author for author in authors if author]
        if not title or not summary:
            continue
        entries.append(
            {
                "title": title,
                "abstract": summary,
                "published": published,
                "paper_id": paper_id,
                "authors": authors[:8],
            }
        )
    return entries


def build_abstract_record(entry: dict[str, object], query: str) -> dict[str, object]:
    title = str(entry.get("title", "")).strip()
    abstract = str(entry.get("abstract", "")).strip()
    paper_id = str(entry.get("paper_id", "")).strip()
    record_text = f"Title: {title}. Abstract: {abstract}"
    return {
        "source": "arxiv_abstract",
        "record_text": record_text,
        "meta": {
            "url": paper_id,
            "source_url": paper_id,
            "source_type": "arxiv_abstract",
            "collector": "arxiv_abstract",
            "query": query,
            "title": title,
            "authors": entry.get("authors", []),
            "published": entry.get("published", ""),
            "source_domain": "arxiv.org",
            "license_hint": "abstract_metadata_only_mixed_license_preprint",
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
    compliance = bot.compliance.decide_for_source("arxiv.org", "web")
    if not compliance.allowed:
        bot._append_dead_letter("collector_plugin:arxiv_abstract", url, "compliance_denied", compliance.reason)
        return False
    decision = bot.quality_gate.evaluate(text)
    if not decision.accepted:
        bot._append_dead_letter("collector_plugin:arxiv_abstract", url, "quality_rejected", decision.reason)
        return False
    if bot._is_duplicate_content(text) or bot._is_semantic_duplicate(text):
        bot._append_dead_letter("collector_plugin:arxiv_abstract", url, "duplicate_content", "arxiv_duplicate")
        return False
    meta["quality"] = decision.score
    meta["compliance_level"] = compliance.level
    meta["compliance_reason"] = compliance.reason
    bot._append_record("arxiv_abstract", text, meta)
    bot._update_language_stats(text)
    bot._count_modality("text")
    priority = bot._compute_training_priority(modality="text", quality=decision.score, source="web")
    bot.training_queue.enqueue(
        {
            "source": "arxiv_abstract",
            "url": url,
            "modality": "text",
            "quality": decision.score,
            "priority": priority,
            "collector": "arxiv_abstract",
        }
    )
    return True


def collect(bot) -> int:
    queries = read_arxiv_queries(bot)
    if not queries:
        return 0
    visited = set(bot.state.visited_urls)
    added = 0
    for query in queries:
        try:
            payload = fetch_text(build_query_url(query, max_results=5), timeout=int(bot.config.request_timeout_sec))
            entries = parse_arxiv_entries(payload)
        except Exception as exc:
            bot._append_dead_letter("collector_plugin:arxiv_abstract", query, "query_failed", str(exc))
            continue
        for entry in entries:
            record = build_abstract_record(entry, query)
            url = str(record.get("meta", {}).get("url", "")) if isinstance(record.get("meta"), dict) else ""
            if url in visited:
                continue
            if _ingest_record(bot, record):
                added += 1
                visited.add(url)
    bot.state.visited_urls = sorted(visited)[-200_000:]
    return added
