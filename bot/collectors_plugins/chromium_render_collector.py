"""Chromium render collector (Phase A skeleton).

- Captures rendered HTML for JS-heavy pages.
- Writes rendered snapshots under data/raw/autobot/rendered/YYYYMMDD/.
- Appends raw/render pair manifest to workspace/autobot/render_pairs.jsonl.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path

REQUIRES_NETWORK = True


def _safe_name(url: str) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).strip() or "index"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", base)[:180]


def _candidate_urls(bot, limit: int) -> list[str]:
    blocked = {d for d, s in bot.state.domain_reputation.items() if float(s) <= -1.5}
    seeds = bot.planner.next_seeds(bot._gap_signal(), blocked_domains=blocked)
    urls = []
    for u in seeds:
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        if not bot.policy.is_allowed_url(u):
            continue
        if not bot._in_my_shard(u):
            continue
        urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def _strip_html(html_text: str) -> str:
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _token_set(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u4e00-\u9fff]+", text.lower())
    return {t for t in toks if len(t) >= 2}


def _raw_render_delta(raw_text: str, rendered_text: str) -> float:
    a = _token_set(raw_text)
    b = _token_set(rendered_text)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    jaccard = inter / max(1, union)
    delta = 1.0 - jaccard
    return round(max(0.0, min(1.0, delta)), 4)


def _render_with_playwright(url: str, timeout_ms: int) -> tuple[str, str, str]:
    # Dynamic import so environments without playwright can still run plugin loader.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        title = page.title() or ""
        rendered_html = page.content()
        visible_text = page.inner_text("body") if page.query_selector("body") else ""
        browser.close()
    return title, rendered_html, visible_text


def collect(bot) -> int:
    # Safety gate: disabled by default unless explicitly enabled.
    if os.environ.get("AUTOBOT_CHROMIUM_ENABLED", "0").strip() not in {"1", "true", "TRUE"}:
        bot.event_logger.emit(
            "collector_plugin_skipped",
            {"plugin": "chromium_render_collector.py", "reason": "env_disabled"},
        )
        return 0

    day = datetime.utcnow().strftime("%Y%m%d")
    out_dir = raw_data_path("autobot", "rendered", day)
    os.makedirs(out_dir, exist_ok=True)

    pair_manifest = workspace_path("autobot", "render_pairs.jsonl")
    ensure_parent_directory(pair_manifest)

    urls = _candidate_urls(bot, limit=max(1, int(bot.policy.max_pages_per_cycle // 3)))
    if not urls:
        return 0

    added = 0
    for url in urls:
        domain = bot._domain_from_url(url)
        compliance = bot.compliance.decide_for_source(domain, "web")
        if not compliance.allowed:
            continue
        if not bot.policy.is_allowed_by_robots(url):
            continue

        try:
            title, rendered_html, visible_text = _render_with_playwright(
                url,
                timeout_ms=max(5000, int(bot.config.request_timeout_sec) * 1000),
            )
        except Exception as exc:
            bot._append_dead_letter("collector_plugin:chromium", url, "render_failed", str(exc))
            continue

        safe = _safe_name(url)
        render_path = os.path.join(out_dir, safe + ".rendered.html")
        text_path = os.path.join(out_dir, safe + ".visible.txt")

        with open(render_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(visible_text)

        # Also fetch raw snapshot for pair baseline when possible.
        raw_path = None
        try:
            raw_path = bot._download_url(url)
        except Exception:
            raw_path = None

        pair = {
            "ts": datetime.utcnow().isoformat(),
            "url": url,
            "domain": domain,
            "raw_path": raw_path,
            "render_path": render_path,
            "visible_text_path": text_path,
            "title": title,
            "compliance_level": compliance.level,
            "compliance_reason": compliance.reason,
            "js_required_hint": len(visible_text.strip()) > 0 and (raw_path is None),
        }

        raw_text = ""
        if raw_path and os.path.exists(raw_path):
            try:
                with open(raw_path, "r", encoding="utf-8", errors="ignore") as rf:
                    raw_text = _strip_html(rf.read(700_000))
            except Exception:
                raw_text = ""
        rendered_text = _strip_html(rendered_html)
        delta = _raw_render_delta(raw_text, rendered_text)
        pair["raw_vs_rendered_delta"] = delta

        # Phase B: use rendered visible text as direct training signal.
        summary_text = visible_text.strip() or rendered_text[:5000]
        if summary_text:
            decision = bot.quality_gate.evaluate(summary_text)
            if decision.accepted and (not bot._is_duplicate_content(summary_text)) and (not bot._is_semantic_duplicate(summary_text)):
                bot._append_record(
                    "web",
                    summary_text,
                    {
                        "url": url,
                        "collector": "chromium_render",
                        "quality": decision.score,
                        "raw_vs_rendered_delta": delta,
                        "compliance_level": compliance.level,
                        "compliance_reason": compliance.reason,
                    },
                )
                bot._update_language_stats(summary_text)
                bot._count_modality("text")
                priority = bot._compute_training_priority(modality="text", quality=decision.score, source="web")
                bot.training_queue.enqueue(
                    {
                        "source": "web",
                        "url": url,
                        "path": render_path,
                        "modality": "text",
                        "quality": decision.score,
                        "priority": priority,
                        "collector": "chromium_render",
                        "raw_vs_rendered_delta": delta,
                        "curriculum_stage": bot._curriculum_stage(decision.score, "web", render_delta=delta),
                    }
                )
                added += 1

        with open(pair_manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return added
