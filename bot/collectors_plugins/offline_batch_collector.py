"""Offline batch collector plugin.

Scans data/raw/offline_batch_inbox recursively and ingests files.
After successful processing, moves files to data/processed/offline_batch_done.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from sara_engine.utils.multimodal_ingest import ingest_file
from sara_engine.utils.project_paths import processed_data_path, raw_data_path


INBOX_DIR = raw_data_path("offline_batch_inbox")
DONE_DIR = processed_data_path("offline_batch_done")


def collect(bot) -> int:
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)

    processed = 0
    seen = set(bot.state.processed_files)

    for entry in sorted(Path(INBOX_DIR).glob("**/*")):
        if not entry.is_file():
            continue
        abs_path = str(entry.resolve())
        if abs_path in seen:
            continue

        try:
            rec = ingest_file(abs_path)
            compliance = bot.compliance.decide_for_source("local.offline_batch", "hot_inbox")
            if not compliance.allowed:
                bot._append_dead_letter("collector_plugin:offline_batch", abs_path, "compliance_denied", compliance.reason)
                continue

            decision = bot.quality_gate.evaluate(rec.summary_text)
            if not decision.accepted:
                bot._append_dead_letter("collector_plugin:offline_batch", abs_path, "quality_rejected", decision.reason)
                continue
            if bot._is_duplicate_content(rec.summary_text):
                bot._append_dead_letter("collector_plugin:offline_batch", abs_path, "duplicate_content", "hash_match")
                continue
            if bot._is_semantic_duplicate(rec.summary_text):
                bot._append_dead_letter("collector_plugin:offline_batch", abs_path, "semantic_duplicate", "simhash_near")
                continue

            bot._append_record(
                "hot_inbox",
                rec.summary_text,
                {
                    "quality": decision.score,
                    "collector": "offline_batch",
                    **rec.metadata,
                },
            )
            bot._update_language_stats(rec.summary_text)
            bot._count_modality(rec.modality)
            priority = bot._compute_training_priority(modality=rec.modality, quality=decision.score, source="hot_inbox")
            bot.training_queue.enqueue(
                {
                    "source": "hot_inbox",
                    "path": abs_path,
                    "modality": rec.modality,
                    "quality": decision.score,
                    "priority": priority,
                    "collector": "offline_batch",
                }
            )

            dest_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + entry.name
            dest_path = os.path.join(DONE_DIR, dest_name)
            shutil.move(abs_path, dest_path)

            seen.add(abs_path)
            bot.state.failed_attempts.pop(abs_path, None)
            processed += 1
        except Exception as exc:
            bot._append_dead_letter("collector_plugin:offline_batch", abs_path, "process_failed", str(exc))
            dropped = bot._record_failure(abs_path, "collector_plugin:offline_batch", str(exc))
            if dropped:
                seen.add(abs_path)

    bot.state.processed_files = sorted(seen)[-100_000:]
    return processed
