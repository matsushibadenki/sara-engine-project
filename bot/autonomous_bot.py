#!/usr/bin/env python3
"""Autonomous multimodal crawler + continual training orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from bot.evaluation_gate import EvaluationGate
from bot.compliance import SourceCompliance
from bot.event_log import EventLogger
from bot.io_utils import atomic_write_json
from bot.model_registry import ModelRegistry
from bot.planner import CapabilityGapSignal, CollectionPlanner
from bot.policy import CrawlPolicy
from bot.quality_gate import QualityGate
from bot.promotion_policy import can_promote, resolve_policy
from bot.promotion_score_gate import evaluate_score_gate
from bot.state_store import BotState, StateStore
from bot.training_queue import TrainingQueue
from sara_engine.utils.multimodal_ingest import ingest_file  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    model_path,
    processed_data_path,
    raw_data_path,
    workspace_path,
)


@dataclass
class BotConfig:
    crawl_seeds: list[str]
    crawl_interval_sec: int
    train_interval_sec: int
    idle_sleep_sec: int
    request_timeout_sec: int
    max_pages_per_cycle: int
    high_queue_threshold: int
    high_failure_threshold: int
    dead_letter_rate_threshold: int
    cooldown_cycles: int
    promotion_policy: str
    canary_min_corpus_lines: int
    max_backup_count: int
    strict_allowlist_mode: bool
    allowed_domains: list[str]
    raw_data_retention_days: int
    max_dead_letter_lines: int
    max_event_lines: int
    compliance_policy_path: str
    compliance_preset: str
    weekly_digest_path: str
    alert_failed_items_threshold: int
    alert_queue_pending_threshold: int
    weekly_gate_max_failed_items: int
    weekly_gate_max_avg_queue: int
    semantic_hamming_threshold: int
    promotion_min_score: float
    max_records_lines: int
    max_corpus_lines: int
    critical_alert_window_minutes: int
    critical_alert_threshold: int
    hot_done_retention_days: int
    quality_block_patterns: list[str]
    audit_export_enabled: bool
    audit_snapshot_path: str
    replay_interval_sec: int
    replay_samples_per_cycle: int
    replay_min_quality: float
    alert_dedup_window_sec: int
    collector_plugins_enabled: bool
    collector_plugins_dir: str
    offline_mode: bool
    shard_id: int
    total_shards: int
    cooperative_training_enabled: bool
    training_leader_shard: int
    curriculum_enabled: bool
    curriculum_easy_ratio: float
    curriculum_medium_ratio: float
    curriculum_hard_ratio: float
    render_delta_medium_threshold: float
    render_delta_hard_threshold: float
    benchmark_min_pass_rate: float
    benchmark_max_latency_ms: float


class AutonomousLearningBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.running = True
        self.last_train_ts = 0.0

        self.hot_inbox_dir = raw_data_path("hot_inbox")
        self.hot_done_dir = processed_data_path("hot_done")
        self.web_raw_dir = raw_data_path("autobot", "web")
        self.records_jsonl = processed_data_path("autobot", "multimodal_records.jsonl")
        self.corpus_path = processed_data_path("autobot", "corpus.txt")
        self.state_path = workspace_path("autobot", "state.json")
        self.model_root_dir = model_path("autobot_self_organized")
        self.pid_path = workspace_path("autobot", "bot.pid")
        self.queue_path = workspace_path("autobot", "train_queue.json")
        self.eval_report_path = workspace_path("autobot", "eval_report.json")
        self.benchmark_latest_path = workspace_path("autobot", "benchmark_latest.json")
        self.registry_meta_path = workspace_path("autobot", "model_registry.json")
        self.dead_letter_path = workspace_path("autobot", "dead_letter.jsonl")
        self.metrics_path = workspace_path("autobot", "metrics.json")
        self.alerts_path = workspace_path("autobot", "alerts.log")
        self.log_path = workspace_path("autobot", "bot.log")
        self.shutdown_path = workspace_path("autobot", "shutdown_status.json")
        self.events_path = workspace_path("autobot", "events.jsonl")
        self.daily_digest_path = workspace_path("autobot", "daily_digest.jsonl")
        self.daily_digest_dir = workspace_path("autobot", "digests")
        self.weekly_digest_text_path = workspace_path("autobot", "weekly_digest.txt")
        self.audit_snapshot_path = workspace_path("autobot", "audit_snapshot.json")
        self.collector_plugins_dir = os.path.join(PROJECT_ROOT, self.config.collector_plugins_dir)

        for p in [
            self.hot_inbox_dir,
            self.hot_done_dir,
            self.web_raw_dir,
            self.records_jsonl,
            self.corpus_path,
            self.state_path,
            self.pid_path,
            self.dead_letter_path,
            self.metrics_path,
            self.alerts_path,
            self.log_path,
            self.shutdown_path,
            self.events_path,
            self.daily_digest_path,
            self.weekly_digest_text_path,
        ]:
            ensure_parent_directory(p)
        os.makedirs(self.daily_digest_dir, exist_ok=True)
        os.makedirs(self.model_root_dir, exist_ok=True)

        self.state_store = StateStore(self.state_path)
        self.state: BotState = self.state_store.load()
        self.policy = CrawlPolicy(max_pages_per_cycle=self.config.max_pages_per_cycle)
        self.policy.strict_allowlist_mode = bool(self.config.strict_allowlist_mode)
        self.policy.allowed_domains = tuple(self.config.allowed_domains)
        self.planner = CollectionPlanner()
        self.quality_gate = QualityGate(extra_block_patterns=list(self.config.quality_block_patterns))
        self.training_queue = TrainingQueue(self.queue_path)
        self.evaluation_gate = EvaluationGate(
            self.eval_report_path,
            benchmark_path=self.benchmark_latest_path,
            benchmark_min_pass_rate=float(self.config.benchmark_min_pass_rate),
            benchmark_max_latency_ms=float(self.config.benchmark_max_latency_ms),
        )
        self.model_registry = ModelRegistry(self.model_root_dir, self.registry_meta_path)
        self.event_logger = EventLogger(self.events_path)
        policy_path = self.config.compliance_policy_path.strip()
        if policy_path and not os.path.isabs(policy_path):
            policy_path = os.path.join(PROJECT_ROOT, policy_path)
        self.compliance = SourceCompliance(policy_path=policy_path)
        self.compliance.apply_preset(self.config.compliance_preset)
        self.promotion_policy = resolve_policy(self.config.promotion_policy)

        self.current_cycle = 0
        self.cooldown_until_cycle = 0
        self.last_dead_letter_count = 0
        self.last_alert_signature = ""
        self._low_reputation_threshold = -1.5
        self.shutdown_reason = "running"
        self.last_digest_date = ""
        self.last_escalation_ts = ""
        self._alert_dedup_cache: dict[str, float] = {}
        self.alert_suppressed_total = 0
        self.last_snapshot: dict[str, object] = {
            "actions": ["normal"],
            "hot_processed": 0,
            "web_processed": 0,
            "new_samples": 0,
            "queue_pending": 0,
            "queue_recovered_count": 0,
            "state_recovered_count": self.state_store.recovered_count(),
            "jp_ratio": 0.0,
            "en_ratio": 0.0,
            "failed_item_count": 0,
            "dead_letter_total": 0,
            "dead_letter_delta": 0,
            "cycle": 0,
            "cooldown_until_cycle": 0,
            "effective_max_pages": self.policy.max_pages_per_cycle,
        }

    def _run_plugin_collectors(self) -> int:
        if not bool(self.config.collector_plugins_enabled):
            return 0
        if not os.path.isdir(self.collector_plugins_dir):
            return 0
        total = 0
        for name in sorted(os.listdir(self.collector_plugins_dir)):
            if not name.endswith(".py") or name.startswith("_"):
                continue
            path = os.path.join(self.collector_plugins_dir, name)
            module_name = f"bot.collector_plugin.{name[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                collect_fn = getattr(mod, "collect", None)
                requires_network = bool(getattr(mod, "REQUIRES_NETWORK", False))
                if bool(self.config.offline_mode) and requires_network:
                    self.event_logger.emit(
                        "collector_plugin_skipped",
                        {"plugin": name, "reason": "offline_mode_network_plugin"},
                    )
                    continue
                if not callable(collect_fn):
                    self.event_logger.emit("collector_plugin_skipped", {"plugin": name, "reason": "missing_collect"})
                    continue
                result = collect_fn(self)
                count = int(result) if result is not None else 0
                total += max(0, count)
                self.event_logger.emit("collector_plugin_ran", {"plugin": name, "collected": max(0, count)})
            except Exception as exc:
                self._append_dead_letter("collector_plugin", path, "plugin_runtime_error", str(exc))
                self.event_logger.emit(
                    "collector_plugin_error",
                    {"plugin": name, "error": str(exc)[:500]},
                )
        return total

    def _in_my_shard(self, key: str) -> bool:
        total = max(1, int(self.config.total_shards))
        shard = max(0, int(self.config.shard_id))
        if total == 1:
            return True
        bucket = int(hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % total
        return bucket == shard

    def _can_train_on_this_shard(self) -> bool:
        if not bool(self.config.cooperative_training_enabled):
            return True
        return int(self.config.shard_id) == int(self.config.training_leader_shard)

    def _write_audit_snapshot(self, snapshot: dict[str, object]) -> None:
        if not bool(self.config.audit_export_enabled):
            return
        registry: dict[str, object] = {}
        try:
            if os.path.exists(self.registry_meta_path):
                with open(self.registry_meta_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    registry = obj
        except Exception:
            registry = {}

        payload = {
            "ts": datetime.utcnow().isoformat(),
            "cycle": int(snapshot.get("cycle", 0)),
            "ingestion": {
                "hot_processed": int(snapshot.get("hot_processed", 0)),
                "web_processed": int(snapshot.get("web_processed", 0)),
                "new_samples": int(snapshot.get("new_samples", 0)),
                "dead_letter_total": int(snapshot.get("dead_letter_total", 0)),
                "failed_item_count": int(snapshot.get("failed_item_count", 0)),
                "modality_counts": dict(self.state.modality_counts),
                "language_counts": dict(self.state.language_counts),
            },
            "queue": self.training_queue.stats().__dict__,
            "promotion": {
                "policy": self.promotion_policy.mode,
                "last_attempt_promoted": registry.get("last_attempt_promoted"),
                "last_attempt_reason": registry.get("last_attempt_reason"),
                "last_promoted_at": registry.get("last_promoted_at"),
                "last_eval_passed": bool(self.state.last_eval_passed),
            },
            "paths": {
                "records_jsonl": self.records_jsonl,
                "corpus": self.corpus_path,
                "events": self.events_path,
                "dead_letter": self.dead_letter_path,
            },
        }

        target = self.config.audit_snapshot_path.strip() or self.audit_snapshot_path
        if not os.path.isabs(target):
            target = os.path.join(PROJECT_ROOT, target)
        ensure_parent_directory(target)
        atomic_write_json(target, payload)

    def _effective_train_interval(self) -> int:
        base = int(self.config.train_interval_sec)
        queue_pending = int(self.last_snapshot.get("queue_pending", 0))
        failed_items = int(self.last_snapshot.get("failed_item_count", 0))
        actions = set(str(x) for x in self.last_snapshot.get("actions", []))

        factor = 1.0
        if queue_pending >= int(self.config.high_queue_threshold):
            factor *= 0.5
        if failed_items >= int(self.config.high_failure_threshold):
            factor *= 1.6
        if "dead_letter_spike" in actions or "high_failure_mode" in actions:
            factor *= 1.8
        if not bool(self.state.last_eval_passed):
            factor *= 1.4
        effective = int(max(30, min(3600, round(base * factor))))
        return effective

    def _curriculum_stage(self, quality: float, source: str, render_delta: float | None = None) -> str:
        q = float(quality)
        src = (source or "").strip().lower()
        if render_delta is not None:
            d = max(0.0, min(1.0, float(render_delta)))
            if d >= float(self.config.render_delta_hard_threshold):
                return "hard"
            if d >= float(self.config.render_delta_medium_threshold):
                return "medium"
        if src == "replay":
            return "hard"
        if q < 0.35:
            return "easy"
        if q < 0.7:
            return "medium"
        return "hard"

    def _select_replay_records(self, limit: int, min_quality: float) -> list[dict[str, object]]:
        if limit < 1 or not os.path.exists(self.records_jsonl):
            return []
        try:
            with open(self.records_jsonl, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-5000:]
        except Exception:
            return []

        candidates: list[dict[str, object]] = []
        for line in lines:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta", {})
            if not isinstance(meta, dict):
                continue
            q = float(meta.get("quality", 0.0) or 0.0)
            if q < min_quality:
                continue
            txt = str(obj.get("record_text", "")).strip()
            if len(txt) < 16:
                continue
            candidates.append({"quality": q, "source": str(obj.get("source", "")), "record_text": txt})
        if not candidates:
            return []
        candidates.sort(key=lambda x: float(x.get("quality", 0.0)), reverse=True)

        cursor = max(0, int(self.state.replay_cursor))
        picked = []
        for i in range(limit):
            idx = (cursor + i) % len(candidates)
            picked.append(candidates[idx])
        self.state.replay_cursor = (cursor + limit) % len(candidates)
        return picked

    def _inject_replay_samples(self) -> int:
        now = time.time()
        interval = int(self.config.replay_interval_sec)
        if interval < 1:
            return 0
        if not hasattr(self, "_last_replay_ts"):
            self._last_replay_ts = 0.0
        if now - float(self._last_replay_ts) < interval:
            return 0
        limit = int(self.config.replay_samples_per_cycle)
        min_quality = float(self.config.replay_min_quality)
        records = self._select_replay_records(limit=limit, min_quality=min_quality)
        if not records:
            return 0
        for rec in records:
            self.training_queue.enqueue(
                {
                    "source": "replay",
                    "path": "records_jsonl",
                    "modality": "text",
                    "quality": float(rec.get("quality", 0.0)),
                    "priority": round(min(1.5, 0.9 + float(rec.get("quality", 0.0)) * 0.4), 4),
                    "curriculum_stage": self._curriculum_stage(float(rec.get("quality", 0.0)), "replay"),
                    "note": "scheduled_high_value_replay",
                }
            )
        self._last_replay_ts = now
        self.event_logger.emit(
            "replay_injected",
            {
                "count": len(records),
                "min_quality": min_quality,
            },
        )
        return len(records)

    def _save_state(self) -> None:
        self.state_store.save(self.state)

    def _gap_signal(self) -> CapabilityGapSignal:
        counts = self.state.modality_counts
        total = max(1, sum(int(v) for v in counts.values()))
        lang = self.state.language_counts
        lang_total = max(1, int(lang.get("jp", 0)) + int(lang.get("en", 0)) + int(lang.get("other", 0)))
        return CapabilityGapSignal(
            text_ratio=float(counts.get("text", 0)) / total,
            image_ratio=float(counts.get("image", 0)) / total,
            audio_ratio=float(counts.get("audio", 0)) / total,
            video_ratio=float(counts.get("video", 0)) / total,
            binary_ratio=float(counts.get("binary", 0)) / total,
            jp_ratio=float(lang.get("jp", 0)) / lang_total,
            en_ratio=float(lang.get("en", 0)) / lang_total,
        )

    def _count_modality(self, modality: str) -> None:
        self.state.modality_counts[modality] = int(self.state.modality_counts.get(modality, 0)) + 1

    def _domain_from_url(self, url: str) -> str:
        try:
            return urllib.parse.urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _domain_score(self, domain: str) -> float:
        return float(self.state.domain_reputation.get(domain, 0.0))

    def _adjust_domain_score(self, domain: str, delta: float) -> None:
        if not domain:
            return
        current = float(self.state.domain_reputation.get(domain, 0.0))
        updated = max(-3.0, min(2.0, current + delta))
        self.state.domain_reputation[domain] = round(updated, 4)

    def _is_domain_low_reputation(self, domain: str) -> bool:
        if not domain:
            return False
        return self._domain_score(domain) <= self._low_reputation_threshold

    def _is_duplicate_content(self, text: str) -> bool:
        normalized = " ".join(text.split()).strip().lower()
        if not normalized:
            return True
        digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()
        seen = set(self.state.content_hashes)
        if digest in seen:
            return True
        self.state.content_hashes.append(digest)
        if len(self.state.content_hashes) > 200_000:
            self.state.content_hashes = self.state.content_hashes[-200_000:]
        return False

    def _text_language(self, text: str) -> str:
        jp = sum(1 for ch in text if ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff"))
        en = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
        if jp > en and jp >= 6:
            return "jp"
        if en > jp and en >= 8:
            return "en"
        return "other"

    def _simhash64(self, text: str) -> int:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u4e00-\u9fff]+", text.lower())
        if not tokens:
            tokens = text.lower().split()
        if not tokens:
            return 0
        vec = [0] * 64
        for tok in tokens[:512]:
            h = int(hashlib.md5(tok.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
            for i in range(64):
                if (h >> i) & 1:
                    vec[i] += 1
                else:
                    vec[i] -= 1
        out = 0
        for i, v in enumerate(vec):
            if v >= 0:
                out |= (1 << i)
        return out

    def _hamming64(self, a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _is_semantic_duplicate(self, text: str) -> bool:
        sig = self._simhash64(text)
        recent = self.state.semantic_hashes[-4000:]
        threshold = int(self.config.semantic_hamming_threshold)
        for other in recent:
            if self._hamming64(sig, int(other)) <= threshold:
                return True
        self.state.semantic_hashes.append(int(sig))
        if len(self.state.semantic_hashes) > 120_000:
            self.state.semantic_hashes = self.state.semantic_hashes[-120_000:]
        return False

    def _update_language_stats(self, text: str) -> None:
        tag = self._text_language(text)
        self.state.language_counts[tag] = int(self.state.language_counts.get(tag, 0)) + 1

    def _compute_training_priority(self, *, modality: str, quality: float, source: str) -> float:
        # Higher priority for high-quality, underrepresented modalities, and human-provided data.
        modality_counts = self.state.modality_counts
        total = max(1, sum(int(v) for v in modality_counts.values()))
        ratio = float(modality_counts.get(modality, 0)) / total
        scarcity = 1.0 - min(1.0, ratio * 5.0)

        source_bonus = 0.12 if source == "hot_inbox" else 0.0
        quality_term = max(0.0, min(1.0, float(quality)))
        score = quality_term * 0.55 + scarcity * 0.33 + source_bonus
        return round(max(0.0, min(1.5, score)), 4)

    def _append_record(self, source: str, record_text: str, meta: dict[str, object]) -> None:
        with open(self.records_jsonl, "a", encoding="utf-8") as jf:
            payload = {
                "source": source,
                "record_text": record_text,
                "meta": meta,
                "ts": datetime.utcnow().isoformat(),
            }
            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with open(self.corpus_path, "a", encoding="utf-8") as cf:
            cf.write(record_text.replace("\n", " ").strip() + "\n")
        self._trim_file_lines(self.records_jsonl, max_lines=int(self.config.max_records_lines))
        self._trim_file_lines(self.corpus_path, max_lines=int(self.config.max_corpus_lines))

    def _append_dead_letter(self, source: str, item: str, reason: str, detail: str = "") -> None:
        self._rotate_dead_letter_if_needed(max_bytes=5_000_000)
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "source": source,
            "item": item,
            "reason": reason,
            "detail": detail[:500],
        }
        with open(self.dead_letter_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._trim_file_lines(self.dead_letter_path, max_lines=int(self.config.max_dead_letter_lines))

    def _rotate_dead_letter_if_needed(self, max_bytes: int = 5_000_000) -> None:
        try:
            if not os.path.exists(self.dead_letter_path):
                return
            if os.path.getsize(self.dead_letter_path) < max_bytes:
                return
            rotated = self.dead_letter_path + ".1"
            if os.path.exists(rotated):
                os.remove(rotated)
            os.replace(self.dead_letter_path, rotated)
            with open(self.dead_letter_path, "w", encoding="utf-8"):
                pass
            print("[INFO] Rotated dead_letter.jsonl due to size limit.")
        except Exception:
            return

    def _trim_file_lines(self, path: str, max_lines: int) -> None:
        if max_lines < 1 or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if len(lines) <= max_lines:
                return
            keep = lines[-max_lines:]
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
        except Exception:
            return

    def _prune_raw_data_by_retention(self) -> int:
        days = int(self.config.raw_data_retention_days)
        if days < 1:
            return 0
        cutoff = datetime.utcnow() - timedelta(days=days)
        removed = 0
        try:
            for name in os.listdir(self.web_raw_dir):
                full = os.path.join(self.web_raw_dir, name)
                if not os.path.isdir(full):
                    continue
                # Expect YYYYMMDD folders.
                try:
                    dt = datetime.strptime(name, "%Y%m%d")
                except ValueError:
                    continue
                if dt < cutoff:
                    shutil.rmtree(full, ignore_errors=True)
                    removed += 1
        except OSError:
            return removed
        return removed

    def _prune_hot_done_by_retention(self) -> int:
        days = int(self.config.hot_done_retention_days)
        if days < 1:
            return 0
        cutoff = datetime.utcnow() - timedelta(days=days)
        removed = 0
        try:
            for name in os.listdir(self.hot_done_dir):
                full = os.path.join(self.hot_done_dir, name)
                if not os.path.isfile(full):
                    continue
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(full))
                except OSError:
                    continue
                if mtime < cutoff:
                    try:
                        os.remove(full)
                        removed += 1
                    except OSError:
                        continue
        except OSError:
            return removed
        return removed

    def _dead_letter_count(self) -> int:
        if not os.path.exists(self.dead_letter_path):
            return 0
        count = 0
        with open(self.dead_letter_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
        return count

    def _record_failure(self, key: str, source: str, detail: str) -> bool:
        attempts = int(self.state.failed_attempts.get(key, 0)) + 1
        self.state.failed_attempts[key] = attempts
        if attempts >= 3:
            self._append_dead_letter(source, key, "max_retries_exceeded", detail)
            return True
        return False

    def _control_actions(self, hot_count: int, crawl_count: int, total_new: int) -> dict[str, object]:
        queue_stats = self.training_queue.stats()
        failed_item_count = len(self.state.failed_attempts)
        dead_letter_total = self._dead_letter_count()
        dead_letter_delta = max(0, dead_letter_total - self.last_dead_letter_count)
        self.last_dead_letter_count = dead_letter_total

        actions: list[str] = []
        should_skip_training = False
        effective_max_pages = self.policy.max_pages_per_cycle

        if self.current_cycle < self.cooldown_until_cycle:
            actions.append("cooldown_active")
            should_skip_training = True
            effective_max_pages = max(1, self.policy.max_pages_per_cycle // 4)

        if queue_stats.pending >= self.config.high_queue_threshold:
            actions.append("queue_backpressure")
            effective_max_pages = min(effective_max_pages, max(1, self.policy.max_pages_per_cycle // 3))

        if failed_item_count >= self.config.high_failure_threshold:
            actions.append("high_failure_mode")
            should_skip_training = True
            self.cooldown_until_cycle = max(self.cooldown_until_cycle, self.current_cycle + self.config.cooldown_cycles)
            effective_max_pages = min(effective_max_pages, max(1, self.policy.max_pages_per_cycle // 5))

        if dead_letter_delta >= self.config.dead_letter_rate_threshold:
            actions.append("dead_letter_spike")
            should_skip_training = True
            self.cooldown_until_cycle = max(self.cooldown_until_cycle, self.current_cycle + self.config.cooldown_cycles)
            effective_max_pages = min(effective_max_pages, max(1, self.policy.max_pages_per_cycle // 5))

        if not actions:
            actions.append("normal")

        return {
            "actions": actions,
            "should_skip_training": should_skip_training,
            "effective_max_pages": effective_max_pages,
            "queue_pending": queue_stats.pending,
            "queue_recovered_count": queue_stats.recovered_count,
            "state_recovered_count": self.state_store.recovered_count(),
            "jp_ratio": self._gap_signal().jp_ratio,
            "en_ratio": self._gap_signal().en_ratio,
            "low_reputation_domain_count": sum(1 for _d, s in self.state.domain_reputation.items() if float(s) <= self._low_reputation_threshold),
            "records_max_lines": int(self.config.max_records_lines),
            "corpus_max_lines": int(self.config.max_corpus_lines),
            "failed_item_count": failed_item_count,
            "dead_letter_total": dead_letter_total,
            "dead_letter_delta": dead_letter_delta,
            "hot_processed": hot_count,
            "web_processed": crawl_count,
            "new_samples": total_new,
            "cycle": self.current_cycle,
            "cooldown_until_cycle": self.cooldown_until_cycle,
        }

    def _write_metrics(self, snapshot: dict[str, object]) -> None:
        metrics = {
            "ts": datetime.utcnow().isoformat(),
            "hot_processed": int(snapshot.get("hot_processed", 0)),
            "web_processed": int(snapshot.get("web_processed", 0)),
            "new_samples": int(snapshot.get("new_samples", 0)),
            "queue_pending": int(snapshot.get("queue_pending", 0)),
            "queue_recovered_count": int(snapshot.get("queue_recovered_count", 0)),
            "state_recovered_count": int(snapshot.get("state_recovered_count", 0)),
            "jp_ratio": float(snapshot.get("jp_ratio", 0.0)),
            "en_ratio": float(snapshot.get("en_ratio", 0.0)),
            "low_reputation_domain_count": int(snapshot.get("low_reputation_domain_count", 0)),
            "records_max_lines": int(snapshot.get("records_max_lines", 0)),
            "corpus_max_lines": int(snapshot.get("corpus_max_lines", 0)),
            "visited_url_count": len(self.state.visited_urls),
            "processed_file_count": len(self.state.processed_files),
            "failed_item_count": int(snapshot.get("failed_item_count", 0)),
            "dead_letter_total": int(snapshot.get("dead_letter_total", 0)),
            "dead_letter_delta": int(snapshot.get("dead_letter_delta", 0)),
            "control_actions": list(snapshot.get("actions", [])),
            "cooldown_until_cycle": int(snapshot.get("cooldown_until_cycle", 0)),
            "effective_max_pages": int(snapshot.get("effective_max_pages", self.policy.max_pages_per_cycle)),
            "modality_counts": self.state.modality_counts,
            "alert_suppressed_total": int(self.alert_suppressed_total),
        }
        atomic_write_json(self.metrics_path, metrics)

    def _rotate_alerts_if_needed(self, max_bytes: int) -> None:
        try:
            if not os.path.exists(self.alerts_path):
                return
            if os.path.getsize(self.alerts_path) < max_bytes:
                return
            rotated = self.alerts_path + ".1"
            if os.path.exists(rotated):
                os.remove(rotated)
            os.replace(self.alerts_path, rotated)
            with open(self.alerts_path, "w", encoding="utf-8"):
                pass
        except Exception:
            return

    def _rotate_log_if_needed(self, max_bytes: int = 10_000_000) -> None:
        try:
            if not os.path.exists(self.log_path):
                return
            if os.path.getsize(self.log_path) < max_bytes:
                return
            rotated = self.log_path + ".1"
            if os.path.exists(rotated):
                os.remove(rotated)
            os.replace(self.log_path, rotated)
            with open(self.log_path, "w", encoding="utf-8"):
                pass
            print("[INFO] Rotated bot.log due to size limit.")
        except Exception:
            return

    def _emit_alert_if_needed(self, snapshot: dict[str, object]) -> None:
        actions = [str(a) for a in snapshot.get("actions", [])]
        actionable = [a for a in actions if a != "normal"]
        if not actionable:
            return

        signature = (
            f"actions={','.join(actionable)}|"
            f"queue={int(snapshot.get('queue_pending', 0))}|"
            f"failed={int(snapshot.get('failed_item_count', 0))}|"
            f"dead_delta={int(snapshot.get('dead_letter_delta', 0))}|"
            f"cooldown={int(snapshot.get('cooldown_until_cycle', 0))}"
        )
        if signature == self.last_alert_signature:
            return
        if not self._should_emit_alert_signature(signature):
            return
        self.last_alert_signature = signature

        self._rotate_alerts_if_needed(max_bytes=2_000_000)

        severity = "WARN"
        if (
            "high_failure_mode" in actionable
            or "dead_letter_spike" in actionable
            or int(snapshot.get("dead_letter_delta", 0)) >= max(1, self.config.dead_letter_rate_threshold)
        ):
            severity = "CRITICAL"

        line = (
            f"ALERT ts={datetime.utcnow().isoformat()} "
            f"severity={severity} "
            f"cycle={int(snapshot.get('cycle', 0))} "
            f"actions={','.join(actionable)} "
            f"queue_pending={int(snapshot.get('queue_pending', 0))} "
            f"failed_items={int(snapshot.get('failed_item_count', 0))} "
            f"dead_letter_delta={int(snapshot.get('dead_letter_delta', 0))} "
            f"cooldown_until={int(snapshot.get('cooldown_until_cycle', 0))}"
        )
        print(f"[ALERT] {line}")
        with open(self.alerts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.event_logger.emit(
            "alert",
            {
                "line": line,
                "actions": actionable,
                "cycle": int(snapshot.get("cycle", 0)),
            },
        )
        self._maybe_emit_escalation()

    def _maybe_emit_escalation(self) -> None:
        if not os.path.exists(self.alerts_path):
            return
        window_min = int(self.config.critical_alert_window_minutes)
        threshold = int(self.config.critical_alert_threshold)
        if window_min < 1 or threshold < 1:
            return

        cutoff = datetime.utcnow() - timedelta(minutes=window_min)
        critical_count = 0
        try:
            with open(self.alerts_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-2000:]
        except Exception:
            return

        for line in reversed(lines):
            if "severity=CRITICAL" not in line:
                continue
            ts_key = "ts="
            i = line.find(ts_key)
            if i < 0:
                continue
            ts_start = i + len(ts_key)
            ts_end = line.find(" ", ts_start)
            raw_ts = line[ts_start:] if ts_end < 0 else line[ts_start:ts_end]
            try:
                dt = datetime.fromisoformat(raw_ts)
            except Exception:
                continue
            if dt < cutoff:
                break
            critical_count += 1

        if critical_count < threshold:
            return

        now_iso = datetime.utcnow().isoformat()
        if self.last_escalation_ts:
            try:
                last = datetime.fromisoformat(self.last_escalation_ts)
                if (datetime.utcnow() - last) < timedelta(minutes=window_min):
                    return
            except Exception:
                pass

        self.last_escalation_ts = now_iso
        line = (
            f"ALERT ts={now_iso} severity=CRITICAL kind=escalation "
            f"critical_count={critical_count} window_minutes={window_min} threshold={threshold}"
        )
        with open(self.alerts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.event_logger.emit(
            "alert_escalation",
            {
                "critical_count": critical_count,
                "window_minutes": window_min,
                "threshold": threshold,
            },
        )
        self._trim_file_lines(self.events_path, max_lines=int(self.config.max_event_lines))

    def _should_emit_alert_signature(self, signature: str) -> bool:
        now = time.time()
        window_sec = int(self.config.alert_dedup_window_sec)
        if window_sec < 1:
            return True
        last = float(self._alert_dedup_cache.get(signature, 0.0))
        if last > 0 and (now - last) < window_sec:
            self.alert_suppressed_total += 1
            return False
        self._alert_dedup_cache[signature] = now
        if len(self._alert_dedup_cache) > 4000:
            # Keep cache bounded.
            items = sorted(self._alert_dedup_cache.items(), key=lambda kv: kv[1], reverse=True)[:2000]
            self._alert_dedup_cache = {k: v for k, v in items}
        return True

    def _run_promotion_flow(self, report: dict[str, object]) -> None:
        score_gate = evaluate_score_gate(
            eval_report=report,
            snapshot=self.last_snapshot,
            min_score=float(self.config.promotion_min_score),
        )
        self.event_logger.emit(
            "promotion_score_gate",
            {
                "passed": score_gate.passed,
                "score": score_gate.score,
                "reasons": score_gate.reasons,
                "threshold": float(self.config.promotion_min_score),
            },
        )
        if not score_gate.passed:
            print(f"[INFO] Promotion skipped by score gate: score={score_gate.score} reasons={','.join(score_gate.reasons)}")
            return

        weekly_allowed, weekly_reason = self._weekly_gate_allows_promotion()
        if not weekly_allowed:
            self.event_logger.emit(
                "promotion_weekly_gate_blocked",
                {"reason": weekly_reason},
            )
            print(f"[INFO] Promotion skipped by weekly gate: {weekly_reason}")
            return

        allowed, reason = can_promote(report, self.promotion_policy)
        if not allowed:
            self.event_logger.emit(
                "promotion_skipped",
                {
                    "reason": reason,
                    "policy": self.promotion_policy.mode,
                    "report_passed": bool(report.get("passed", False)),
                },
            )
            print(f"[INFO] Promotion skipped: {reason}")
            return

        promotion = self.model_registry.promote_candidate(report)
        self.event_logger.emit(
            "promotion_attempt",
            {
                "promoted": promotion.promoted,
                "reason": promotion.reason,
                "policy": self.promotion_policy.mode,
            },
        )
        print(f"[INFO] Promotion: promoted={promotion.promoted} reason={promotion.reason} production={promotion.production_dir}")

        canary_report = self.evaluation_gate.evaluate(self.corpus_path, self.model_registry.production_dir)
        canary_ok = bool(canary_report.get("passed", False)) and int(canary_report.get("corpus_lines", 0)) >= int(self.config.canary_min_corpus_lines)
        self.event_logger.emit(
            "canary_result",
            {
                "passed": canary_ok,
                "corpus_lines": int(canary_report.get("corpus_lines", 0)),
                "min_required": int(self.config.canary_min_corpus_lines),
            },
        )
        if not canary_ok:
            rolled_back = self.model_registry.rollback_latest()
            self.event_logger.emit(
                "rollback",
                {
                    "rolled_back": rolled_back,
                    "reason": "canary_failed",
                },
            )
            print(f"[WARN] Canary failed, rollback={rolled_back}")

        removed = self.model_registry.prune_backups(max_keep=int(self.config.max_backup_count))
        if removed > 0:
            self.event_logger.emit("backup_pruned", {"removed": removed})

    def _weekly_gate_allows_promotion(self) -> tuple[bool, str]:
        target = self.config.weekly_digest_path.strip() or workspace_path("autobot", "weekly_digest.json")
        if not os.path.isabs(target):
            target = os.path.join(PROJECT_ROOT, target)
        if not os.path.exists(target):
            return True, "no_weekly_digest"
        try:
            with open(target, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return True, "invalid_weekly_digest"
            max_failed = int(payload.get("max_failed_items", 0) or 0)
            avg_queue = float(payload.get("avg_queue_pending", 0.0) or 0.0)
            if max_failed > int(self.config.weekly_gate_max_failed_items):
                return False, f"max_failed_items:{max_failed}>{self.config.weekly_gate_max_failed_items}"
            if avg_queue > float(self.config.weekly_gate_max_avg_queue):
                return False, f"avg_queue_pending:{avg_queue}>{self.config.weekly_gate_max_avg_queue}"
            return True, "ok"
        except Exception:
            return True, "weekly_digest_read_error"

    def _write_shutdown_status(self) -> None:
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "reason": self.shutdown_reason,
            "cycle": int(self.current_cycle),
            "last_snapshot": self.last_snapshot,
        }
        atomic_write_json(self.shutdown_path, payload)

    def _write_daily_digest(self, snapshot: dict[str, object]) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        payload = {
            "date": today,
            "cycle": int(snapshot.get("cycle", 0)),
            "new_samples": int(snapshot.get("new_samples", 0)),
            "queue_pending": int(snapshot.get("queue_pending", 0)),
            "failed_item_count": int(snapshot.get("failed_item_count", 0)),
            "dead_letter_total": int(snapshot.get("dead_letter_total", 0)),
            "actions": list(snapshot.get("actions", [])),
            "low_reputation_domain_count": int(snapshot.get("low_reputation_domain_count", 0)),
            "policy": self.promotion_policy.mode,
        }

        # Append one JSON line for automation-friendly history.
        with open(self.daily_digest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        # Keep only latest 4000 lines in digest history.
        self._trim_file_lines(self.daily_digest_path, max_lines=4000)

        # Write one human-readable summary file per day.
        summary_file = os.path.join(self.daily_digest_dir, f"{today}.txt")
        lines = [
            f"Date: {today}",
            f"Cycle: {payload['cycle']}",
            f"New samples: {payload['new_samples']}",
            f"Queue pending: {payload['queue_pending']}",
            f"Failed items: {payload['failed_item_count']}",
            f"Dead letters total: {payload['dead_letter_total']}",
            f"Low reputation domains: {payload['low_reputation_domain_count']}",
            f"Actions: {','.join(payload['actions'])}",
            f"Promotion policy: {payload['policy']}",
        ]
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _update_weekly_digest(self) -> None:
        source = self.daily_digest_path
        target = self.config.weekly_digest_path.strip() or workspace_path("autobot", "weekly_digest.json")
        if not os.path.isabs(target):
            target = os.path.join(PROJECT_ROOT, target)
        ensure_parent_directory(target)
        if not os.path.exists(source):
            return
        try:
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return
            rows = []
            for ln in lines[-2000:]:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
            if not rows:
                return
            recent = rows[-7:]
            avg_new = sum(int(r.get("new_samples", 0)) for r in recent) / max(len(recent), 1)
            avg_queue = sum(int(r.get("queue_pending", 0)) for r in recent) / max(len(recent), 1)
            max_failed = max(int(r.get("failed_item_count", 0)) for r in recent)
            payload = {
                "updated_at": datetime.utcnow().isoformat(),
                "days_count": len(recent),
                "avg_new_samples": round(avg_new, 2),
                "avg_queue_pending": round(avg_queue, 2),
                "max_failed_items": int(max_failed),
            }
            atomic_write_json(target, payload)

            lines = [
                "SARA Autobot Weekly Digest",
                f"Updated: {payload['updated_at']}",
                f"Days included: {payload['days_count']}",
                f"Average new samples: {payload['avg_new_samples']}",
                f"Average queue pending: {payload['avg_queue_pending']}",
                f"Max failed items: {payload['max_failed_items']}",
            ]
            with open(self.weekly_digest_text_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            return

    def _emit_threshold_alerts(self, snapshot: dict[str, object]) -> None:
        failed_items = int(snapshot.get("failed_item_count", 0))
        queue_pending = int(snapshot.get("queue_pending", 0))
        if failed_items >= int(self.config.alert_failed_items_threshold):
            signature = (
                f"threshold_failed_items|value={failed_items}|"
                f"threshold={int(self.config.alert_failed_items_threshold)}"
            )
            if self._should_emit_alert_signature(signature):
                alert_line = (
                    f"ALERT ts={datetime.utcnow().isoformat()} severity=CRITICAL "
                    f"kind=threshold_failed_items value={failed_items} "
                    f"threshold={int(self.config.alert_failed_items_threshold)}"
                )
                with open(self.alerts_path, "a", encoding="utf-8") as f:
                    f.write(alert_line + "\n")
                self.event_logger.emit(
                    "threshold_alert",
                    {
                        "kind": "failed_items",
                        "value": failed_items,
                        "threshold": int(self.config.alert_failed_items_threshold),
                    },
                )
        if queue_pending >= int(self.config.alert_queue_pending_threshold):
            signature = (
                f"threshold_queue_pending|value={queue_pending}|"
                f"threshold={int(self.config.alert_queue_pending_threshold)}"
            )
            if self._should_emit_alert_signature(signature):
                alert_line = (
                    f"ALERT ts={datetime.utcnow().isoformat()} severity=WARN "
                    f"kind=threshold_queue_pending value={queue_pending} "
                    f"threshold={int(self.config.alert_queue_pending_threshold)}"
                )
                with open(self.alerts_path, "a", encoding="utf-8") as f:
                    f.write(alert_line + "\n")
                self.event_logger.emit(
                    "threshold_alert",
                    {
                        "kind": "queue_pending",
                        "value": queue_pending,
                        "threshold": int(self.config.alert_queue_pending_threshold),
                    },
                )

    def _download_url(self, url: str) -> str | None:
        day = datetime.utcnow().strftime("%Y%m%d")
        target_dir = os.path.join(self.web_raw_dir, day)
        os.makedirs(target_dir, exist_ok=True)

        req = urllib.request.Request(url, headers={"User-Agent": "SARA-Autobot/1.0"})
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as res:
            data = res.read()
            ctype = res.headers.get("Content-Type", "")

        safe_name = re.sub(
            r"[^a-zA-Z0-9._-]",
            "_",
            urllib.parse.urlparse(url).netloc + urllib.parse.urlparse(url).path,
        )
        if not safe_name or safe_name.endswith("_"):
            safe_name += "index"

        ext = ""
        if "text/html" in ctype:
            ext = ".html"
        elif "application/json" in ctype:
            ext = ".json"
        elif "/" in ctype:
            ext = "." + ctype.split("/")[-1].split(";")[0]

        file_path = os.path.join(target_dir, safe_name[:160] + ext)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path

    def _extract_links(self, html_path: str, base_url: str) -> list[str]:
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read(500_000)
        except Exception:
            return []
        links = re.findall(r'href=["\'](.*?)["\']', html, flags=re.IGNORECASE)
        out: list[str] = []
        for link in links:
            abs_link = urllib.parse.urljoin(base_url, link)
            if abs_link.startswith("http://") or abs_link.startswith("https://"):
                out.append(abs_link.split("#")[0])
        return out[:100]

    def process_hot_inbox(self) -> int:
        processed_count = 0
        seen = set(self.state.processed_files)
        for entry in sorted(Path(self.hot_inbox_dir).glob("**/*")):
            if not entry.is_file():
                continue
            abs_path = str(entry.resolve())
            if not self._in_my_shard(abs_path):
                continue
            if abs_path in seen:
                continue
            try:
                rec = ingest_file(abs_path)
                compliance = self.compliance.decide_for_source("local.hot_inbox", "hot_inbox")
                if not compliance.allowed:
                    self._append_dead_letter("hot_inbox", abs_path, "compliance_denied", compliance.reason)
                    continue
                decision = self.quality_gate.evaluate(rec.summary_text)
                if not decision.accepted:
                    self._append_dead_letter("hot_inbox", abs_path, "quality_rejected", decision.reason)
                    continue
                if self._is_duplicate_content(rec.summary_text):
                    self._append_dead_letter("hot_inbox", abs_path, "duplicate_content", "hash_match")
                    continue
                if self._is_semantic_duplicate(rec.summary_text):
                    self._append_dead_letter("hot_inbox", abs_path, "semantic_duplicate", "simhash_near")
                    continue
                self._append_record("hot_inbox", rec.summary_text, {"quality": decision.score, **rec.metadata})
                self._update_language_stats(rec.summary_text)
                self._count_modality(rec.modality)
                priority = self._compute_training_priority(modality=rec.modality, quality=decision.score, source="hot_inbox")
                self.training_queue.enqueue({
                    "source": "hot_inbox",
                    "path": abs_path,
                    "modality": rec.modality,
                    "quality": decision.score,
                    "priority": priority,
                    "curriculum_stage": self._curriculum_stage(decision.score, "hot_inbox"),
                })
                dest_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + entry.name
                dest_path = os.path.join(self.hot_done_dir, dest_name)
                shutil.move(abs_path, dest_path)
                seen.add(abs_path)
                self.state.failed_attempts.pop(abs_path, None)
                processed_count += 1
            except Exception as e:
                print(f"[WARN] Hot inbox processing failed for {abs_path}: {e}")
                dropped = self._record_failure(abs_path, "hot_inbox", str(e))
                if dropped:
                    seen.add(abs_path)

        self.state.processed_files = sorted(seen)[-100_000:]
        return processed_count

    def crawl_web(self, max_pages_override: int | None = None) -> int:
        visited = set(self.state.visited_urls)
        q: queue.Queue[str] = queue.Queue()
        blocked = {d for d, s in self.state.domain_reputation.items() if float(s) <= self._low_reputation_threshold}
        seeds = self.planner.next_seeds(self._gap_signal(), blocked_domains=blocked)
        for seed in seeds:
            if self._in_my_shard(seed):
                q.put(seed)

        added = 0
        max_pages = max_pages_override if max_pages_override is not None else self.policy.max_pages_per_cycle
        while not q.empty() and added < max_pages:
            url = q.get().strip()
            if not self._in_my_shard(url):
                continue
            domain = self._domain_from_url(url)
            if self._is_domain_low_reputation(domain):
                continue
            compliance = self.compliance.decide_for_source(domain, "web")
            if not compliance.allowed:
                self._append_dead_letter("web", url, "compliance_denied", compliance.reason)
                self.event_logger.emit(
                    "compliance_denied",
                    {"url": url, "domain": domain, "reason": compliance.reason},
                )
                continue
            if not self.policy.is_allowed_url(url):
                continue
            if not self.policy.is_allowed_by_robots(url):
                continue
            if url in visited:
                continue
            try:
                local_file = self._download_url(url)
                if not local_file:
                    continue
                rec = ingest_file(local_file)
                decision = self.quality_gate.evaluate(rec.summary_text)
                if not decision.accepted:
                    self._append_dead_letter("web", url, "quality_rejected", decision.reason)
                    self._adjust_domain_score(domain, -0.05)
                    visited.add(url)
                    continue
                if self._is_duplicate_content(rec.summary_text):
                    self._append_dead_letter("web", url, "duplicate_content", "hash_match")
                    self._adjust_domain_score(domain, -0.02)
                    visited.add(url)
                    continue
                if self._is_semantic_duplicate(rec.summary_text):
                    self._append_dead_letter("web", url, "semantic_duplicate", "simhash_near")
                    self._adjust_domain_score(domain, -0.02)
                    visited.add(url)
                    continue
                self._append_record(
                    "web",
                    rec.summary_text,
                    {
                        "url": url,
                        "quality": decision.score,
                        "compliance_level": compliance.level,
                        "compliance_reason": compliance.reason,
                        **rec.metadata,
                    },
                )
                self._update_language_stats(rec.summary_text)
                if compliance.level != "allow":
                    self.event_logger.emit(
                        "compliance_warning",
                        {
                            "url": url,
                            "domain": domain,
                            "level": compliance.level,
                            "reason": compliance.reason,
                        },
                    )
                self._count_modality(rec.modality)
                priority = self._compute_training_priority(modality=rec.modality, quality=decision.score, source="web")
                self.training_queue.enqueue({
                    "source": "web",
                    "url": url,
                    "path": local_file,
                    "modality": rec.modality,
                    "quality": decision.score,
                    "priority": priority,
                    "curriculum_stage": self._curriculum_stage(decision.score, "web"),
                })
                visited.add(url)
                self._adjust_domain_score(domain, +0.08)
                self.state.failed_attempts.pop(url, None)
                added += 1
                if local_file.endswith(".html"):
                    for link in self._extract_links(local_file, url):
                        if link not in visited and self.policy.is_allowed_url(link) and self._in_my_shard(link):
                            q.put(link)
            except Exception as e:
                print(f"[WARN] Crawl failed for {url}: {e}")
                self._adjust_domain_score(domain, -0.25)
                dropped = self._record_failure(url, "web", str(e))
                if dropped:
                    visited.add(url)

        self.state.visited_urls = sorted(visited)[-200_000:]
        return added

    def maybe_train(self, new_samples: int, skip_training: bool = False) -> None:
        now = time.time()
        replay_count = self._inject_replay_samples()
        if new_samples <= 0 and replay_count <= 0:
            return
        if not self._can_train_on_this_shard():
            self.event_logger.emit(
                "training_skipped_non_leader_shard",
                {
                    "shard_id": int(self.config.shard_id),
                    "training_leader_shard": int(self.config.training_leader_shard),
                    "total_shards": int(self.config.total_shards),
                },
            )
            return
        if skip_training:
            print("[INFO] Training skipped by control action.")
            return
        effective_interval = self._effective_train_interval()
        if now - self.last_train_ts < effective_interval:
            return

        if bool(self.config.curriculum_enabled):
            batch = self.training_queue.drain_curriculum(
                limit=256,
                easy_ratio=float(self.config.curriculum_easy_ratio),
                medium_ratio=float(self.config.curriculum_medium_ratio),
                hard_ratio=float(self.config.curriculum_hard_ratio),
            )
        else:
            batch = self.training_queue.drain(limit=256)
        if not batch:
            return

        print("[INFO] Starting continual self-organized training...")
        from scripts.train.train_self_organized import train_self_organized

        train_self_organized(
            corpus_path=self.corpus_path,
            save_dir=self.model_registry.candidate_dir,
            vocab_size=65536,
            sdr_size=128,
            context_window=15,
        )
        report = self.evaluation_gate.evaluate(self.corpus_path, self.model_registry.candidate_dir)
        self.state.last_eval_passed = bool(report.get("passed", False))
        print(f"[INFO] Evaluation gate: passed={report.get('passed')} corpus_lines={report.get('corpus_lines')}")
        self._run_promotion_flow(report)
        self.last_train_ts = now

    def run(self) -> None:
        with open(self.pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

        print("[INFO] Autonomous learning bot started.")
        try:
            while self.running:
                self.current_cycle += 1
                cycle_start = time.time()
                hot_count = self.process_hot_inbox()
                plugin_count = self._run_plugin_collectors()
                pre_snapshot = self._control_actions(hot_count=hot_count, crawl_count=0, total_new=hot_count)
                crawl_count = 0
                if not bool(self.config.offline_mode):
                    crawl_count = self.crawl_web(
                        max_pages_override=int(pre_snapshot.get("effective_max_pages", self.policy.max_pages_per_cycle))
                    )
                total_new = hot_count + crawl_count + plugin_count
                snapshot = self._control_actions(hot_count=hot_count, crawl_count=crawl_count, total_new=total_new)
                self.last_snapshot = snapshot

                print(f"[INFO] cycle complete: hot={hot_count}, plugins={plugin_count}, web={crawl_count}, total_new={total_new}")
                print(f"[INFO] control actions: {','.join(str(x) for x in snapshot.get('actions', []))}")
                self._rotate_log_if_needed()
                self._write_metrics(snapshot)
                self._emit_alert_if_needed(snapshot)
                self._save_state()
                removed_dirs = self._prune_raw_data_by_retention()
                removed_hot_done = self._prune_hot_done_by_retention()
                if removed_dirs > 0 or removed_hot_done > 0:
                    self.event_logger.emit(
                        "retention_pruned",
                        {
                            "removed_raw_dirs": removed_dirs,
                            "removed_hot_done_files": removed_hot_done,
                        },
                    )
                self.event_logger.emit(
                    "cycle",
                    {
                        "cycle": int(self.current_cycle),
                        "hot": int(hot_count),
                        "plugins": int(plugin_count),
                        "web": int(crawl_count),
                        "new": int(total_new),
                        "actions": list(snapshot.get("actions", [])),
                    },
                )
                self._trim_file_lines(self.events_path, max_lines=int(self.config.max_event_lines))
                self._write_daily_digest(snapshot)
                self._update_weekly_digest()
                self._emit_threshold_alerts(snapshot)
                self._write_audit_snapshot(snapshot)
                self.maybe_train(total_new, skip_training=bool(snapshot.get("should_skip_training", False)))

                elapsed = time.time() - cycle_start
                sleep_sec = max(1, self.config.crawl_interval_sec - int(elapsed))
                for _ in range(sleep_sec):
                    if not self.running:
                        break
                    time.sleep(self.config.idle_sleep_sec)
        finally:
            self._write_metrics(self.last_snapshot)
            self._save_state()
            self._write_shutdown_status()
            print("[INFO] Autonomous learning bot stopped.")


DEFAULT_CONFIG = BotConfig(
    crawl_seeds=[
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD",
        "https://arxiv.org/list/cs.AI/recent",
    ],
    crawl_interval_sec=120,
    train_interval_sec=300,
    idle_sleep_sec=1,
    request_timeout_sec=20,
    max_pages_per_cycle=24,
    high_queue_threshold=1200,
    high_failure_threshold=200,
    dead_letter_rate_threshold=60,
    cooldown_cycles=3,
    promotion_policy="balanced",
    canary_min_corpus_lines=120,
    max_backup_count=7,
    strict_allowlist_mode=False,
    allowed_domains=[],
    raw_data_retention_days=30,
    max_dead_letter_lines=20000,
    max_event_lines=50000,
    compliance_policy_path="bot/compliance_policy.example.json",
    compliance_preset="balanced",
    weekly_digest_path="workspace/autobot/weekly_digest.json",
    alert_failed_items_threshold=250,
    alert_queue_pending_threshold=1500,
    weekly_gate_max_failed_items=500,
    weekly_gate_max_avg_queue=2000,
    semantic_hamming_threshold=3,
    promotion_min_score=0.52,
    max_records_lines=300000,
    max_corpus_lines=300000,
    critical_alert_window_minutes=20,
    critical_alert_threshold=8,
    hot_done_retention_days=30,
    quality_block_patterns=[],
    audit_export_enabled=True,
    audit_snapshot_path="workspace/autobot/audit_snapshot.json",
    replay_interval_sec=1800,
    replay_samples_per_cycle=24,
    replay_min_quality=0.45,
    alert_dedup_window_sec=300,
    collector_plugins_enabled=True,
    collector_plugins_dir="bot/collectors_plugins",
    offline_mode=False,
    shard_id=0,
    total_shards=1,
    cooperative_training_enabled=True,
    training_leader_shard=0,
    curriculum_enabled=True,
    curriculum_easy_ratio=0.25,
    curriculum_medium_ratio=0.5,
    curriculum_hard_ratio=0.25,
    render_delta_medium_threshold=0.2,
    render_delta_hard_threshold=0.45,
    benchmark_min_pass_rate=0.8,
    benchmark_max_latency_ms=5000.0,
)


def _validate_config(cfg: BotConfig) -> BotConfig:
    if cfg.crawl_interval_sec < 5:
        raise ValueError("crawl_interval_sec must be >= 5")
    if cfg.train_interval_sec < 10:
        raise ValueError("train_interval_sec must be >= 10")
    if cfg.request_timeout_sec < 1:
        raise ValueError("request_timeout_sec must be >= 1")
    if cfg.max_pages_per_cycle < 1:
        raise ValueError("max_pages_per_cycle must be >= 1")
    if cfg.high_queue_threshold < 1:
        raise ValueError("high_queue_threshold must be >= 1")
    if cfg.high_failure_threshold < 1:
        raise ValueError("high_failure_threshold must be >= 1")
    if cfg.dead_letter_rate_threshold < 1:
        raise ValueError("dead_letter_rate_threshold must be >= 1")
    if cfg.cooldown_cycles < 1:
        raise ValueError("cooldown_cycles must be >= 1")
    if cfg.promotion_policy.strip().lower() not in {"strict", "balanced", "exploratory"}:
        raise ValueError("promotion_policy must be one of: strict, balanced, exploratory")
    if cfg.canary_min_corpus_lines < 1:
        raise ValueError("canary_min_corpus_lines must be >= 1")
    if cfg.max_backup_count < 1:
        raise ValueError("max_backup_count must be >= 1")
    if cfg.raw_data_retention_days < 1:
        raise ValueError("raw_data_retention_days must be >= 1")
    if cfg.max_dead_letter_lines < 100:
        raise ValueError("max_dead_letter_lines must be >= 100")
    if cfg.max_event_lines < 100:
        raise ValueError("max_event_lines must be >= 100")
    if cfg.compliance_preset.strip().lower() not in {"strict", "balanced", "open"}:
        raise ValueError("compliance_preset must be one of: strict, balanced, open")
    if cfg.alert_failed_items_threshold < 1:
        raise ValueError("alert_failed_items_threshold must be >= 1")
    if cfg.alert_queue_pending_threshold < 1:
        raise ValueError("alert_queue_pending_threshold must be >= 1")
    if cfg.weekly_gate_max_failed_items < 1:
        raise ValueError("weekly_gate_max_failed_items must be >= 1")
    if cfg.weekly_gate_max_avg_queue < 1:
        raise ValueError("weekly_gate_max_avg_queue must be >= 1")
    if cfg.semantic_hamming_threshold < 0 or cfg.semantic_hamming_threshold > 16:
        raise ValueError("semantic_hamming_threshold must be between 0 and 16")
    if cfg.promotion_min_score < 0.0 or cfg.promotion_min_score > 1.0:
        raise ValueError("promotion_min_score must be between 0.0 and 1.0")
    if cfg.max_records_lines < 1000:
        raise ValueError("max_records_lines must be >= 1000")
    if cfg.max_corpus_lines < 1000:
        raise ValueError("max_corpus_lines must be >= 1000")
    if cfg.critical_alert_window_minutes < 1:
        raise ValueError("critical_alert_window_minutes must be >= 1")
    if cfg.critical_alert_threshold < 1:
        raise ValueError("critical_alert_threshold must be >= 1")
    if cfg.hot_done_retention_days < 1:
        raise ValueError("hot_done_retention_days must be >= 1")
    if not isinstance(cfg.quality_block_patterns, list):
        raise ValueError("quality_block_patterns must be a list")
    if cfg.replay_interval_sec < 30:
        raise ValueError("replay_interval_sec must be >= 30")
    if cfg.replay_samples_per_cycle < 1:
        raise ValueError("replay_samples_per_cycle must be >= 1")
    if cfg.replay_min_quality < 0.0 or cfg.replay_min_quality > 1.0:
        raise ValueError("replay_min_quality must be between 0.0 and 1.0")
    if cfg.alert_dedup_window_sec < 1:
        raise ValueError("alert_dedup_window_sec must be >= 1")
    if not cfg.collector_plugins_dir.strip():
        raise ValueError("collector_plugins_dir must not be empty")
    if cfg.total_shards < 1:
        raise ValueError("total_shards must be >= 1")
    if cfg.shard_id < 0 or cfg.shard_id >= cfg.total_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < total_shards")
    if cfg.training_leader_shard < 0 or cfg.training_leader_shard >= cfg.total_shards:
        raise ValueError("training_leader_shard must satisfy 0 <= training_leader_shard < total_shards")
    if cfg.curriculum_easy_ratio < 0 or cfg.curriculum_medium_ratio < 0 or cfg.curriculum_hard_ratio < 0:
        raise ValueError("curriculum ratios must be >= 0")
    if (cfg.curriculum_easy_ratio + cfg.curriculum_medium_ratio + cfg.curriculum_hard_ratio) <= 0:
        raise ValueError("sum of curriculum ratios must be > 0")
    if cfg.render_delta_medium_threshold < 0.0 or cfg.render_delta_medium_threshold > 1.0:
        raise ValueError("render_delta_medium_threshold must be between 0.0 and 1.0")
    if cfg.render_delta_hard_threshold < 0.0 or cfg.render_delta_hard_threshold > 1.0:
        raise ValueError("render_delta_hard_threshold must be between 0.0 and 1.0")
    if cfg.render_delta_medium_threshold > cfg.render_delta_hard_threshold:
        raise ValueError("render_delta_medium_threshold must be <= render_delta_hard_threshold")
    if cfg.benchmark_min_pass_rate < 0.0 or cfg.benchmark_min_pass_rate > 1.0:
        raise ValueError("benchmark_min_pass_rate must be between 0.0 and 1.0")
    if cfg.benchmark_max_latency_ms < 1.0:
        raise ValueError("benchmark_max_latency_ms must be >= 1.0")
    return cfg


def _install_signal_handlers(bot: AutonomousLearningBot) -> None:
    def _handler(signum: int, _frame: object) -> None:
        bot.shutdown_reason = f"signal_{int(signum)}"
        bot.running = False

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autonomous multimodal learning bot.")
    parser.add_argument("--config", default="", help="Optional JSON config path.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = json.load(f)
        config = BotConfig(
            crawl_seeds=list(raw.get("crawl_seeds", DEFAULT_CONFIG.crawl_seeds)),
            crawl_interval_sec=int(raw.get("crawl_interval_sec", DEFAULT_CONFIG.crawl_interval_sec)),
            train_interval_sec=int(raw.get("train_interval_sec", DEFAULT_CONFIG.train_interval_sec)),
            idle_sleep_sec=int(raw.get("idle_sleep_sec", DEFAULT_CONFIG.idle_sleep_sec)),
            request_timeout_sec=int(raw.get("request_timeout_sec", DEFAULT_CONFIG.request_timeout_sec)),
            max_pages_per_cycle=int(raw.get("max_pages_per_cycle", DEFAULT_CONFIG.max_pages_per_cycle)),
            high_queue_threshold=int(raw.get("high_queue_threshold", DEFAULT_CONFIG.high_queue_threshold)),
            high_failure_threshold=int(raw.get("high_failure_threshold", DEFAULT_CONFIG.high_failure_threshold)),
            dead_letter_rate_threshold=int(raw.get("dead_letter_rate_threshold", DEFAULT_CONFIG.dead_letter_rate_threshold)),
            cooldown_cycles=int(raw.get("cooldown_cycles", DEFAULT_CONFIG.cooldown_cycles)),
            promotion_policy=str(raw.get("promotion_policy", DEFAULT_CONFIG.promotion_policy)),
            canary_min_corpus_lines=int(raw.get("canary_min_corpus_lines", DEFAULT_CONFIG.canary_min_corpus_lines)),
            max_backup_count=int(raw.get("max_backup_count", DEFAULT_CONFIG.max_backup_count)),
            strict_allowlist_mode=bool(raw.get("strict_allowlist_mode", DEFAULT_CONFIG.strict_allowlist_mode)),
            allowed_domains=[str(x).strip().lower() for x in raw.get("allowed_domains", DEFAULT_CONFIG.allowed_domains) if str(x).strip()],
            raw_data_retention_days=int(raw.get("raw_data_retention_days", DEFAULT_CONFIG.raw_data_retention_days)),
            max_dead_letter_lines=int(raw.get("max_dead_letter_lines", DEFAULT_CONFIG.max_dead_letter_lines)),
            max_event_lines=int(raw.get("max_event_lines", DEFAULT_CONFIG.max_event_lines)),
            compliance_policy_path=str(raw.get("compliance_policy_path", DEFAULT_CONFIG.compliance_policy_path)),
            compliance_preset=str(raw.get("compliance_preset", DEFAULT_CONFIG.compliance_preset)),
            weekly_digest_path=str(raw.get("weekly_digest_path", DEFAULT_CONFIG.weekly_digest_path)),
            alert_failed_items_threshold=int(raw.get("alert_failed_items_threshold", DEFAULT_CONFIG.alert_failed_items_threshold)),
            alert_queue_pending_threshold=int(raw.get("alert_queue_pending_threshold", DEFAULT_CONFIG.alert_queue_pending_threshold)),
            weekly_gate_max_failed_items=int(raw.get("weekly_gate_max_failed_items", DEFAULT_CONFIG.weekly_gate_max_failed_items)),
            weekly_gate_max_avg_queue=int(raw.get("weekly_gate_max_avg_queue", DEFAULT_CONFIG.weekly_gate_max_avg_queue)),
            semantic_hamming_threshold=int(raw.get("semantic_hamming_threshold", DEFAULT_CONFIG.semantic_hamming_threshold)),
            promotion_min_score=float(raw.get("promotion_min_score", DEFAULT_CONFIG.promotion_min_score)),
            max_records_lines=int(raw.get("max_records_lines", DEFAULT_CONFIG.max_records_lines)),
            max_corpus_lines=int(raw.get("max_corpus_lines", DEFAULT_CONFIG.max_corpus_lines)),
            critical_alert_window_minutes=int(raw.get("critical_alert_window_minutes", DEFAULT_CONFIG.critical_alert_window_minutes)),
            critical_alert_threshold=int(raw.get("critical_alert_threshold", DEFAULT_CONFIG.critical_alert_threshold)),
            hot_done_retention_days=int(raw.get("hot_done_retention_days", DEFAULT_CONFIG.hot_done_retention_days)),
            quality_block_patterns=[str(x) for x in raw.get("quality_block_patterns", DEFAULT_CONFIG.quality_block_patterns)],
            audit_export_enabled=bool(raw.get("audit_export_enabled", DEFAULT_CONFIG.audit_export_enabled)),
            audit_snapshot_path=str(raw.get("audit_snapshot_path", DEFAULT_CONFIG.audit_snapshot_path)),
            replay_interval_sec=int(raw.get("replay_interval_sec", DEFAULT_CONFIG.replay_interval_sec)),
            replay_samples_per_cycle=int(raw.get("replay_samples_per_cycle", DEFAULT_CONFIG.replay_samples_per_cycle)),
            replay_min_quality=float(raw.get("replay_min_quality", DEFAULT_CONFIG.replay_min_quality)),
            alert_dedup_window_sec=int(raw.get("alert_dedup_window_sec", DEFAULT_CONFIG.alert_dedup_window_sec)),
            collector_plugins_enabled=bool(raw.get("collector_plugins_enabled", DEFAULT_CONFIG.collector_plugins_enabled)),
            collector_plugins_dir=str(raw.get("collector_plugins_dir", DEFAULT_CONFIG.collector_plugins_dir)),
            offline_mode=bool(raw.get("offline_mode", DEFAULT_CONFIG.offline_mode)),
            shard_id=int(raw.get("shard_id", DEFAULT_CONFIG.shard_id)),
            total_shards=int(raw.get("total_shards", DEFAULT_CONFIG.total_shards)),
            cooperative_training_enabled=bool(raw.get("cooperative_training_enabled", DEFAULT_CONFIG.cooperative_training_enabled)),
            training_leader_shard=int(raw.get("training_leader_shard", DEFAULT_CONFIG.training_leader_shard)),
            curriculum_enabled=bool(raw.get("curriculum_enabled", DEFAULT_CONFIG.curriculum_enabled)),
            curriculum_easy_ratio=float(raw.get("curriculum_easy_ratio", DEFAULT_CONFIG.curriculum_easy_ratio)),
            curriculum_medium_ratio=float(raw.get("curriculum_medium_ratio", DEFAULT_CONFIG.curriculum_medium_ratio)),
            curriculum_hard_ratio=float(raw.get("curriculum_hard_ratio", DEFAULT_CONFIG.curriculum_hard_ratio)),
            render_delta_medium_threshold=float(raw.get("render_delta_medium_threshold", DEFAULT_CONFIG.render_delta_medium_threshold)),
            render_delta_hard_threshold=float(raw.get("render_delta_hard_threshold", DEFAULT_CONFIG.render_delta_hard_threshold)),
            benchmark_min_pass_rate=float(raw.get("benchmark_min_pass_rate", DEFAULT_CONFIG.benchmark_min_pass_rate)),
            benchmark_max_latency_ms=float(raw.get("benchmark_max_latency_ms", DEFAULT_CONFIG.benchmark_max_latency_ms)),
        )

    try:
        config = _validate_config(config)
    except ValueError as exc:
        print(f"[ERROR] Invalid config: {exc}")
        return 1

    bot = AutonomousLearningBot(config)
    _install_signal_handlers(bot)
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
