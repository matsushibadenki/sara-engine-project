# Autonomous Learning Bot

This bot continuously crawls web content, ingests multimodal files, and retrains the SNN model without human intervention.

## Start

```bash
bash bot/start.sh
```

## Stop

```bash
bash bot/stop.sh
```

## Status

```bash
bash bot/status.sh
```

```bash
bash bot/status.sh --json
bash bot/status.sh --verbose
```

`status.sh` includes latest promotion attempt/result from `workspace/autobot/model_registry.json`.
It also handles broken JSON gracefully and reports parse errors instead of crashing.

## Dashboard

```bash
bash bot/dashboard_start.sh
```

Open: `http://127.0.0.1:8765`

```bash
bash bot/dashboard_stop.sh
```

## Benchmark Suite

```bash
bash bot/benchmark.sh
```

Hybrid comparison:

```bash
bash bot/benchmark.sh compare
```

- Cases file: `bot/benchmark_cases.json`
- Latest result: `workspace/autobot/benchmark_latest.json`
- History: `workspace/autobot/benchmark_history.jsonl`
- Hybrid compare: `workspace/autobot/benchmark_hybrid_compare.json`
- Use run tags: `bash bot/benchmark.sh --tag raw` and `bash bot/benchmark.sh --tag hybrid`
- Promotion/evaluation gate reads `benchmark_latest.json` when present.
- Gate thresholds: `benchmark_min_pass_rate`, `benchmark_max_latency_ms`.

## Production Inference (Fixed Path)

Use this command for inference. It always loads:

- `models/autobot_self_organized/production/`

```bash
bash bot/chat.sh
```

## Hot Folder

Put files into:

- `data/raw/hot_inbox/`

After ingestion, files are moved to:

- `data/processed/hot_done/`

## Outputs

- Records JSONL: `data/processed/autobot/multimodal_records.jsonl`
- Training corpus: `data/processed/autobot/corpus.txt`
- Bot state: `workspace/autobot/state.json`
- Logs: `workspace/autobot/bot.log`
- Model candidate: `models/autobot_self_organized/candidate/`
- Model production: `models/autobot_self_organized/production/`
- Model backups: `models/autobot_self_organized/backups/`
- Registry metadata: `workspace/autobot/model_registry.json`
- Train queue: `workspace/autobot/train_queue.json`
- Eval report: `workspace/autobot/eval_report.json`
- Runtime metrics: `workspace/autobot/metrics.json`
- Shutdown status: `workspace/autobot/shutdown_status.json`
- Structured events: `workspace/autobot/events.jsonl`
- Daily digest history: `workspace/autobot/daily_digest.jsonl`
- Daily digest text: `workspace/autobot/digests/YYYY-MM-DD.txt`
- Weekly digest: `workspace/autobot/weekly_digest.json`
- Weekly digest text: `workspace/autobot/weekly_digest.txt`
- Dead letters: `workspace/autobot/dead_letter.jsonl`
- Rotated dead letters: `workspace/autobot/dead_letter.jsonl.1`
- Alerts log: `workspace/autobot/alerts.log`
- Rotated alerts: `workspace/autobot/alerts.log.1`
- Runtime log: `workspace/autobot/bot.log`
- Rotated runtime log: `workspace/autobot/bot.log.1`
- Dashboard log: `workspace/autobot/dashboard.log`

## Runtime Architecture

- `bot/autonomous_bot.py`: runtime orchestrator loop.
- `bot/collectors_plugins/*.py`: drop-in collector plugins (`collect(bot) -> int`).
- `bot/planner.py`: capability-gap-aware seed planning.
- `bot/policy.py`: URL and crawl policy guardrails.
- `bot/quality_gate.py`: ingestion quality/safety filtering.
- `bot/training_queue.py`: asynchronous training queue.
- `bot/evaluation_gate.py`: post-training regression gate.
- `bot/state_store.py`: persistent runtime state.
- `bot/model_registry.py`: candidate/production promotion and backups.

## Safety and Compliance

- The crawler checks `robots.txt` before visiting URLs.
- Quality gate blocks low-information and possible-secret records.
- Start script sets writable cache paths for stable headless runtime.

## Autonomous Control Actions

- Backpressure mode: reduces crawl pages when queue is too large.
- High-failure mode: starts cooldown and temporarily skips training.
- Dead-letter spike mode: starts cooldown to avoid cascading failures.
- Control decisions are written to `workspace/autobot/metrics.json`.
- Non-normal control states also append `ALERT` lines to `workspace/autobot/alerts.log`.
- Alerts include `severity` (`WARN` / `CRITICAL`) and `status.sh` shows recent counts.
- `bot.log` is auto-rotated around 10MB (startup and runtime checks).

## Data Quality Intelligence

- Content deduplication uses normalized text hash to skip repeated records.
- Semantic deduplication uses SimHash near-duplicate detection (`semantic_hamming_threshold`).
- Domain reputation is tracked and low-score domains are automatically deprioritized/skipped.
- JP/EN language mix is tracked and seed planning is adjusted to reduce language imbalance.

## Autonomous Curriculum

- Training queue uses priority-based draining.
- Priority blends quality score, modality scarcity, and source type (hot folder gets a boost).
- Explicit curriculum stages are assigned (`easy` / `medium` / `hard`) and drained by configured ratios.
- Training cadence adapts to queue pressure and recent eval outcome (faster under backlog, slower after failed eval/high-failure mode).
- High-value replay periodically re-injects strong historical samples (`replay_interval_sec`, `replay_samples_per_cycle`, `replay_min_quality`).
- Curriculum knobs: `curriculum_enabled`, `curriculum_easy_ratio`, `curriculum_medium_ratio`, `curriculum_hard_ratio`.

## Promotion Governance

- Promotion policy modes: `strict`, `balanced`, `exploratory` (`config.example.json`).
- Canary validation runs after promotion and triggers rollback on failure.
- Backup retention is pruned automatically with `max_backup_count`.
- Benchmark pass-rate and latency are included in evaluation/promotion gating when benchmark output exists.

## Security & Compliance Controls

- Optional strict allowlist crawl mode (`strict_allowlist_mode`, `allowed_domains`).
- Raw web data retention pruning (`raw_data_retention_days`).
- Processed hot-folder retention pruning (`hot_done_retention_days`).
- Dead-letter and structured event line caps (`max_dead_letter_lines`, `max_event_lines`).
- Domain compliance rules can be externalized via `compliance_policy_path` JSON.
- Source-type compliance rules are supported (`source_types.web`, `source_types.hot_inbox`).
- Compliance behavior can be tuned with presets (`compliance_preset`: `strict` / `balanced` / `open`).
- PII/secret filtering is enforced by quality gate and can be extended by `quality_block_patterns`.

## Audit Export

- The bot exports an audit snapshot to `workspace/autobot/audit_snapshot.json` each cycle.
- Snapshot includes ingestion, queue, and promotion metadata to trace ingestion-to-promotion state.
- Audit export can be toggled with `audit_export_enabled` and redirected with `audit_snapshot_path`.

## Threshold Alerts

- Emits structured `threshold_alert` events when failed-items or queue-pending exceed configured thresholds.
- Threshold alerts are also appended to `workspace/autobot/alerts.log` with severity.
- Repeated CRITICAL alerts in a rolling window trigger escalation events (`critical_alert_window_minutes`, `critical_alert_threshold`).
- Alert dedup window suppresses repetitive identical alerts for a short period (`alert_dedup_window_sec`).

## Status Visibility

- `status.sh --json` now includes `audit` and `state` snapshots when available.
- Text mode also displays replay/runtime state and replay-related config baselines.
- `status` now reports `critical_alerts_recent` (within configured `critical_alert_window_minutes`).
- `metrics.json` includes `alert_suppressed_total` for dedup effectiveness monitoring.

## Collector Plugins

- Enable or disable via `collector_plugins_enabled`.
- Plugin directory is configured by `collector_plugins_dir` (default: `bot/collectors_plugins`).
- Each plugin is a Python file exporting `collect(bot) -> int`.
- Failures are isolated and logged as dead letters/events; core loop continues.
- Included example: `bot/collectors_plugins/rss_collector.py` (RSS/Atom ingestion).
- Optional feed list file: `workspace/autobot/rss_feeds.txt` (one URL per line). Sample: `bot/rss_feeds.example.txt`.
- Included offline collector: `bot/collectors_plugins/offline_batch_collector.py`.
- Included Chromium render collector (Phase A skeleton): `bot/collectors_plugins/chromium_render_collector.py`.
- Chromium collector is opt-in: set `AUTOBOT_CHROMIUM_ENABLED=1` before start.
- Chromium collector outputs:
  - rendered snapshots: `data/raw/autobot/rendered/YYYYMMDD/`
  - pair manifest: `workspace/autobot/render_pairs.jsonl`
- Chromium collector computes `raw_vs_rendered_delta` and enqueues rendered-text samples with curriculum stage reflection.
- Delta thresholds: `render_delta_medium_threshold`, `render_delta_hard_threshold`.

## Offline Batch Mode

- Set `offline_mode=true` to disable built-in web crawling.
- In offline mode, plugins marked `REQUIRES_NETWORK=True` are auto-skipped.
- Drop files into `data/raw/offline_batch_inbox/` for offline batch ingestion.
- Processed files are moved to `data/processed/offline_batch_done/`.

## Multi-Bot Sharding

- Set `total_shards` to the number of bot workers.
- Set each worker `shard_id` in `[0, total_shards-1]`.
- URL/file ownership is assigned by hash; each worker processes only its shard.
- Set `cooperative_training_enabled=true` to centralize training/promotion.
- Set `training_leader_shard` to the shard that performs training/promotion; others stay collection-focused.
- Example:
  - worker A: `shard_id=0`, `total_shards=2`
  - worker B: `shard_id=1`, `total_shards=2`

## Weekly Promotion Gate

- Promotion can be blocked when weekly trend degrades (`weekly_gate_max_failed_items`, `weekly_gate_max_avg_queue`).
- Promotion also requires a multi-factor score gate (`promotion_min_score`) over quality/stability/queue health.

## Reliability Notes

- Critical JSON outputs (`state`, `queue`, `metrics`, `eval`, `registry`) use atomic write strategy.
- Startup validates config thresholds and rejects invalid values with clear errors.
- If `train_queue.json` is corrupted, it is auto-backed up as `train_queue.json.corrupt.<timestamp>.json` and reset.
- If `state.json` is corrupted, it is auto-backed up as `state.json.corrupt.<timestamp>.json` and reset.
- On shutdown signals, bot writes a final checkpoint (`state`, `metrics`, `shutdown_status`).
- `dead_letter.jsonl` is auto-rotated around 5MB.
- `multimodal_records.jsonl` and `corpus.txt` are line-capped (`max_records_lines`, `max_corpus_lines`).

## Optional custom config

```bash
python3 bot/autonomous_bot.py --config bot/config.example.json
```
