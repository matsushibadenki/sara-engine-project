# BOT ROADMAP

## Vision
Build a fully autonomous, CPU-first multimodal learning bot that continuously collects, filters, learns, evaluates, and safely promotes models with minimal human intervention (`start/stop` only).

## Current State (Implemented)
- Continuous runtime loop (`bot/autonomous_bot.py`)
- Web crawl + hot folder ingestion
- Multimodal ingestion abstraction (`src/sara_engine/utils/multimodal_ingest.py`)
- Quality/safety filtering (`bot/quality_gate.py`)
- Training queue (`bot/training_queue.py`)
- Candidate/production model promotion with backups (`bot/model_registry.py`)
- Evaluation gate integration (`bot/evaluation_gate.py`)
- Robots policy checks (`bot/policy.py`)
- Dead-letter handling and retry limits
- Metrics and alert logging
- Ops scripts: `start.sh`, `stop.sh`, `status.sh`, `chat.sh`

## Progress Status
- Phase 1 Reliability Hardening: `in progress` (atomic writes, config validation, graceful shutdown checkpoints implemented)
- Phase 2 Data Quality Intelligence: `in progress` (dedup hash, domain reputation control implemented)
- Phase 3 Autonomous Curriculum & Planning: `implemented` (gap planning + explicit easy/medium/hard curriculum + adaptive cadence + replay)
- Phase 4 Evaluation & Promotion Governance: `in progress` (policy modes, canary, rollback, backup pruning, benchmark gating implemented)
- Phase 5 Operations & Observability: `in progress` (structured events, status visibility, alert severity/rotation)
- Phase 6 Security & Compliance: `in progress` (PII guard, source-type compliance policy, retention + audit snapshot)
- Backlog Item (Plugin Collectors): `implemented` (drop-in collector loader under `bot/collectors_plugins`)
- 🟢 [Done] Local dashboard pages for Overview, Queue, Alerts and Diagnostics, with English, Japanese and Simplified Chinese navigation and live status. [Page implementation](dashboard_pages.py)
- 🟢 [Done] Verify the four pages through the real status reader using isolated PID, metrics and log fixtures. Missing metrics and logs are explicit, empty logs show zero, and unreadable status and positive alert counts are visible. Chrome checks covered desktop and narrow layouts, polling failure and recovery. [Regression tests](../tests/test_autobot_dashboard.py)
- 🟢 [Done] Keep the four dashboard routes navigable during status collection failures. An initial failure shows a localized retry message without values, and polling failures retain the last good panel with an explicit stale-status message. The status API returns HTTP 503 with a stable error code. [Dashboard recovery regression](../tests/test_autobot_dashboard.py)
- 🟢 [Done] Bound both manual and automatic dashboard refreshes to five seconds. A stalled request keeps the last good data visible, shows the refresh warning and resumes polling. Manual refresh uses the same single-request path.
- 🟢 [Done] Use the live producer's reported alert window for status counts. Older snapshots and unavailable runtime settings use an explicitly labeled fallback. [Dashboard alert regression](../tests/test_autobot_dashboard.py)
- 🟢 [Done] Verify the alert window across a real Autobot restart in an isolated offline workspace. With the first metrics write paused, the alert page used the 20-minute fallback and counted one alert at both starts, including the second start with the old metrics file intact. After the respective writes it switched to 90 minutes and three alerts, then 45 minutes and two alerts. [Process restart regression](../tests/test_autobot_alert_restart.py)
- 🟢 [Done] Verify browser polling through an actual offline restart with the Japanese alert page left open. Chrome at 320px showed 90 minutes / three alerts, 20 minutes / one alert after stopping and during the restart with unchanged old metrics, then 45 minutes / two alerts after the new write. The document marker survived and the main frame navigated only on initial load. The fallback notice appeared during both fallback stages and cleared on the new metrics.
- 🟢 [Done] Verify abrupt Autobot termination with SIGKILL. The same Japanese alert page in Chrome at 320px switched from 90 minutes / three alerts to the labeled 20-minute fallback / one alert while the original PID file and metrics remained intact. It kept the fallback after restart until the new producer wrote metrics, then showed 45 minutes / two alerts. The main frame navigated only at initial load. The process restart regression covers both graceful shutdown and SIGKILL.
- 🟢 [Done] Require the metrics launch ID to match a lease held by Autobot for its process lifetime. Isolated SIGKILL verification injected PID reuse by assigning the PID of a real unrelated process to the stale PID and metrics records while retaining the old launch ID. Chrome kept the 20-minute fallback and one alert, then switched to the new 45-minute window and two alerts after restart without another navigation. OS PID allocation was not forced. [Process restart regression](../tests/test_autobot_alert_restart.py)
- 🟢 [Done] Require a live lease whose owner matches the PID file before Overview reports Running. Chrome kept the original Japanese Overview open at 320px through SIGKILL, an injected recycled PID belonging to a live unrelated process, and restart. It showed Stopped during PID reuse and Running once the new lease was published, before the first metrics write. Alerts switched to the runtime window only after new metrics matched the launch ID. Bot deployments using the previous lease format need a restart to publish the PID-bearing format.
- 🟢 [Done] Gate the Overview metrics timestamp and all Queue metrics on the same live PID and launch-ID match as Alerts. Keep the previous snapshot available in Diagnostics with its current-status flag. Real offline subprocess and Chrome checks covered SIGKILL, injected PID reuse, the interval before the new producer's first write, and recovery in the original documents.
- Backlog Item (Multi-bot Sharding): `implemented` (hash-based shard ownership with `shard_id`/`total_shards`)
- Backlog Item (Benchmark Suite): `implemented` (`bot/benchmark_suite.py`, `bot/benchmark_cases.json`, `bot/benchmark.sh`)

## Guiding Principles
- CPU-first and energy-efficient operation
- No destructive fallback behavior on failures
- Safe-by-default crawling and ingestion
- Observable runtime state at all times
- Production model stability over training speed

## Phase 1: Reliability Hardening (Next)
1. Atomic file writes for metrics/state/queue
2. Queue corruption recovery and integrity checks
3. Alert severity levels (`INFO`, `WARN`, `CRITICAL`)
4. Config schema validation on startup
5. Graceful shutdown checkpointing for in-flight cycle

## Phase 2: Data Quality Intelligence
1. Source reputation scoring and domain trust tiers
2. Duplicate detection across modalities (hash + semantic)
3. Better document extraction quality (PDF/docx fallback strategies)
4. Automatic low-value source suppression
5. Language balance control (JP/EN targets)

## Phase 3: Autonomous Curriculum & Planning
1. Capability-gap analyzer from eval and modality stats
2. Dynamic crawl strategy per weakness area
3. Curriculum batches (easy -> medium -> hard)
4. Adaptive training cadence from queue pressure and eval trend
5. Scheduled replay from high-value historical samples

## Phase 4: Evaluation & Promotion Governance
1. Multi-gate scoring (quality, stability, regression, latency)
2. Promotion policies (`strict`, `balanced`, `exploratory`)
3. Canary evaluation before full production replacement
4. Automatic rollback trigger from post-promotion degradation
5. Promotion report snapshots for audit trail

## Phase 5: Operations & Observability
1. Structured logs (JSONL) with event types
2. Daily/weekly digest generation from metrics
3. Alert dedup windows and escalation rules
4. Storage quotas + auto-pruning policies
5. Health command expansion (`status --json`, `status --verbose`)

## Phase 6: Security & Compliance
1. PII pattern detection extension and configurable blocklists
2. License policy checks per source type
3. Crawl allowlist mode for strict environments
4. Data retention controls by source and modality
5. Full audit export for ingestion-to-promotion path

### Phase 6 Implementation Notes
- `quality_gate.py` now blocks likely PII and supports configurable `quality_block_patterns`.
- `compliance.py` now supports `source_types` rules (e.g., `web`, `hot_inbox`) in policy JSON.
- Retention controls cover both raw web cache and processed hot-folder outputs.
- `workspace/autobot/audit_snapshot.json` exports ingestion/queue/promotion audit state each cycle.

### Phase 3 Implementation Notes
- Adaptive train cadence is now driven by queue pressure, failure signals, and last evaluation result.
- Scheduled replay now injects high-quality historical samples into the training queue.
- Queue now supports explicit curriculum stage draining with configurable easy/medium/hard ratios.

### Phase 4 Implementation Notes
- Evaluation gate now reads `workspace/autobot/benchmark_latest.json` when available.
- Promotion scoring now penalizes failed benchmark pass-rate and high average latency.
- Benchmark thresholds are configurable with `benchmark_min_pass_rate` and `benchmark_max_latency_ms`.

## Backlog (Longer-Term)
- Plugin-style collector interface (drop-in new collectors) ✅ implemented
- Optional offline batch mode for air-gapped environments ✅ implemented
- Lightweight dashboard for local monitoring ✅ implemented
- Multi-bot sharding and cooperative training ✅ implemented (hash sharding + leader-shard cooperative training)
- Benchmark suite specific to autobot production model quality ✅ implemented
- Hybrid Web Learning (raw HTML + Chromium rendered DOM) for JS-heavy pages

## Idea: Hybrid Web Learning (Raw + Rendered)
Goal:
- Learn from both `curl`-downloadable raw HTML and Chromium-rendered post-JS content to improve coverage and factual extraction quality.

Motivation:
- Raw HTML is lightweight, stable, and cheap to store/process.
- Rendered DOM captures SPA/dynamic content not visible in raw HTML.
- Pairing both enables robust extraction and better curriculum difficulty control.

Data Collection Design:
1. Raw fetch collector:
- Use existing web downloader (`urllib`/`curl`-equivalent path) to save raw HTML + headers.
- Store under `data/raw/autobot/web/YYYYMMDD/`.
2. Rendered fetch collector:
- Add Chromium-based collector plugin (headless) to capture:
- final rendered HTML snapshot
- visible text blocks
- metadata (title, canonical URL, timing, JS-required flag)
3. Pairing key:
- Match raw/rendered by normalized URL + crawl timestamp window.
- Save a pair manifest in `workspace/autobot/` for downstream training jobs.

Training Strategy:
1. Dual-view sample format:
- Input features include both raw and rendered summaries.
- Prefer rendered text when dynamic-content confidence is high.
2. Curriculum usage:
- `easy`: static pages where raw≈rendered
- `medium`: minor DOM deltas
- `hard`: large raw/rendered divergence (JS-heavy pages)
3. Quality checks:
- Reject rendered snapshots with low visible-text density or obvious placeholder skeletons.

Operational Plan (Phased):
1. Phase A:
- Create `chromium_render_collector.py` plugin skeleton. ✅ implemented
- Persist rendered snapshots + pairing manifest. ✅ implemented
2. Phase B:
- Add diff scorer (`raw_vs_rendered_delta`) and feed into curriculum stage assignment. ✅ implemented
3. Phase C:
- Add benchmark slice for JS-heavy domains and compare pass-rate/latency before vs after hybrid ingestion. ✅ implemented

Safety/Compliance Notes:
- Reuse existing robots/compliance gates before rendering.
- Respect retention controls for both raw and rendered artifacts.
- Keep rendered artifacts within managed directories (`data/`, `workspace/`).

## Definition of Done (Per Milestone)
- Tests added for new logic paths
- `status.sh` reflects new runtime state
- README updated with usage and operational behavior
- No writes outside managed directories
- No regression in start/stop/chat workflows

## Suggested Milestone Order
1. Reliability Hardening
2. Evaluation & Promotion Governance
3. Data Quality Intelligence
4. Autonomous Curriculum & Planning
5. Operations & Observability
6. Security & Compliance
