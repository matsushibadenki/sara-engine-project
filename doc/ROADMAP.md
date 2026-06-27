# SARA Engine Roadmap

This roadmap starts after the v1.1 release-hardening work. Completed implementation history is kept in [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md). The archived pre-v1.1 roadmap is kept in [old/ROADMAP_v1_1_completion_archive.md](old/ROADMAP_v1_1_completion_archive.md).

## Current Release Baseline

SARA Engine v1.1 is release-ready by the active gates:

- v1.1 release gate: `15/15` checks passing
- Research product completion gate: `14/14` checks passing
- ANN-efficiency roadmap gate: `6/6` stages passing
- Full test suite: `1008` tests passing in the project Python 3.10 environment

The project is now structurally ready as a CPU-first SNN research product. The next phase should convert proxy evidence into stronger external evidence, improve real-world usefulness, and keep the sparse/runtime policy intact.

* **ROADMAP closure audit:**
  * DONE: release-critical path is complete for the v1.1 baseline.
  * DONE: observed-only evidence remains labeled until stronger external or physical measurements pass.
  * DONE: long-term research backlog is separated into Phase 6 and later roadmap work.
  * DONE: roadmap completion audit is present for the current baseline.
  * DONE: research product completion gate is part of the active release evidence surface.

## Execution Priority

The active research order is:

1. **Phase 6 physical energy evidence.** Establish paired SARA-versus-ANN `joule_per_success` under identical task, success, CPU, and measurement conditions.
2. **Phase 8 stronger external ANN baselines.** Replace weak proxy-only comparisons with credible offline CPU baselines.
3. **Phase 7 autonomous learning data preparation.** Expand source-aware learning material only with strict train/evaluation separation and anti-self-evaluation controls.
4. Continue later architecture phases only when they support the first three priorities or do not delay them. Phase 19 and Phase 20 are conditional accuracy experiments, not reasons to postpone Phase 6, Phase 8, or Phase 7.

This order reflects SARA's central research claim: local learning, sparse events, low energy use, and continual adaptation should produce better useful-task performance per joule on bounded tasks. New architectural novelty is secondary until this claim has physical and externally comparable evidence.

## North-Star Objective

The long-term goal is to exceed ANN-style systems on useful task performance per unit of energy, starting from bounded tasks and expanding only when evidence remains stable.

The project should not chase dense ANN or LLM capability on their own terms. It should compete where SARA's design is strongest:

- Sparse event routing
- Local learning and adaptation
- Bounded memory and explicit forgetting
- Low idle cost and event-budgeted execution
- Interpretable traces
- CPU-first and neuromorphic-portable execution
- Reliable operation under managed release gates

## Non-Negotiable Policy

New work must follow [policy.md](policy.md):

- No runtime dependency on backpropagation
- No dense-matrix-first runtime design
- No GPU requirement for correctness or normal operation
- Managed outputs only under `data/`, `workspace/`, and `models/`
- Accuracy and energy efficiency must be evaluated together
- New mechanisms should attach to sparse events, bounded memory, local updates, or operational observability

## Phase 6: Physical Energy Evidence

Goal: move from proxy-only ANN-efficiency claims to paired physical measurements.

**Priority: 1 / highest. This phase takes precedence over new architecture work.**

### Fair Comparison Contract

- Run SARA and ANN on the same physical CPU, core allocation, frequency/power mode, memory limits, operating-system state, and background-process policy.
- Use the same task inputs, corpus/index contents, task order, number of warm-up runs, measured repetitions, timeout, and random-seed policy.
- Use one machine-checkable success criterion per task. Do not count lower-quality outputs as successful merely because they are cheaper.
- Measure the same execution boundary for both systems, including equivalent index/model loading policy and excluding only predeclared setup steps.
- Record idle baseline, wall-clock duration, average watts or integrated joules, success count, failures, retries, CPU model, thread count, process affinity, and measurement-tool metadata.
- Alternate or randomized-block the SARA and ANN run order to reduce thermal, caching, and battery/AC-order bias.
- Report median and dispersion across repeated paired runs, not only the best run.
- Compute `joule_per_success = measured_joules / successful_trials` for each system and task.
- Keep proxy claims separate from physical claims. A physical win requires comparable quality and a lower paired `joule_per_success`, not merely fewer abstract event operations.

### Deliverables

- Collect paired SARA and ANN measurements for the tasks already listed in `workspace/evaluation/energy_measurement_session_plan.json`.
- Use `python scripts/sara_cli.py run-physical-energy-session-batch` to convert the duplicated per-system session-plan rows into concrete pair-level frozen runs before a laboratory session.
- Use `python scripts/sara_cli.py eval-physical-energy-session-progress` during the session to verify which planned pairs are complete, partial, invalid, or still missing before claiming that Phase 6 data collection is done.
- Store rows in `data/raw/energy_measurements.jsonl` through `python scripts/sara_cli.py record-energy-measurement`.
- Extend the measurement rows and session report with CPU/environment identity, run-order block, warm-up count, measured repetition count, success-criterion ID, measurement boundary, and meter/tool identity.
  - DONE: fairness schema v2 requires these fields and rejects mismatched pairs.
  - DONE: session planning now exposes `pair_command_template`, pair-level managed meter-template paths, and a batch expander that materializes concrete replicate runs under `workspace/evaluation/physical_energy_session_batch.json`.
  - DONE: session progress monitoring now compares planned pair runs against recorded rows and surfaces completion, partial coverage, invalid fairness matches, and orphan measurements under `workspace/evaluation/physical_energy_session_progress.json`.
  - DONE: `energy_measurement_readiness.py` now emits that same session-progress surface directly from the managed session plan, so partial laboratory sessions appear in the main Phase 6 readiness artifact instead of hiding behind aggregate row counts.
- Add at least one retrieval task and one continual-adaptation task where SARA's sparse/local design is expected to matter.
- Produce per-task paired rows and an aggregate report without hiding task-level losses.
- Regenerate:
  - `energy_measurement_readiness.py`
  - `ann_efficiency_roadmap_gate.py`
  - `research_product_completion_gate.py`
  - `v1_release_gate.py`
- Add a short measurement protocol document under `doc/` once the first real session is complete.
  - DONE for pre-session protocol: `doc/ENERGY_MEASUREMENT_PROTOCOL.md` defines fixed conditions, alternating run order, paired repetitions, aggregation, and managed artifacts. Append observed laboratory details after the first physical session.

### Acceptance Criteria

- Every measured task has both SARA and ANN rows.
- Every pair shares the same environment fingerprint, task fixture hash, success-criterion ID, and measurement protocol version.
- `joule_per_success` is computed from validated rows.
- Quality or success-rate parity passes before an energy advantage is credited.
- Repeated paired runs report median, spread, run-order balance, and invalid-run reasons.
- The minimum paired ANN/SARA joule-per-success ratio passes the configured threshold.
- Reports continue to label claims as proxy-only until physical measurements pass.
- Measurement commands and results remain reproducible from managed artifacts.

## Phase 7: Autonomous Learning Data Preparation

Goal: reduce human involvement in collecting and shaping training data by turning the existing `bot/` runtime into an automatic learning-material preparation loop.

**Priority: 3. Data generation must not compromise evaluation independence.**

The bot should not simply crawl more pages. It should transform collected material into safe, source-aware, curriculum-ready learning data that supports SARA's sparse retrieval, abstention, contrastive reasoning, and energy-efficiency goals.

### Evaluation Independence Contract

- Generated or transformed learning materials must never be inserted into the held-out evaluation set used to claim quality, abstention, or energy gains.
- Split by source document, source revision/hash, domain, and collection time before material generation so near-duplicate claims cannot cross train/evaluation boundaries.
- Keep external benchmark fixtures immutable during a measured comparison cycle.
- Report overlap checks for source URL/path, material hash, sparse signature similarity, and extracted claim similarity.
- Preserve negative queries, contrastive pairs, unsupported questions, and abstention cases from independent sources.
- Label synthetic, transformed, replay, repair, and directly observed materials separately.
- Evaluation failures may request new training data, but the failed held-out examples themselves must not become the next reported test set.

### Implementation Steps

1. Add `bot/dataset_builder.py`.
   - DONE: deterministic source-backed builder reads accepted autobot records and emits candidate/accepted materials plus an operator report.
   - Read accepted records from `data/processed/autobot/multimodal_records.jsonl`.
   - Generate structured learning materials:
     - `summary`
     - `qa_pairs`
     - `definition_cards`
     - `contrastive_pairs`
     - `negative_queries`
     - `procedural_steps`
     - `source_claims`
   - Write intermediate artifacts under `data/interim/autobot/`.

2. Add `bot/learning_material_gate.py`.
   - DONE: gate rejects short, duplicate, unsupported, secret-like, and PII-like material before processed outputs are written.
   - Reject generated samples that are too short, unsupported by source text, duplicate, unsafe, or likely to contain secrets/PII.
   - Preserve source URL/path, collection time, quality score, language, source type, license/compliance hint, and rejection reason.
   - Write rejected items to `data/interim/autobot/rejected_learning_materials.jsonl`.

3. Add `bot/curriculum_manifest.py`.
   - DONE: accepted materials are assigned to `easy`, `medium`, `hard`, `repair`, or `replay` with gap-aware priority.
   - Assign each accepted item to `easy`, `medium`, `hard`, `repair`, or `replay`.
   - Set priority from quality score, source reliability, rarity, modality scarcity, and current evaluation gaps.
   - Write `data/processed/autobot/curriculum_manifest.jsonl`.

4. Extend `bot/training_queue.py` integration.
   - DONE: `TrainingQueue.enqueue_learning_materials` can queue manifest rows while boosting repair items and bounding replay items.
   - Queue accepted learning materials by curriculum stage and material type.
   - Prefer repair data when recent gates fail.
   - Keep replay items bounded and high-value only.

5. Extend `bot/planner.py`.
   - DONE: planner can convert weak external-validity signals into machine-readable material requests under `workspace/autobot/`.
   - Convert evaluation failures into collection/material requests:
     - weak summary coverage -> summary and source-claim data
     - weak negative controls -> negative-query data
     - weak contrastive controls -> near-miss/contrastive pairs
     - weak retrieval grounding -> source-backed QA
     - language imbalance -> language-targeted collection
   - Keep all planning outputs machine-readable under `workspace/autobot/`.

6. Add source-aware collector plugins.
   - DONE: opt-in official documentation and arXiv abstract collector plugins are present and remain skipped in offline mode.
   - `official_docs_collector.py`: allowlisted official documentation and stable reference pages.
   - `arxiv_abstract_collector.py`: paper metadata, abstracts, and topic summaries without treating papers as unrestricted training text.
   - Optional extension: GitHub README/docs collector for repositories with acceptable license/compliance signals.

7. Add a dataset quality report.
   - DONE: `workspace/autobot/dataset_builder_report.json` and summary text record accepted/rejected counts, material types, languages, domains, duplicate rejections, and curriculum distribution.
   - Write `workspace/autobot/dataset_builder_report.json`.
   - Summarize accepted/rejected counts, material-type counts, language balance, source domains, duplicate rate, compliance decisions, and curriculum distribution.
   - Add a compact text summary for operator review.

8. Add tests and docs.
   - DONE: unit tests cover extraction, gating, curriculum assignment, planner feedback, queue integration, and CLI dispatch.
   - Unit-test material extraction, gating, curriculum assignment, and planner feedback.
   - Update `bot/README.md` and `doc/TOOLS.md`.
   - Keep generated data out of the repository root.

9. Add an end-to-end managed gap-repair loop.
   - DONE: `bot/run_gap_loop.py` and `python scripts/sara_cli.py run-autobot-gap-loop` now chain dataset build, collection-target generation, deterministic gap-material synthesis, and queue enqueue in one managed pass.
   - Preserve configurable candidate/rejected interim paths so repeated experiments can isolate intermediate artifacts without writing outside managed directories.
   - Treat this loop as the default operational bridge from Event Memory evaluation gaps to repair/replay curriculum preparation.

10. Add Phase 7 loop-readiness evidence.
   - DONE: `python scripts/sara_cli.py eval-autobot-gap-loop-readiness` now validates that accepted source-backed materials, collection targets, deterministic supplements, and queue injection remain connected as one managed repair loop.
   - Report slot-level target coverage, enqueue coverage, skip ratio, and repair/replay curriculum share so the project can distinguish "targets were requested" from "targets were actually converted into trainable materials".
   - Keep this evidence separate from Phase 6 energy proof and Phase 8 ANN baseline proof; it is an autonomy/readiness check, not a task-quality claim by itself.
   - DONE: the research benchmark suite now surfaces this readiness report alongside closed-loop gap reduction so Phase 7 can be monitored as both "did coverage improve?" and "did the loop operationally complete?".
   - DONE: `research_product_completion_gate.py` now requires this readiness report, so the high-level research-product gate fails loudly when the autonomous gap-repair loop stops producing usable repair/replay curriculum.

### Future Slice: Synchronized Experience Data

Goal: prepare small, high-integrity time-aligned multimodal learning records for Phase 16 and Phase 20 without turning broad web video scraping into the collection strategy.

The useful signal is not raw channel count. It is the density of independently informative, correctly aligned, rights-cleared channels. Audio, transcript, visual, action, reaction, or sensor events are valuable only when their timestamps, provenance, uncertainty, and semantic relationship are reliable.

1. Define a synchronized experience record.
   - Proposed schema: `sara-synchronized-experience-v1`.
   - Store a source/session ID, monotonic timestamp, duration, modality, sparse or source-native payload reference, confidence, alignment uncertainty, speaker/track ID when available, evidence type, license, consent/compliance state, and source hash.
   - Preserve original timestamps before optional 25/32/40 ms chunking.
   - Distinguish observed events, automatic transcripts, translated captions, inferred visual labels, reactions, actions, and causal hypotheses.
   - Store both session-relative time/delay and a source-clock anchor when available; relative delay supports SNN learning, while anchored time preserves cross-device synchronization, reproducibility, and deletion/audit lineage.
   - Treat this as an event-memory substrate, not a finalized semantic database. The first stored unit should be a bounded, auditable time-local record with modality, change/evidence type, confidence, and lineage, even when object, action, reward, or state labels are still unknown.

2. Use a conservative source order.
   - First: project-created recordings and sensor sessions with explicit participant consent.
   - Second: public-domain or clearly reusable datasets whose licenses permit the intended storage, transformation, and model-training use.
   - Third: creator-provided or partner-provided media with explicit written permission and downloadable source files.
   - Last and opt-in only: platform-hosted media where both platform terms and content rights permit the exact access and storage path.
   - Do not scrape YouTube pages, obtain scraped YouTube content, or download/store YouTube audiovisual content without the permissions required by YouTube and the rights holder.
   - Do not treat a public URL, visible subtitle, or Creative Commons search label alone as sufficient provenance; record the actual license evidence and attribution requirements.

3. Start with audio, transcript, and timestamps.
   - Prefer human-aligned or creator-supplied transcripts.
   - Preserve word, segment, pause, overlap, speaker-turn, and non-speech event timing when licensed and available.
   - Record transcript origin as `human`, `creator_caption`, `automatic_caption`, `local_asr`, or `translated`, and never merge these quality levels silently.
   - Compare timestamp-aware learning against transcript-only and timestamp-shuffled controls.

4. Add visual alignment only after audio-text integrity passes.
   - Sample sparse scene-change, object-presence, motion, and speaker-visibility events rather than retaining every frame in the learning IR.
   - Treat co-occurrence as a binding candidate, not proof that the spoken noun names the visible object.
   - Add hard negatives: off-screen narration, montage, stock footage, subtitles that lag, multiple visible objects, and scene cuts.
   - Require repeated cross-source consistency or verification before durable cross-modal links are admitted.

5. Add interaction timing as a separate evidence type.
   - Represent turn onset, pause duration, overlap, interruption, response latency, laughter/no-laughter, and repair events when consent and rights permit.
   - Do not equate audience laughter, applause, engagement, or sentiment with semantic correctness or reward.
   - Compare true timing against shuffled-pause and transcript-only controls.

6. Define synchronization quality metrics.
   - `informative_channel_count`: channels that add non-duplicate evidence.
   - `alignment_confidence`: confidence that events refer to the same local occurrence.
   - `alignment_uncertainty_ms`: estimated timestamp error.
   - `cross_modal_consistency`: repeated agreement across aligned channels.
   - `contradiction_rate`: frequency of incompatible channel evidence.
   - `synchronization_density`: verified informative bindings per second, not raw streams or bytes.
   - `learning_gain_per_event` and eventually `joule_per_success`: useful gain under equal event and energy budgets.

7. Protect evaluation independence and privacy.
   - Split complete recordings, speakers, channels, creators, series, source hashes, and collection periods before deriving clips or events.
   - Keep clips from the same original media in one split.
   - Do not train on held-out timestamps, reactions, captions, or alternate uploads of evaluation material.
   - Minimize faces, voices, names, location, biometrics, children-related content, and bystander data unless explicit consent and policy permit use.
   - Support source revocation and deletion propagation through every derived event and manifest.

8. Add a small observed-only pilot before collection at scale.
   - Use rights-cleared audio plus transcript/timestamps first.
   - Add sparse visual events only after the audio-text pilot beats transcript-only and shuffled-timestamp controls.
   - Add conversation timing after speaker and reaction labels are independently auditable.
   - Do not add robotics or embodied sensor collection until the same event schema, consent rules, and source lineage work on smaller data.

9. Add hierarchical event compression.
   - Proposed pipeline: `raw evidence -> change events -> episodes -> relation hypotheses -> verified invariants`.
   - For ANN-free extraction, prefer `raw signal -> change detection -> temporal eventization -> synchronization detection -> frequent sequence mining -> prediction-gain test -> Event Memory`.
   - Keep a second optional ingestion lane for ANN-assisted proposals: `raw evidence -> candidate proposal extraction -> candidate episodes/relations -> contradiction/redundancy review -> prediction-gain/verification gate -> Event Memory`.
   - Both lanes must converge on the same typed verification boundary so research can continue in either direction without rewriting the durable memory design.
   - Do not begin by hand-labeling `state`, `event`, `result`, or `reward` from continuous media. Start from source-native streams, then derive higher abstractions only after lower-level change and synchronization evidence is stable.
   - Level 0 raw evidence is not the training IR, but retain bounded audit windows, calibration samples, and rights-cleared exemplars long enough to detect extractor errors and support later reprocessing.
   - Level 1 stores sensor deltas, scene changes, acoustic onsets, motion changes, touch/force changes, pause/turn boundaries, and transcript boundaries. This is the first mandatory compression layer.
   - Initial ANN-free feature families should stay close to signal statistics:
     - vision: frame difference, optical-flow summary, edge-change bursts, color-histogram shift, scene cut, motion onset/offset, region appearance/disappearance
     - audio: onset/pause, band-energy shift, pitch change, spectral centroid/bandwidth, turn boundary, periodic burst candidates
     - touch/proprioception: pressure delta, acceleration peak, contact start/end, force ramp, vibration burst
     - text/subtitles: token boundary, pause, segment boundary, repetition, mismatch to nearby non-text channels
   - Level 2 stores synchronization candidates across channels such as `vision-change near syllable onset`, `subtitle span near speaker turn`, or `touch spike near motion burst`, with uncertainty and channel provenance.
   - Level 3 stores provisional unlabeled event-dictionary entries and clusters such as `visual_cluster_018`, `audio_cluster_044`, `motion_pattern_007`, or `touch_event_003`, with confidence, prototype statistics, extractor identity, and occurrence count.
   - Optional ANN-assisted candidates may enter at Level 3 or Level 4 only when typed as proposals, for example `candidate_event`, `candidate_episode`, or `candidate_relation`, and never as observed facts.
   - Level 4 stores bounded episodes that group temporally and source-consistent events.
   - Level 5 stores relation or causal hypotheses, never observed facts.
   - Level 6 stores only verified reusable invariants with source counts, contradiction history, scope, and expiry/revalidation policy.
   - Every higher-level record must retain lineage to the lower-level event IDs and source hashes used to construct it.
   - Require explicit promotion tests between levels so the system can prove that a derived abstraction improves downstream quality or compression efficiency without silently discarding needed evidence.
   - Delay human-readable semantic labels until repeated synchrony or source-backed language evidence justifies them; the early Event Memory should work even when events are only cluster IDs.
   - If ANN proposals are used, preserve the same delay rule: candidate labels may aid search and audit, but promotion to durable memory must still depend on repeated support, contradiction checks, and prediction value.
   - DONE: a unified `EventMemoryIngestPipeline` now runs `change detection -> temporal eventization -> bounded episode segmentation -> frequent sequence mining -> synchrony/prediction-gain relation proposals -> verification`, while keeping observed records and ANN-assisted candidate records distinct.

10. Add change-triggered eventization without deleting baseline context.
   - Use modality-specific delta thresholds, hysteresis, refractory windows, and scene/segment boundaries to avoid storing repeated unchanged samples.
   - Prefer a `change store` before a rich semantic event store. Early ingestion should answer `what changed, when, and with what confidence` before it claims `what this means`.
   - Keep a low-rate baseline reservoir of ordinary states so the system can estimate normality, class frequency, calibration drift, and false-surprise rates.
   - Preserve pre-event and post-event context windows for high-value changes; an isolated `cat_appeared` event is insufficient without evidence of what changed and what followed.
   - Store explicit `state_continues` summaries for long stable intervals instead of duplicating frames.
   - Compare eventized data with uniform sparse sampling under equal byte, event, and compute budgets.
   - When a later extractor infers semantics, keep both the original change evidence and the derived semantic label so the system can retract or revise the higher-level claim without losing the underlying observation.
   - Add explicit detector diagnostics such as threshold, refractory interval, calibration version, local baseline estimate, and suppressed-change count so later failures can be traced to extraction rather than learning.
   - If an ANN proposer is present, record proposer model ID, version, proposal prompt/configuration hash when applicable, confidence, and calibration state separately from the observation that triggered the proposal.

11. Add balanced experience selection.
   - Do not retain or replay data by prediction error alone.
   - Compute a bounded retention priority from novelty/surprise, expected learning gain, uncertainty, safety/rarity, representative coverage, source reliability, contradiction value, and storage/event cost.
   - Clip each component so sensor noise, adversarial novelty, or model miscalibration cannot monopolize memory.
   - Reserve explicit quotas for routine representative experience, rare/high-risk events, negative/no-change examples, contradictions, and recovery outcomes.
   - Re-estimate priority after learning; once-surprising events should decay unless they remain useful, rare, safety-critical, or required for coverage.
   - Distinguish raw prediction error from validated surprise. A large model mismatch should not become a durable learning target unless the underlying event survives source checks, synchronization checks, and contradiction review.
   - Prefer `prediction gain` over mere co-occurrence when deciding whether a relation is worth keeping:
     - compare `P(B|A)` with `P(B)` or an equivalent bounded lift statistic
     - retain a relation only when it improves bounded next-event or next-state prediction over the no-relation baseline
     - store evidence count and counterexample count together so frequent but weak associations do not crowd out stronger predictive structure

12. Add a sparse temporal relation graph.
   - Proposed schema: `sara-temporal-relation-graph-v1`.
   - Store event nodes plus bounded typed edges such as `before`, `after`, `overlaps`, `same_episode`, `predicts`, `action_precedes_result`, and `causal_hypothesis`.
   - Represent delay as a bounded interval or compact distribution, not only a single millisecond value.
   - Keep a session clock anchor for audit while using relative delay edges in local learning.
   - Cap outgoing edges, episode size, graph depth, and query expansion.
   - Require counterexamples, repeated sources, action/result evidence, or deterministic verification before promoting a temporal relation to a durable causal invariant.
   - Support unlabeled relation learning first; a node may remain `visual_cluster_018` while still participating in useful `predicts`, `follows`, `suppresses`, or `same_episode` edges.
   - Permit ANN-generated relation candidates only as `candidate_relation` records with explicit proposer provenance; they must pass the same counterexample and prediction-gain checks as ANN-free relation hypotheses.
   - DONE: relation-level concept review can now consume `FrequentSequence` support as a separate signal, so repeated ordered episode patterns can raise revalidation priority and optionally become an explicit admission requirement without collapsing observed facts and proposal labels into one layer.

14. Add an ANN-free bootstrap implementation path.
   - Proposed low-level modules:
     - `src/sara_engine/ingest/change_detection.py`
     - `src/sara_engine/ingest/temporal_eventizer.py`
     - `src/sara_engine/ingest/episode_segmentation.py`
     - `src/sara_engine/ingest/synchrony_detector.py`
     - `src/sara_engine/ingest/frequent_sequence.py`
     - `src/sara_engine/ingest/prediction_gain.py`
   - Keep the first version deterministic, thresholded, and auditable.
   - Prefer simple clustering or prototype matching over semantic classification.
   - Treat any future ANN-assisted labels as optional overlays on top of the ANN-free event stream, never as the sole source of Event Memory records.
   - DONE: deterministic `EpisodeSegmenter` and `FrequentSequenceMiner` now provide bounded episode construction and repeated-sequence extraction without requiring ANN semantics.

15. Add an optional ANN-assisted proposal path.
   - Proposed low-level modules:
     - `src/sara_engine/ingest/candidate_proposals.py`
     - `src/sara_engine/ingest/proposal_verifier.py`
     - `src/sara_engine/ingest/proposal_lineage.py`
   - Restrict ANN use to proposal generation for event, episode, relation, subtitle/ASR alignment, or coarse object/action hints.
   - Do not let ANN outputs write directly into observed state, verified relations, or concept crystals.
   - Preserve five distinct record types:
     - `observed_event`
     - `candidate_event`
     - `candidate_relation`
     - `verified_relation`
     - `concept_crystal_candidate`
   - Require the ANN-assisted path to emit the same managed diagnostics as the ANN-free path: evidence count, counterexample count, prediction gain, contradiction status, lineage, and retention decision.
   - Make the ANN-assisted path removable at runtime and benchmark time so SARA can be evaluated with and without proposal help under the same downstream verification logic.
   - DONE: observed and proposal-assisted events now converge through the same bounded episode segmentation interface, so either lane can be enabled without changing downstream sequence-mining and verification boundaries.

13. Make compression reversible enough to audit.
   - Record extractor version, thresholds, calibration, source segment, compression decision, discarded-event counts, and uncertainty.
   - Keep deterministic manifests for raw-to-event, event-to-episode, episode-to-relation, and relation-to-invariant transitions.
   - When raw retention expires, preserve hashes, consent/license lineage, deletion state, and sufficient aggregate diagnostics without retaining prohibited content.
   - Rebuild or invalidate dependent concepts when an extractor changes materially, a source is revoked, or contradiction exceeds the configured threshold.
   - Do not claim compression quality from byte reduction alone; measure downstream quality loss, false-event rate, missed-event rate, and energy saved per successful task.

### Managed Outputs

- Raw collected source material: `data/raw/autobot/`
- Rights-cleared raw synchronized sessions: `data/raw/synchronized_experience/`
- Extracted candidate text: `data/interim/autobot/extracted_text.jsonl`
- Candidate synchronized events: `data/interim/synchronized_experience/candidate_events.jsonl`
- Candidate episodes and relation hypotheses: `data/interim/synchronized_experience/candidate_relations.jsonl`
- Generated candidate materials: `data/interim/autobot/candidate_learning_materials.jsonl`
- Rejected generated materials: `data/interim/autobot/rejected_learning_materials.jsonl`
- Rejected or quarantined synchronized events: `data/interim/synchronized_experience/rejected_events.jsonl`
- Accepted QA data: `data/processed/autobot/qa_pairs.jsonl`
- Accepted contrastive data: `data/processed/autobot/contrastive_pairs.jsonl`
- Accepted negative-query data: `data/processed/autobot/negative_queries.jsonl`
- Accepted synchronized event manifest: `data/processed/synchronized_experience/manifest.jsonl`
- Accepted temporal relation graph: `data/processed/synchronized_experience/temporal_relation_graph.jsonl`
- Verified invariant manifest: `data/processed/synchronized_experience/verified_invariants.jsonl`
- Curriculum manifest: `data/processed/autobot/curriculum_manifest.jsonl`
- Dataset builder report: `workspace/autobot/dataset_builder_report.json`
- Dataset builder summary: `workspace/autobot/dataset_builder_summary.txt`
- Synchronization quality report: `workspace/evaluation/synchronized_experience_quality.json`
- Event compression and retention report: `workspace/evaluation/synchronized_experience_compression.json`
- Raw-to-invariant lineage manifest: `workspace/autobot/synchronized_experience_lineage.jsonl`
- Rights, consent, attribution, and deletion ledger: `workspace/autobot/synchronized_experience_compliance.json`

### Acceptance Criteria

- The bot can generate multiple learning-material types from already accepted records without human rewriting.
- Generated materials pass quality/compliance gates before entering the training queue.
- Evaluation failures can influence the next data-preparation cycle.
- Negative-query and contrastive-pair generation improve real-data external-validity coverage without weakening abstention behavior.
- Train/evaluation source hashes and near-duplicate checks show no prohibited overlap.
- Reported gains reproduce on a frozen independently sourced evaluation set.
- Timestamp-aware multimodal material outperforms transcript-only and timestamp-shuffled controls before collection is expanded.
- Synchronization density counts verified informative bindings rather than duplicate channels, frames, or raw bytes.
- Every accepted synchronized record has auditable license/consent state, source lineage, alignment uncertainty, and deletion propagation.
- Eventization reduces stored bytes and processed events over uniform sparse sampling without exceeding predefined missed-event, false-event, quality-loss, or energy budgets.
- Retention contains routine baseline, negative/no-change, rare/safety, contradiction, and recovery strata; surprise-only selection is not permitted.
- Temporal relation edges remain bounded and causal hypotheses cannot become verified invariants from temporal order alone.
- Higher-level episodes and invariants retain deterministic lineage to source hashes and lower-level event IDs.
- All artifacts follow the managed output policy.
- Human review becomes an audit step, not the default data-preparation path.

## Phase 8: Stronger External Baselines

Goal: make the ANN comparison harder and more credible without importing ANN assumptions into the SARA runtime.

**Priority: 2. Proxy baselines are insufficient for the main research claim.**

### Deliverables

- Add at least one stronger offline ANN-style retrieval baseline for comparison.
  - DONE: `real_data_external_validity.py` includes an offline BM25-style lexical baseline alongside dense-scan and hashed dense-embedding proxies.
- Distinguish actual baseline implementations from proxy estimates in every report and chart.
- Add CPU-only reference implementations, starting small:
  - BM25 with a standard library implementation or independently validated equivalent.
  - A lightweight pretrained sentence-embedding retriever with exact cosine search.
  - A FAISS CPU index when the optional dependency is available, with exact-search fallback and identical embeddings.
  - A tiny pretrained Transformer or cross-encoder retrieval/reranking baseline on the same candidates.
  - DONE: `real_data_external_validity.py` now exposes `reference_readiness`, separating `not_configured`, `missing_directory`, and dependency/runtime failures for optional local embedding, FAISS, and cross-encoder references.
- Add one comparison artifact that separates proxy baselines, offline reference baselines, and physical energy evidence in one place.
  - DONE: `python scripts/sara_cli.py eval-sara-ann-comparison` now writes a managed report that labels `proxy`, `offline_reference`, and `physical` evidence tiers separately, highlights the strongest currently available baseline, and surfaces the next missing comparison actions.
- Record model/index identity, parameter count, embedding dimension, quantization, thread count, index build policy, retrieval latency, peak RSS, success/quality, and measured joules when Phase 6 instrumentation is available.
- Compare cold-start, warm-index, and repeated-query conditions separately.
- Keep ANN model download/training and dense index construction outside SARA's production runtime, but include their cost when the compared use case requires them.
- Expand real-data tasks beyond the current small curated fixtures.
- Add noisy, adversarial, and delayed-recall retrieval cases.
  - DONE: repository fixture probe runs noisy, adversarial, and delayed-recall cases through sparse retrieval and records observed metrics.
- Keep dense baselines outside the production runtime path.
- Add per-task external validity summaries that separate quality, cost, abstention, and failure type.
  - DONE: `real_data_external_validity.py` writes `per_task_external_validity_summary` with quality, cost, abstention, and `failure_type` fields.

### Acceptance Criteria

- SARA keeps sparse-event cost advantage on bounded tasks.
- SARA is compared against at least BM25 plus one real pretrained embedding or tiny-Transformer baseline, not only hashed-vector or dense-scan proxies.
- All systems use the same corpus, candidates, query set, success criteria, CPU constraints, and measurement boundary.
- Accuracy, abstention, latency, memory, and energy are reported together; no efficiency win is claimed below the required quality floor.
- Near-miss and partial-evidence controls remain passing.
- External baselines are clearly labeled as offline references.
- Runtime policy remains CPU-first and sparse.

## Phase 9: Research-Grade Benchmark Package

Goal: make SARA easier for other researchers to evaluate and reproduce.

### Deliverables

- Add a single benchmark entry command that runs the recommended research suite.
  - DONE: `python scripts/sara_cli.py eval-research-benchmark-suite` runs the compact suite and writes a managed manifest.
- Create a compact benchmark README or protocol under `doc/`.
  - DONE: `doc/BENCHMARK_PROTOCOL.md` documents commands, outputs, proven claims, and unproven claims.
- Produce a machine-readable benchmark manifest under `workspace/evaluation/`.
  - DONE: `workspace/evaluation/research_benchmark_manifest.json` is produced by the benchmark suite.
- Add example datasets or fixtures small enough for repository-safe validation.
  - DONE: `data/processed/benchmark_fixtures/external_validity_cases.jsonl` covers QA, abstention, contrastive, noisy, adversarial, and delayed-recall examples.
- Add clear "what is proven" and "what is not proven" sections to release notes or benchmark docs.
  - DONE: the benchmark protocol and manifest include explicit proven and not-proven sections.

### Acceptance Criteria

- A new researcher can reproduce the v1.1 gate surface and ANN-efficiency proxy evidence from documented commands.
- Physical energy experiments have a clear optional path when meters or target hardware are available.
- The benchmark suite avoids hidden root outputs and follows managed path policy.

## Phase 10: Rust Sparse Runtime Hardening

Goal: turn the existing PyO3 Rust core from an acceleration helper into a better-tested, benchmarked, policy-aligned sparse runtime foundation.

The current Rust crate builds successfully and exposes useful primitives through `sara_rust_core`, but it still needs stronger tests, English-only comments, clearer API coverage, and real performance evidence before it can be treated as the main sparse runtime foundation.

### Implementation Steps

1. Clean up Rust source comments and public API notes.
   - Convert Japanese code comments in `src/sara_engine/lib.rs` to English.
   - Keep comments short and focused on non-obvious runtime behavior.
   - Document which primitives are policy-critical and which are compatibility helpers.

2. Add Rust unit tests.
   - Test `calculate_sdr_overlap`.
   - Test `sparse_propagate_threshold` with dict-style, tuple-list, and dense-list Python-compatible weight shapes.
   - Test `SpikeEngine` propagation, decay, STDP update, and reset behavior.
   - Test `SpikeWTARouter` top-k selection, threshold adaptation, and weight decay.
   - Test `LIFNetwork` threshold firing and reset.
   - Test `CausalSynapses` delay-aware potential calculation and prediction-error learning.
   - Test `ScalableSDRMemory` top-k overlap search and empty-query behavior.
   - Test `RewardModulatedSTDP` eligibility trace and reward update bounds.
   - Test `build_direct_synapses`, `batch_tokens_to_sdr`, and `apply_homeostatic_scaling`.

3. Align Python/Rust API expectations.
   - Resolve the Python references to `RustSpikeAttention` by either implementing it in Rust or removing/renaming the optional path.
   - Add a small Python smoke test that imports `sara_rust_core` and checks the expected exported symbols.
   - Keep Python fallbacks explicit and tested so the package remains usable when the extension is not built.

4. Add Rust benchmark coverage.
   - Add a benchmark script or test harness comparing Rust and Python fallback paths for:
     - sparse propagation
     - SDR overlap/search
     - direct synapse construction
     - batch token-to-SDR conversion
   - Write benchmark reports under `workspace/evaluation/`.
   - Track speedup and output-equivalence separately.

5. Use Rayon where it actually helps.
   - Parallelize safe, deterministic hot paths such as batch SDR generation and large direct-synapse accumulation.
   - Keep deterministic output where tests require it.
   - Avoid parallelism that increases memory pressure enough to undermine energy-efficiency goals.

6. Improve error handling and input validation.
   - Return clear Python exceptions for malformed weights, out-of-range parameters, empty unsupported structures, and invalid thresholds.
   - Add bounds for density, vocab size, context window, top-k, and decay rates.

7. Connect Rust runtime status to release evidence.
   - Add a managed Rust core readiness report under `workspace/evaluation/rust_core_readiness.json`.
   - Include version alignment, exported symbols, smoke results, unit-test status, and benchmark summary.
   - Consider adding this report to research-product completion only after it is stable across several runs.

### Acceptance Criteria

- `cargo test` runs meaningful Rust tests, not only a zero-test build.
- Python import smoke tests confirm the expected `sara_rust_core` API.
- Optional Python fallbacks remain working and explicit.
- Rust benchmark reports show where Rust is faster and where Python remains the reference.
- No new Rust path violates the no-backprop, no-dense-runtime, CPU-first policy.
- Rust comments and logs are in English.

## Phase 11: Hardware And Neuromorphic Portability

Goal: make the sparse-event runtime easier to map to hardware-oriented backends.

### Deliverables

- Formalize the sparse event IR used by runtime traces and backend profiles.
  - DONE: `src/sara_engine/edge/neuromorphic.py` emits `sara-spike-event-ir-v1` and a `sara-neuromorphic-capability-matrix-v1` summary for backend profiles.
- Expand neuromorphic HAL smoke tests into a backend capability matrix.
  - DONE: `python scripts/sara_cli.py eval-neuromorphic-capability-matrix` writes the managed backend profile matrix.
- Add hardware-profile reports for event budget, routing hints, memory footprint, and unsupported operations.
  - DONE: `workspace/evaluation/neuromorphic_capability_matrix.json` records event headroom, state budget, routing hints, update policies, adapter policy, and unsupported checks per profile.
- Keep hardware-specific adapters optional.
  - DONE: the matrix records Lava/SpiNNaker/Akida profile readiness without requiring any accelerator runtime.

### Acceptance Criteria

- CPU behavior remains the reference path.
- Hardware profile reports can explain what maps cleanly and what falls back.
- No release-critical feature requires a specific accelerator.

## Phase 12: Usability And Research Operator Experience

Goal: make the product usable by researchers without requiring them to read every gate script.

### Deliverables

- Consolidate common release and research commands into clearer docs and CLI aliases.
- Add a compact operational dashboard or text summary generator for the most important artifacts.
- Improve examples for corpus import, curriculum training, memory inspection, energy measurement, and gate review.
- Add troubleshooting notes for Python version, missing optional dependencies, and managed output violations.

### Acceptance Criteria

- The active docs remain short and role-specific.
- `doc/old/` keeps historical material out of the main path.
- A release operator can identify the next action from one summary report or manifest.

### Optional Candidate: Local LLM Operator Assistant

Source idea: Google, "Gemma 4: Our most capable open models to date", Google Blog, 2026-04-02.

Adopt the useful product idea that a compact local or edge LLM can improve developer and researcher workflows through structured JSON output, function calling, long-context document handling, code assistance, and multimodal inspection. This must remain an optional operator-assistance layer, not a SARA core intelligence mechanism.

Policy boundary:

- Do not make any LLM required for correctness, release gates, runtime learning, or SNN inference.
- Do not use dense LLM training, backpropagation, GPU-first inference, or hidden dense embeddings as SARA's primary runtime path.
- Do not allow an LLM to directly mutate training data, models, release manifests, or managed reports without deterministic validation.
- Keep all LLM-generated proposals source-linked, reviewable, and reproducible enough to be rejected or replayed by non-LLM tooling.

SARA-native implementation direction:

- Add an optional adapter that converts local LLM output into constrained action proposals, never direct actions.
  - DONE: `src/sara_engine/operator/llm_assistant.py` evaluates proposal JSON without executing it.
- Use strict JSON schemas for proposal types such as `roadmap_patch`, `evaluation_summary`, `dataset_candidate_review`, `collector_request`, `triage_note`, and `operator_next_action`.
  - DONE: `src/sara_engine/operator/llm_proposal_schema.py` validates proposal type, safe action type, source references, managed output paths, and secret-like text.
- Route every proposal through existing policy gates, managed-output validation, source-material checks, and deterministic benchmark scripts.
  - DONE: `scripts/eval/operator_llm_assistant_readiness.py` writes a managed readiness report and rejects invalid JSON, unsupported actions, unmanaged paths, secret-like text, and direct mutation attempts.
- Prefer local CPU-capable or edge-capable configurations for routine use; GPU or cloud execution may be documented only as optional research infrastructure.
- Treat multimodal or long-context LLM capabilities as ingestion and review helpers for documents, charts, screenshots, and code, not as learned SARA state.

Proposed artifacts:

- Reference module: `src/sara_engine/operator/llm_assistant.py`
- Schema module: `src/sara_engine/operator/llm_proposal_schema.py`
- Evaluator script: `scripts/eval/operator_llm_assistant_readiness.py`
- Report: `workspace/evaluation/operator_llm_assistant_readiness.json`
- Summary: `workspace/evaluation/operator_llm_assistant_readiness_summary.txt`

Acceptance criteria:

- The assistant is disabled by default and the full benchmark suite remains green without it.
  - DONE: readiness passes without importing or calling any LLM runtime.
- Every accepted proposal records source references, schema version, policy-gate result, and deterministic validator output.
  - DONE: validation entries include schema version, source count, action count, rejection reasons, and acceptance status.
- Invalid JSON, unsupported actions, missing sources, unmanaged paths, secret-like text, and direct model/data mutation attempts are rejected.
  - DONE: unit tests and the readiness script cover these rejection classes.
- The readiness report makes latency, token budget, proposal acceptance rate, rejection reasons, and fallback behavior visible.
  - DONE: the managed report records zero-runtime latency/token fields, acceptance rate, rejection counts, and fallback behavior.
- Operator docs clearly state that the local LLM assistant is a convenience layer, not a release-critical dependency.

## Phase 13: Capability Expansion Under Sparse Constraints

Goal: improve useful intelligence while preserving the energy-policy advantage.

### Candidate Directions

- Larger continual-memory experiments with bounded replay.
- Event-camera or DVS classification and association tasks.
- Stronger local credit assignment without runtime backpropagation.
- Better route selection between specialist submodels.
- Improved sparse verifier behavior for uncertain retrieval and reasoning.
- Multimodal association tasks that start with classification, grounding, and prediction instead of generation.
- Sparse reasoning priors for forecasting and future-state prediction, inspired by reasoning-aware time-series forecasting but implemented without dense LLM/TSFM fusion.
- Sparse synesthetic multimodal binding that treats language, vision, audio, and tactile events as equal sparse event sources rather than language-centered multimodal inputs.
- Sparse verifiable planning traces that decompose action plans into state-action-next-state checks, inspired by PDDL-INSTRUCT but implemented without LLM instruction tuning.

### Focused Candidate: Sparse Reasoning Prior For Future-State Prediction

Source paper: Md Atik Ahamed et al., "Reasoning-Aware Training for Time Series Forecasting", arXiv:2605.08625v1.

Adopt the useful idea that an explicit reasoning prior can guide a numerical or temporal forecast and make the prediction easier to inspect. Do not port STRIDE directly: SARA should not adopt end-to-end backpropagation, LoRA training, dense LLM hidden-state projection, TSFM embedding fusion, quantile-loss training, or LLM-as-a-judge release gates.

SARA-native implementation direction:

- Build sparse reasoning priors from source-backed claims, event traces, retrieval evidence, external-event records, counterfactual lanes, and future-state route traces.
  - DONE: `sparse_reasoning_prior.py` builds bounded source-backed evidence signatures and machine-checkable traces.
- Inject the prior as a bounded sparse event/control signal into future-state prediction or route-selection evaluators, not as a dense embedding.
  - DONE: the first observed path emits bounded direction, magnitude, route, confidence, and abstention controls.
- Add a deterministic `logic_to_state_consistency` evaluator that checks whether the reasoning trace supports the predicted direction, magnitude bucket, route choice, or abstention decision.
  - DONE: fixture expectations are checked deterministically without an LLM judge.
- Add `event_relevance` and `external_event_missing_abstention` checks so the system abstains or raises uncertainty when sudden shifts require external context that is not present.
  - DONE: unsupported evidence and sudden shifts without external events abstain with explicit reasons.
- Keep all reasoning-prior evidence observed-only until it improves or preserves forecast quality, counterfactual consistency, abstention integrity, and event cost.
  - DONE: the benchmark is research-suite evidence and does not alter production forecasting.

Proposed artifacts:

- Reference module: `src/sara_engine/reasoning/sparse_reasoning_prior.py`
- Evaluator script: `scripts/eval/sparse_reasoning_prior_benchmark.py`
- Fixture: `data/processed/benchmark_fixtures/sparse_reasoning_prior_cases.jsonl`
- Report: `workspace/evaluation/sparse_reasoning_prior_benchmark.json`
- Summary: `workspace/evaluation/sparse_reasoning_prior_benchmark_summary.txt`
- Trace: `workspace/evaluation/sparse_reasoning_prior_traces.jsonl`

First implementation slice:

1. Add source-backed sparse evidence aggregation for direction and magnitude buckets.
   - DONE.
2. Add deterministic logic-to-state consistency and event-relevance checks.
   - DONE.
3. Add external-event-missing abstention for sudden-shift cases.
   - DONE.
4. Add a repository-safe fixture and managed observed-only benchmark.
   - DONE.
5. Add CLI command `python scripts/sara_cli.py eval-sparse-reasoning-prior`.
   - DONE.
6. Add the benchmark to the compact research suite without making it release-critical.
   - DONE.

### Focused Candidate: Sparse Verifiable Planning Trace

Source paper: Pulkit Verma, Ngoc La, Anthony Favier, Swaroop Mishra, and Julie A. Shah, "Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning", arXiv:2509.13351v1.

Adopt the useful planning idea that a generated plan should be decomposed into explicit, verifiable logical steps: precondition satisfaction, action applicability, effect application, invariant preservation, state transition, and goal progress. Do not port PDDL-INSTRUCT directly: SARA should not add LLM instruction tuning, dense chain-of-thought training, GPU-first fine-tuning, or hidden natural-language reasoning as a trusted internal mechanism.

SARA-native implementation direction:

- Represent each plan as sparse `state -> action -> next_state` event triples with explicit preconditions, add/delete effects, violated invariants, and goal-progress markers.
  - DONE: `src/sara_engine/reasoning/sparse_plan_trace.py` verifies STRIPS-like sparse plan traces and records machine-checkable step results.
- Add a deterministic sparse plan verifier that checks action applicability, state transition consistency, invariant preservation, and final goal satisfaction.
  - DONE: `scripts/eval/sparse_plan_trace_verifier.py` checks missing preconditions, wrong effects, missing frame persistence, invariant violations, and unmet goals.
- Convert invalid plans into source-backed repair and replay materials so failures can strengthen future route selection without runtime backpropagation.
  - DONE: invalid cases emit managed `plan_trace_repair` materials under `data/processed/autobot/`.
- Store reasoning traces as machine-checkable sparse facts, not free-form CoT text; optional LLM operator assistance may summarize reports but must not validate plans.
- Connect verified plan traces to future-state consistency, counterfactual lanes, sparse reasoning priors, Phase 15 dendritic feedback traces, and Phase 16 route selection.
- Keep PDDL support optional and narrow; start with a repository-safe STRIPS-like fixture rather than full PDDL coverage.
  - DONE: the first fixture is a repository-safe STRIPS-like JSONL fixture, not a full PDDL parser.

Proposed artifacts:

- Reference module: `src/sara_engine/reasoning/sparse_plan_trace.py`
- Evaluator script: `scripts/eval/sparse_plan_trace_verifier.py`
- Fixture data: `data/processed/benchmark_fixtures/sparse_plan_trace_cases.jsonl`
- Repair material output: `data/processed/autobot/plan_trace_repair_materials.jsonl`
- Report: `workspace/evaluation/sparse_plan_trace_verifier.json`
- Summary: `workspace/evaluation/sparse_plan_trace_verifier_summary.txt`

Acceptance criteria:

- Plan traces are sparse, CPU-first, bounded-state, and backpropagation-free.
  - DONE: verifier uses sparse sets and deterministic local checks only.
- Invalid action preconditions, wrong effects, invariant violations, missing frame persistence, and unmet goals are detected deterministically.
  - DONE: unit tests and managed evaluator cover these failure classes.
- Repair/replay materials remain source-backed and managed under `data/processed/autobot/`.
  - DONE: repair output is written to `data/processed/autobot/plan_trace_repair_materials.jsonl`.
- The verifier reports step count, event cost, state budget, invalid-step count, repair-material count, and abstention/fallback behavior.
  - DONE: report includes case count, invalid step count, event cost, state budget, repair-material count, and fallback behavior.
- No release-critical runtime behavior depends on LLM-generated chain-of-thought.
  - DONE: report is observed-only and contains no LLM dependency.

### Promotion Rule

Each candidate starts as one of:

- A small primitive
- A trace builder
- A focused evaluator
- An observed-only report
- A bounded runtime module

It can become release-critical only after quality, energy cost, state budget, traceability, and regression behavior remain stable across the managed suite.

## Phase 14: Sparse Own-Latent Learning

Goal: implement a SARA-native version of the "learn from your own latents, not from tokens" direction from arXiv:2605.27734v1, while preserving CPU-first, sparse-event, no-runtime-backprop policy.

Source paper: Daniel J. Korchinski, Alessandro Favero, and Matthieu Wyart, "Learn from your own latents and not from tokens: A sample-complexity theory", arXiv:2605.27734v1.

This phase should not port data2vec, JEPA, or the paper's gradient-based SLC experiments directly. Instead, it should convert the useful idea into sparse event-cluster prediction, local latent consolidation, and sample-efficiency evaluators that fit SARA's runtime constraints.

### Adoption Boundary

- Adopt the sample-efficiency hypothesis: predicting own latent structure can reduce the amount of source material needed to recover useful hierarchy.
- Adopt the recursive latent-clustering framing as a SARA evaluator and sparse data structure.
- Do not adopt GPU-first data2vec training, dense teacher-student regression, large dense contrastive batches, or runtime backpropagation.
- Do not make this release-critical until observed reports show stable quality, abstention, contrastive behavior, and event-cost advantage.

### Deliverables

1. Add a policy-compatible sparse latent signature builder.
   - Proposed module: `src/sara_engine/learning/own_latent.py`.
   - DONE: `own_latent.py` provides deterministic sparse signatures and local own-latent prediction state.
   - Build compact latent signatures from existing source-backed materials, retrieval traces, spike event traces, and direct-memory associations.
   - Use sparse sets, hashed event signatures, MinHash/Jaccard-style sketches, or bounded counters rather than dense embedding matrices.
   - Write intermediate signature artifacts under `data/interim/autobot/` or `workspace/evaluation/`.

2. Add an own-latent prediction primitive.
   - Proposed reference class: `SparseOwnLatentPredictor`.
   - DONE: `SparseOwnLatentPredictor` predicts latent labels from sparse context signatures using local co-occurrence updates.
   - Predict future or masked sparse latent signatures from nearby context events.
   - Use local update rules, STDP-style eligibility, homeostatic normalization, or bounded co-occurrence updates.
   - Do not require backpropagation, GPU execution, large dense contrastive batches, or unbounded replay.
   - Keep Python fallback behavior explicit; consider a Rust sparse hot path only after the Python reference is stable.

3. Add latent-cluster consolidation for dataset-builder outputs.
   - Proposed script: `scripts/eval/own_latent_manifest_builder.py`.
   - DONE: source-backed autobot learning materials can be converted into `data/processed/autobot/latent_manifest.jsonl` with sparse signatures and preserved source metadata.
   - Connect `bot/dataset_builder.py` outputs (`qa_pair`, `source_claim`, `contrastive_pair`, `negative_query`, summaries) to latent cluster IDs or event signatures.
   - Preserve source URL/path, material hash, language, quality score, and compliance metadata in the latent manifest.
   - Write final processed manifests under `data/processed/autobot/`.

4. Add a focused sample-efficiency evaluator.
   - Proposed script: `scripts/eval/own_latent_learning_benchmark.py`.
   - DONE: observed-only benchmark compares sparse own-latent prediction with a token-overlap baseline at multiple train sizes.
   - Compare token-level retrieval/training signals against sparse own-latent prediction at multiple data sizes.
   - Track accuracy, abstention integrity, contrastive near-miss behavior, event cost, state budget, and sample count.
   - Write managed reports under `workspace/evaluation/own_latent_learning_benchmark.json` and a compact summary text file.

5. Add an RHM-inspired synthetic fixture.
   - Proposed generator: `scripts/eval/build_own_latent_rhm_fixture.py`.
   - DONE: deterministic fixture generator writes `own_latent_rhm_cases.jsonl` under repository-safe benchmark fixtures.
   - Generate a small deterministic hierarchical sparse-event dataset that can test whether latent clusters are recovered with fewer examples than token-level baselines.
   - Keep it repository-safe and CPU-light.
   - Store fixture data under `data/processed/benchmark_fixtures/` when finalized, or `data/interim/` while experimental.

6. Add regression gates only after observed evidence is stable.
   - DONE: the research benchmark suite records own-latent evidence as observed-only, not as a release-critical gate.
   - Start as observed-only evidence in the research benchmark suite.
   - Promote to a required gate only after it improves or preserves external-validity quality, abstention, sparse event cost, and state budget across repeated runs.

### Managed Outputs

- Candidate latent signatures: `data/interim/autobot/latent_signatures.jsonl`
- Accepted latent manifest: `data/processed/autobot/latent_manifest.jsonl`
- Synthetic own-latent fixture: `data/processed/benchmark_fixtures/own_latent_rhm_cases.jsonl`
- Own-latent benchmark report: `workspace/evaluation/own_latent_learning_benchmark.json`
- Own-latent benchmark summary: `workspace/evaluation/own_latent_learning_benchmark_summary.txt`
- Own-latent benchmark history: `workspace/evaluation/own_latent_learning_history.json`

### First Implementation Slice

1. Implement `SparseOwnLatentPredictor` with deterministic sparse-set signatures and local co-occurrence updates.
   - DONE.
2. Build `own_latent_rhm_cases.jsonl` with a tiny fixed hierarchy and expected latent groups.
   - DONE.
3. Add a benchmark that compares token-overlap recovery against own-latent cluster recovery at 4, 8, 16, and 32 examples.
   - DONE.
4. Add CLI command `python scripts/sara_cli.py eval-own-latent-learning`.
   - DONE.
5. Add the observed-only report to `eval-research-benchmark-suite` after the benchmark is stable.
   - DONE.

### Acceptance Criteria

- Own-latent learning remains sparse, CPU-first, bounded-state, and backpropagation-free at runtime.
- Token-level baselines stay outside the production runtime path and are clearly labeled as comparison references.
- The benchmark reports sample efficiency and event/energy proxy together, not separately.
- Negative-query and contrastive controls do not regress when latent-cluster training data is added.
- All generated artifacts follow the managed output policy.
- The implementation is useful even when no Rust extension or accelerator is available.

## Phase 15: Sparse Dendritic Feedback Gate

Goal: implement a SARA-optimized, sparse-event version of the dendritic implicit-bias idea from arXiv:2605.30370v2, improving robustness and sample efficiency without importing dense implicit ANN layers.

Source paper: Raul Mohedano et al., "Updating the standard neuron model in artificial neural networks", arXiv:2605.30370v2.

This phase should not port IBNN directly. SARA should adopt the biological insight that dendritic nonlinearities, local lateral interactions, and bAP-like feedback can make units more expressive and robust, but implement it as a bounded sparse gate over events and latent units.

### Adoption Boundary

- Adopt dendritic-like local nonlinear bias as a sparse event gate.
- Adopt bAP-inspired local feedback as a bounded state signal from a unit's recent output back into its local input context.
- Adopt lateral sparse interactions only among a small neighborhood of active units or latent clusters.
- Do not solve dense implicit equations with unbounded gradient descent at runtime.
- Do not add dense all-to-all neuron comparisons, GPU requirements, or backpropagation-dependent learning.
- Do not make the mechanism release-critical until robustness and event-cost evidence are stable.

### Deliverables

1. Add a sparse dendritic gate primitive.
   - Proposed module: `src/sara_engine/learning/dendritic_feedback.py`.
   - Proposed reference class: `SparseDendriticFeedbackGate`.
   - DONE: `SparseDendriticFeedbackGate` gates sparse events with local potential, recent-output feedback, sparse neighbor activity, and trace output.
   - Inputs: active event IDs, local potentials, recent output spikes, sparse neighbor activity, and homeostatic state.
   - Output: gated sparse events plus a trace explaining which local dendritic terms changed the decision.

2. Use bounded fixed-point style updates.
   - DONE: the first reference path uses a bounded one-step update and records convergence steps, event cost, state budget, and fallback behavior.
   - Run at most 1-3 local update steps per event window.
   - Record convergence status, step count, event cost, and state budget.
   - Fall back to the ungated sparse path when convergence fails or the event budget is exceeded.

3. Add local learning and stabilization.
   - DONE: local co-occurrence updates and homeostatic clipping are implemented for the reference gate.
   - Update dendritic gate weights with local co-occurrence, STDP-style eligibility, and homeostatic clipping.
   - Encourage independent sparse representations when active latent clusters are too similar.
   - Keep all trainable or adaptive state bounded and serializable.

4. Integrate only into observed paths first.
   - Start with `real_data_external_validity.py` noisy/adversarial/contrastive fixtures and own-latent cluster stabilization.
   - Optionally add a Rust sparse hot path after the Python reference has stable tests and reports.
   - Do not alter default inference behavior until observed reports show stable benefit.

5. Add a robustness and memorization benchmark.
   - Proposed script: `scripts/eval/dendritic_feedback_gate_benchmark.py`.
   - DONE: observed-only benchmark writes managed robustness, cost, fallback, convergence, and trace evidence.
   - Compare baseline sparse routing against dendritic-gated sparse routing on:
     - noisy retrieval
     - adversarial near-miss retrieval
     - contrastive controls
     - small-sample latent clustering
     - mislabeled or conflicting source-backed materials
   - Track accuracy, abstention integrity, robustness delta, memorization suppression, event cost, state budget, fallback rate, and convergence steps.

6. Connect to hardware portability.
   - Represent dendritic gate traces in the neuromorphic capability matrix as optional local feedback operations.
   - Mark unsupported profiles as adapter fallback rather than blocking CPU reference behavior.

### Managed Outputs

- Dendritic gate benchmark report: `workspace/evaluation/dendritic_feedback_gate_benchmark.json`
- Dendritic gate benchmark summary: `workspace/evaluation/dendritic_feedback_gate_benchmark_summary.txt`
- Optional gate trace samples: `workspace/evaluation/dendritic_feedback_gate_traces.jsonl`
- Optional processed gate state: `data/processed/autobot/dendritic_feedback_gate_state.json`

### First Implementation Slice

1. Implement `SparseDendriticFeedbackGate` with fixed sparse neighbor maps, bAP-like recent-output feedback, and 1-step bounded updates.
   - DONE.
2. Add unit tests for gating, convergence fallback, homeostatic clipping, and trace generation.
   - DONE.
3. Add a compact benchmark over existing noisy/adversarial/contrastive fixture cases.
   - DONE.
4. Add CLI command `python scripts/sara_cli.py eval-dendritic-feedback-gate`.
   - DONE.
5. Keep the report observed-only until robustness improves without increasing event cost beyond the configured budget.
   - DONE.

### Acceptance Criteria

- The gate remains sparse, CPU-first, bounded-state, and backpropagation-free.
- Gating never creates unbounded recurrent loops; every update has a hard step limit.
- Robustness or contrastive accuracy improves, or the report clearly records a negative result without promotion.
- Event cost, state budget, fallback rate, and convergence steps are visible in the report.
- Default production inference remains unchanged until repeated observed reports justify promotion.
- Generated artifacts stay under managed `workspace/`, `data/processed/`, or `data/interim/` paths.

## Phase 16: Sparse Synesthetic Multimodal Binding

Goal: build a SARA-native multimodal binding layer inspired by synesthesia, sensory substitution, cortical plasticity, and thalamic gating, while keeping every modality equal and preserving sparse SNN constraints.

This phase should not implement a dense LLM/LVM/LSM/LTSM mesh. It should translate the useful idea into sparse event synchronization, bounded cross-modal co-activation, and observed-only missing-modality prediction. Language must not become the hub, and no modality should be treated as the privileged representation.

Design principle: treat modality differences as differences in input statistics, timing, topology, and routing history rather than as reasons to create unrelated learning algorithms. SARA should use a shared sparse cortical primitive across modalities, with modality-specific adapters at the edges and common local plasticity in the middle.

### Adoption Boundary

- Adopt equal-modality processing for language, vision, audio, and tactile streams as sparse event sources.
- Adopt common time chunking for cross-modal temporal binding, starting with a configurable 25-40 ms equivalent event window.
- Adopt synesthetic cross-wiring as bounded sparse cross-modal links, not all-to-all dense cross-attention.
- Adopt sensory substitution as uncertainty-aware missing-modality event prediction, not ungrounded generation.
- Adopt thalamic gating as a sparse route selector over event pathways, not softmax/MoE dense weighting.
- Adopt a pluggable sparse cortical-column primitive whose local update rule is shared across language, vision, audio, and tactile streams.
- Adopt Hebbian/STDP-style local plasticity as the shared learning rule; modality specialization should emerge from input distribution, timing, sparse topology, and connection history.
- Do not use a universal dense embedding space, dense hidden-state fusion, GPU-first multimodal training, or runtime backpropagation.

### Deliverables

1. Add a shared sparse multimodal event IR.
   - Proposed module: `src/sara_engine/multimodal/synesthetic_binding.py`.
   - Represent each event with modality, time chunk, source ID, sparse signature, confidence, uncertainty, and energy/event cost.
   - Support at least `language`, `vision`, `audio`, and `tactile` event sources while allowing missing modalities.
   - Keep the IR compatible with the neuromorphic capability matrix and existing spike-event reports.

2. Add a pluggable sparse cortical primitive.
   - Proposed helper: `SparsePluggableCorticalColumn`.
   - Use the same sparse event intake, local co-activation, STDP-style update, homeostatic clipping, and bounded readout path for all modalities.
   - Keep modality-specific logic in adapters such as `LanguageEventAdapter`, `VisionEventAdapter`, `AudioEventAdapter`, and `TactileEventAdapter`.
   - Allow "plug swapping" experiments where a modality adapter is routed into a different cortical column while preserving the same learning rule.
   - Record whether specialization comes from input statistics, timing profile, topology, prior links, or gate history.

3. Add common time chunking.
   - Proposed helper: `SparseTemporalBinder`.
   - Convert asynchronous event streams into bounded windows with stable `time_chunk_id` values.
   - Start with 25 ms, 32 ms, and 40 ms profiles and report which profile preserves cross-modal alignment with the lowest event cost.
   - Avoid dense frame stacking or unbounded token buffering.

4. Add bounded synesthetic cross-links.
   - Proposed helper: `SparseSynestheticLinker`.
   - Learn local cross-modal links from co-occurrence, STDP-style timing, own-latent cluster agreement, and source-backed evidence.
   - Permit direct audio-to-tactile, tactile-to-vision, vision-to-language, and other non-language-centered routes.
   - Cap links per event, per modality pair, and per time chunk.

5. Add a sparse thalamic gate.
   - Proposed helper: `SparseThalamicGate`.
   - Select which event routes open from uncertainty, novelty, recent success, event cost, and task context.
   - Support an equal-processing mode and a focused mode, but record every route suppression and route amplification decision.
   - Reuse or interoperate with Phase 15 dendritic feedback traces when possible.

6. Add sensory-substitution prediction as observed-only evidence.
   - Predict missing modality event signatures from available modalities with uncertainty attached.
   - Label outputs as `predicted_missing_modality_events`, never as observed data.
   - Reject or abstain when source evidence, confidence, or cross-modal agreement is insufficient.
   - Use this first for benchmark fixtures and operator reports, not production inference.

7. Add a focused benchmark.
   - Proposed script: `scripts/eval/synesthetic_multimodal_binding_benchmark.py`.
   - Test temporal alignment, shared-column plug-swapping behavior, cross-modal link precision, missing-modality abstention, non-language route usefulness, event cost, state budget, and traceability.
   - Include small repository-safe fixtures for paired and partially missing language/vision/audio/tactile event streams.
   - Keep all evidence observed-only until repeated reports show quality gains without event-cost blowup.

8. Connect to existing research surfaces.
   - Feed own-latent cluster IDs into multimodal event signatures when available.
     - DONE: modality adapters optionally incorporate source-backed `latent_manifest.jsonl` cluster IDs and sparse signatures.
   - Feed dendritic feedback and thalamic gate traces into neuromorphic capability reporting as optional local feedback/routing operations.
     - DONE: bounded dendritic route hints feed the thalamic trace, and both appear as optional neuromorphic state-trace routing hints.
   - Keep optional LLM operator assistance outside the runtime path; it may summarize multimodal reports but must not create learned state.
     - DONE: no LLM runtime or proposal path is imported by the multimodal implementation.

### Managed Outputs

- Multimodal fixture data: `data/processed/benchmark_fixtures/synesthetic_multimodal_cases.jsonl`
- Candidate cross-modal links: `data/interim/autobot/synesthetic_cross_links.jsonl`
- Accepted binding manifest: `data/processed/autobot/synesthetic_binding_manifest.jsonl`
- Benchmark report: `workspace/evaluation/synesthetic_multimodal_binding_benchmark.json`
- Benchmark summary: `workspace/evaluation/synesthetic_multimodal_binding_benchmark_summary.txt`
- Optional trace samples: `workspace/evaluation/synesthetic_multimodal_binding_traces.jsonl`
- Optional plug-swap report: `workspace/evaluation/sparse_cortical_column_plug_swap_report.json`

### First Implementation Slice

1. Implement `SparsePluggableCorticalColumn` with shared local sparse updates and homeostatic clipping.
   - DONE.
2. Implement `SparseTemporalBinder` with deterministic time chunking and sparse event normalization.
   - DONE.
3. Implement `SparseSynestheticLinker` with bounded pairwise modality links and no dense attention.
   - DONE.
4. Implement `SparseThalamicGate` as a sparse route selector with equal-processing and focused modes.
   - DONE.
5. Add a tiny repository-safe fixture covering language, vision, audio, tactile, missing-modality, and plug-swapping cases.
   - DONE.
6. Add CLI command `python scripts/sara_cli.py eval-synesthetic-multimodal-binding`.
   - DONE.
7. Keep the report observed-only and outside release-critical gates until quality, abstention, event cost, and state budget remain stable.
   - DONE: the benchmark is included in the research suite as observed-only evidence and is not a release-critical gate.

### Second Implementation Slice

1. Add modality-edge adapters for language, vision, audio, and tactile feature streams.
   - DONE: all adapters emit the same `SparseMultimodalEvent` IR while recording specialization factors.
2. Compare 25 ms, 32 ms, and 40 ms temporal binding profiles.
   - DONE: the benchmark records all profiles and deterministically selects the best alignment/cost profile.
3. Integrate source-backed own-latent signatures when a managed latent manifest is available.
   - DONE: adapter traces preserve source reference, latent cluster ID, and `own_latent` specialization evidence.
4. Convert bounded dendritic feedback results into auditable thalamic route hints.
   - DONE: route hints are clipped to `[-0.25, 0.25]` and recorded per routed or suppressed event.
5. Expose dendritic and synesthetic routing operations in the neuromorphic capability matrix.
   - DONE: both operations are represented as optional state traces with backend adapter policy.

### Acceptance Criteria

- All modality streams are represented as sparse events, not dense universal embeddings.
- Language is not required as a hub for cross-modal binding or route selection.
- The same sparse cortical primitive can process multiple modality adapters without changing its learning rule.
- Modality specialization is explained by input statistics, timing windows, topology, and local plasticity traces rather than by hidden dense model-specific logic.
- Missing-modality predictions remain clearly labeled, uncertainty-aware, and source-bounded.
- Cross-modal links are capped, auditable, and reversible; no all-to-all dense cross-attention is introduced.
- Event cost, state budget, temporal alignment quality, route decisions, and abstention behavior are visible in managed reports.
- The implementation remains CPU-first, bounded-state, backpropagation-free at runtime, and useful without GPU or external large models.
- Generated artifacts stay under managed `workspace/`, `data/processed/`, or `data/interim/` paths.

## Phase 17: Verified Sparse Resonance Credit

Goal: make SARA's learning identity more original and coherent by coordinating its existing local plasticity mechanisms through verified multi-signal resonance rather than adding another isolated learning rule.

SARA should update a sparse local eligibility trace only when several independent evidence channels agree. Candidate channels include local spike coincidence, own-latent or future-state prediction consistency, deterministic verifier confidence, cross-modal agreement, reward, novelty, and metabolic headroom. Contradiction, abstention, unverified sources, or low resource headroom must freeze plasticity.

### Adoption Boundary

- Reuse STDP, three-factor eligibility, homeostasis, reasoning verification, multimodal agreement, and metabolic budgets.
- Add a bounded update-permission layer, not dense end-to-end credit assignment.
- Require source-backed and machine-checkable evidence before durable reinforcement.
- Permit signed positive or negative local credit after verification.
- Do not claim biological equivalence or Transformer-level superiority from synthetic fixtures.
- Keep the first implementation observed-only and outside production learning.

### First Implementation Slice

1. Add `SparseResonanceCreditAssigner` under `src/sara_engine/learning/resonance_credit.py`.
   - DONE: it combines six bounded sparse evidence channels and applies updates only after configurable multi-channel agreement.
2. Freeze plasticity on verifier contradiction, abstention, missing source backing, insufficient resonance, and low metabolic headroom.
   - DONE: every freeze reason is explicit in the trace.
3. Bound link count, weight magnitude, event cost, and serialized adaptive state.
   - DONE: state uses sparse link dictionaries with hard link and weight caps.
4. Compare resonance gating with a reward-only update policy on harmful-update fixtures.
   - DONE: the fixture records cases where reward-only learning would update despite contradiction, abstention, resource pressure, or single-channel noise.
5. Add CLI command `python scripts/sara_cli.py eval-resonance-credit`.
   - DONE.
6. Add observed-only evidence to the compact research suite.
   - DONE.

### Second Implementation Slice

1. Add a deterministic evidence bridge from managed SARA evaluator reports.
   - DONE: `resonance_evidence.py` derives resonance channels from reasoning-prior, plan-verifier, multimodal, dendritic, own-latent, and metabolic reports.
2. Keep evidence channels independent and auditable.
   - DONE: every channel records its report field derivation, trust status, source schema, and observed-only state.
3. Reject missing, failed, non-observed, or schema-less evidence as unverified.
   - DONE: incomplete report bundles produce `freeze_unverified_source`.
4. Recompute metabolic headroom through the bounded structural-budget evaluator.
   - DONE: integration execution does not depend on stale consolidation artifacts.
5. Add integration fault cases for verifier contradiction, source loss, abstention regression, and metabolic pressure.
   - DONE: all four faults freeze plasticity with distinct reasons.
6. Add CLI command `python scripts/sara_cli.py eval-resonance-credit-integration`.
   - DONE.
7. Add the bridge benchmark to the compact research suite after its source reports.
   - DONE.

### Managed Outputs

- Fixture: `data/processed/benchmark_fixtures/resonance_credit_cases.jsonl`
- Report: `workspace/evaluation/resonance_credit_benchmark.json`
- Summary: `workspace/evaluation/resonance_credit_benchmark_summary.txt`
- Trace: `workspace/evaluation/resonance_credit_traces.jsonl`
- Observed state: `workspace/evaluation/resonance_credit_state.json`
- Integration report: `workspace/evaluation/resonance_credit_integration_benchmark.json`
- Integration summary: `workspace/evaluation/resonance_credit_integration_benchmark_summary.txt`
- Integration traces: `workspace/evaluation/resonance_credit_integration_traces.jsonl`

### Acceptance Criteria

- Verified useful cases receive signed local credit.
- Contradiction, abstention, unverified-source, low-budget, and weak-resonance cases do not mutate learning state.
- Harmful-update suppression and decision integrity are visible in managed reports.
- Learning remains sparse, bounded, CPU-first, backpropagation-free, and compatible with existing eligibility traces.
- Production learning remains unchanged until larger source-aware experiments justify promotion.

## Phase 18: Verified Hierarchical Event-State Caching

**Priority: HIGH. This is the primary implementation phase after the completed Phase 17 work, except for physical-energy measurements that require external hardware.**

Goal: adapt the useful growing-memory idea from *Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281) into a SARA-native memory system that preserves useful event-state snapshots across long intervals without introducing dense recurrent states, runtime backpropagation, GPU dependence, or unbounded memory growth.

The paper's important contribution for SARA is the separation of current computation from selectively retrieved historical states. SARA should adopt this capability, but replace dense hidden-state caching and learned dense retrieval with verified sparse event-state records, bounded hierarchical retention, source-aware retrieval, and explicit forgetting.

Design role in SARA: this phase is the primary home of **Event Memory**. Recurrent dynamics may support short-lived processing elsewhere, but durable experience, delayed recall, contradiction tracking, and reusable `state -> event -> state` episodes should land here rather than being hidden inside transient recurrent state.

### Adoption Boundary

- Adopt segment-level state caching so delayed evidence can be recovered without replaying the entire event history.
- Adopt selective retrieval from multiple historical timescales.
- Adopt a logarithmic or similarly bounded hierarchy so long histories do not require linear active-memory growth.
- Store sparse event signatures, own-latent IDs, causal links, temporal metadata, source references, verifier state, uncertainty, and measured event cost.
- Permit useful unlabeled event identities such as stable cluster IDs or prototype IDs in durable memory when they repeatedly improve prediction or retrieval, even before human-readable semantics are attached.
- Keep ANN-assisted candidate labels, event proposals, or relation proposals as separate pre-verification record types; Phase 18 may admit only observed or verifier-cleared records into durable Event Memory.
- Use Phase 17 verified sparse resonance credit to decide whether a candidate state is promoted into durable memory.
- Use metabolic budget, utility, age, contradiction status, and redundancy to merge, demote, expire, or forget cache entries.
- Keep retrieval CPU-first and sparse, using bounded overlap, Jaccard-style similarity, causal adjacency, temporal proximity, or existing sparse memory primitives.
- Do not adopt dense RNN hidden-state vectors, dense query/key/value attention, parameter averaging such as Memory Soup, AdamW training, runtime backpropagation, or GPU-first kernels.
- Do not claim growing memory is beneficial unless delayed-recall quality improves under explicit event-cost and state-budget limits.

### First Implementation Slice

1. Add a sparse event-state cache primitive.
   - Proposed module: `src/sara_engine/memory/event_state_cache.py`.
   - Define a compact cache entry with sparse signature, source reference, time segment, own-latent ID, causal predecessors, confidence, uncertainty, verifier status, resonance score, access count, event cost, and expiry metadata.
   - Enforce hard limits for signature width, causal links, entries per tier, total entries, and serialized state size.
   - DONE: `VerifiedHierarchicalEventStateCache` implements the bounded sparse entry IR, lifecycle trace, hard signature/link/entry limits, and observed state serialization.

2. Add bounded hierarchical retention policies.
   - Support fixed, linear, and logarithmic retention profiles for observed comparison.
   - Keep recent exact event states in a short-lived tier and consolidate only verified, non-redundant states into longer-lived tiers.
   - Make promotion, merge, demotion, expiry, and eviction decisions deterministic and traceable.
   - DONE: fixed, linear, and logarithmic profiles use deterministic tier capacities, verified duplicate merging, utility eviction, and explicit expiry.

3. Add sparse retrieval and reactivation.
   - Rank candidates with bounded sparse overlap, own-latent agreement, causal relevance, temporal relevance, source reliability, and verifier status.
   - Return a hard-capped top-k result with an abstention path when evidence is weak or contradictory.
   - Reactivation may emit sparse routing hints or eligibility events, but must not silently mutate durable state.
   - DONE: hard-capped retrieval combines sparse overlap, own-latent, causal, temporal, source, confidence, and reliability evidence with an explicit abstention path.
   - DONE: Event Memory ingest can now consume reactivation hints through a bounded persistent self-state controller, so retrieval may help sustain short-term internal activity without turning recurrent state into the durable store.
   - DONE: retrieval ranking now accepts bounded self-state alignment as a separate component, allowing current internally maintained state to bias which verified memories are easiest to reactivate while preserving the primary role of sparse overlap, source integrity, and verification.
   - DONE: idle replay planning now closes the loop from Event Memory to persistent self-state and back into bounded replay candidate selection, so spontaneous/recurrent internal activity can choose verified memories for offline replay without letting recurrent state overwrite the durable store.
   - DONE: idle consolidation orchestration now connects replay selection, sleep-style replay evaluation, and concept review priority, so verified memories reactivated during low-input periods can feed bounded offline strengthening and concept admission without bypassing verifier-controlled memory boundaries.
   - DONE: verified Event Memory entries can now accept bounded consolidation-refresh feedback from replay observations, letting utility and tier move cautiously based on retention/noise/health outcomes instead of raw internal activation alone.
   - DONE: idle consolidation now derives `liquid / glass / crystal` memory-phase traces from replay outcomes and feeds those phases back into cautious cache-tier refresh, aligning phase-aware consolidation evidence with `recent / consolidated / durable` organization.
   - DONE: idle consolidation now also projects the same phase-enriched replay outcomes into delta-retention policy events, so phase-aware retention gates, cache-tier refresh, and sleep-style replay evidence all operate over one consistent bounded maintenance trace.

4. Connect verification and resonance credit.
   - Promote a candidate only when source backing, deterministic verification, resonance agreement, and metabolic headroom pass.
   - Block or quarantine entries associated with contradiction, failed verification, ungrounded prediction, or abstention.
   - Preserve predicted and observed states as distinct record types.
   - DONE: admission requires observed, source-backed, verified, sufficiently resonant states with metabolic headroom; contradiction, prediction-only, abstention, and failed verification have distinct block decisions.

5. Add metabolic consolidation and forgetting.
   - Prefer merging redundant verified entries over retaining duplicates.
   - Evict low-utility, stale, contradicted, or expensive entries before high-value source-backed states.
   - Report cache growth, retained utility, retrieval event cost, merge count, eviction count, and state-budget headroom.
   - DONE: redundant entries merge, tier budgets evict low-utility states, expiry is explicit, and managed state reports lifecycle counts and bounded tier occupancy.
   - DONE: cache utility and retrieval scoring now accept sequence-backed support as an explicit bounded factor, so repeated ordered episode evidence can improve durable retention priority without replacing source, resonance, or verification requirements.

6. Add a focused benchmark.
   - Proposed script: `scripts/eval/event_state_cache_benchmark.py`.
   - Compare no cache, fixed-window cache, linear hierarchy, and logarithmic hierarchy on delayed recall, distractor resistance, contradiction handling, source integrity, abstention, event cost, and state growth.
   - Include long-gap and repeated-evidence fixtures that expose whether hierarchical caching improves useful recall rather than merely retaining more data.
   - Keep all results observed-only until source-aware fixtures show repeatable gains.
   - DONE: the repository-safe benchmark compares no-cache, fixed, linear, and logarithmic profiles across 23 candidates and 3 delayed/negative queries.

7. Add CLI and research-suite integration.
   - Proposed command: `python scripts/sara_cli.py eval-event-state-cache`.
   - Run after Phase 17 resonance-credit integration so promotion decisions use live verified evidence.
   - Keep the benchmark outside release-critical gates until bounded growth and retrieval integrity remain stable.
   - DONE: `python scripts/sara_cli.py eval-event-state-cache` runs the benchmark, and the compact research suite includes it after resonance-credit integration.

### Second Implementation Slice

1. Bridge live Phase 17 resonance evidence directly into cache-admission candidates instead of relying only on fixture fields.
   - DONE: `event_state_evidence.py` derives promotion candidates from the six managed Phase 17 evidence sources and `SparseResonanceCreditAssigner`.
2. Add source-aware delayed-recall cases built from accepted autobot materials and preserve source revision/hash metadata.
   - DONE: the integration benchmark reads `latent_manifest.jsonl`, uses source references and material hashes, and compares fixed with logarithmic retention.
3. Emit bounded reactivation hints for causal routing and eligibility without mutating durable cache state during retrieval.
   - DONE: retrieval emits hard-capped `verified_event_state_reactivation` hints explicitly marked `mutates_durable_state = false`.
4. Add persistence round-trip, corrupted-state rejection, and schema migration tests.
   - DONE: strict v1 state loading validates schema, entry types, signature/link budgets, verified-observed status, duplicate IDs, and tier values; round-trip and corruption tests pass.
5. Evaluate a Rust sparse-overlap hot path only if Python profiling shows retrieval cost warrants it.
   - DONE: the source-aware benchmark reports a maximum retrieval event cost of `38`; Rust acceleration is deferred until larger profiling demonstrates a need.
6. Keep this slice observed-only until repeated source-aware runs preserve recall, abstention, source integrity, and bounded growth.
   - DONE: `python scripts/sara_cli.py eval-event-state-cache-integration` is included in the research suite as observed-only evidence and remains outside release-critical gates.

### Managed Outputs

- Fixture: `data/processed/benchmark_fixtures/event_state_cache_cases.jsonl`
- Candidate cache records: `data/interim/event_state_cache/candidates.jsonl`
- Accepted cache manifest: `data/processed/event_state_cache/manifest.jsonl`
- Benchmark report: `workspace/evaluation/event_state_cache_benchmark.json`
- Benchmark summary: `workspace/evaluation/event_state_cache_benchmark_summary.txt`
- Retrieval and lifecycle traces: `workspace/evaluation/event_state_cache_traces.jsonl`
- Observed cache state: `workspace/evaluation/event_state_cache_state.json`
- Source-aware integration report: `workspace/evaluation/event_state_cache_integration_benchmark.json`
- Source-aware integration summary: `workspace/evaluation/event_state_cache_integration_benchmark_summary.txt`
- Source-aware integration traces: `workspace/evaluation/event_state_cache_integration_traces.jsonl`
- Persistence round-trip state: `workspace/evaluation/event_state_cache_round_trip_state.json`

### Acceptance Criteria

- Delayed-recall success improves over no-cache and fixed-window controls without exceeding configured event-cost or state-growth budgets.
- Logarithmic or bounded hierarchical retention preserves useful long-range evidence with lower active-state growth than a linear cache.
- Contradicted, unverified, predicted-only, and abstained states cannot be promoted as verified durable memory.
- Retrieval remains sparse, top-k bounded, source-aware, deterministic, and auditable.
- Cache growth is bounded by hard tier limits, metabolic headroom, consolidation, expiry, and explicit forgetting.
- Production memory behavior remains unchanged until larger source-aware experiments justify promotion.
- The implementation remains CPU-first, backpropagation-free at runtime, dense-matrix-independent, and compatible with existing SNN event traces.
- All generated artifacts stay under managed `data/`, `workspace/`, or `models/` paths.

### Recurrent Boundary

- Do not try to make recurrent activity itself the durable memory store.
- Permit only bounded transient recurrence outside this phase, with explicit decay, hard occupancy limits, deterministic replay, and auditable handoff into verified event-memory records.
- If a behavior can be explained by a verified event-memory retrieval path, prefer that over adding larger or more persistent recurrent state.
- DONE: `PersistentSelfStateController` now provides bounded self-sustaining activity by combining tonic spontaneous firing, sparse recurrent reuse, Event Memory reactivation hints, and local transition prediction, while keeping the durable store in Phase 18 Event Memory rather than in unbounded recurrent state.
- DONE: self-state continuity now feeds back into relation verification and concept revalidation as a small bounded preference signal, so SARA can keep pursuing coherent hypotheses across sparse or intermittent input without letting recurrent state override source checks or verifier gates.
- DONE: bounded idle replay selection now uses persistent self-state continuity, Event Memory utility, sequence support, and optional astro-style modulation to decide which verified memories are worth replaying during low-input periods.
- DONE: bounded idle consolidation now returns replay-selected concept priorities into the concept review loop and derives sleep-style retention/noise/health traces from those same replay decisions, keeping internal maintenance auditable across time.
- DONE: bounded idle consolidation can now refresh verified cache utility/tier from observed replay outcomes, creating a cautious path from offline replay evidence into `recent/consolidated/durable` memory organization.
- DONE: `persistent_self_state_benchmark.py` now records observed-only evidence that sparse spontaneous firing, Event Memory reactivation, and internal next-state prediction can preserve a bounded self-state across idle steps.
- DONE: `idle_replay_benchmark.py` now records observed-only evidence that verified memories can be replay-ranked under an explicit event budget using self-state alignment, reactivation hints, and astro-style modulation.
- DONE: `internal_maintenance_efficiency_benchmark.py` now records fixed-loop `maintenance_selected_count`, `maintenance_refresh_count`, `maintenance_event_cost`, and normalized `maintenance_event_cost_per_selected`, providing a pre-physical reference surface for the same maintenance fields exported by Phase 6 energy-pair runs.
- DONE: the compact research benchmark suite now includes both persistent self-state and idle replay maintenance surfaces, so internal continuity evidence is tracked beside Event Memory, consolidation, and ANN-comparison evidence.

## Phase 19: Sparse Liquid Time-Constant Spiking Dynamics

**Priority: CONDITIONAL / after Phase 6, Phase 8, and Phase 7 evidence work. Do not activate this phase merely for architectural novelty.**

Goal: test whether a small, SARA-native subset of Liquid Neural Network and continuous-time modeling ideas can improve the temporal accuracy of fixed-time-constant SNN paths without sacrificing sparse-event execution, local learning, bounded state, CPU operation, or energy efficiency.

The useful idea is not to replace SARA with a dense ODE network. It is to let selected spiking neurons adapt bounded local time constants, leak, or threshold dynamics from sparse input history. Continuous local state must still pass through an axon-hillock-style event conversion boundary before routing, memory, or downstream learning. Relevant research references include Liquid Time-constant Networks (arXiv:2006.04439) and closed-form continuous-time models (arXiv:2106.13898), but SARA should adopt only policy-compatible primitives.

### Activation Gate

- Start implementation only after Phase 6 has a usable physical-energy comparison path and Phase 8 has credible external baselines, unless a narrowly scoped prototype directly helps those measurements.
- Require an observed fixed-time-constant failure mode on an irregular-timing, delayed-response, multimodal-binding, or continual-adaptation task.
- Define the expected accuracy gain, event-cost ceiling, state ceiling, latency ceiling, and energy non-regression threshold before implementation.
- Keep the fixed-time-constant SNN as the control and default production path.
- Do not promote the liquid path when its quality gain is absent, unstable, fixture-specific, or achievable with a simpler fixed-time-constant SNN adjustment.
- Stop and archive the experiment if sparse-event advantage, local-learning compatibility, bounded execution, or paired `joule_per_success` is materially worse than the fixed SNN control.

### Adoption Boundary

- Permit per-neuron or per-small-column adaptive time constants, bounded leak, adaptive threshold, and compact continuous local state.
- Prefer closed-form, piecewise, or event-driven updates evaluated only when an input event or scheduled decay boundary occurs.
- Use sparse local inputs, local eligibility traces, homeostasis, verified resonance credit, and metabolic gating for adaptation.
- Hard-bound time-constant range, state variables, update frequency, arithmetic operations per event, and serialized adaptive state.
- Preserve deterministic reset, replay, state inspection, abstention, and traceability.
- Treat this phase as an optional refinement of **short-term recurrent dynamics**, not as a replacement for Phase 18 event memory.
- Do not introduce a general-purpose numerical ODE solver into the runtime hot path.
- Do not introduce dense recurrent matrices, all-to-all continuous coupling, BPTT, runtime backpropagation, GPU dependence, or opaque pretrained LNN state.
- Do not describe the mechanism as an LNN advantage unless the tested implementation remains recognizably SNN-first and the benefit survives fixed-SNN controls.

### First Experimental Slice

1. Define a `SparseLiquidTimeConstantNeuron` or equivalent small-column primitive under `src/sara_engine/nn/`.
   - Maintain membrane state, adaptive threshold, and one bounded liquid time constant.
   - Convert continuous state to sparse spikes through the existing axon-hillock-style threshold boundary.
   - Keep update equations closed-form or event-driven and expose operation counts in the trace.
2. Add deterministic fixed-versus-liquid temporal fixtures.
   - Cover irregular event gaps, short/long dependency switching, delayed recall, distractors, and abrupt context changes.
   - Include negative controls where fixed time constants should already be sufficient.
3. Add a focused observed-only benchmark.
   - Proposed script: `scripts/eval/sparse_liquid_time_constant_benchmark.py`.
   - Compare fixed SNN, bounded liquid-time-constant SNN, and a simpler multi-timescale fixed-SNN control.
   - Report temporal accuracy, abstention, event count, arithmetic/update count, latency, peak state, adaptation stability, and replay determinism.
4. Integrate only as an optional route.
   - Candidate uses are Phase 16 temporal binding, Phase 18 event-state reactivation, and sensor/audio/tactile streams with irregular timestamps.
   - Do not modify production defaults or durable learning until the acceptance criteria pass on independently sourced held-out cases.
5. Connect to physical energy measurement after functional evidence.
   - Reuse the Phase 6 fairness contract and identical execution boundaries.
   - Compare fixed and liquid SNN variants with the same task, success criterion, CPU, run order, repetitions, and meter.

### Managed Outputs

- Fixture: `data/processed/benchmark_fixtures/sparse_liquid_time_constant_cases.jsonl`
- Benchmark report: `workspace/evaluation/sparse_liquid_time_constant_benchmark.json`
- Benchmark summary: `workspace/evaluation/sparse_liquid_time_constant_benchmark_summary.txt`
- State and event traces: `workspace/evaluation/sparse_liquid_time_constant_traces.jsonl`
- Optional physical-energy rows: `data/raw/energy_measurements.jsonl`

### Acceptance Criteria

- The liquid variant improves predefined held-out temporal quality over both the fixed SNN and the simpler multi-timescale fixed-SNN control.
- Accuracy gains reproduce across seeds or deterministic input-order variants and are not limited to self-generated training fixtures.
- Event count, update count, latency, and state remain within the predeclared ceilings.
- Runtime remains sparse, bounded, CPU-first, GPU-independent, and free of backpropagation.
- Adaptation uses local information and does not require dense matrix computation as the primary mechanism.
- Replay is deterministic when adaptive state and event history are fixed.
- No energy advantage is claimed before paired physical measurement; promotion requires non-regressing or improved `joule_per_success` at the required quality floor.
- If these criteria fail, retain the fixed-time-constant SNN design and make no production integration.

## Phase 20: Semantic Echo Field for Sparse Temporal Language

**Priority: CONDITIONAL / after Phase 6, Phase 8, and Phase 7 evidence work. Phase 19 is optional and not a prerequisite.**

Goal: improve SARA's bounded text understanding by representing language as sparse temporal events and retaining only salient semantic activity as finite multi-timescale echoes. Later language events should bind to compatible echoes through bounded local resonance rather than dense all-token Attention.

Design paper: [Semantic Echo Field v2: Event-Centric World Model Extension for SARA Engine](idea/Semantic_Echo_Field_Sparse_Temporal_Language_Architecture_v2.md).

### Activation Gate

- Do not delay Phase 6 physical energy evidence, Phase 8 external ANN baselines, or Phase 7 evaluation-isolated data work.
- Require a frozen independently sourced language benchmark with long-distance dependency, role binding, negation, anaphora, noisy text, unsupported queries, and delayed-recall cases.
- Declare accuracy, abstention, event, state, latency, and energy thresholds before implementation.
- Keep the current fixed SNN language path as the default and control.
- Compare against a single-decay fixed SNN and a simpler multi-timescale fixed SNN before crediting Semantic Echo Field.
- Reject the mechanism if gains depend on an external ANN parser, self-generated evaluation material, unbounded echo retention, or dense candidate comparison.

### Adoption Boundary

- Treat language as one sparse modality over a broader event-centric world model, not as the privileged hub of system meaning.
- Keep `state -> event -> state` records provisional and sparse; inferred state, reward, and causal structure must remain typed hypotheses until verified by source-backed evidence or deterministic checks.
- Treat Semantic Echo Field as the primary experimental home of **bounded recurrent SNN-style short-term language dynamics** in SARA: finite echoes, local resonance, explicit decay, and hard event/state ceilings.
- Represent surface form, optional phonology, source-backed sparse semantic features, provisional role hypotheses, boundaries, prediction errors, and causal hypotheses through a shared bounded event IR.
- Treat orthographic, phonological, semantic, predictive, and causal-hypothesis information as asynchronous sparse axes with independent time resolution, confidence, evidence type, expiry, and event cost; do not stack them into a dense 4D/5D tensor.
- Start with surface events and fixed fast/medium/slow echo tiers; open optional channels only under an explicit sparse gate.
- Maintain hard limits for active echoes, signature width, links per echo, resonance comparisons per input, decay lifetime, role slots, updates, and serialized state.
- Compute resonance only among active local candidates using semantic overlap, temporal compatibility, provisional role compatibility, source reliability, verified history, contradiction, and event cost.
- Treat phoneme or mora expansion as finer temporal resolution, not free information; compare it against repeated-event and equal-budget controls.
- Require cross-axis links to use repeated source-consistent co-activation and negative controls; temporal coincidence alone cannot create durable binding.
- Label causal events as `causal_hypothesis` until independent evidence and deterministic verification support promotion; STDP order alone is not causal proof.
- Treat role assignments as competing hypotheses rather than trusted parser output.
- Treat phase as an optional short-lived local role-binding mechanism, not a universal fixed mapping from grammatical roles to global phases.
- Treat internal phonological recoding as an optional ambiguity and boundary route, not a mandatory text-understanding path.
- Permit bounded input-triggered dynamic semantic modes composed of sparse event sets and relative timing, but prohibit always-on concept oscillators that increase idle cost.
- Separate transient echo state, predicted state, externally proposed features, and verified durable concepts.
- Reposition Semantic Echo Field as a cortical-style resonance subsystem that guides retrieval, ambiguity resolution, and next-state prediction over event memory, rather than as the sole center of language understanding.
- Use Phase 17 verified resonance credit and Phase 18 cache admission for crystallization; a single co-occurrence must never become durable knowledge.
- Use Phase 19 adaptive time constants only if fixed multi-timescale echoes show a reproducible limitation and Phase 19's own acceptance criteria pass.
- Do not introduce dense all-pairs Attention, dense recurrent matrices, runtime backpropagation, GPU dependence, or a general-purpose ODE solver.

### Role Split

- Phase 20 handles transient recurrent-like language processing: active echoes, local bindings, ambiguity resolution, and next-event or next-state hints.
- Phase 18 handles durable event memory: verified episodes, contradiction-aware retention, and delayed recall beyond the bounded echo horizon.
- Phase 19 is optional and only refines the short-term temporal dynamics of Phase 20 or other event routes when fixed-timescale recurrence is provably insufficient.
- Reject designs that blur these roles by hiding long-term memory in unbounded recurrent state or by turning event memory into a passive log with no verified retrieval path.

### First Experimental Slice

1. Add a bounded language-event adapter.
   - Proposed module: `src/sara_engine/language/semantic_events.py`.
   - Emit source-labeled sparse surface, boundary, optional phonological, semantic-feature, prediction-error, and causal-hypothesis events.
   - Preserve whether each feature is observed, dictionary-assisted, self-learned, or externally proposed.
   - Keep the adapter compatible with the broader synchronized-experience/event-memory schemas so text-derived events can later bind to non-language episodes without format conversion.
2. Add a fixed multi-timescale echo field.
   - Proposed module: `src/sara_engine/language/semantic_echo.py`.
   - Implement deterministic fast, medium, and slow decay tiers with hard occupancy and comparison limits.
   - Emit dependency, role, reactivation, contradiction, or abstention candidates without mutating durable state.
3. Add the simplest credible controls first.
   - Compare current SARA language behavior, single-decay retention, fixed multi-timescale retention, and Semantic Echo Field under identical state and event ceilings.
   - Record when the proposed mechanism offers no benefit and retain the simpler control.
4. Connect verified crystallization.
   - Convert repeated source-backed useful echo patterns into Phase 17 evidence candidates.
   - Admit durable concepts through Phase 18 only after verification, independent-source support, metabolic headroom, and negative/contrastive checks.
   - Route accepted language episodes through Phase 18 as event-memory candidates and expose an optional external audit trail for WordPress or other operator dashboards without making CMS state part of runtime truth.
5. Add optional features only through separate ablations.
   - Test phase-role slots after the echo-only path.
   - Test phonological recoding only on ambiguity, speech-text alignment, and phrase-boundary cases.
   - Keep both disabled by default until they improve held-out quality within budget.
6. Add asynchronous axis binding after the surface-only echo path.
   - Proposed module: `src/sara_engine/language/axis_binding.py`.
   - Bind only locally active orthographic, phonological, and semantic events under per-axis and per-window budgets.
   - Compare against simple event concatenation, repeated-event temporal expansion, and surface-only controls.
7. Add bounded dynamic semantic modes only after axis binding is stable.
   - Proposed module: `src/sara_engine/language/dynamic_semantic_mode.py`.
   - Represent a concept with a capped event set, partial order, delay windows, confidence, source state, and deterministic expiry.
   - Trigger modes only from input or verified reactivation hints and require zero additional steady firing after expiry.

### Focused Benchmark

- Proposed script: `scripts/eval/semantic_echo_field_benchmark.py`.
- Use frozen raw-text and source-aware fixtures covering:
  - local order and morphology
  - long-distance subject/predicate and object/predicate relations
  - embedded clauses and word-order variation
  - anaphora, omission, negation, and scope
  - noisy and adversarial near-miss text
  - unsupported queries and abstention
  - delayed recall and continual adaptation
  - source revision and contradiction
  - unknown words, morphology, reading-speed variation, and phonological ambiguity
  - speech-text alignment and deliberate mismatch
  - semantic-neighbor versus unrelated-concept resonance
  - temporal correlation versus source-supported causal hypotheses
- Keep external parser-assisted and external LLM-assisted conditions separate from raw-text-only SARA results.
- Include at least one language with different word-order properties after the initial Japanese slice.
- Report accuracy/F1, role-binding precision/recall, abstention, contradiction handling, retention, harmful crystallization, events and bindings per axis, false cross-axis resonance, dynamic-mode reactivation stability, idle spikes, unverified causal promotions, active echoes, state bytes, latency, peak RSS, and measured joules when Phase 6 instrumentation is available.

### Managed Outputs

- Fixture: `data/processed/benchmark_fixtures/semantic_echo_field_cases.jsonl`
- Optional source-backed feature proposals: `data/interim/semantic_echo/feature_proposals.jsonl`
- Verified concept manifest: `data/processed/semantic_echo/concept_manifest.jsonl`
- Benchmark report: `workspace/evaluation/semantic_echo_field_benchmark.json`
- Benchmark summary: `workspace/evaluation/semantic_echo_field_benchmark_summary.txt`
- Event, echo, binding, and crystallization traces: `workspace/evaluation/semantic_echo_field_traces.jsonl`
- Optional observed state: `workspace/evaluation/semantic_echo_field_state.json`
- Axis-binding ablation report: `workspace/evaluation/semantic_echo_axis_binding_ablation.json`
- Dynamic semantic-mode ablation report: `workspace/evaluation/dynamic_semantic_mode_ablation.json`

### Acceptance Criteria

- Semantic Echo Field improves predefined independently sourced held-out language quality over both single-decay and fixed multi-timescale SNN controls, or matches quality at a meaningfully lower measured cost.
- Negative-query, contrastive, contradiction, and abstention behavior do not regress.
- Improvements remain when external ANN parser and LLM feature proposals are disabled.
- Phase-role binding and phonological recoding are promoted independently only when their ablations show reproducible net benefit.
- Asynchronous multi-axis binding outperforms surface-only and simple-concatenation controls under equal event and state budgets; phonological expansion must also beat a repeated-event control.
- Dynamic semantic modes improve concept reactivation or context-sensitive meaning over static sparse signatures without persistent idle firing.
- Causal hypotheses cannot enter observed or durable state solely from temporal order, STDP, or cross-axis synchronization.
- Active echoes, resonance comparisons, links, updates, latency, and serialized state remain within predeclared hard ceilings.
- Durable crystallization requires source backing, repeated verified utility, contradiction checks, and metabolic headroom.
- Runtime remains sparse, bounded, CPU-first, GPU-independent, dense-Attention-free, and backpropagation-free.
- No energy advantage is claimed before paired physical measurement; production promotion requires non-regressing or improved `joule_per_success` at the required quality floor.
- If these criteria fail, retain the simpler fixed SNN language path and record the negative result.

## Immediate Next Actions

1. Freeze the Phase 6 fairness contract and extend the measurement schema with environment fingerprints, protocol IDs, paired run blocks, and quality-parity fields.
   - DONE: fairness schema v2 validates matched environments, fixtures, criteria, boundaries, tools, repetitions, trial counts, run order, quality parity, median ratios, and MAD.
2. Run the first paired SARA/ANN physical-energy session on the same CPU and compute per-task `joule_per_success`.
   - IN PROGRESS: `run-physical-energy-pair` now freezes and executes an alternating-order SARA/BM25 retrieval pair with identical corpus, tasks, repetitions, thread environment, and exact-match success criteria. The frozen workload now also exports bounded maintenance traces from `PersistentSelfStateController + Event Memory + IdleConsolidationLoop`, and the runner records those values into optional Phase 6 maintenance fields beside the physical joule row. Pair reports now emit a resume-append command and exact per-system `record-energy-measurement` command templates. The runner can also ingest a managed meter-reading JSON with direct joules or `average_watts * duration_seconds`, reject pair/replicate mismatches, append both rows from the frozen manifest, and generate a fill-in meter template JSON beside the pair report so the first real session has less transcription friction. The pair report can now also compare the physical SARA maintenance surface against the observed-only internal maintenance benchmark, exposing event-cost and self-state drift in the same artifact instead of requiring manual cross-reading. The readiness session plan now also exposes per-run `run-physical-energy-pair` command templates and managed meter-template paths, so multiple replicates can be executed through the same frozen pair workflow instead of by manual row entry alone. Physical joule capture remains pending.
   - IN PROGRESS: `ann_efficiency_roadmap_gate.py` now carries the observed-only internal maintenance reference surface alongside Phase 6 physical-measurement follow-up and Phase 8 external-reference follow-up, so missing `eval-internal-maintenance-efficiency` evidence becomes a first-class repair action in the same roadmap loop.
   - VERIFIED: pilot execution produced equal quality (`48/48` successful trials for both systems), and replicate 2 dry-run reversed order to `ANN -> SARA`.
   - BLOCKED EXTERNALLY: local `powermetrics` is available but requires an interactive macOS sudo password. Use an authorized powermetrics session or external meter, then provide the measured pair joules to the runner.
3. Publish the Phase 6 measurement protocol and regenerate readiness, ANN-efficiency, research-product, and release evidence.
   - PARTIAL: the pre-session protocol and v2 session plan are generated; physical rows and post-session observations remain pending.
   - IN PROGRESS: the readiness and session-progress summaries can now also surface the observed-only internal maintenance reference, so pre-physical maintenance efficiency and post-physical maintenance rows stay aligned in the same Phase 6 reading surface. The readiness aggregate now also compares physical SARA maintenance-event cost per selected replay against that reference, and the roadmap gate can raise a dedicated maintenance-alignment drift action when physical self-state cost moves too far from the bounded internal benchmark.
4. Strengthen Phase 8 with one real lightweight pretrained embedding baseline, followed by optional FAISS CPU and tiny-Transformer retrieval baselines.
   - IN PROGRESS: `eval-sara-ann-comparison` now reads the same physical-versus-reference maintenance alignment surface used by Phase 6 readiness and exposes missing-alignment or high-drift follow-up directly in the comparison report, so baseline-strength discussion stays tied to persistent-self-state efficiency rather than only task cost.
   - IN PROGRESS: the comparison report now also reads the Event Memory ingest compression surface (`eventization_emission_ratio`, `episode_compression_ratio`, `relation_verification_yield`, self-state continuity) so changes in concept formation or compression can be judged against Phase 6/8 evidence instead of being discussed separately.
5. Expand Phase 8 onto frozen independently sourced noisy, adversarial, negative-query, contrastive, and delayed-recall tasks.
6. Audit Phase 7 train/evaluation isolation by source hash, revision, domain, time split, and near-duplicate signature before generating more autonomous material.
   - IN PROGRESS: Event Memory compression quality now propagates into the research benchmark manifest and operational runbook generation via the comparison report, so weak episode compression or relation-verification yield can surface as managed repair work instead of remaining an isolated evaluator detail.
7. Keep completed Phase 13-18 mechanisms stable; do not prioritize further architecture expansion over Phase 6, Phase 8, and Phase 7 evidence.
   - IN PROGRESS: `event_memory_maintenance_coupling_benchmark.py` now compares bounded Event Memory compression profiles (`tight`, `balanced`, `wide`) against persistent self-state continuity and maintenance-load proxy, so concept compression changes can be screened for hidden self-state cost before they are treated as accuracy or efficiency wins.
8. Keep Phase 19 inactive until its activation gate is satisfied; use it only to address an observed temporal-accuracy limitation without erasing SNN efficiency.
9. Keep Phase 20 inactive until a frozen independent language benchmark and cost ceilings exist; begin with echo-only fixed-timescale controls before phase binding, phonological recoding, or adaptive time constants.

## Completed Work Reference

Do not re-add completed implementation history to this roadmap. Add completed items to [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md), and keep this file focused on what should happen next.
