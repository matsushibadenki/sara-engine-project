# Implemented Features

This document is the canonical list of completed SARA Engine features as of v1.1. It replaces the long completed-history section that previously lived in `doc/ROADMAP.md`.

## Release State

- Target version: `1.1.0`
- v1.1 release gate: `15/15` checks passing
- Research product completion gate: `14/14` checks passing
- Full test suite: `1008` tests passing in the Python 3.10 project environment
- Current release posture: ready for v1.1 release, with physical joule measurements intentionally left as the next evidence loop

## Core Policy And Runtime Constraints

- CPU-first runtime direction is established.
- SNN-based sparse event processing is the primary implementation style.
- Runtime learning does not depend on backpropagation.
- Dense matrix computation is not the primary runtime path.
- GPU execution is not required for correctness or normal operation.
- Generated artifacts are constrained to managed paths under `data/`, `workspace/`, and `models/`.
- Code comments and logs are expected to remain in English.

## Foundation And Runtime

- Rust/PyO3-backed event-driven SNN core is integrated.
- Python package surface is import-hardened with lazy exports to reduce optional dependency failures.
- Core SNN model families, spiking language components, local learning modules, and memory utilities are present under `src/sara_engine/`.
- Shared sparse-event and bounded-state design is used across release gates and research evaluators.
- TurboQuant-style compact checkpoint utilities are integrated into SNN memory and model save/load paths.

## CLI And Data Operations

- Unified CLI: `scripts/sara_cli.py`.
- Corpus database import, status, preview, export, activation, deactivation, and reset flows are implemented.
- Managed data export writes to policy-approved data paths.
- Self-organized SNN training is available through `train-self-org`.
- Staged real-data curriculum is available through `train-curriculum --stage small|medium|large`.
- Legacy distilled chat and training paths remain available for compatibility, but are not the preferred production path.
- Memory inspection, upgrade, pruning, replay-data generation, direct-memory repair, cleanup, and DB maintenance commands are implemented.
- Real-energy measurement rows can be appended through `record-energy-measurement`.

## Memory And Continual Learning

- Direct memory serialization and restoration are unified across active paths.
- Unsafe active-path `eval()` loading has been removed in favor of safe parsers and shared helpers.
- Direct memory, hippocampal memory, long-term memory, structural memory, and replay/consolidation flows are connected to evaluation gates.
- Memory health diagnostics expose conversational readiness, session memory, diagnostic hits, predictive state, and future-state runtime state.
- Direct-memory repair can write managed repair reports.
- Stage D continual consolidation minimums are implemented and release-gated.
- Nested-memory readiness is tracked as an observed-only memory scheduling signal.

## Dialogue, Inference, And Agent Runtime

- `SaraInference`, `SpikingLLM`, and `SaraAgent` paths have lightweight benchmark coverage.
- Fast intent handling, practical fallback behavior, session memory, and next-step assistance are implemented.
- Retrieval hygiene metrics include off-topic suppression, multi-turn consistency, retrieval grounding, and retrieval stability.
- Direction-shift following and predictive command stability are tracked in Phase 3 and release summaries.
- Safe arithmetic parsing replaced unsafe calculator evaluation.
- Agent diagnostics are bounded and exposed through CLI and release summaries.

## World Model And Cognitive Runtime

- Stage B world-model minimums are implemented and gated.
- Future-state prediction, command hints, runtime tracking, shift tracking, reward/policy checks, and energy-aware action preference checks are implemented.
- Sparse 2D room-geometry fixtures support top-down room hypotheses, occlusion reasoning, topology checks, counterfactual alternatives, route planning, affordance selection, route execution, and invalid-action rollback.
- Stage E modular cognitive runtime minimums are implemented and gated.
- Common spike space, temporal compression, dendritic context gating, reverse reasoning traces, causal candidate traces, orchestration, counterfactual lanes, action trace observability, and runtime trace replay are checked.

## Phase 3 Quality Gates

- Phase 3 accuracy suite is implemented.
- Phase 3 completion gate validates completion score and failed check entries, not only top-level pass flags.
- Accuracy, retrieval, adaptation, predictive, efficiency, few-shot, continual, and dialogue signals are aggregated into managed reports.
- Trend and delta tracking exist for release-relevant quality metrics.

## Phase 4 Scale And Operational Cycle

- Phase 4 scale/continual benchmark is implemented.
- Phase 4 completion gate validates structural plasticity stability, hippocampal transfer, scale-out retention, continual drift recovery, latency-sensitive quality metrics, and threshold results.
- Phase 4 operational cycle runner is implemented.
- Scheduled/manual GitHub Actions workflow support exists for Phase 4 operational-cycle validation.
- CI dependency files are separated for release and Phase 4 operational workflows.

## Phase 5 Completion Surface

- Phase 5 predictive-coding benchmark is implemented.
- Phase 5 entry gate validates latent transition alignment, prediction-error observability, correction-event coverage, anti-collapse diversity, counterfactual separation, multi-step chains, horizon stability, macro actions, subgoals, depth routing, and micro-ES refinement.
- Phase 5 completion gate validates Phase 4, Phase 5 entry, macro/subgoal, micro-ES, and sparse diffusion block readiness signals.
- Phase 5 completion currently reports `55/55` passing checks.

## Sparse Diffusion Block Integration

- SARA-compatible sparse diffusion block readiness is implemented.
- Checks cover equal-mass uncertainty partitioning, independent sparse-event blocks, local denoising, event-cost advantage, block-count ablation, recurrent single-pass compression, and policy compatibility.
- Sparse diffusion block evidence is propagated into Phase 5 completion, real-data external validity, scale ladder, ANN-efficiency roadmap, and research-product completion gates.

## Real-Data External Validity

- Real-data external-validity benchmark is implemented.
- The benchmark compares SARA sparse retrieval against ANN-style dense-scan, dense-embedding proxy, offline BM25-style lexical baselines, and an optional local pretrained embedding reference when a managed local model path is available.
- Real-data QA, summary keyword coverage, continual memory, negative controls, partial-evidence abstention, contrastive near-miss behavior, noisy/adversarial/delayed repository fixtures, metabolic sparse routing, and sparse diffusion block behavior are checked.
- Small/medium/large external-validity ladder is implemented.
- External-validity reports include thresholds, check details, benchmark-context fingerprints, per-task quality/cost/abstention/failure summaries, repository fixture probe metrics, and managed history tracking.

## Research Benchmark Package

- Compact research benchmark suite is implemented through `python scripts/sara_cli.py eval-research-benchmark-suite`.
- Benchmark protocol is documented in `doc/BENCHMARK_PROTOCOL.md`.
- Repository-safe benchmark fixtures live under `data/processed/benchmark_fixtures/`.
- The suite writes a managed manifest under `workspace/evaluation/research_benchmark_manifest.json`.
- The manifest records proven and not-proven claims, including the current Rust built-extension limitation when the optional PyO3 module is not installed.
- The suite now surfaces both gap-closure evidence and Phase 7 gap-loop readiness evidence, so deterministic supplement quality and managed queue handoff are visible as separate research signals.
- The research-product completion gate now also checks Phase 7 autobot gap-loop readiness, so top-level completion evidence fails when managed autonomous gap repair stops converting requests into queueable repair or replay curriculum.

## Autobot Dataset Preparation

- Source-aware autobot dataset builder is implemented through `python scripts/sara_cli.py build-autobot-dataset`.
- The autobot dataset builder now consumes fixture-driven material request plans, merges their evaluation gaps into curriculum prioritization, and writes managed collection-target manifests so missing transcript, counterexample, repair-note, and revision-note evidence can flow directly into the next collection pass.
- A deterministic gap-materials builder now turns those collection targets into source-backed `transcript_segment`, `counterexample`, `repair_note`, and `revision_note` supplements under managed `data/processed/autobot` outputs, so own-latent manifest generation can consume concrete follow-up evidence instead of only abstract requests.
- Gap materials now produce their own managed curriculum manifest, with `counterexample` / `repair_note` / `revision_note` prioritized as repair-stage items and `transcript_segment` routed into replay when retrieval grounding is the active gap, so the training queue and own-latent manifest builder can consume the same follow-up evidence without a separate manual merge step.
- A managed gap-curriculum enqueue runner now pushes those repair/replay gap manifests straight into `workspace/autobot/train_queue.json`, preserving the originating curriculum path on each queue item so the next training cycle can trace every injected follow-up example back to its gap-material source.
- A gap-materials closed-loop benchmark now compares accepted-only versus accepted-plus-gap-material latent coverage, so the project can quantify whether deterministic supplements actually shrink fixture-driven coverage gaps instead of only assuming they help.
- The builder reads accepted records from `data/processed/autobot/multimodal_records.jsonl`.
- Deterministic extraction creates summaries, QA pairs, source claims, definition cards, negative queries, procedural steps, and contrastive pairs.
- Learning-material gating rejects short, duplicate, unsupported, secret-like, and PII-like samples before processed outputs are written.
- Accepted materials are written under `data/processed/autobot/`; rejected and candidate materials remain under `data/interim/autobot/`.
- Curriculum manifests assign `easy`, `medium`, `hard`, `repair`, and `replay` stages with gap-aware priorities.
- Planner feedback can convert weak summary, negative-control, contrastive-control, retrieval-grounding, and language-balance signals into machine-readable material requests under `workspace/autobot/`.
- Source-aware collector plugins include opt-in official documentation URLs and arXiv metadata/abstract queries; both remain optional network plugins and are skipped in offline mode.

## Rust Sparse Runtime Hardening

- Rust sparse-runtime source readiness is checked through `scripts/eval/rust_core_readiness.py`.
- The optional Rust Python extension has been rebuilt with maturin in the project Python 3.10 environment, and readiness now reports `ready`.
- Rust unit tests cover sparse overlap, propagation, WTA routing, LIF behavior, causal synapses, scalable SDR memory, reward-modulated STDP, direct synapse construction, batch token-to-SDR conversion, and homeostasis.
- Python fallback behavior remains explicit for environments where the optional Rust extension is not built.
- Rust sparse-runtime benchmark reporting is implemented under `workspace/evaluation/rust_core_benchmark.json`.

## Sparse Own-Latent Learning

- Observed-only sparse own-latent learning benchmark is implemented through `python scripts/sara_cli.py eval-own-latent-learning`.
- `SparseOwnLatentPredictor` uses deterministic sparse signatures and local co-occurrence updates without runtime backpropagation, dense embedding matrices, or GPU requirements.
- A deterministic RHM-style fixture generator writes `data/processed/benchmark_fixtures/own_latent_rhm_cases.jsonl`.
- Source-backed autobot materials can be converted into `data/processed/autobot/latent_manifest.jsonl` through `python scripts/sara_cli.py build-own-latent-manifest`.
- The latent manifest preserves material hash, source URL/path, language, quality score, license hint, compliance level, sparse signature, and observed-only status.
- The benchmark compares sparse own-latent prediction against a token-overlap reference at multiple train sizes.
- Reports are written under `workspace/evaluation/own_latent_learning_benchmark.json` and remain observed-only evidence in the research benchmark suite.

## Sparse Dendritic Feedback Gate

- Observed-only sparse dendritic feedback gate benchmark is implemented through `python scripts/sara_cli.py eval-dendritic-feedback-gate`.
- `SparseDendriticFeedbackGate` uses bounded local potentials, recent-output feedback, sparse neighbor activity, local co-occurrence updates, and homeostatic clipping.
- The benchmark compares baseline sparse gating with dendritic-gated sparse routing on noisy, adversarial, contrastive, and conflicting-material cases.
- Reports are written under `workspace/evaluation/dendritic_feedback_gate_benchmark.json` and include robustness delta, event cost, state budget, fallback rate, convergence steps, and trace samples.
- Default production inference remains unchanged; the gate is observed-only evidence.

## Sparse Verifiable Planning Trace

- Observed-only sparse plan-trace verification is implemented through `python scripts/sara_cli.py eval-sparse-plan-trace-verifier`.
- `verify_sparse_plan_trace` checks STRIPS-like sparse `state -> action -> next_state` traces without LLM chain-of-thought, dense training, or runtime backpropagation.
- The verifier detects missing preconditions, wrong effects, missing frame persistence, invariant violations, unknown actions, empty plans, and unmet goals.
- Invalid traces emit managed repair materials under `data/processed/autobot/plan_trace_repair_materials.jsonl`.
- Reports are written under `workspace/evaluation/sparse_plan_trace_verifier.json` and include event cost, state budget, invalid-step count, repair-material count, and fallback behavior.

## Sparse Reasoning Prior

- Observed-only future-state reasoning-prior evaluation is implemented through `python scripts/sara_cli.py eval-sparse-reasoning-prior`.
- Source-backed sparse evidence is converted into bounded direction, magnitude, route, confidence, uncertainty, and abstention controls.
- `logic_to_state_consistency` checks expected direction, magnitude, and abstention deterministically without an LLM judge.
- Sudden-shift cases require relevant external-event evidence; missing context produces `request_external_context` rather than an unsupported forecast.
- Reports expose event relevance, source-backed integrity, event cost, state budget, sparse signatures, and per-evidence trace contributions.

## Verified Sparse Resonance Credit

- `SparseResonanceCreditAssigner` provides a SARA-specific coordination layer over existing local eligibility and plasticity mechanisms.
- Durable local updates require resonance among spike coincidence, prediction consistency, verifier confidence, cross-modal agreement, reward, and novelty channels.
- Contradiction, abstention, missing source backing, insufficient resonance, and metabolic pressure freeze updates with explicit reasons.
- Signed positive and negative credit remains sparse, link-bounded, weight-clipped, CPU-first, and backpropagation-free.
- The observed-only benchmark demonstrates suppression of harmful reward-only updates without changing production learning.
- `resonance_evidence.py` converts managed reasoning, planning, multimodal, dendritic, own-latent, and metabolic reports into auditable resonance channels.
- The integration benchmark reinforces only a complete healthy report bundle and isolates verifier contradiction, missing-source, abstention-regression, and metabolic-pressure failures.

## Sparse Synesthetic Multimodal Binding

- Observed-only equal-modality binding is implemented through `python scripts/sara_cli.py eval-synesthetic-multimodal-binding`.
- `SparseTemporalBinder` maps language, vision, audio, and tactile sparse signatures into deterministic bounded time chunks.
- Language, vision, audio, and tactile adapters convert modality-specific feature names into the same sparse event IR and record specialization factors.
- `SparsePluggableCorticalColumn` applies the same local Hebbian update and homeostatic clipping rule to every modality.
- `SparseSynestheticLinker` learns capped direct cross-modal routes, including non-language audio-to-tactile prediction, and abstains on unsupported signatures.
- `SparseThalamicGate` records equal and focused route decisions without dense softmax or runtime backpropagation.
- The focused benchmark compares 25/32/40 ms binding profiles, optionally incorporates source-backed own-latent signatures, and accepts bounded dendritic route hints.
- Neuromorphic capability reports expose dendritic feedback and equal-modality thalamic routing as optional state-trace operations.
- Managed reports expose adapter integrity, temporal alignment, plug-swap integrity, cross-modal precision, abstention, route traceability, event cost, and state budget.

## Verified Hierarchical Event-State Caching

- `VerifiedHierarchicalEventStateCache` stores source-backed observed sparse states without dense recurrent vectors, runtime backpropagation, or GPU requirements.
- The active design split is explicit: durable experience belongs in verified event memory, while transient recurrent-style processing remains bounded and separate.
- Admission requires deterministic verification, sufficient resonance, and metabolic headroom; contradiction, prediction-only, abstention, missing source backing, and failed verification are blocked explicitly.
- Fixed, linear, and logarithmic retention profiles support bounded comparison with deterministic merging, eviction, expiry, and lifecycle traces.
- Retrieval combines sparse overlap, own-latent agreement, causal context, temporal relevance, source agreement, confidence, and reliability with hard top-k and abstention limits.
- `python scripts/sara_cli.py eval-event-state-cache` writes managed fixture, candidate, manifest, trace, state, report, and summary artifacts.
- The first observed-only benchmark records delayed recall `1.0`, negative-query abstention `1.0`, blocked-decision integrity `1.0`, logarithmic state count `6` versus linear `12`, and maximum retrieval event cost `17`.
- `event_state_evidence.py` converts managed Phase 17 evidence and source-aware own-latent manifest rows into auditable cache-promotion candidates.
- Source revisions preserve material hashes, while retrieval emits bounded read-only reactivation hints rather than mutating durable state.
- Strict state restoration rejects unsupported schemas, malformed entries, budget overflow, duplicate IDs, invalid tiers, and unverified durable records.
- The source-aware integration benchmark records logarithmic delayed recall `1.0` versus fixed-window `0.0`, round-trip integrity `1.0`, corruption rejection `1.0`, source revision integrity `1.0`, and maximum retrieval event cost `38`.

## Typed Ingest Proposal Boundary

- `src/sara_engine/ingest/` now provides typed ANN-free and ANN-assisted ingestion records rather than mixing observations, proposals, verified relations, and concept candidates.
- `ObservedEvent`, `CandidateEvent`, `CandidateRelation`, `VerifiedRelation`, and `ConceptCrystalCandidate` are distinct schema-bearing record types with explicit lineage.
- `ProposalVerifier` enforces bounded promotion rules so candidate labels or relation hints do not directly enter durable Event Memory without evidence, prediction gain, and counterexample checks.
- ANN-assisted metadata remains proposal-scoped through lineage records and does not become runtime truth by default.
- Deterministic bootstrap ingest primitives now include scalar change detection, temporal eventization, near-time cross-modal synchrony detection, and bounded prediction-gain relation proposal before Event Memory admission.
- Cross-context relation stability scoring is implemented so repeated relations can be ranked by prediction-gain persistence across episodes and sources before they are proposed as concept-crystal candidates.
- Concept crystallization now has a deterministic guard that quarantines candidates under source-revision conflicts, weak source diversity, or excessive counterexample pressure before durable promotion.
- Audited concept candidates can now be bridged into Phase 18 `EventStateCandidate` entries, while quarantined candidates are preserved as explicit revalidation-queue records instead of being silently dropped.
- Revalidation scheduling is implemented for quarantined concepts, with deterministic cooldowns, attempt budgets, recovery checks for source diversity and revision conflicts, and explicit retry priorities before reassessment.
- A deterministic concept review loop now reconnects the ready revalidation queue to relation restabilization, concept rebuilding, guarded admission, and carry-forward revalidation when evidence is still insufficient.
- The concept revalidation queue can now be persisted under managed `workspace/` paths, reloaded across review cycles, and written with compact review reports so the Phase 18 loop can continue beyond a single in-memory process.
- The event-state cache integration benchmark now exercises one deterministic persisted concept-revalidation cycle and reports whether quarantined concept candidates can recover, admit, and drain their managed queue under source-aware evidence.
- A source-aware concept revalidation fixture builder now generates mixed recoverable and blocked cases under managed benchmark-fixture paths so harder follow-up coverage can be created directly from latent manifest rows.
- Fixture-builder reports now emit deterministic expansion priorities so blocked-heavy case types can directly drive the next Event Memory follow-up wave in the research benchmark summary.
- Fixture-builder reports now also summarize available manifest material types and missing preferred material types so follow-up collection can target the right source-aware evidence mix instead of only increasing raw case counts.
- The own-latent manifest builder now consumes fixture-feedback expansion plans and reports live material coverage gaps, so latent-manifest generation can be steered by missing transcript, counterexample, repair-note, or revision-note evidence before the next Event Memory evaluation pass.
- Fixture-feedback gaps are now converted into managed `workspace/autobot` material request plans, so dataset collection and curriculum prioritization can consume the same source-diversity, counterexample, repair-support, and revision-conflict requests that surfaced in Event Memory evaluation.
- The autobot gap loop is now runnable as one managed end-to-end path through `python scripts/sara_cli.py run-autobot-gap-loop`, which builds source-backed materials, emits collection targets, synthesizes deterministic gap materials, and enqueues the resulting repair curriculum without leaving managed directories.
- `python scripts/sara_cli.py eval-autobot-gap-loop-readiness` now converts the latest managed gap-loop artifacts into an auditable Phase 7 readiness report, including requested-slot coverage, enqueue coverage, skip ratio, and repair/replay curriculum share.

## Neuromorphic Capability Matrix

- Chip-neutral sparse event IR is summarized as a neuromorphic capability matrix.
- `python scripts/sara_cli.py eval-neuromorphic-capability-matrix` writes managed event-budget, routing-hint, state-budget, adapter-policy, and unsupported-operation evidence.
- Lava, SpiNNaker, and Akida-style profiles remain optional profile checks; no accelerator is required for correctness or release evidence.

## Optional Local LLM Operator Assistant

- Optional local LLM operator-assistant readiness is implemented through `python scripts/sara_cli.py eval-operator-llm-assistant-readiness`.
- The assistant path is disabled by default and no LLM runtime is required for readiness.
- `src/sara_engine/operator/llm_proposal_schema.py` validates strict proposal JSON for allowed proposal types, safe action types, source references, managed output paths, and secret-like text.
- Direct file, data, model, release, and git mutation actions are rejected; accepted proposals are not executed by the readiness check.
- Reports are written under `workspace/evaluation/operator_llm_assistant_readiness.json` and include acceptance rate, rejection counts, zero-runtime token/latency fields, and fallback behavior.

## ANN-Efficiency Roadmap Gate

- ANN-efficiency roadmap gate is implemented with six stages:
  - Sparse proxy instrumentation
  - Limited real-data advantage
  - Scale-ladder advantage
  - Strict operational regression guard
  - Neuromorphic transfer readiness
  - Real joule measurement readiness
- The gate currently reports `6/6` stages passing.
- The roadmap gate separates proxy evidence from physical energy claims.
- Next physical-measurement actions are generated as managed `next_evidence_actions`.

## Energy Measurement Readiness

- Energy measurement readiness schema v2 is implemented.
- Measurement rows are validated from `data/raw/energy_measurements.jsonl`.
- Rows support either direct joules or derived joules from average watts and duration.
- Real-joule claims require matched SARA and ANN rows for the same pair and replicate.
- Pair validation requires matching environment fingerprint, task fixture hash, success criterion, measurement boundary, tool, CPU, thread count, affinity, power mode, warm-up count, measured repetitions, and trial count.
- Quality parity is checked before an energy advantage is credited.
- Per-task evidence reports median `joule_per_success`, median absolute deviation, invalid-pair reasons, and run-order balance.
- The default protocol requires at least three paired replicates per task with alternating or randomized-block run order.
- Pending paired measurement commands are written into a measurement plan and standalone measurement session plan.
- `doc/ENERGY_MEASUREMENT_PROTOCOL.md` documents the fixed conditions and laboratory workflow.
- `run-physical-energy-pair` freezes the corpus/task hash, exact-match criterion, Apple CPU identity, thread environment, warm-up and repetition counts, and alternating run order before launching separate SARA and BM25 workload processes.
- The pilot runner execution preserved quality at `48/48` trials for both systems and writes unmeasured candidates only under `workspace/evaluation/`.
- Current state: fairness protocol and validation are complete; physical paired measurements are pending.

## Operational Readiness And Release Gates

- Release soak supports quick, release, and extended profiles.
- Release gate validates soak, Phase 3 readiness, Phase 5 completion, and external-validity evidence.
- Operational readiness supports strict-production checks, artifact refresh, runbooks, action manifests, repair plans, retry queues, failure focus, and runbook action hygiene.
- Operational readiness imports ANN-efficiency next evidence actions into the operator manifest.
- v1.1 release gate validates operational readiness, Phase 3, Stage B/D/E, Phase 4, Phase 5, external validity, research-product completion, and version alignment.
- v1.1 release gate currently reports `15/15` passing checks.

## Research-Product Completion Gate

- Research-product completion gate is implemented as the top-level completion surface for the research product.
- It validates:
  - Policy core constraints
  - Roadmap closure audit
  - Phase 3 completion
  - Phase 4 completion
  - Phase 5 completion
  - Strict operational readiness
  - ANN-efficiency roadmap
  - Energy measurement session plan
  - Research fixture readiness
  - Rust core readiness
  - Sparse diffusion block readiness
  - Neuromorphic HAL smoke behavior
  - Managed output boundary
  - Memory repair operations
- It currently reports `14/14` passing checks with `completion_score = 1.0`.

## Documentation And Release Artifacts

- Active documentation hub is present.
- Policy, tools, training manual, release checklist, release notes, architecture review, competitive analysis, implemented feature list, and current roadmap are maintained as active docs.
- Long completed roadmap history has been archived under `doc/old/`.
- Exploratory research notes remain under `doc/idea/` or `doc/old/`; the active Semantic Echo Field design reference is `doc/idea/Semantic_Echo_Field_Sparse_Temporal_Language_Architecture_v2.md`, while historical or superseded idea assets should be archived under `doc/idea/old/`.

## Known Non-Blocking Backlog

These are not v1.1 blockers, but remain important research/product work:

- Physical paired SARA/ANN joule measurements on target hardware.
- Larger real-data continual-learning experiments.
- Wider external baselines beyond current proxy baselines.
- Native event-camera dataset integration and augmentation.
- Hardware backend adapters beyond current mock/HAL smoke behavior.
- Interactive observability dashboard for sparse events, memory, routing, and energy traces.
- Stronger user-facing documentation and examples for third-party researchers.
