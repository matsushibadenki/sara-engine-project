# Implemented Features

This document is the canonical list of completed SARA Engine features as of v1.1. It replaces the long completed-history section that previously lived in `doc/ROADMAP.md`.

## Release State

- Target version: `1.1.0`
- v1.1 release gate: `15/15` checks passing
- Research product completion gate: `12/12` checks passing
- Full test suite: `904` tests passing in the Python 3.10 project environment
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
- The benchmark compares SARA sparse retrieval against ANN-style dense-scan and dense-embedding proxy baselines.
- Real-data QA, summary keyword coverage, continual memory, negative controls, partial-evidence abstention, contrastive near-miss behavior, metabolic sparse routing, and sparse diffusion block behavior are checked.
- Small/medium/large external-validity ladder is implemented.
- External-validity reports include thresholds, check details, benchmark-context fingerprints, and managed history tracking.

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

- Energy measurement readiness report is implemented.
- Measurement rows are validated from `data/raw/energy_measurements.jsonl`.
- Rows support either direct joules or derived joules from average watts and duration.
- Real-joule claims require paired SARA and ANN rows for the same task.
- Pending paired measurement commands are written into a measurement plan and standalone measurement session plan.
- Current state: measurement protocol is complete; physical paired measurements are pending.

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
  - Sparse diffusion block readiness
  - Neuromorphic HAL smoke behavior
  - Managed output boundary
  - Memory repair operations
- It currently reports `12/12` passing checks with `completion_score = 1.0`.

## Documentation And Release Artifacts

- Active documentation hub is present.
- Policy, tools, training manual, release checklist, release notes, architecture review, competitive analysis, implemented feature list, and current roadmap are maintained as active docs.
- Long completed roadmap history has been archived under `doc/old/`.
- Exploratory research notes remain under `doc/idea/` or `doc/old/`.

## Known Non-Blocking Backlog

These are not v1.1 blockers, but remain important research/product work:

- Physical paired SARA/ANN joule measurements on target hardware.
- Larger real-data continual-learning experiments.
- Wider external baselines beyond current proxy baselines.
- Native event-camera dataset integration and augmentation.
- Hardware backend adapters beyond current mock/HAL smoke behavior.
- Interactive observability dashboard for sparse events, memory, routing, and energy traces.
- Stronger user-facing documentation and examples for third-party researchers.
