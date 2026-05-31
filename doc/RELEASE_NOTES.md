# Release Notes

## Current v1.1 Release Candidate

This release candidate focuses on research-product hardening for the CPU-first SNN stack, ANN-efficiency roadmap evidence, and final v1.1 gate readiness.

### Highlights

- Added the ANN-efficiency roadmap gate, with staged checks for sparse proxy instrumentation, limited real-data advantage, scale-ladder advantage, strict operational regression blocking, neuromorphic transfer readiness, and real joule-measurement readiness.
- Added real-data external-validity negative controls for absent queries, partial-evidence abstention, contrastive near-miss retrieval, and an offline dense embedding ANN-style baseline without making dense vectors part of the runtime path.
- Promoted sparse diffusion block checks into real-data external validity, the small/medium/large scale ladder, and ANN-efficiency roadmap Stage 2/3 so denoising, event-cost advantage, partition integrity, and single-pass integrity must hold on real-corpus tasks.
- Added real-energy measurement readiness, including validated SARA/ANN `joule_per_success` rows and support for deriving joules from `average_watts * duration_seconds`.
- Hardened real-joule acceptance so measured tasks must be paired across SARA and ANN, with the minimum per-task ANN/SARA joule-per-success ratio enforced by the roadmap gate.
- Added an energy `measurement_plan` that lists missing SARA/ANN task pairs, weak paired ratios, and command templates for the next managed measurement rows.
- Added an energy `measurement_session_plan` that expands pending/weak pairs into stable run-id templates and `real_energy_session` command templates for the next paired measurement session.
- Added standalone energy measurement session plan artifacts under `workspace/evaluation/` so lab operators can review the next paired-measurement commands without opening the full readiness report.
- Propagated energy measurement session plan items into ANN-efficiency roadmap `next_evidence_actions` so the top-level roadmap summary keeps the real-joule evidence loop visible.
- Propagated ANN-efficiency `next_evidence_actions` into the operational runbook action manifest as `ann_efficiency_next_evidence`, so real-joule measurement follow-up is visible in the release operator queue.
- Added sparse diffusion block readiness, a SARA-compatible adaptation of DiffusionBlocks-style decomposition that validates equal-mass uncertainty partitioning, independent local denoising blocks, block-count ablation, and recurrent single-pass compression without runtime backpropagation.
- Added a research-product completion gate that combines policy constraints, roadmap closure, Phase 3/4/5 completion, strict operational readiness, neuromorphic HAL smoke coverage, managed-output boundaries, memory repair operations, sparse diffusion block readiness, the energy measurement session plan, and the ANN-efficiency roadmap.
- Added the research-product completion report to the final v1.1 release gate, so release promotion now requires the ANN-efficiency and measurement-session checks to pass.
- Raised the final release gate target to v1.1.0 so matching Python/Rust package versions below the target can no longer pass promotion.
- Unified direct-memory serialization and restoration across `SpikingLLM`, `SaraInference`, training scripts, evaluation scripts, and maintenance utilities.
- Removed unsafe `eval()` usage from the active loading path and replaced it with safe parsers and shared helpers.
- Added lightweight runtime diagnostics for `SaraAgent`, including session persistence of recent issues and optional CLI diagnostics display.
- Replaced the chat calculator `eval()` path with a safe arithmetic parser that supports only basic operations.
- Added stronger FORCE artifact validation to fail fast on malformed shapes and mismatched metadata.
- Fixed the `UnifiedSNNModel` FORCE readout path so one logical step updates reservoir state only once.
- Added TurboQuant-style shared quantization utilities and integrated them into `SaraInference`, `SpikingTransformerModel`, and `SpikingLLM` save/load paths.
- Extended `train_snn_lm.py` with optional TurboQuant-style checkpoint compression flags for managed model outputs.
- Added retrieval-hygiene instrumentation for `SaraAgent`, including `off_topic_suppression`, `multi_turn_consistency`, and `retrieval_stability`.
- Added context-aware retrieval stabilization across `SaraAgent`, `SparseMemoryStore` / hippocampus, and `SaraInference` direct memory so noise recall is reduced at multiple layers.
- Unified retrieval diagnostics formatting and storage across `SaraAgent` and `SaraInference` through a shared utility module.

### Reliability Work Included

- Added regression tests for direct-memory round-trips, inference memory compatibility, agent session recovery, CLI dispatch, and calculator safety.
- Added lightweight soak tests for repeated `SaraAgent` dialogue turns and repeated `SaraInference` memory updates.
- Added explicit release soak acceptance thresholds so the release gate now checks minimum workload, bounded diagnostics, and memory round-trip integrity together.
- Added a lightweight `SaraAgent` dialogue benchmark report under `workspace/evaluation/` so Phase 3 quality work can be tracked numerically.
- Added lightweight Phase 3 accuracy benchmarks for `SaraInference` and `SpikingLLM`, plus an aggregated `phase3_accuracy_suite.py` report under `workspace/evaluation/`.
- Added Phase 3 accuracy history tracking and optional embedding of the latest accuracy summary into `release_soak.py` reports so shipping checks can review reliability and lightweight quality gates together.
- Added `retrieval_hygiene` focus summaries and trend/delta reporting to Phase 3 and release summaries so recall quality changes can be tracked across runs.
- Expanded the release gate so `stage_b_readiness` is now a required shipping check, including explicit world-model prototype minimums for predicted transitions, command hints, predictor snapshots, and runtime/shift tracking.
- Added a lightweight stochastic-computing prototype path for low-precision score aggregation, including opt-in edge runtime support and `stochastic_readout_integrity` observability in efficiency benchmarks and release summaries.
- Added opt-in low-precision persistence in the edge exporter so readout weights can be serialized in quantized form while keeping `SaraEdgeRuntime` compatibility.
- Marked `scripts/old/` as legacy and documented that it is not the recommended production path.
- Expanded CLI end-to-end coverage for `sara-chat` and `sara-train`, including managed-output validation and training-runtime failure handling.
- Embedded checklist-oriented release review state into `release_soak.py` summaries so managed output paths, release notes review state, and final review readiness can be audited from one report.
- Added runtime-backed Stage B operator tracing for `SaraInference`, including `transition_operator`, speculative draft/verify acceptance tracking, rollback observability, and counterfactual branch viability ratios in release summaries and memory health reports.
- Added a fluid-inspired supplementary dynamics layer for predictive support tracing, including bounded scalar field propagation and `fluid_trace` observability in `SaraInference`, Phase 3 predictive benchmarks, and release summaries.
- Added operational readiness runbooks and action manifests so `failure_focus`, iterative repair actions, retry queues, and fallback actions can be reviewed and queued from managed artifacts.
- Added `phase4_operational_cycle.py` and a scheduled GitHub Actions workflow for release/extended operational-cycle validation.
- Strengthened Phase 3 completion validation so `completion_score` and failed `checks` entries are verified, not only the top-level pass flag.
- Added Phase 4 `quality_metrics` for structural stability, hippocampal transfer retention, scale-out retention/latency, and continual drift recovery, with matching completion-gate validation.

### Operational Notes

- The project continues to prioritize SNN-friendly efficiency: no backpropagation, no required GPU path, and no matrix-heavy runtime dependency for the newly added reliability features.
- ANN-comparison claims remain proxy-only until paired SARA/ANN real joule measurements are recorded; v1.1 ships the ingestion and claim-guard path, not fabricated physical measurements.
- Completed roadmap history is now consolidated in `doc/IMPLEMENTED_FEATURES.md`, while `doc/ROADMAP.md` is reserved for post-v1.1 direction.
- New diagnostics are intentionally lightweight and bounded to avoid turning observability into a hidden energy cost.
- Retrieval diagnostics now use a shared schema, making it easier to compare `SaraAgent` and `SaraInference` behavior during Phase 3 / Phase 4 tuning.
- Release soak now supports `quick`, `release`, and `extended` profiles so longer CPU-only shipping checks can use a fixed acceptance baseline.
- `extended` remains the required profile for final shipping decisions, while `release` remains useful for routine pre-release validation.
- Shipping review now expects Stage B predictive summaries to show both logical readiness and operator-level agreement, so reviewers can detect silent divergence between planned and verified transition traces without inspecting raw JSON.
- Phase 4 completion now expects the benchmark report to include both binary required metrics and numeric quality metrics. Older thin reports should be regenerated with `python scripts/eval/phase4_scale_continual_benchmark.py`.
- Operational readiness now writes a Markdown runbook and JSON action manifest under `workspace/release/`; review these before using repair-log automation.

### Known Gaps Before Full Production Release

- The `extended` soak profile should still be run on the actual target environment before a public production release.
- Real joule measurements should be collected on the target hardware before making physical energy-efficiency claims beyond the proxy and protocol-ready evidence.
- End-to-end CLI scenario coverage is improved, but command families outside the primary `sara-chat` / `sara-train` path still deserve broader coverage.
- Legacy scripts remain available for reference, but production usage should prefer `src/sara_engine`, `scripts/train`, `scripts/eval`, and `scripts/sara_cli.py`.
- Phase 5 research work should continue to preserve the Phase 1-4 completion gates before adding new acceptance criteria.
- The archived pre-v1.1 roadmap and broad research survey are retained under `doc/old/` for historical reference.
