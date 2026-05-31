# SARA Engine Tools

This document lists the active command surface. Older broad tool descriptions have been moved to `doc/idea/old/legacy_bout-Tools-*.md` as reference material.

Run commands from the repository root unless a script says otherwise.

## Core CLI

- `python scripts/sara_cli.py db-import <file>`: import text or chat JSONL into `data/sara_corpus.db`.
- `python scripts/sara_cli.py db-status --format json`: inspect corpus counts, categories, and review metadata.
- `python scripts/sara_cli.py db-list`: preview candidate training materials.
- `python scripts/sara_cli.py db-export`: export active material to managed training data.
- `python scripts/sara_cli.py db-activate` / `db-deactivate`: manage training-material inclusion.
- `python scripts/sara_cli.py train-self-org`: run the preferred SNN self-organized learning path.
- `python scripts/sara_cli.py train-curriculum --stage <small|medium|large>`: run staged real-data curriculum (export -> training -> gates).
- `python scripts/sara_cli.py train-distill`: legacy conversational-memory path.
- `python scripts/sara_cli.py chat-self-org`: chat with the self-organized model.
- `python scripts/sara_cli.py chat-distill`: chat with the legacy distilled agent state.
- `python scripts/sara_cli.py inspect-memory`: inspect saved SNN memory and retrieval diagnostics.
- `python scripts/sara_cli.py upgrade-memory`: rewrite older memory artifacts into the current managed format.
- `python scripts/sara_cli.py fix-memory`: remove or decay one direct-memory association and write a managed repair report.
- `python scripts/sara_cli.py build-replay-data`: generate replay token JSONL from chat JSONL.
- `python scripts/sara_cli.py eval-external-validity`: compare sparse SARA retrieval with an ANN-style dense-scan proxy on real-data tasks.
  - supports `--history-path`, `--regression-tolerance`, and `--no-history-update` for managed trend checks.
- `python scripts/sara_cli.py prune`: prune low-value memory weights.
- `python scripts/sara_cli.py clean`: clean interim and processed data outputs.

## Evaluation Gates

- `python scripts/eval/phase3_accuracy_suite.py`: aggregate Phase 3 accuracy, readiness, and Stage B-E metrics.
  - `--regression-tolerance` can be used to tune trend-regression sensitivity when repeated runs produce small nondeterministic drift.
  - summary reports both `regression_count` for observed diagnostics and `gate_regression_count` for release-blocking quality regressions.
  - Includes `nested_memory_readiness` as an observed-only focus for Nested Learning-inspired memory scheduling.
- `python scripts/eval/phase3_completion_gate.py`: validate Phase 3 completion from the managed Phase 3 report.
- `python scripts/eval/phase4_scale_continual_benchmark.py`: run Phase 4 scale-out and continual-learning checks.
- `python scripts/eval/phase4_completion_gate.py`: validate Phase 4 completion, including quality metrics.
- `python scripts/eval/phase5_predictive_coding_benchmark.py`: run the Phase 5 Spiking H-JEPA predictive-coding entry benchmark.
- `python scripts/eval/phase5_entry_gate.py`: validate the Phase 5 predictive-coding entry benchmark report.
- `python scripts/eval/phase5_completion_gate.py`: validate Phase 5 completion from Phase 4 benchmark, Phase 5 benchmark, Phase 5 entry gate, and sparse diffusion block readiness artifacts; summary output includes macro/subgoal and micro-ES detail values for release review.
- `python scripts/eval/research_product_completion_gate.py`: validate the research-product completion surface across policy, ROADMAP closure, Phase 3/4/5, strict operational readiness, ANN-efficiency roadmap, energy measurement session plan, memory repair operations, managed output policy, and neuromorphic HAL smoke behavior.
- `python scripts/eval/release_soak.py --profile release --include-accuracy`: routine release soak with embedded accuracy summary.
- `python scripts/eval/release_soak.py --profile extended --include-accuracy`: final shipping-profile soak.
- `python scripts/eval/release_gate.py`: validate release soak, Phase 3 readiness, and Phase 5 completion gate readiness.
  - also validates `workspace/evaluation/phase5_completion_gate_report.json` unless `--skip-phase5-completion-gate` is provided.
  - also validates `workspace/evaluation/real_data_external_validity.json` unless `--skip-external-validity-gate` is provided.
- `python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy --strict-production`: refresh all major artifacts, including Phase 5 entry gate artifacts, and validate operational readiness against them.
  - refresh sequence now includes `phase5_completion_gate.py`, so Phase 5 completion artifact is regenerated and validated in the same strict cycle.
  - operational summary includes Phase 5 completion macro/subgoal and micro-ES detail values when the completion gate artifact is present.
  - refresh sequence also includes `real_data_external_validity.py`, so real-data QA, summary, continual-memory, and ANN-cost advantage evidence is regenerated before release soak.
  - `--phase3-regression-tolerance 0.05` can be used to pass a less-sensitive trend threshold to `phase3_accuracy_suite.py` during refresh.
  - `--v1-actions-path workspace/release/v1_release_gate_actions.json` can be used to merge v1 recovery actions into the operational runbook action manifest.
  - `--v1-actions-max-age-seconds 86400` can be used to accept only fresh v1 recovery actions and skip stale or missing-timestamp entries.
  - `--ann-efficiency-roadmap-report-path workspace/evaluation/ann_efficiency_roadmap_gate.json` can be used to merge ANN-efficiency `next_evidence_actions` into the operational runbook action manifest.
  - `--runbook-max-per-source 1` can be used to cap action concentration from a single source (`iterative_next_action`, `retry_queue`, `fallback_action`, `v1_recovery_action`).
  - `--runbook-max-actions 25` can be used to cap total runbook action count.
  - `--runbook-drop-rate-threshold 0.8` can be used to tune checklist sensitivity for manifest drop pressure.
  - operational summary now includes `runbook_action_total` and `runbook_action_source_count:*` for manifest distribution visibility.
  - operational summary also includes build-time skip counters (`runbook_action_skipped_*`) to show dropped candidates caused by duplicate/source-cap/max-actions filtering.
  - `runbook_action_skipped_source_cap_by_source:*` shows which source bucket was capped most.
  - `runbook_action_skipped_max_actions_by_source:*` shows which source lost candidates due to total manifest cap.
  - `runbook_action_skipped_duplicate_by_source:*` shows where dedup suppression is concentrated.
  - `runbook_action_skipped_empty_command_by_source:*` shows where invalid/empty action candidates are entering.
  - `runbook_action_*_drop_rate` fields provide normalized manifest drop pressure for quick comparison across runs.
  - checklist now includes `runbook_drop_rate_ok` (warns when manifest drop rate is unusually high).
  - when `runbook_drop_rate_ok=false`, runbook action manifest adds a medium-priority recovery command that relaxes runbook caps and re-runs operational readiness.
  - runbook markdown `Execution Manifest` now includes `Considered candidates` and skip reasons so operators can audit candidate drops without opening JSON artifacts.
  - runbook markdown `Execution Manifest` also shows configured `max actions` / `max per source` values used for that run.
  - if stale/missing-timestamp v1 actions are rejected, runbook actions will include `python scripts/eval/v1_release_gate.py` as a high-priority hygiene recovery command.
- `python scripts/eval/phase4_operational_cycle.py --dry-run`: validate planned release/extended operational cycle commands.
- `python scripts/eval/phase4_operational_cycle.py`: run the full release/extended operational cycle.
  - `--runbook-max-actions`, `--runbook-max-per-source`, `--runbook-drop-rate-threshold`, `--v1-actions-max-age-seconds` can be used to keep Phase4 periodic cycle behavior aligned with operational runbook policy.
  - `.github/workflows/phase4-operational-cycle.yml` `workflow_dispatch` exposes the same runbook controls for manual CI runs.
  - CI test dependencies for this workflow are centralized in `requirements-ci-phase4.txt` to avoid import-chain drift between local and GitHub Actions runs.
- `python scripts/eval/v1_release_gate.py`: validate final v1.1 promotion prerequisites.
  - defaults to `--target-version 1.1.0`, so matching Python/Rust package versions below v1.1 cannot pass promotion.
  - v1 gate now requires `workspace/evaluation/phase5_completion_gate_report.json` in addition to Phase 5 entry snapshots.
  - v1 summary includes Phase 5 completion macro/subgoal and micro-ES detail values for final promotion review.
  - v1/release gates require sparse diffusion block completion checks through the Phase 5 completion artifact.
  - v1 gate also requires `workspace/evaluation/research_product_completion_gate_report.json`, including the energy measurement session plan check.
  - v1 gate also validates `workspace/evaluation/real_data_external_validity.json` so external validity and ANN-cost advantage regressions block promotion.
  - Also writes `workspace/release/v1_release_gate_actions.json` (priority-sorted recovery action manifest for failed categories).
- `.github/workflows/release.yml`: tag release workflow now validates release/operational contract tests before `maturin publish`.
  - CI dependencies are centralized in `requirements-ci-release.txt` (which currently references `requirements-ci-phase4.txt`).

## Focused Benchmarks

- `python scripts/eval/agent_dialogue_benchmark.py`: dialogue quality and retrieval behavior.
- `python scripts/eval/inference_accuracy_benchmark.py`: `SaraInference` sequence and retrieval accuracy.
- `python scripts/eval/spiking_llm_accuracy_benchmark.py`: `SpikingLLM` memory and streaming behavior.
- `python scripts/eval/task_switch_adaptation_benchmark.py`: task-switch and adaptation readiness.
- `python scripts/eval/future_state_consistency_benchmark.py`: world-model and predictive-runtime consistency.
  - includes a lightweight room-geometry fixture that reconstructs a top-down room hypothesis from sparse 2D wall/door/occlusion events without 3DGS or GPU reconstruction.
  - reports `future_state_spatial_projection_integrity`, `future_state_spatial_topology_consistency`, and `future_state_spatial_occlusion_reasoning`.
  - also ranks counterfactual room hypotheses and reports `future_state_spatial_counterfactual_selection`.
  - includes a connected two-room topology fixture and reports `future_state_spatial_adjacency_consistency`, `future_state_spatial_door_connectivity_integrity`, and `future_state_spatial_multi_room_counterfactual_selection`.
  - also evaluates topology-guided route planning with `future_state_spatial_route_planning_integrity`, `future_state_spatial_affordance_action_selection`, and `future_state_spatial_energy_aware_route_selection`.
  - executes the selected route and reports `future_state_spatial_route_state_update_integrity`, `future_state_spatial_invalid_action_rejection`, `future_state_spatial_route_rollback_observability`, and `future_state_spatial_route_execution_cost_bound`.
- `python scripts/eval/energy_efficiency_benchmark.py`: performance-per-energy, ANN-cost advantage proxy, sparse event cost, and low-precision readout signals.
- `python scripts/eval/continual_consolidation_benchmark.py`: replay, consolidation, and memory-health readiness.
- `python scripts/eval/nested_memory_readiness_benchmark.py`: Nested Learning-inspired multi-rate continuum memory controller readiness.
- `python scripts/eval/cognitive_runtime_benchmark.py`: Stage E modular cognitive runtime readiness.
- `python scripts/eval/phase5_predictive_coding_benchmark.py`: latent transition, prediction-error, correction-event, anti-collapse, and counterfactual separation readiness.
- `python scripts/eval/real_data_external_validity.py`: real-corpus retrieval, extractive summary, and continual-memory benchmark with ANN-style dense-scan cost comparison.
  - writes `workspace/evaluation/real_data_external_validity_history.json` by default and fails `trend.no_regressions` when external-validity quality or ANN-cost advantage regresses beyond tolerance.
  - stores corpus/task fingerprints in `benchmark_context`; trend comparison is skipped when the benchmark context changes.
  - report includes `thresholds` and `check_details` so each gate decision can be audited from the JSON artifact alone.
  - SARA retrieval uses metabolic sparse routing: rare-token-first search, confidence-based early stop, and verified fallback for hard high-candidate cases.
  - report includes metabolic diagnostics such as `sara_metabolic_cost_reduction_proxy`, `sara_metabolic_early_stop_rate`, and processed-token counts.
  - also includes absent-query negative controls (`negative_control_abstention_integrity`, `negative_control_cost_advantage_proxy`) so sparse routing must reject no-hit prompts instead of selecting a dense-scan fallback answer.
  - partial-evidence controls (`partial_evidence_abstention_integrity`, `partial_evidence_cost_advantage_proxy`) verify that common-token overlap alone is not enough to force an answer.
  - contrastive near-miss controls (`contrastive_control_accuracy`, `contrastive_control_cost_advantage_proxy`) verify that rare discriminative tokens win over common overlap in similar documents.
  - dense embedding baseline controls (`dense_embedding_ann_proxy_qa_accuracy`, `dense_embedding_ann_cost_advantage_proxy`) compare sparse routing against an offline hashed-vector ANN-style baseline without making dense vectors part of the runtime path.
  - real-data sparse diffusion block controls (`sparse_diffusion_real_data_denoise_accuracy`, `sparse_diffusion_real_data_event_cost_advantage`, `sparse_diffusion_real_data_partition_integrity`, `sparse_diffusion_real_data_single_pass_integrity`) verify that uncertainty-partitioned sparse denoising holds on real-corpus tasks.
- `python scripts/eval/real_data_external_validity_ladder.py`: runs the small/medium/large external-validity scale ladder and aggregates minimum QA, ANN-cost advantage, performance-energy ratio, and sparse diffusion block scores across profiles.
  - writes `workspace/evaluation/real_data_external_validity_ladder.json` and `workspace/evaluation/real_data_external_validity_ladder_summary.txt`.
  - aggregates minimum absent-query, partial-evidence, contrastive near-miss, dense embedding baseline, and real-data sparse diffusion block results across profiles.
  - operational readiness refresh now runs this after the single-profile external-validity benchmark and before release soak.
- `python scripts/eval/ann_efficiency_roadmap_gate.py`: aggregates the ANN-style accuracy-per-energy roadmap into staged checks.
  - validates sparse proxy instrumentation, limited real-data advantage, scale-ladder advantage, strict operational regression blocking, and neuromorphic transfer readiness.
  - Stage 2/3 require absent-query negative controls and real-data sparse diffusion block controls to pass before ANN-efficiency evidence is accepted.
  - Stage 6 validates `energy_measurement_readiness.py`, separating proxy-only evidence from real joule-per-success evidence.
  - propagates `measurement_session_plan.planned_runs` into top-level `next_evidence_actions` and the roadmap summary when available, falling back to `measurement_plan.pending_pairs` and `measurement_plan.weak_pairs`.
  - operational readiness imports these `next_evidence_actions` into `workspace/release/operational_readiness_runbook_actions.json` as `ann_efficiency_next_evidence` actions when the roadmap artifact is present.
  - writes `workspace/evaluation/ann_efficiency_roadmap_gate.json` and `workspace/evaluation/ann_efficiency_roadmap_gate_summary.txt`.
  - use `--refresh-artifacts` to regenerate the energy, external-validity, and ladder evidence before evaluating the roadmap.
- `python scripts/eval/energy_measurement_readiness.py`: validates the real-energy measurement schema and optional paired SARA/ANN joule evidence.
  - writes `workspace/evaluation/energy_measurement_readiness.json` and `workspace/evaluation/energy_measurement_readiness_summary.txt`.
  - also writes the standalone lab plan artifacts `workspace/evaluation/energy_measurement_session_plan.json` and `workspace/evaluation/energy_measurement_session_plan.txt`.
  - accepts `--measurement-path data/raw/energy_measurements.jsonl`; required fields are `run_id`, `system`, `task`, `success_count`, and `joules`.
  - use `--append-measurement --run-id <id> --system sara|ann --task <name> --success-count <n> --joules <J>` to append a validated measurement row before regenerating the readiness report.
  - alternatively pass `--average-watts <W> --duration-seconds <s>` with `--joules 0` or omitted; the tool records `joules = average_watts * duration_seconds` and keeps the derivation in the measurement row.
  - real joule evidence is accepted only when each measured `task` has both SARA and ANN rows and the minimum per-task ANN/SARA joule-per-success ratio passes the configured threshold.
  - the report includes `measurement_plan.pending_pairs` with command templates for missing SARA/ANN task rows and `measurement_plan.weak_pairs` for paired tasks that need repeat measurement or trace inspection.
  - the report also includes `measurement_session_plan.planned_runs` with stable run-id templates and `real_energy_session` command templates for the next paired measurement session.
- `python scripts/eval/sparse_diffusion_block_readiness.py`: evaluates the SARA-compatible sparse diffusion block research gate.
  - validates equal-mass uncertainty partitioning, independent sparse-event blocks, local denoising accuracy, event-cost advantage, block-count ablation, recurrent single-pass compression, and policy compatibility.
  - writes `workspace/evaluation/sparse_diffusion_block_readiness.json` and `workspace/evaluation/sparse_diffusion_block_readiness_summary.txt`.
  - use `python scripts/sara_cli.py eval-sparse-diffusion-block-readiness` to run the same gate from the unified CLI.

## Real-Data Curriculum

- `python scripts/train/run_real_data_curriculum.py --stage small --dry-run`: inspect the small-stage command plan without executing it.
- `python scripts/train/run_real_data_curriculum.py --stage small --preflight-only`: write only the data readiness report without executing training commands.
- `python scripts/train/run_real_data_curriculum.py --stage small`: run the verified pilot path against `data/processed/corpus.txt`, then Phase 3 and Phase 5 gates.
- `python scripts/train/run_real_data_curriculum.py --stage medium`: run medium-scale curriculum with Phase 3/4/5 gates.
- `python scripts/train/run_real_data_curriculum.py --stage large`: run large-scale curriculum with strict operational readiness appended.
- `python scripts/train/run_real_data_curriculum.py --stage large --skip-gates`: run training-only path (export + self-org + SNN-LM) for rapid iteration.
- Phase 5 gate path now appends `real_data_external_validity.py` to record real-data QA accuracy, summary coverage, continual-memory hit rate, `performance_energy_ratio_proxy`, and `ann_cost_advantage_proxy`.
  - curriculum runs use stage-specific external-validity history files (`real_data_external_validity_<stage>_history.json`) so small/medium/large trend checks do not cross-contaminate.
- Use `python scripts/sara_cli.py eval-external-validity-ladder` to run the same scale ladder through the unified CLI.
- Use `python scripts/sara_cli.py eval-ann-efficiency-roadmap --refresh-artifacts` to refresh the ANN-efficiency evidence path and evaluate the staged research roadmap.
- Use `python scripts/sara_cli.py record-energy-measurement --run-id <id> --system sara|ann --task <name> --success-count <n> --joules <J>` to append managed real-energy evidence through the unified CLI.
- If the meter or `powermetrics` output is averaged power, use `python scripts/sara_cli.py record-energy-measurement --run-id <id> --system sara|ann --task <name> --success-count <n> --average-watts <W> --duration-seconds <s>` and the CLI will derive joules before regenerating the readiness report.
- Managed report output: `workspace/reports/real_data_curriculum_small.json`
- Managed report output: `workspace/reports/real_data_curriculum_medium.json`
- Managed report output: `workspace/reports/real_data_curriculum_large.json`

## Managed Outputs

- Evaluation reports: `workspace/evaluation/`
  - Phase 5 entry gate report: `workspace/evaluation/phase5_entry_gate_report.json`
  - Phase 5 entry gate summary: `workspace/evaluation/phase5_entry_gate_summary.txt`
  - Phase 5 completion gate report: `workspace/evaluation/phase5_completion_gate_report.json`
  - Phase 5 completion gate summary: `workspace/evaluation/phase5_completion_gate_summary.txt`
  - Nested memory readiness report: `workspace/evaluation/nested_memory_readiness_benchmark.json`
  - Nested memory readiness summary: `workspace/evaluation/nested_memory_readiness_summary.txt`
  - Real-data external validity report: `workspace/evaluation/real_data_external_validity.json`
  - Real-data external validity summary: `workspace/evaluation/real_data_external_validity_summary.txt`
  - Real-data external validity history: `workspace/evaluation/real_data_external_validity_history.json`
  - Real-data external validity ladder report: `workspace/evaluation/real_data_external_validity_ladder.json`
  - Real-data external validity ladder summary: `workspace/evaluation/real_data_external_validity_ladder_summary.txt`
- Release and operational reports: `workspace/release/`
- Interim preprocessing outputs: `data/interim/`
- Processed training data: `data/processed/`
- Raw imported/exported data: `data/raw/`
- Final model artifacts: `models/`

Do not add generated outputs to the repository root.
