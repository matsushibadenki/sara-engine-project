# SARA Engine

SARA (Spiking Architecture for Reasoning and Adaptation) Engine is a CPU-first spiking AI framework for learning, inference, continual memory, and lightweight agent workflows without backpropagation-heavy runtime assumptions.

The project combines an event-driven Rust core with Python model, evaluation, and CLI layers. It prioritizes biological plausibility, bounded runtime state, managed output paths, and release-gated operation suitable for local and edge-oriented deployment.

## Design Policy

SARA Engine may borrow ideas from JEPA, linear RNNs, SSMs, predictive coding, local plasticity, neuromorphic hardware research, sparse verifier/search methods, and modular cognitive architectures. These ideas must remain coherent under one simple spine:

- The shared interface is sparse events, not dense tensors. New mechanisms should exchange compact spike/event traces, bounded state snapshots, route records, correction events, and human-auditable summaries.
- Runtime learning should stay backpropagation-free and matrix-light. Dense ANN, Transformer, GPU, or large-batch training ideas may be used only as references, baselines, or optional offline comparisons, not as the production runtime path.
- Prefer small composable primitives over large monolithic models. Memory, world model, value, language, math, body-control, retrieval, verifier, and self-monitoring functions should be separable modules with explicit connect/disconnect and local update traces.
- Every experimental architecture starts as observed-only. It can be promoted only after it improves or preserves quality, sparse event cost, state budget, traceability, and release-gate stability across the managed evaluation suite.
- Scaling should improve capability by adding sparse capacity: more specialist submodels, better routing, compact memories, local manifolds, event budgets, sleep/replay consolidation, and hardware-aware event IR. Scaling should not require unbounded context, uncontrolled state growth, or GPU-resident dense computation.
- Implementations should be compact. A new feature should first be a small trace builder, evaluator, or bounded state module with focused tests before becoming a runtime dependency.
- Interpretability is a first-class requirement. Important decisions must expose route evidence, counterfactual separation, local credit, memory steering, prediction error, correction coverage, and stable digests whenever possible.
- Project output policy is mandatory. Generated data, intermediate artifacts, working files, and model outputs must stay under `data/`, `workspace/`, or `models`, preferably through `sara_engine.utils.project_paths`.

In short: SARA scales through sparse, inspectable, locally adaptive submodels connected by a common event protocol. Research ideas are welcome when they make that spine stronger, simpler, and easier to evaluate.

## Key Features

- Rust-accelerated event-driven SNN core for CPU-focused execution.
- Backpropagation-free, matrix-light learning flows centered on STDP, predictive coding, FORCE, and direct memory updates.
- Performance-per-energy is the primary optimization target, with energy proxies tracking sparse event cost, ANN-style reference cost, and brain-efficiency alignment before real joule measurements are available.
- Spiking language and agent components including `SaraInference`, `SpikingLLM`, and `SaraAgent`.
- Managed release validation with soak reports, release gates, operational readiness reports, runbooks, and action manifests.
- v1.1 release gating validates Phase 3/4/5 completion, strict operational readiness, real-data external validity, sparse diffusion block readiness, research-product completion, and version alignment.
- ANN-efficiency roadmap gating tracks sparse proxy wins, real-data advantage, scale-ladder advantage, operational regression protection, neuromorphic transfer readiness, and paired real-joule measurement readiness.
- Sparse diffusion block readiness checks equal-mass uncertainty partitioning, independent sparse-event denoising blocks, event-cost advantage, block-count ablation, and recurrent single-pass compression.
- Shared TurboQuant-style quantization utilities for compact SNN memory and checkpoint handling.
- Fluid-inspired supplementary dynamics for bounded predictive support tracing without backpropagation or matrix-heavy runtime dependencies.

## Installation

Ensure Python 3.10+ and a working Rust toolchain are available.

```bash
git clone https://github.com/matsushibadenki/sara-engine-project.git
cd sara-engine-project
pip install -e .
```

If Rust core changes are not reflected, rerun `pip install -e .`.

## CLI And Training

Interactive chat with a saved memory model:

```bash
sara-chat --model models/distilled_sara_llm.msgpack
```

Dialogue memory training from JSONL:

```bash
sara-train data/raw/chat_data.jsonl --model models/distilled_sara_llm.msgpack
```

JSONL format:

```json
{"user": "こんにちは", "sara": "こんにちは。SARAです。"}
{"user": "SARAって何？", "sara": "私はスパイキングニューラルネットワークで動くローカルAIエンジンです。"}
```

Subword SNN LM training with optional TurboQuant-style checkpoint compression:

```bash
python scripts/train/train_snn_lm.py \
  --corpus data/processed/corpus.txt \
  --save-dir models/snn_lm_pretrained \
  --turboquant
```

The active integrated CLI is `scripts/sara_cli.py`. It covers corpus import/export, self-organized training, legacy distilled agent memory, chat entry points, memory inspection, artifact upgrade, replay-data generation, pruning, and cleanup:

```bash
python scripts/sara_cli.py db-status
python scripts/sara_cli.py db-import data/raw/example.txt --category document --lang en
python scripts/sara_cli.py db-export
python scripts/sara_cli.py train-self-org
python scripts/sara_cli.py train-curriculum --stage small --dry-run
python scripts/sara_cli.py eval-external-validity --regression-tolerance 0.05
python scripts/sara_cli.py inspect-memory
python scripts/sara_cli.py fix-memory --context-tokens "1,2,3" --wrong-token-id 7 --dry-run
python scripts/eval/research_product_completion_gate.py
```

Staged real-data curriculum training (small/medium/large) with managed report output:

```bash
python scripts/train/run_real_data_curriculum.py --stage small --dry-run
python scripts/train/run_real_data_curriculum.py --stage small --preflight-only
python scripts/train/run_real_data_curriculum.py --stage medium
python scripts/train/run_real_data_curriculum.py --stage large
```

The curriculum gate path also records a real-data external-validity report that compares sparse SARA retrieval with an ANN-style dense-scan proxy for QA accuracy, summary keyword coverage, continual memory, event-cost advantage, and real-corpus sparse diffusion block behavior.

See [doc/TOOLS.md](doc/TOOLS.md), [doc/SARA-Engine_Training_Manual.md](doc/SARA-Engine_Training_Manual.md), and [doc/IMPLEMENTED_FEATURES.md](doc/IMPLEMENTED_FEATURES.md) for the current command and feature map.

## Managed Output Policy

Generated artifacts must stay inside managed directories:

- `data/raw`, `data/interim`, `data/processed`
- `workspace`
- `models`

Repository-root outputs and ad hoc directories are not part of the supported production flow. Path helpers live in [src/sara_engine/utils/project_paths.py](src/sara_engine/utils/project_paths.py).

## Validation And Release Gates

Recommended pre-release flow:

```bash
python scripts/eval/real_data_external_validity.py
python scripts/eval/real_data_external_validity_ladder.py
pytest -q tests/test_release_soak.py tests/test_sara_cli_dispatch.py tests/test_cli_entrypoints.py tests/test_inference_reliability.py tests/test_inference_memory_io.py tests/test_spiking_llm_memory_io.py tests/test_direct_map_utils.py tests/test_chat_agent_calculator.py tests/test_sara_agent_dialogue.py tests/test_practical_reliability.py
pytest -q tests/test_phase3_accuracy_benchmarks.py tests/test_phase4_operational_cycle.py tests/test_operational_readiness.py tests/test_release_soak.py tests/test_release_gate.py
python scripts/eval/phase3_accuracy_suite.py
python scripts/eval/phase3_completion_gate.py
python scripts/eval/phase4_scale_continual_benchmark.py
python scripts/eval/phase4_completion_gate.py
python scripts/eval/phase5_predictive_coding_benchmark.py
python scripts/eval/phase5_entry_gate.py
python scripts/eval/phase5_completion_gate.py
python scripts/eval/real_data_external_validity.py
python scripts/eval/real_data_external_validity_ladder.py
python scripts/eval/ann_efficiency_roadmap_gate.py
python scripts/eval/release_soak.py --profile release --include-accuracy
python scripts/eval/release_gate.py
```

The release path expects Phase 3 results to satisfy Stage A plus Stage B-E readiness. Human-readable summaries should show `PASS` for Stage B world-model minimums, Stage C meta-adaptation, Stage D continual consolidation, Stage E modular cognitive runtime checks, Phase 5 entry readiness, and Phase 5 completion readiness. Phase 4 reports must include both required metrics and `quality_metrics`.

For final shipping decisions, use the extended soak profile and strict operational readiness:

```bash
python scripts/eval/release_soak.py --profile extended --include-accuracy
python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy --strict-production --phase3-regression-tolerance 0.05
python scripts/eval/phase4_operational_cycle.py --dry-run
python scripts/eval/v1_release_gate.py
```

The v1.1 gate explicitly validates strict operational readiness, Phase 3 completion, Stage B reward/policy minimums, Phase 4 quality, target-version alignment, Phase 5 entry metrics from the Phase 3 report, the Phase 5 operational snapshot propagated into `operational_readiness_report.json`, the standalone `phase5_completion_gate_report.json` artifact, and the research-product completion report. The strict operational refresh path regenerates the standalone Phase 5 predictive-coding benchmark, Phase 5 entry gate, sparse diffusion block readiness, and Phase 5 completion gate artifacts before release soak validation. The Phase 5 completion, operational, and v1 summaries expose macro/subgoal efficiency, micro-ES low-rank/event-cost detail values, and sparse diffusion block completion checks for release review.

It also validates `workspace/evaluation/real_data_external_validity.json`, which keeps real-data QA, summary coverage, continual memory, ANN-style cost-advantage, and real-data sparse diffusion block regressions visible at the final promotion gate.
Strict operational readiness also validates `workspace/evaluation/real_data_external_validity_ladder.json`, which aggregates small/medium/large external-validity evidence and blocks promotion when the scale ladder loses sparse-vs-ANN energy advantage or sparse diffusion block integrity.
The external-validity benchmark keeps a managed history at `workspace/evaluation/real_data_external_validity_history.json` and reports `trend.no_regressions` for slow quality or energy-ratio drift.
It fingerprints the corpus and generated task set so trend comparisons are only active for comparable benchmark contexts.
The JSON report includes `thresholds` and `check_details` for direct audit of every external-validity gate decision.
The ladder report exposes the minimum QA score, minimum ANN-cost advantage, minimum performance-energy ratio, and minimum sparse diffusion block denoising/cost/integrity scores across all scale profiles.
The ANN-efficiency roadmap gate combines the energy benchmark, real-data external-validity report, scale ladder, strict operational report, and neuromorphic portability signals into staged evidence for the top-level goal: beat ANN-style systems on task success per event/energy cost.
It also propagates energy `measurement_plan` items into `next_evidence_actions`, so the roadmap report itself lists the managed commands needed for the next real-joule evidence loop.
Strict operational readiness imports those roadmap `next_evidence_actions` into `workspace/release/operational_readiness_runbook_actions.json` when the roadmap artifact is present, keeping the real-joule evidence loop visible in the operator action manifest.
The SARA side uses metabolic sparse retrieval: rare-token-first routing, early stop when confidence is sufficient, and a verified fallback for hard high-candidate cases.
External-validity reports also include absent-query negative controls, so sparse retrieval must abstain cheaply instead of overselecting a document when no corpus evidence exists.
They also include partial-evidence controls where only common terms match, forcing the sparse route to reject weak evidence instead of turning overlap into a false answer.
Contrastive near-miss controls check that rare discriminative terms are processed before common overlap when similar documents compete.
Dense embedding ANN-style controls add an offline hashed-vector baseline for comparison while keeping dense vectors out of the production runtime path.
Energy measurement readiness accepts paired SARA/ANN joule logs from `data/raw/energy_measurements.jsonl` and keeps claims labeled as proxy-only until real `joule_per_success` evidence is present.
Measurements can be appended through `python scripts/sara_cli.py record-energy-measurement ...`, which validates rows before updating the readiness report. The command accepts either direct `--joules <J>` or `--average-watts <W> --duration-seconds <s>` and records the joule derivation.
Real joule evidence must be task-paired: every measured task needs both SARA and ANN rows, and the minimum per-task ANN/SARA joule-per-success ratio must pass before the roadmap gate accepts the claim.
The readiness report also includes a `measurement_plan` plus a `measurement_session_plan` with pending SARA/ANN task pairs, stable run-id templates, and command templates for the next rows to collect.
The session plan is also written separately as `workspace/evaluation/energy_measurement_session_plan.json` and `workspace/evaluation/energy_measurement_session_plan.txt` for lab use.

Sparse diffusion block readiness adapts the DiffusionBlocks idea to SARA policy without adding runtime backpropagation, dense-matrix-first execution, or GPU requirements. `python scripts/sara_cli.py eval-sparse-diffusion-block-readiness` checks equal-mass uncertainty partitioning, independent sparse-event blocks, local denoising, event-cost advantage, block-count ablation, and recurrent single-pass compression.
The world-model benchmark now includes a sparse room-geometry fixture that converts 2D wall/door/occlusion events into a top-down room hypothesis, keeping early spatial reasoning CPU-first and constraint-based.
It also ranks counterfactual room hypotheses, so hidden-wall and wrong-depth alternatives are rejected by topology, area, occlusion, and event-cost constraints.
The spatial benchmark now includes a connected two-room topology case, so overlap and disconnected-room alternatives are rejected by door connectivity, adjacency, area, and event-cost constraints.
It also uses that topology for low-cost route selection, preferring a valid door affordance over wall-crossing or no-progress alternatives.
The selected spatial route is then executed as a sparse state update, while invalid wall-crossing remains an observable rollback instead of corrupting the room state.

Managed outputs:

- Soak report: `workspace/release/release_soak_report.json`
- Soak summary: `workspace/release/release_soak_summary.txt`
- Phase 3 accuracy suite: `workspace/evaluation/phase3_accuracy_suite.json`
- Phase 4 benchmark: `workspace/evaluation/phase4_scale_continual_benchmark.json`
- Phase 5 predictive coding benchmark: `workspace/evaluation/phase5_predictive_coding_benchmark.json`
- Phase 5 entry gate report: `workspace/evaluation/phase5_entry_gate_report.json`
- Phase 5 entry gate summary: `workspace/evaluation/phase5_entry_gate_summary.txt`
- Phase 5 completion gate report: `workspace/evaluation/phase5_completion_gate_report.json`
- Phase 5 completion gate summary: `workspace/evaluation/phase5_completion_gate_summary.txt`
- Operational readiness report: `workspace/release/operational_readiness_report.json`
- Operational runbook: `workspace/release/operational_readiness_runbook.md`
- Operational action manifest: `workspace/release/operational_readiness_runbook_actions.json`
- Phase 4 operational cycle report: `workspace/release/phase4_operational_cycle_report.json`

## Core Modules

- `sara_engine.core`: low-level spiking layers and Rust-facing building blocks.
- `sara_engine.models`: prebuilt SNN language, classifier, reservoir, and multimodal models.
- `sara_engine.learning`: plasticity, predictive coding, FORCE, and structural update rules.
- `sara_engine.memory`: SDR, hippocampal memory, long-term memory, and vector-store components.
- `sara_engine.agent`: bounded-state agent runtime and tool integration.
- `sara_engine.evaluation`: release, reliability, Phase 3/4/5, and cognitive runtime benchmark evaluators.

## Documentation

- [doc/SARA-Engine_Documentation_Hub.md](doc/SARA-Engine_Documentation_Hub.md): active documentation entry point.
- [doc/policy.md](doc/policy.md): project constraints and managed output policy.
- [doc/IMPLEMENTED_FEATURES.md](doc/IMPLEMENTED_FEATURES.md): completed v1.1 feature inventory and gate status.
- [doc/ARCHITECTURE_REVIEW.md](doc/ARCHITECTURE_REVIEW.md): design-health review, architecture spine, research adoption rules, and risk controls.
- [doc/TOOLS.md](doc/TOOLS.md): current CLI, evaluation, release, and maintenance commands.
- [doc/ROADMAP.md](doc/ROADMAP.md): post-v1.1 priorities and next milestone plan.
- [doc/RELEASE_CHECKLIST.md](doc/RELEASE_CHECKLIST.md): pre-release validation and packaging checklist.
- [doc/RELEASE_NOTES.md](doc/RELEASE_NOTES.md): current release-candidate changes and known gaps.
- [doc/idea](doc/idea): research notes, archived ideas, and legacy references.
- [doc/old](doc/old): archived completed roadmaps and non-canonical historical notes.

## License

MIT License.
