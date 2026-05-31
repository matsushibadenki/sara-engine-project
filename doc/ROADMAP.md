# SARA Engine Roadmap

This roadmap starts after the v1.1 release-hardening work. Completed implementation history is kept in [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md). The archived pre-v1.1 roadmap is kept in [old/ROADMAP_v1_1_completion_archive.md](old/ROADMAP_v1_1_completion_archive.md).

## Current Release Baseline

SARA Engine v1.1 is release-ready by the active gates:

- v1.1 release gate: `15/15` checks passing
- Research product completion gate: `12/12` checks passing
- ANN-efficiency roadmap gate: `6/6` stages passing
- Full test suite: `904` tests passing in the project Python 3.10 environment

The project is now structurally ready as a CPU-first SNN research product. The next phase should convert proxy evidence into stronger external evidence, improve real-world usefulness, and keep the sparse/runtime policy intact.

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

### Deliverables

- Collect paired SARA and ANN measurements for the tasks already listed in `workspace/evaluation/energy_measurement_session_plan.json`.
- Store rows in `data/raw/energy_measurements.jsonl` through `python scripts/sara_cli.py record-energy-measurement`.
- Regenerate:
  - `energy_measurement_readiness.py`
  - `ann_efficiency_roadmap_gate.py`
  - `research_product_completion_gate.py`
  - `v1_release_gate.py`
- Add a short measurement protocol document under `doc/` once the first real session is complete.

### Acceptance Criteria

- Every measured task has both SARA and ANN rows.
- `joule_per_success` is computed from validated rows.
- The minimum paired ANN/SARA joule-per-success ratio passes the configured threshold.
- Reports continue to label claims as proxy-only until physical measurements pass.
- Measurement commands and results remain reproducible from managed artifacts.

## Phase 7: Autonomous Learning Data Preparation

Goal: reduce human involvement in collecting and shaping training data by turning the existing `bot/` runtime into an automatic learning-material preparation loop.

The bot should not simply crawl more pages. It should transform collected material into safe, source-aware, curriculum-ready learning data that supports SARA's sparse retrieval, abstention, contrastive reasoning, and energy-efficiency goals.

### Implementation Steps

1. Add `bot/dataset_builder.py`.
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
   - Reject generated samples that are too short, unsupported by source text, duplicate, unsafe, or likely to contain secrets/PII.
   - Preserve source URL/path, collection time, quality score, language, source type, license/compliance hint, and rejection reason.
   - Write rejected items to `data/interim/autobot/rejected_learning_materials.jsonl`.

3. Add `bot/curriculum_manifest.py`.
   - Assign each accepted item to `easy`, `medium`, `hard`, `repair`, or `replay`.
   - Set priority from quality score, source reliability, rarity, modality scarcity, and current evaluation gaps.
   - Write `data/processed/autobot/curriculum_manifest.jsonl`.

4. Extend `bot/training_queue.py` integration.
   - Queue accepted learning materials by curriculum stage and material type.
   - Prefer repair data when recent gates fail.
   - Keep replay items bounded and high-value only.

5. Extend `bot/planner.py`.
   - Convert evaluation failures into collection/material requests:
     - weak summary coverage -> summary and source-claim data
     - weak negative controls -> negative-query data
     - weak contrastive controls -> near-miss/contrastive pairs
     - weak retrieval grounding -> source-backed QA
     - language imbalance -> language-targeted collection
   - Keep all planning outputs machine-readable under `workspace/autobot/`.

6. Add source-aware collector plugins.
   - `official_docs_collector.py`: allowlisted official documentation and stable reference pages.
   - `arxiv_abstract_collector.py`: paper metadata, abstracts, and topic summaries without treating papers as unrestricted training text.
   - Optional extension: GitHub README/docs collector for repositories with acceptable license/compliance signals.

7. Add a dataset quality report.
   - Write `workspace/autobot/dataset_builder_report.json`.
   - Summarize accepted/rejected counts, material-type counts, language balance, source domains, duplicate rate, compliance decisions, and curriculum distribution.
   - Add a compact text summary for operator review.

8. Add tests and docs.
   - Unit-test material extraction, gating, curriculum assignment, and planner feedback.
   - Update `bot/README.md` and `doc/TOOLS.md`.
   - Keep generated data out of the repository root.

### Managed Outputs

- Raw collected source material: `data/raw/autobot/`
- Extracted candidate text: `data/interim/autobot/extracted_text.jsonl`
- Generated candidate materials: `data/interim/autobot/candidate_learning_materials.jsonl`
- Rejected generated materials: `data/interim/autobot/rejected_learning_materials.jsonl`
- Accepted QA data: `data/processed/autobot/qa_pairs.jsonl`
- Accepted contrastive data: `data/processed/autobot/contrastive_pairs.jsonl`
- Accepted negative-query data: `data/processed/autobot/negative_queries.jsonl`
- Curriculum manifest: `data/processed/autobot/curriculum_manifest.jsonl`
- Dataset builder report: `workspace/autobot/dataset_builder_report.json`
- Dataset builder summary: `workspace/autobot/dataset_builder_summary.txt`

### Acceptance Criteria

- The bot can generate multiple learning-material types from already accepted records without human rewriting.
- Generated materials pass quality/compliance gates before entering the training queue.
- Evaluation failures can influence the next data-preparation cycle.
- Negative-query and contrastive-pair generation improve real-data external-validity coverage without weakening abstention behavior.
- All artifacts follow the managed output policy.
- Human review becomes an audit step, not the default data-preparation path.

## Phase 8: Stronger External Baselines

Goal: make the ANN comparison harder and more credible without importing ANN assumptions into the SARA runtime.

### Deliverables

- Add at least one stronger offline ANN-style retrieval baseline for comparison.
- Expand real-data tasks beyond the current small curated fixtures.
- Add noisy, adversarial, and delayed-recall retrieval cases.
- Keep dense baselines outside the production runtime path.
- Add per-task external validity summaries that separate quality, cost, abstention, and failure type.

### Acceptance Criteria

- SARA keeps sparse-event cost advantage on bounded tasks.
- Near-miss and partial-evidence controls remain passing.
- External baselines are clearly labeled as offline references.
- Runtime policy remains CPU-first and sparse.

## Phase 9: Research-Grade Benchmark Package

Goal: make SARA easier for other researchers to evaluate and reproduce.

### Deliverables

- Add a single benchmark entry command that runs the recommended research suite.
- Create a compact benchmark README or protocol under `doc/`.
- Produce a machine-readable benchmark manifest under `workspace/evaluation/`.
- Add example datasets or fixtures small enough for repository-safe validation.
- Add clear "what is proven" and "what is not proven" sections to release notes or benchmark docs.

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
- Expand neuromorphic HAL smoke tests into a backend capability matrix.
- Add hardware-profile reports for event budget, routing hints, memory footprint, and unsupported operations.
- Keep hardware-specific adapters optional.

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

## Phase 13: Capability Expansion Under Sparse Constraints

Goal: improve useful intelligence while preserving the energy-policy advantage.

### Candidate Directions

- Larger continual-memory experiments with bounded replay.
- Event-camera or DVS classification and association tasks.
- Stronger local credit assignment without runtime backpropagation.
- Better route selection between specialist submodels.
- Improved sparse verifier behavior for uncertain retrieval and reasoning.
- Multimodal association tasks that start with classification, grounding, and prediction instead of generation.

### Promotion Rule

Each candidate starts as one of:

- A small primitive
- A trace builder
- A focused evaluator
- An observed-only report
- A bounded runtime module

It can become release-critical only after quality, energy cost, state budget, traceability, and regression behavior remain stable across the managed suite.

## Immediate Next Actions

1. Run the first paired real-energy measurement session from `workspace/evaluation/energy_measurement_session_plan.txt`.
2. Implement the `bot/` Auto Dataset Builder path so accepted records become QA, negative-query, contrastive, and curriculum-ready learning data.
3. Harden the Rust sparse runtime with unit tests, API alignment, English comments, and benchmark reports.
4. Add a measurement protocol document after the first session.
5. Add a stronger offline ANN retrieval baseline while keeping it outside runtime.
6. Create a compact research benchmark protocol for third-party reproduction.
7. Expand the neuromorphic backend profile from smoke behavior to a capability matrix.

## Completed Work Reference

Do not re-add completed implementation history to this roadmap. Add completed items to [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md), and keep this file focused on what should happen next.
