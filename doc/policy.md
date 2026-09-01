# Project Policy

SARA Engine is developed as a CPU-first, SNN-based AI project. New work should preserve the engineering constraints below unless the roadmap explicitly promotes an exception.

These constraints are research-shaping defaults, not untouchable dogma. If a clearly better design, benchmark-backed implementation path, or decisive research opportunity requires a policy exception, the project may intentionally revise or override the policy. Such exceptions should be documented explicitly in the roadmap or a closely related design note, together with the reason, scope, and expected tradeoff.

## Core Design Rules

- Do not make correctness or normal runtime learning depend on global gradient backpropagation, end-to-end differentiation, or storage of a whole-network backward graph.
- Backward information is permitted and encouraged when bounded and auditable. Examples include branch-local activity history, bAP-like events, prediction-error or outcome events, global reward/modulatory scalars, replayed consequences, and source-linked correction signals.
- A backward signal must not be described as a gradient unless it is mathematically a derivative. Local eligibility, temporal proximity, branch contribution, novelty, reward, and modulation may assign credit without computing a global differential gradient.
- Prefer hierarchical local credit assignment: trace recently eligible branches, sub-branches, contacts, and synapses through explicit sparse structure, then apply local plasticity and structural-plasticity rules under fixed state/event budgets.
- Treat learning as an internal distributed capability of each bounded unit, not only as an external optimizer. A unit may compute, retain recent eligibility, receive feedback/modulation, update local parameters, and propose structural change through the same sparse event contract.
- Prefer algorithmic self-similarity over prescribed biological shape: the same bounded `local processing -> integration -> feedback -> plasticity` contract may recur at contact, branch, neuron, circuit, and module scales, while topology is allowed to emerge only through controlled growth and pruning.
- Do not make dense matrix operations the primary runtime design.
- Do not require GPUs for correctness or normal operation.
- Prefer sparse event routing, local plasticity, STDP-style updates, homeostasis, replay, and bounded memory structures.
- Prioritize accuracy and energy efficiency together; do not improve one by hiding unbounded cost in diagnostics or repair loops.
- Prefer Python plus Rust/PyO3 for practical performance, while keeping the public interface usable from Python.
- Code comments and logs should be written in English.

## Output Locations

Generated files must stay in managed directories:

- `data/raw/`: collected source data and exported chat JSONL files.
- `data/processed/`: cleaned corpora and finalized processed datasets.
- `data/interim/`: temporary preprocessing artifacts.
- `workspace/`: reports, summaries, operational runbooks, scratch files, and non-final artifacts.
- `models/`: final model artifacts.

Do not write generated artifacts to the repository root. Prefer `src/sara_engine/utils/project_paths.py` for new read/write paths.

## Release And Gate Policy

- Phase 3 completion requires `phase3_completion.passed == true`, `completion_score >= 1.0`, and no failed check entries.
- Phase 4 completion requires required metrics, threshold results, and numeric quality metrics to pass.
- Final production promotion should use the `extended` soak profile and `operational_readiness.py --strict-production`.
- Operational repair automation should be traceable through managed reports, runbooks, action manifests, and repair logs under `workspace/release/`.

## Research Intake Policy

Research ideas may be stored under `doc/idea/`. Promote them into active implementation only when they can be expressed as:

- A policy-compatible primitive.
- A benchmark or acceptance criterion.
- A bounded runtime feature.
- A managed artifact or release-gate signal.

Ideas that require GPU-first kernels, dense matrix training as the core mechanism, or global-gradient-backpropagation-dependent runtime learning should remain research notes unless the project policy changes or the roadmap explicitly approves a bounded research exception because it materially improves the project's long-term objective. Bounded backward information and local credit propagation are policy-compatible when they satisfy the core rules above.
