# Project Policy

SARA Engine is developed as a CPU-first, SNN-based AI project. New work should preserve the engineering constraints below unless the roadmap explicitly promotes an exception.

## Core Design Rules

- Do not make runtime learning depend on backpropagation.
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

Ideas that require GPU-first kernels, dense matrix training as the core mechanism, or backpropagation-dependent runtime learning should remain research notes unless the project policy changes.
