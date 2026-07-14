# SARA Research Benchmark Protocol

This protocol describes the compact reproducibility path for the v1.1 research surface. It is designed for CPU-first execution and writes only to managed project locations.

## Recommended Command

Run the full compact suite from the repository root:

```bash
python scripts/sara_cli.py eval-research-benchmark-suite
```

For a non-executing command manifest:

```bash
python scripts/sara_cli.py eval-research-benchmark-suite --dry-run
```

The command writes:

- `workspace/evaluation/research_benchmark_manifest.json`
- `workspace/evaluation/research_benchmark_summary.txt`

After the suite completes, validate the Phase 9 package:

```bash
python scripts/sara_cli.py eval-phase9-completion
```

The completion gate writes `workspace/evaluation/phase9_completion_gate.json` and
`workspace/evaluation/phase9_completion_gate_summary.txt`. It rejects dry-run
manifests, failed commands, missing managed outputs, incomplete fixtures, and
protocols without explicit proven and not-proven sections.

The repository-safe fixture file is:

- `data/processed/benchmark_fixtures/external_validity_cases.jsonl`

## Included Checks

- Research fixture readiness for QA, abstention, contrastive, noisy, adversarial, and delayed-recall cases.
- Rust sparse-runtime readiness with `cargo test` evidence.
- Rust sparse-runtime benchmark report with Python reference comparisons.
- Neuromorphic capability matrix for chip-neutral sparse event IR and Lava/SpiNNaker/Akida-style profile coverage.
- Sparse own-latent learning as observed-only sample-efficiency evidence.
- Source-backed own-latent manifest generation from autobot learning materials.
- Sparse dendritic feedback gate behavior as observed-only robustness evidence.
- Sparse plan-trace verification and repair-material generation as observed-only planning evidence.
- Sparse future-state reasoning priors with deterministic logic consistency, event relevance, and external-context abstention.
- Verified sparse resonance credit with reward-only harmful-update comparison and explicit plasticity-freeze reasons.
- Managed-report resonance integration across reasoning, planning, multimodal, dendritic, own-latent, and metabolic evidence.
- Sparse synesthetic multimodal binding, modality adapters, 25/32/40 ms window comparison, own-latent integration, dendritic/thalamic route traces, plug-swapping, and missing-modality abstention as observed-only evidence.
- Verified hierarchical event-state caching with fixed, linear, and logarithmic retention comparisons, delayed recall, blocked admission, abstention, event cost, and state-growth evidence.
- Source-aware event-state cache integration with managed resonance evidence, autobot latent materials, read-only reactivation, persistence round trips, and corruption rejection.
- Research-product completion gate across policy, ROADMAP closure, Rust readiness, managed outputs, and release evidence.
- Integrated v1.1 release gate.
- Phase 13 sparse capability-expansion aggregation across reasoning, planning, multimodal binding, local credit, own-latent, hierarchical cache, and structural-plasticity evidence.
- Phase 14 sparse own-latent completion validation across the local predictor benchmark, source-backed latent manifest, RHM fixture, and bounded-state policy.
- Phase 15 sparse dendritic feedback completion validation across robustness, bounded convergence, fallback, traceability, event cost, and state-budget evidence.
- Phase 16 sparse synesthetic multimodal completion validation across equal-modality routing, temporal alignment, bundle separability, missing-modality abstention, and bounded event/state budgets.
- Phase 17 verified sparse resonance credit completion validation across multi-signal agreement, freeze safety, harmful-update controls, source-backed integration, and bounded local updates.
- Phase 18 verified hierarchical Event Memory completion validation across delayed recall, bounded retention, source-aware admission, persistence, corruption rejection, and negative-query abstention.
- Phase 19 sparse liquid-time-constant completion validation across fixed controls, temporal error improvement, event/update/state bounds, replay determinism, and CPU/backpropagation policy.

## What Is Proven

- The managed v1.1 release gate can be reproduced from repository commands.
- A compact example fixture exists for repository-safe validation of external-validity case types.
- The Rust sparse-runtime source is covered by meaningful unit tests and policy-readiness checks.
- Backend-portability evidence records event budgets, routing hints, state budget, adapter policy, and unsupported operations without requiring accelerator hardware.
- Source-backed sparse own-latent manifests can be generated from managed autobot learning materials.
- Observed-only sparse own-latent and dendritic feedback reports can be reproduced without changing production inference.
- Observed-only sparse plan-trace verifier reports can be reproduced without using LLM chain-of-thought.
- Observed-only sparse reasoning-prior reports can be reproduced without dense LLM/TSFM fusion or an LLM judge.
- Observed-only resonance-credit reports can reproduce verified multi-signal update gating without runtime backpropagation.
- Managed evaluator reports can be bridged into a single sparse plasticity decision while preserving source and failure traceability.
- Observed-only equal-modality sparse binding can be reproduced without a language hub, dense universal embeddings, or runtime backpropagation.
- Observed-only verified event-state caching can reproduce bounded delayed recall and sparse retrieval without dense hidden-state caching.
- Managed Phase 17 evidence can gate source-aware cache promotion while preserving material hashes and strict persistence validation.
- Benchmark and gate outputs stay inside `workspace/` or `workspace/release/` managed paths.
- Python fallback behavior remains explicit when the Rust extension is not built.

## What Is Not Proven

- Physical joule-per-success measurements are not proven unless paired SARA and ANN meter rows are supplied under `data/raw/energy_measurements.jsonl`.
- Rust extension speedup is not proven when `sara_engine.sara_rust_core` is not built in the active Python environment.
- Broad external generalization requires additional source-aware datasets beyond the included fixtures.
- Event-state cache gains remain unproven for production memory until larger source-aware delayed-recall evaluations pass.

## Output Policy

Do not redirect benchmark artifacts to the repository root. Use `--manifest-path` and `--summary-path` only with managed locations such as `workspace/evaluation/`.
### Phase 20 Semantic Echo Field

The Phase 20 benchmark compares single-decay and fixed multi-timescale sparse controls against bounded local Semantic Echo Field dynamics. It records role binding, delayed recall, contradiction and unsupported-query abstention, active echoes, comparisons, updates, and idle spikes. External parser/LLM assistance is disabled; physical energy claims remain out of scope until paired measurement.
