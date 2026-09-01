# SARA Engine Architecture Review

Last reviewed: 2026-05-31

This document is a design-health review for SARA Engine. It does not replace the roadmap or release checklist. Its purpose is to keep the architecture coherent as the project absorbs many research ideas.

## Executive Judgment

The current architecture has a good foundation. The strongest property is not any single module, but the consistency of the project constraints:

- CPU-first operation.
- SNN-based sparse event processing.
- No runtime dependency on global gradient backpropagation; backward information may modulate bounded local eligibility and structural credit.
- No GPU requirement for correctness or normal operation.
- Bounded memory, managed artifacts, and release-gated validation.
- Performance-per-energy treated as the primary objective, with accuracy and capability improvements accepted only when their runtime cost stays bounded.

This is a good design direction because it gives SARA Engine a distinct optimization target instead of trying to imitate dense ANN systems on worse hardware terms. The project should optimize for the best possible capability per unit of event cost, moving toward brain-like efficiency rather than raw dense throughput.

## Primary Optimization Target

The most important project target is the ratio between useful performance and energy cost. SARA Engine should prefer a lower raw score with dramatically lower event cost over a higher raw score that requires ANN-like dense computation.

The long-term direction is to move closer to the human brain's operating style:

- Sparse spikes instead of dense activation sweeps.
- Local updates driven by forward activity plus bounded backward information instead of mandatory global differential gradients.
- Bounded working memory instead of unbounded context expansion.
- Sleep/replay/consolidation during low-pressure windows.
- Energy-aware routing, skipping, pruning, and temporal compression.
- Performance reports that always include cost proxies.

For current software-only evaluation, the project uses explicit proxies rather than fake watt measurements:

- `sara_energy_cost_units`: bounded state units plus route work units.
- `ann_reference_cost_units`: a dense ANN-style fallback reference for the same fixture.
- `performance_energy_ratio_proxy`: successful outcomes per SARA cost unit.
- `ann_cost_advantage_proxy`: reference ANN-style cost divided by SARA cost.
- `brain_efficiency_alignment_proxy`: sparse event cost and ANN-cost advantage combined into a brain-efficiency direction signal.

These proxies should eventually be replaced or augmented with measured joules on target hardware, but they are the correct first discipline: every new intelligence feature must report its metabolic price.

The main risk is conceptual sprawl. The roadmap now includes world-model work, Spiking H-JEPA, Nested Learning, structural plasticity, astrocyte-inspired modulation, stochastic computing, fluid-inspired traces, self-distillation, multimodal integration, and operational repair automation. These are valuable only if they converge into a small runtime spine. If they remain separate features, the system will become difficult to reason about and easy to overfit to lightweight gates.

## Design Spine

Future work should converge on this spine.

| Layer | Role | Design Standard |
| --- | --- | --- |
| Sparse event schema | Common representation for spikes, memory events, predictions, corrections, and diagnostics. | New modules should emit or consume sparse events rather than invent isolated data formats. |
| Bounded memory continuum | Short-term, direct, hippocampal, long-term, and structural memory should behave like one managed continuum. | Memory growth, consolidation, retrieval, and forgetting must be explicit and measurable. |
| Lightweight world model | Predict next state, counterfactual outcome, and action consequence without dense generative modeling. | Prefer state-transition traces, prediction-error signals, and correction events over full reconstruction. |
| Energy-aware scheduler | Decide when to update, search, replay, consolidate, or skip work. | Accuracy improvements must improve or preserve performance-per-energy through spike counts, update counts, latency, route work, or equivalent proxies. |
| Operational observability | Release gates, readiness reports, runbooks, and action manifests make the system auditable. | Gates should catch real regressions, not only confirm toy fixtures. |

Any new feature should attach to one of these layers. If it does not, it should remain under `doc/idea/`.

## Current Strengths

- The project policy is clear and technically meaningful: CPU-first, sparse, event-driven, no backpropagation-heavy runtime, no GPU-first assumption.
- Release and operational gates are unusually strong for a research project. Phase 3, Phase 4, Phase 5 entry, Stage B-E readiness, v1 release checks, and operational runbooks give the project a traceable quality system.
- Import-chain hardening and lazy exports improved package hygiene. Evaluation scripts should not fail because optional audio, visualization, or transformer dependencies are pulled in accidentally.
- The research intake pattern is healthy. New ideas are usually translated into lightweight primitives, benchmarks, or observed-only signals before being promoted.
- Nested Learning is being introduced in the right way: as a multi-rate memory controller and readiness benchmark before it becomes a shipping requirement.
- The managed output policy keeps generated reports, scratch artifacts, data, and models from polluting the repository root.

## Current Risks

- Too many gates can create false confidence. Several current checks are lightweight and may pass with perfect scores on narrow fixtures while real use remains weak.
- The project can over-accumulate concepts. If each paper becomes a separate subsystem, the architecture will drift away from the sparse runtime spine.
- Some optional dependencies still represent philosophical and operational tension. Packages such as `transformers`, `torch`, `matplotlib`, and `numpy` may be useful for tooling or compatibility, but runtime paths should not quietly depend on them.
- Phase 5 and later wording can sound broader than the implementation currently proves. The practical release position should remain "limited CPU-first SNN assistant/research engine" until external baselines prove otherwise.
- Operational repair automation is powerful, but it must not become a substitute for model quality. Repair loops should expose root causes, not hide unstable behavior behind repeated retries.
- Stage D continual consolidation and Nested Learning memory work overlap. They should be unified rather than developed as parallel memory theories.

## Research Adoption Rules

Use the following filter before promoting an idea from `doc/idea/` or the roadmap into implementation.

| Question | Promote If Yes | Keep As Idea If No |
| --- | --- | --- |
| Can it be expressed as sparse events, local state, or bounded memory? | Implement as a primitive or adapter. | Keep as research note. |
| Does it improve accuracy, energy efficiency, stability, or observability under a measurable metric? | Add a benchmark or observed-only report first. | Do not add runtime complexity yet. |
| Can it run correctly on CPU without GPU-specific kernels? | Continue evaluation. | Keep out of the release path. |
| Does it avoid runtime backpropagation as a requirement? | Continue evaluation. | Keep as non-runtime research only. |
| Does it reduce conceptual duplication? | Merge into the existing spine. | Do not create a new subsystem. |

## Feature Positioning

### Keep And Strengthen

- Sparse event schema and common cognitive runtime.
- Direct memory, hippocampal memory, long-term memory, and structural plasticity as one bounded memory continuum.
- Stage B world-model minimums, especially prediction, reward/policy preference, energy-aware action selection, and counterfactual trace checks.
- Stage E modular cognitive runtime checks.
- Phase 5 Spiking H-JEPA entry gate, as long as it remains transition-oriented rather than dense reconstruction-oriented.
- Nested Learning as a multi-rate memory scheduler and consolidation policy, initially observed-only.
- Release, operational readiness, runbooks, and action manifests.

### Use Carefully

- Self-distillation: useful only as spike-pattern stabilization, replay curation, or teacher-free consistency checking. Avoid ANN-style large teacher dependence as a core runtime assumption.
- Astrocyte-inspired modulation: useful as a slow homeostatic gain or plasticity regulator. Avoid adding a separate complex simulator until it improves measured stability.
- Fluid-inspired dynamics: useful as bounded predictive support tracing. Avoid making it a parallel physics engine.
- Stochastic computing: useful as an optional low-cost approximation or noise robustness probe. Avoid letting probabilistic encodings obscure correctness.
- Multimodal integration: start with classification, association, and prediction support. Do not chase full generative multimodal capability before the text/memory/runtime spine is stable.

### Avoid For Now

- GPU-first optimization paths as release-critical features.
- Dense matrix training as the main runtime mechanism.
- Backpropagation-dependent online learning.
- New paper-inspired modules that do not connect to memory, world modeling, energy scheduling, or operational observability.
- Claims of general ANN/LLM competitiveness without external baselines, scale tests, and energy measurements.

## Gate Quality Review

The gate system is a strength, but it should be made harder in a few specific ways.

- Add adversarial and noisy retrieval cases for direct memory, hippocampal memory, and long-term memory.
- Add scale-sensitive tests that measure latency, candidate counts, memory growth, and retrieval precision together.
- Add long-context and delayed-recall scenarios instead of only short fixed prompts.
- Track energy proxies alongside quality: spike count, update count, search count, route work, replay count, consolidation count, and performance-per-energy.
- Treat repeated perfect scores on tiny fixtures as a prompt to increase fixture difficulty.
- Keep observed-only metrics for new research features until they pass multiple benchmark styles.

## Recommended Post-v1.1 Focus

Before adding more major research ideas, post-v1.1 work should prioritize stronger evidence and integration quality.

1. Keep paired SARA/ANN real-joule measurement protocol artifacts frozen on indefinite hold; do not block software evidence on them.
2. Add stronger offline ANN-style baselines while keeping them outside the runtime path.
3. Expand external-validity fixtures toward noisy, delayed, and adversarial retrieval cases.
4. Formalize the sparse event IR and backend capability matrix for neuromorphic portability.
5. Improve operator usability so release and research evidence can be reproduced without reading every evaluator.

## Release Positioning

The honest v1.1 positioning should be:

SARA Engine is a limited-scope, CPU-first SNN research/runtime engine for sparse memory, lightweight dialogue assistance, predictive traces, continual-learning experiments, ANN-efficiency proxy evidence, and release-gated operational evaluation.

It should not yet be positioned as a general replacement for ANN-based LLMs. The competitive claim should remain narrower and evidence-based:

- Better alignment with CPU-only and edge-oriented constraints.
- Lower conceptual dependence on dense training and GPU runtime.
- Stronger inspectability of memory, prediction, and operational readiness.
- Potential performance-per-energy advantages on sparse, continual, low-data workloads.
- Protocol-ready real-joule comparison path, with physical claims deferred until paired measurements pass.

## Architecture Decision

The design is worth continuing. The next architectural priority is not to add more isolated intelligence mechanisms. The priority is to make the existing mechanisms behave like one sparse, bounded, energy-aware cognitive runtime.

The project should use this rule:

If a proposed feature does not strengthen the sparse event spine, bounded memory continuum, lightweight world model, energy-aware scheduler, or operational observability, it should not enter active implementation yet.
