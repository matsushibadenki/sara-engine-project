# SARA Engine

SARA (Spiking Architecture for Reasoning and Adaptation) Engine is a CPU-first research engine for sparse, event-driven intelligence. It explores how useful reasoning, continual memory, and adaptive behavior can be built from spikes, local state, structural relations, replay, and bounded verification rather than runtime backpropagation or a dense matrix-first architecture.

SARA is not currently a general replacement for ANN-based LLMs. Its practical scope is a local SNN research/runtime system with lightweight dialogue, sparse memory, predictive traces, structural reasoning experiments, and reproducible promotion gates.

## Design Thesis

The project is organized around one principle:

> Intelligence should emerge from reusable sparse interactions among bounded local components, while every important prediction remains traceable to experience, structure, and correction evidence.

This leads to the following design commitments:

- **Sparse events are the common interface.** Modules exchange spikes, typed events, bounded state snapshots, routes, prediction errors, corrections, and audit records instead of unrestricted dense tensors.
- **Runtime learning is local.** Production-oriented learning must not require backpropagation, GPU execution, or dense matrix sweeps. ANN and Transformer systems may be used as offline references or comparison baselines only.
- **Time is part of the representation.** Firing time, order, interval, phase, delay, fatigue, and short-lived state are potential computational resources rather than values to average away automatically.
- **Knowledge is not identical to weights.** Experiences, relations, provenance, revisions, contradictions, and evidence should survive changes in internal representation. Weights or effective interactions may be regenerated, cached, or revised from that state.
- **Structure and parameters are complementary.** Connectivity determines possible computation; local strengths, delays, roles, and state tune it. The project does not assume that either topology or scalar weight alone is sufficient.
- **Memory must remain revisitable.** A learned representation may change, but the system should preserve a route back to the experience and evidence from which it arose.
- **Growth must be bounded.** New capability should come from sparse capacity, selective routing, consolidation, structural reuse, and local specialization—not unbounded context or uncontrolled state accumulation.
- **Claims follow evidence.** A new mechanism begins as default-off and observed-only. It enters a runtime path only after preregistered comparison, deterministic replay, bounded-cost checks, independent evidence, and explicit promotion review.

## Cognitive Spine

SARA's intended information flow is:

```text
typed sensory or language events
        -> sparse local fragments
        -> bounded temporal and memory state
        -> reusable relations and structural hypotheses
        -> prediction, action, or retrieval proposal
        -> verifier and counterfactual checks
        -> local correction, replay, or consolidation
        -> auditable evidence record
```

The system favors small, composable components over one monolithic model. Language, memory, world modeling, value, routing, verification, and self-monitoring should be independently observable and connect through the sparse event contract.

## RISA-Aligned Structural Memory

RISA-aligned work treats knowledge as more than a collection of facts or one scalar per connection. A memory record can include entities, events, roles, temporal relations, context, confidence, provenance, and revision history. Repeated experiences can then share reusable structural resources while preserving their exceptions.

The long-term research model is:

```text
Memory   = Experience + Structure + Evidence + Revision
Concept  = a reusable invariant supported across experiences
Learning = local reorganization that explains new experience with bounded change
```

Three different maturity levels must not be confused:

- The current RISA kernel implements explicit, human-readable action/effect patterns, structural interpolation, typed feedback, contradiction handling, and evidence-linked prediction.
- Registered later experiments study explicit structural invariants, canonical deltas, transformation memory, and minimum-description-length selection.
- Anonymous concepts formed only by repeated local circuit reuse—without semantic labels, global graph-isomorphism search, or global clustering—remain an unproven Phase 39 hypothesis.

If anonymous reuse is implemented later, a latent assembly will count as useful structure only when it recurs across independent contexts, improves held-out prediction or compression, survives deterministic replay, has a targeted ablation effect, and stays within state and event budgets. Opaque activity alone is not evidence of an emergent concept.

## Learning And Plasticity

Implemented and experimental learning paths include local STDP-family updates, reward modulation, predictive error traces, direct memory updates, structural plasticity, replay, and consolidation. Repetition-dependent memory work models the observation that repeated, spaced, and successfully retrieved experiences may become more stable, while contradictions and excessive repetition must not receive automatic reinforcement.

Several biologically inspired ideas are intentionally isolated until evidence is stronger:

- time-dependent effective interactions generated from recent spike state;
- multi-contact and dendritic edge microcircuits;
- overlapping spatiotemporal expert fields without fixed expert boundaries;
- evidence-preserving migration between evolving learning systems;
- explicit structure-plus-delta memory compared with differences derived from shared/non-shared active resources;
- usage-driven anonymous latent structures and emergent hierarchies.

These are research candidates, not descriptions of the production runtime.

## Current Status

Roadmap labels have strict meanings:

- `[Done]` implemented in the current codebase, with the evidence stated by the relevant gate.
- `[Next]` the highest-priority unfinished work.
- `[Later]` planned research that is blocked on prerequisites, independent evidence, or an explicit operator decision.

Current high-level state:

| Status | Area | Evidence boundary |
| --- | --- | --- |
| [Done] | Sparse CPU-first Python/Rust runtime, local plasticity, managed memory, evaluation, and release tooling | Implemented and covered by repository tests/gates |
| [Done] | RISA structural interpolation, predictive structural feedback, contradiction freeze, and Event Memory boundary checks | Explicit typed structures; not anonymous concept emergence |
| [Done] | Repetition-dependent consolidation contract | Observed-only; not connected to production recall |
| [Done] | Structured-edge and checkpoint-cache registered experiments | Observed-only; promotion remains closed where gates or independent evidence are missing |
| [Next] | Human source-alignment review for six hash-bound Phase 34 excerpts | Required before semantic delayed-recall preregistration |
| [Later] | Temporal effective interactions, emergent expert fields, architecture evolution, structural invariant/delta memory, and anonymous local reuse | Must follow the dependency and preregistration order in the roadmap |
| [Later] | Physical energy claims | Proxy metrics are not joule measurements; reopening requires an explicit operator decision |

For the authoritative status, dependencies, negative results, and acceptance gates, see [doc/ROADMAP_NEXT_LEVEL.md](doc/ROADMAP_NEXT_LEVEL.md).

## What SARA Optimizes

The primary target is useful task performance per event and energy cost, not dense throughput in isolation. Until paired physical measurements exist, the project reports explicit software proxies such as sparse event work, bounded state units, update counts, route work, latency, and ANN-style reference cost.

Proxy advantages must be labeled as proxies. They must not be presented as measured power or joule superiority.

## Installation

Python 3.10 or newer and a working Rust toolchain are required for the integrated extension build.

```bash
git clone https://github.com/matsushibadenki/sara-engine-project.git
cd sara-engine-project
python3 -m pip install -e .
```

If Rust core changes are not reflected in the installed package, rerun `python3 -m pip install -e .`.

Optional ANN reference and visualization dependencies are not required by the normal CPU-first runtime:

```bash
python3 -m pip install -e '.[ann-reference]'
python3 -m pip install -e '.[visualization]'
```

## Quick Start

Interactive chat with a saved memory model:

```bash
sara-chat --model models/distilled_sara_llm.msgpack
```

Train dialogue memory from managed JSONL input:

```bash
sara-train data/raw/chat_data.jsonl --model models/distilled_sara_llm.msgpack
```

Example JSONL:

```json
{"user": "こんにちは", "sara": "こんにちは。SARAです。"}
{"user": "What is SARA?", "sara": "SARA is a local sparse spiking research engine."}
{"user": "SARA是什么？", "sara": "SARA是一个本地稀疏脉冲研究引擎。"}
```

The integrated research and data CLI is `scripts/sara_cli.py`:

```bash
python3 scripts/sara_cli.py db-status
python3 scripts/sara_cli.py db-import data/raw/example.txt --category document --lang en
python3 scripts/sara_cli.py db-export
python3 scripts/sara_cli.py train-self-org
python3 scripts/sara_cli.py train-curriculum --stage small --dry-run
python3 scripts/sara_cli.py inspect-memory
python3 scripts/sara_cli.py eval-research-benchmark-suite --dry-run
python3 scripts/sara_cli.py eval-own-latent-learning --no-history-update
python3 scripts/sara_cli.py eval-neuromorphic-capability-matrix
```

Subword SNN language-model training with optional compact checkpoint encoding:

```bash
python3 scripts/train/train_snn_lm.py \
  --corpus data/processed/corpus.txt \
  --save-dir models/snn_lm_pretrained \
  --turboquant
```

See [doc/TOOLS.md](doc/TOOLS.md) and [doc/SARA-Engine_Training_Manual.md](doc/SARA-Engine_Training_Manual.md) before running a full training or evaluation cycle.

## Validation

Run the full repository test suite:

```bash
pytest -q
```

Run the main research and release evidence paths separately when their required fixtures are available:

```bash
python3 scripts/sara_cli.py eval-research-benchmark-suite --dry-run
python3 scripts/eval/real_data_external_validity.py
python3 scripts/eval/real_data_external_validity_ladder.py
python3 scripts/eval/release_soak.py --profile release --include-accuracy
python3 scripts/eval/release_gate.py
```

A passing synthetic or observed-only benchmark proves only its registered scope. Promotion decisions must also inspect source isolation, independent evidence, negative controls, event/state budgets, deterministic replay, and unresolved failure conditions.

## Managed Output Policy

New generated files must stay in the repository's managed locations:

- source data and imported records: `data/raw/`;
- temporary preprocessing artifacts: `data/interim/`;
- finalized datasets and fixtures: `data/processed/`;
- reports, preregistrations, and scratch work: `workspace/`;
- final model artifacts: `models/`.

Do not write generated artifacts to the repository root or create ad hoc output directories. New code should use [src/sara_engine/utils/project_paths.py](src/sara_engine/utils/project_paths.py) and validate output paths before writing.

## Repository Map

- `src/sara_engine/core` and the Rust extension: low-level spiking runtime primitives.
- `src/sara_engine/neuro`: neuron, synapse, dendritic, and structured-edge mechanisms.
- `src/sara_engine/learning`: local plasticity, prediction, consolidation, and observed-only learners.
- `src/sara_engine/memory`: sparse, hippocampal, long-term, event, and checkpoint memory components.
- `src/sara_engine/risa`: explicit structural memory and prediction kernel.
- `src/sara_engine/multimodal`: typed structural binding and verification.
- `src/sara_engine/agent`: bounded agent, planning, tool, and verification loops.
- `src/sara_engine/evaluation`: preregistration, benchmark, promotion, and release evidence.
- `scripts/sara_cli.py`: integrated operator and research CLI.
- `tests`: unit, contract, benchmark, and gate coverage.

## Documentation

- [Documentation Hub](doc/SARA-Engine_Documentation_Hub.md): canonical documentation entry point.
- [Next-Level Roadmap](doc/ROADMAP_NEXT_LEVEL.md): current implementation order, research dependencies, and acceptance gates.
- [Architecture Review](doc/ARCHITECTURE_REVIEW.md): runtime spine, adoption rules, and evidence limits.
- [Implemented Features](doc/IMPLEMENTED_FEATURES.md): completed feature inventory.
- [Tool Reference](doc/TOOLS.md): CLI, evaluation, release, and maintenance commands.
- [Training Manual](doc/SARA-Engine_Training_Manual.md): data and training workflows.
- [Release Checklist](doc/RELEASE_CHECKLIST.md): release validation requirements.

## License

MIT License.
