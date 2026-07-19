# SARA Engine Next-Level Roadmap

この文書は、v1.1の実装完了後にSARAを「構造を保持するSNN」から「構造を検証しながら成長させる知能」へ進めるための新しいロードマップです。

現行の完了状況は [ROADMAP.md](ROADMAP.md) と [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md) を正とします。この文書は新しい研究・製品レベルの目標だけを扱い、完了済みの実装履歴は再掲しません。

## Status Convention

- [Done] Current code and evidence already satisfy the item.
- [Next] Highest-priority unfinished work that can be executed internally.
- [Later] Planned work that must wait for stronger evidence, independent data, or an explicit operator decision.

## Level-2 North Star

SARAを、次の閉ループを持つCPU-firstの構造型エージェントへ進化させます。

```text
observe
  -> sparse eventization
  -> anonymous concept formation
  -> structural prediction
  -> typed prediction error
  -> source-aware verification
  -> bounded structural edit
  -> Event Memory consolidation
  -> later retrieval and correction
```

到達目標は、単発の正解率ではなく、未知状況・矛盾・情報更新・長い時間差に対して、適切に予測し、分からないときに停止し、検証済みの知識だけを再利用できることです。

## Non-Negotiable Boundaries

- Runtime learning remains backpropagation-free, sparse-event-first, CPU-first, and bounded-state.
- Durable knowledge requires observed evidence, provenance, verification, contradiction checks, and Event Memory admission.
- Structural proposals never mutate durable graph state directly.
- Prediction error is not treated as a numeric gradient; it becomes an auditable typed edit proposal.
- Unknown, unsupported, contradictory, or oscillating hypotheses must abstain or freeze.
- Dense ANN/LLM components may be offline comparison references, but not hidden runtime dependencies.
- Physical joule measurement remains `[Later]` and indefinitely pending. No proxy may be promoted to a physical-energy claim.
- All artifacts remain under `data/`, `workspace/`, or `models/`.

## Current Baseline

- [Done] v1.1 release gate: `15/15`.
- [Done] Research product completion: `17/17`.
- [Done] Full Python test suite: `1343 passed` in the project Python 3.10 environment.
- [Done] Phase 7 independent split: 24 train and 24 evaluation records with isolation checks passing.
- [Done] Phase 8 local pretrained embedding reference: `nomic-ai/nomic-embed-text-v1`.
- [Done] Phase 17-20 observed-only mechanisms: resonance credit, Event Memory, liquid dynamics, Semantic Echo.
- [Done] RISA structural interpolation, typed predictive feedback, contradiction freeze, and Event Memory boundary benchmarks.
- [Done] Existing independent migration manifest: 6 records from 2 source domains with preserved revision and horizon metadata.
- [Later] External meter-based joule evidence, FAISS/cross-encoder optional references, and broader independent long-horizon evidence.

## Promotion Ladder

Every major mechanism must pass this ladder before it can affect production defaults:

1. **Unit safety:** deterministic tests cover normal, missing, contradictory, malformed, and budget-exceeded inputs.
2. **Observed-only fixture:** a small frozen benchmark proves the intended behavior and records negative results.
3. **Independent data:** source, revision, domain, collection time, license, and near-duplicate metadata are preserved.
4. **Long-horizon test:** the mechanism is evaluated after revision, replay, forgetting, and delayed correction.
5. **Ablation:** compare the mechanism with the simplest existing control under equal data, state, event, and task budgets.
6. **Operational gate:** CLI, report, summary, repair action, and managed output boundary are connected.
7. **Promotion review:** production defaults change only after all previous evidence is reproducible.

## Phase 21: Structural Cognition Core

**Goal:** turn RISA structural proposals into compositional reasoning over verified subgraphs without allowing uncontrolled graph growth.

### [Done]

- Structural interpolation from independent verified relation evidence.
- Confidence interpolation with bounded source and revision metadata.
- Typed predictive feedback: `strengthen_relation`, `cut_relation`, `request_more_evidence`, and `freeze_subgraph`.
- Oscillation and contradiction freeze with rollback state.
- Event Memory admission and reactivation boundary.
- Bounded verified subgraph composition and relation-signature structural analogy.
- Phase 21 fixture and CLI benchmark for supported composition, unsupported abstention, analogy, and durable-mutation blocking.

### [Next]

- Add bounded subgraph composition for two or more verified relations.
- Add structural analogy between concept neighborhoods using relation-type-aware overlap.
- Add explicit `create_provisional_node` proposals without durable admission.
- Add rollback snapshots for multi-edit proposals, not only individual edits.
- Benchmark compositional queries, unsupported neighbors, revision conflict, and relation-type confusion.

### Acceptance Gate

- Compositional answer accuracy improves over single-edge retrieval on independent held-out cases.
- Unsupported compositions abstain rather than inventing a missing edge.
- Proposals remain capped by active-subgraph size, edit count, state bytes, and event cost.
- A rejected or rolled-back proposal leaves the prior verified graph byte-equivalent.

## Phase 22: Continual Horizon Intelligence

**Goal:** measure whether SARA can learn a sequence of revisions without catastrophic interference or hidden leakage.

### [Done]

- Added the observed-only `eval-continual-horizon` benchmark with fixed-initial-state, Event Memory, resonance-credit, and structural-feedback profiles.
- Added frozen fixture cases for 10, 30, and 100 ordered episodes with source revision, contradiction, delayed verification, and bounded distractor pressure.
- Reports now expose revision uptake latency, useful recall, contradiction blocking, abstention integrity, state growth, maintenance cost, replay count, independent source hashes/domains, and fixed/linear/logarithmic retention profiles without making a physical-energy claim.
- Delayed-verification and protected-knowledge retention ablations now expose correction latency and catastrophic-interference checks under distractor pressure.

### [Next]

- Repeat delayed correction and forgetting ablations with independently collected records and equal event/state budgets.
- Replace the compact fixture source metadata with independently collected manifests and domain-transfer rows.
- Keep all horizon reports observed-only until independent source coverage is sufficient.

### Acceptance Gate

- Later corrections are incorporated without erasing unrelated verified knowledge.
- Useful recall after correction exceeds the frozen control.
- Contradiction repair completes within a bounded number of episodes.
- State growth and maintenance cost remain within declared ceilings.
- No train/evaluation source hash, revision, domain, or near-duplicate leakage is present.

## Phase 23: Multimodal Structural World Model

**Goal:** bind vision, audio, text, and temporal signals through shared sparse events while preserving modality-local evidence.

### [Next]

- Extend the current common spike-space path with modality-specific event provenance.
- Represent cross-modal relations as hypotheses first, not immediate concepts.
- Add asynchronous binding windows and modality dropout tests.
- Add cross-modal contradiction cases where one modality is noisy or delayed.
- Connect verified bundles to RISA subgraphs and Event Memory episodes.

### Acceptance Gate

- Cross-modal binding improves over independent modality controls under equal event/state budgets.
- Missing modality predictions are uncertainty-aware and may abstain.
- False cross-modal links remain below the declared ceiling.
- Payloads remain separable after binding; shared IDs cannot collapse evidence.
- Durable cross-modal concepts require repeated source-backed verification.

## Phase 24: Causal and Counterfactual Structure

**Goal:** distinguish temporal correlation from causal support and use verified structures for bounded counterfactual reasoning.

### [Next]

- Separate `precedes`, `correlates_with`, `causes_candidate`, and `causes_verified` relation types.
- Require intervention-like or contrastive evidence before promoting causal hypotheses.
- Add counterfactual branch records with bounded depth and explicit rollback.
- Return supporting event paths and alternative explanations with every causal answer.
- Freeze causal promotion under source conflict or unstable feedback.

### Acceptance Gate

- Temporal order alone never creates `causes_verified`.
- Counterfactual answers preserve source and context boundaries.
- Unsupported causal questions abstain.
- Branch count, depth, event cost, and serialized state remain bounded.

## Phase 25: Verifiable Agent Loop

**Goal:** connect perception, memory, planning, action, and outcome correction into a safe bounded agent loop.

### [Next]

- Use RISA structural predictions to propose actions, not to bypass policy checks.
- Require a plan trace, expected outcome, risk estimate, and rollback action before execution.
- Store action outcomes as Event Memory candidates only after observation and verification.
- Add task interruption, goal change, stale-plan, and unexpected-outcome cases.
- Compare action selection with and without structural feedback under equal event budgets.

### Acceptance Gate

- The agent follows goal changes without retaining obsolete action plans as facts.
- Invalid actions are rejected or rolled back.
- Action traces identify the concept, evidence, prediction, and outcome that caused each decision.
- Unknown state triggers safe abstention or information gathering.

## Phase 26: Self-Evaluation and Research Memory

**Goal:** make SARA able to detect when its own evidence is weak and prioritize the next useful experiment.

### [Next]

- Add a bounded research journal for hypothesis, evidence, result, confidence, negative result, and next test.
- Link every promotion proposal to the benchmark report and source manifest that justified it.
- Detect repeated failed experiments and suppress duplicate work.
- Generate repair priorities without automatically editing ROADMAP or production configuration.
- Track metric drift across benchmark runs and distinguish data drift from code regression.

### Acceptance Gate

- The system never self-promotes an unverified hypothesis to durable knowledge.
- Negative results remain queryable and alter future experiment priority.
- Every suggested action is reproducible from a managed command and artifact path.
- Human approval remains required for roadmap and production-default changes.

## Phase 27: Portable Sparse Runtime

**Goal:** make the verified SARA runtime portable across Python, Rust, and constrained edge targets without changing semantics.

### [Later]

- Define a canonical sparse event IR and versioned state migration contract.
- Add Python/Rust replay equivalence for Event Memory, RISA proposals, and predictive feedback.
- Add low-memory, ARM64, and optional neuromorphic capability profiles.
- Measure latency, state bytes, event count, and deterministic replay across targets.
- Keep hardware energy claims separate from software portability evidence.

### Acceptance Gate

- Canonical event traces replay to equivalent decisions across supported runtimes.
- State migrations are explicit, reversible, and reject incompatible versions.
- Unsupported hardware capabilities fail clearly without corrupting state.

## Phase 28: Level-2 Promotion Review

**Goal:** decide whether SARA has become a stronger general-purpose research prototype rather than a collection of passing mechanisms.

### [Later]

- Run the complete promotion ladder on Phases 21-27.
- Require at least one independent long-horizon workload and one multimodal workload.
- Compare against frozen controls, not only against prior SARA versions.
- Publish a capability matrix with accuracy, abstention, revision, state, event cost, latency, and provenance quality.
- Keep physical joule claims explicitly unresolved while Phase 6 remains pending.

### Level-2 Promotion Criteria

- Structural reasoning improves on held-out compositional tasks.
- Continual revision improves without catastrophic interference.
- Cross-modal binding is useful and source-auditable.
- Causal and counterfactual outputs remain conservative and traceable.
- Agent action loops are verifiable and rollback-safe.
- Runtime remains sparse, bounded, CPU-first, and backpropagation-free.
- All negative results and unresolved evidence gaps are visible.

## Immediate Execution Order

1. [Done] Implement bounded RISA subgraph composition and structural analogy.
2. [Next] Build the 10/30/100-episode continual horizon benchmark.
3. [Next] Connect the horizon benchmark to Event Memory retention profiles.
4. [Next] Add multimodal structural contradiction and missing-modality cases.
5. [Later] Activate causal/counterfactual promotion only after horizon evidence is stable.
6. [Later] Activate portable runtime work after canonical event IR is frozen.
7. [Later] Reopen physical-energy evidence only by explicit operator decision.

## Required Managed Outputs

- `data/processed/benchmark_fixtures/next_level_structural_cases.jsonl`
- `data/processed/benchmark_fixtures/continual_horizon_cases.jsonl`
- `workspace/evaluation/next_level_structural_benchmark.json`
- `workspace/evaluation/continual_horizon_benchmark.json`
- `workspace/evaluation/multimodal_structural_benchmark.json`
- `workspace/evaluation/causal_counterfactual_benchmark.json`
- `workspace/evaluation/next_level_research_journal.jsonl`
- `workspace/evaluation/next_level_promotion_gate.json`

## Review Rule

新しいアイデアは、まずこのロードマップの対象Phase、仮説、最小実験、失敗条件、昇格条件を明記してから実装します。実装が先行しても、独立データ・否定例・再現可能な評価がない限り、知能の向上や本番昇格とは扱いません。
