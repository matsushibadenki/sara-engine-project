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
- Any time-dependent effective interaction must remain a bounded local sparse cache derived from event/state history. It must not become a hidden dense weight matrix, a backpropagation path, or durable knowledge without the normal verification boundary.
- Tokenizer acceleration must preserve token IDs, boundaries, ordering, special-token behavior, normalization, and decode semantics exactly. A faster but semantically different tokenizer is a separate experiment, not an implementation optimization.
- Physical joule measurement remains `[Later]` and indefinitely pending. No proxy may be promoted to a physical-energy claim.
- All artifacts remain under `data/`, `workspace/`, or `models/`.

## Adopted Kimi K3 Design References

The user-provided *Kimi K3: Open Frontier Intelligence Technical Report* was reviewed as an external Transformer-system reference. SARA does not inherit its architecture, benchmark scores, scaling claims, or hardware assumptions. The following principles are adopted only after translation into sparse, local, bounded mechanisms:

- **Bounded state dynamics:** Kimi Delta Attention lower-bounds log-decay and SiTU-GLU smoothly caps activations. SARA may reuse the principle of finite decay/activation ranges for local temporal state, fatigue, and plasticity, but not KDA's dense recurrent matrices, learned projections, backpropagation, or Tensor Core implementation.
- **Selective access across depth:** Attention Residuals retrieves selected prior-layer representations instead of uniformly compressing every layer into one residual stream. SARA will test a block-bounded sparse analogue that selects a small number of prior event/state summaries by local overlap, resonance, and verified context. Dense attention and softmax over all layers are excluded.
- **Quantile-calibrated sparse routing:** Kimi K3's Quantile Balancing separates dispatch bias from expert mixture weights, applies the update only to the next step, and estimates large-batch quantiles with fixed-cost histograms. SARA will test whether bounded sparse histograms can calibrate WTA expert thresholds without gradients while preserving fixed Top-k routing and deterministic event budgets.
- **State-aware immutable checkpointing:** Kimi K3 decouples fine prefix-hash boundaries from coarse physical storage, uses chained hashes, restores read-only checkpoints into private copy-on-write state, and invalidates dependent cache groups atomically. SARA will translate this into canonical sparse-event checkpoints whose reuse requires matching state-schema, source revision, tokenizer/runtime identity, and all required state groups.
- **Replay instead of snapshot proliferation:** Kimi K3 caches compact projected inputs and replays an accepted speculative prefix instead of snapshotting a large recurrent state at every draft position. SARA will evaluate compact canonical event/input replay for rollback, while retaining exact digest validation against the pre-action state.
- **Resumable long-horizon agents:** partial rollouts, pause/resume, fork-for-judging, and periodic snapshots are relevant to Phase 25. Any SARA adoption must preserve isolated reversible state, bounded staleness, deterministic tool-call/result pairing, and no external side effects before approval.

The following Kimi K3 mechanisms remain comparison-only and are not SARA runtime candidates: dense KDA/MLA attention, full or block softmax AttnRes, LatentMoE projections, Muon optimization, RL/distillation gradients, quantization-aware backpropagation, GPU kernels, expert-parallel communication, and reported model/benchmark scaling results.

## Adopted Multi-Contact and Dendritic Design References

The multi-synapse discussion is adopted as a research question, not as a claim that biological detail automatically improves SARA. Primary studies support four relevant premises:

- Reconstructed rat neocortical neuron pairs showed multiple anatomical contacts and a correlation between putative contact count and functional release sites ([Journal of Neuroscience, 2010](https://pubmed.ncbi.nlm.nih.gov/20107071/)).
- A recent computational study found that contact-specific nonlinear transmission across parallel synapses can increase classification capacity in its tested models ([PLOS Computational Biology, 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012285)).
- Nearby inputs on one thin dendritic branch can integrate differently from separated inputs, supporting branch-local computational subunits rather than one global sum ([Nature Neuroscience, 2004](https://www.nature.com/articles/nn1253)).
- Timing-dependent cooperation and distance-dependent competition can organize synaptic clusters in a biophysical model ([Nature Communications, 2021](https://www.nature.com/articles/s41467-021-23557-3)), but another controlled modeling study found that small random clusters need not materially change somatic responses ([Nature Communications, 2020](https://www.nature.com/articles/s41467-020-15147-6)).
- The TwinProp preprint reports that a detailed pyramidal-cell simulation lost substantial 4-bit parity accuracy when voltage-dependent dendritic mechanisms, morphology, or NMDA-mediated integration were ablated. It also reports broader recruitment of dendritic compartments as task dimensionality increased ([bioRxiv preprint, 2026](https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full)). This is useful as a capacity-probe and causal-ablation reference, but it is not peer-reviewed evidence for a biologically plausible learning rule.

SARA therefore adopts only the testable principle that one sparse outer relation may contain a bounded set of typed contacts and branch-local interactions. It does not assume that more contacts, biological realism, or moving computation inside edges will improve accuracy, reduce total complexity, or save physical energy. Existing `Synapse`, `DendriticTree`, dendritic-feedback, and structural-plasticity code are implementation references and controls, not evidence that this hypothesis has passed.

TwinProp's training path remains comparison-only: it fits a DNN digital twin, differentiates through that twin, optimizes synaptic strengths and locations with Adam and repeated restarts on GPUs, and validates the result in a detailed NEURON simulation. SARA will not adopt that optimizer, the dense twin, detailed ion-channel simulation, PCA as a runtime mechanism, thousands of contacts per case, or the reported DNN-decoder upper bounds. The transferable question is narrower: whether deterministic bounded branch-local coincidence and a slow local interaction state add value over equal-budget linear contacts when all arms use the same minimal spike-count/threshold readout.

## Current Baseline

- [Done] v1.1 release gate: `15/15`.
- [Done] Research product completion: `17/17`.
- [Done] Full Python test suite: `1398 passed` in the project Python 3.10 environment after Phase 23 cross-modal hypothesis hardening.
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
- Added source-backed `create_provisional_node` feedback proposals; unknown nodes remain review-only and cannot enter the durable graph directly.
- Added bounded multi-edit staging with deterministic graph snapshots, exact-digest rollback validation, edit/node/edge budgets, and byte-equivalent restoration after any late edit failure.
- The Phase 21 benchmark now verifies provisional-node isolation, successful two-edit staging, and atomic rollback after a partially staged batch.

### [Next]

- Repeat compositional and analogy benchmarks with independent held-out cases.

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
- Added `eval-continual-horizon-external`, which validates the existing independent migration manifest before any Phase 22 promotion review; it deliberately keeps `promotion_allowed=false`.
- The external gate now separates manifest quality from horizon promotion: the current 0-5 record span passes input validation but remains blocked until every domain reaches the 10/30/100 buckets.
- Added `build-continual-horizon-collection-request`, which writes managed per-domain collection targets and required provenance fields without fabricating missing records.

### [Next]

- Repeat delayed correction and forgetting ablations with independently collected records and equal event/state budgets.
- Expand the external manifest to 10/30/100 episode coverage per domain before independent long-horizon claims.
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

### [Done]

- Added the observed-only `eval-phase23-structural-fusion` benchmark.
- Added modality-local provenance verification for supported, missing, contradictory, and temporally misaligned evidence.
- Missing-modality outputs remain provisional, contradictions and delay conflicts abstain, and every decision blocks direct durable mutation.
- Structural fusion decisions now gate the existing Event Memory multimodal admission boundary: only `verify_cross_modal_structure` may promote, while all other decisions freeze.
- The production ingest path now requires a receipt-backed structural decision; missing, stale, or forged decisions freeze before Event Memory admission.
- Cross-modal claims are separated from modality-local labels so relations such as vision `dog` and audio `bark` can support one shared claim.
- Added `eval-phase23-external-multimodal`, which audits independent provenance, rights metadata, source and near-duplicate uniqueness, decision-family coverage, verifier accuracy, and Event Memory admission integrity.
- Added `build-phase23-multimodal-collection-request`; missing evidence now produces managed aligned, missing-modality, contradiction, and temporal-misalignment targets instead of relaxing the gate.
- The independent Phase 23 gate is connected to Level-2 promotion review, capability-matrix provenance, evidence-bound human approval, and scale-up readiness.
- Added a bounded cross-modal hypothesis ledger. A verified bundle first becomes `provisional_hypothesis`; at least two verified observations from distinct source identities are required for `eligible_for_review`.
- Reused modality source references, duplicate observations, stale receipts, mismatched claim keys, missing provenance, and state-budget overflow are rejected before support is counted.
- Contradictory structural evidence freezes the hypothesis. Even review-eligible hypotheses retain `durable_mutation_allowed=false` and enter the RISA adapter as `verified=false`.
- The Phase 23 benchmark now covers provisional-to-review transition, contradiction freeze, and the non-promoting RISA hypothesis boundary.
- Added explicit asynchronous boundary cases at 31 ms and 33 ms around the 32 ms structural binding limit.
- Added symmetric modality-dropout coverage: both missing audio and missing vision remain provisional rather than being verified or durably admitted.
- Connected receipt-backed verified bundles to bounded Event Memory episodes with deterministic multi-source hashes; rejected bundles, incomplete evidence, single-modality inputs, and episode-budget overflow remain isolated.
- Added a bounded RISA subgraph projection for verified bundle members. The projection preserves verified evidence edges and source-revision context without directly mutating durable graph state; unverified and edge-budget-exceeded inputs return an empty projection.

### [Next]

- Collect the five-case minimum across at least two independent recording/source domains and pass the external Phase 23 gate.

### Acceptance Gate

- Cross-modal binding improves over independent modality controls under equal event/state budgets.
- Missing modality predictions are uncertainty-aware and may abstain.
- False cross-modal links remain below the declared ceiling.
- Payloads remain separable after binding; shared IDs cannot collapse evidence.
- Durable cross-modal concepts require repeated source-backed verification.

## Phase 24: Causal and Counterfactual Structure

**Goal:** distinguish temporal correlation from causal support and use verified structures for bounded counterfactual reasoning.

### [Done]

- Added the observed-only `eval-phase24-causal` benchmark and bounded causal reasoner.
- Temporal or correlational evidence remains `causes_candidate`; intervention plus contrastive support is required for `causes_verified`.
- Source conflict and unsupported counterfactuals abstain, branch depth/count are bounded, and durable mutation remains blocked.
- Causal output is connected to Event Memory admission: only `causes_verified` is eligible, while temporal candidates and abstentions are rejected as unverified.
- Intervention and contrastive evidence can now promote `causes_candidate`; callers no longer need to pre-label evidence as `causes_verified`.
- Counterfactual results now contain deterministic branch records with bounded depth, branch count, event cost, and serialized state bytes. Each branch retains supporting event paths, context tags, alternative explanations, and a non-durable rollback action.
- Added explicit counterfactual rollback. Rollback returns a separate `rolled_back` result, preserves the original staged records unchanged, and consumes budget reserved during branch staging.
- Every causal answer now exposes supporting source/event paths and alternative explanations. Source conflict and unstable feedback both freeze causal promotion before Event Memory admission.

### [Next]

- Repeat causal and counterfactual evaluation with independent intervention records and delayed feedback revisions.

### Acceptance Gate

- Temporal order alone never creates `causes_verified`.
- Counterfactual answers preserve source and context boundaries.
- Unsupported causal questions abstain.
- Branch count, depth, event cost, and serialized state remain bounded.

## Phase 25: Verifiable Agent Loop

**Goal:** connect perception, memory, planning, action, and outcome correction into a safe bounded agent loop.

### [Done]

- Added the observed-only `eval-phase25-agent-loop` benchmark and bounded plan decision layer.
- Structural predictions require a valid sparse plan trace, expected outcome, rollback action, acceptable risk, and current goal match.
- Invalid, high-risk, stale-goal, or rollback-less plans are rejected; observed outcomes remain non-durable candidates.
- Verified observed outcomes now enter the existing Event Memory admission path; rejected plans and mismatched outcomes produce no candidate.
- Unexpected outcomes now produce an explicit rollback requirement and are excluded from Event Memory admission.
- `SaraAgent` now exposes the bounded plan, verified outcome, causal reasoning, and Event Memory admission path.
- Successful outcome text requires a source-backed observation receipt before it can become a memory candidate.
- Added an equal-event-envelope action-selection ablation. The control and structural-feedback arms scan and pay for the same candidate and feedback events; the control masks feedback content while the structural arm may use only verified, stable, source-backed feedback.
- Action-selection traces now retain the selected concept, evidence reference, structural prediction, expected outcome, feedback sources, per-action scores, and side-effect status. Malformed candidates, policy-ineligible actions, and event/state budget overflow abstain.
- The observed-only fixture records a positive structural-feedback selection change under equal four-event budgets without promoting it to an independent action-quality claim.
- Added a bounded transactional tool-state adapter. Accepted plans may stage only allow-listed, verified, source-backed JSON state edits under edit, event, and state-byte ceilings; tool goal, expected outcome, and rollback action must match the plan.
- Expected outcomes commit the staged operational state atomically. Unexpected outcomes, late staged-edit failures, malformed values, and budget violations reject or roll back to the exact pre-transaction digest. External side effects remain disabled.
- Added immutable resumable candidate execution with explicit `pause`, `resume`, `fork_for_judging`, and `snapshot` transitions. Resume requires exact goal, plan, source revision, state digest, remaining event budget, and sandbox checkpoint identity.
- Judging forks are read-only copy-on-write views and cannot mutate their source trajectory. The observed-only Phase 25 benchmark verifies exact resume, judging-fork isolation, and stale-revision rejection.
- Added indexed typed pairing for bounded parallel tool calls. Calls require canonical zero-based indexes, unique IDs, declared JSON result types, and fixed event/state ceilings; missing, duplicate, reordered, failed, tool-mismatched, or type-mismatched results block commit.
- The transactional adapter revalidates the complete paired batch immediately before its existing atomic commit path. The observed-only Phase 25 benchmark verifies exact pairing commit and byte-equivalent state preservation after reordered-result rejection.
- Added deterministic partial-rollout scheduling with fixed event slices, equal-turn candidate selection, bounded trajectory/state capacity, and a queue-coverable pause-staleness ceiling.
- Every paused trajectory retains and revalidates its goal, plan, source revision, state digest, remaining event budget, and sandbox checkpoint identity. The observed-only Phase 25 benchmark verifies `A -> B -> A` scheduling, bounded wait ticks, exact budget completion, and stale-source rejection.
- Added an isolated reversible tool sandbox that owns a canonical private JSON state rather than a caller mapping. Verified tool commits advance only the sandbox revision; external side effects and durable mutation remain disabled.
- Sandbox checkpoints bind sandbox ID, revision, and state digest. Restore rejects foreign, future, or tampered checkpoints and the observed-only Phase 25 benchmark verifies private commit, caller-state isolation, and byte-equivalent rollback.

### [Next]

- Repeat the isolated reversible sandbox boundary with independently recorded action/outcome traces before enabling any external side effect.

### Acceptance Gate

- The agent follows goal changes without retaining obsolete action plans as facts.
- Invalid actions are rejected or rolled back.
- Action traces identify the concept, evidence, prediction, and outcome that caused each decision.
- Unknown state triggers safe abstention or information gathering.

## Phase 26: Self-Evaluation and Research Memory

**Goal:** make SARA able to detect when its own evidence is weak and prioritize the next useful experiment.

### [Done]

- Added `eval-next-level-promotion-review` with a bounded research journal and a separate promotion gate.
- The review links Phase 21-25 report artifacts, records negative results, emits reproducible next actions, and never self-promotes production defaults.
- Physical joule measurement remains excluded and independent long-horizon/multimodal evidence remains visible as unresolved.
- Added evidence-bound human approval manifests; any report change invalidates stale approval.
- Added stable experiment fingerprints over hypothesis, evidence status, negative results, next tests, and promotion state. Repeated unchanged failures are detected from the bounded research-journal window.
- Duplicate successful internal benchmark reruns are suppressed when the same blocked experiment recurs, while required independent-data collection actions remain active. Changed evidence or failure content creates a new experiment fingerprint.
- Identical journal entries are no longer appended repeatedly; legacy entries remain readable and the first fingerprinted migration record is preserved.
- Added per-phase metric snapshots with separate numeric-metric, data-provenance, and benchmark-implementation fingerprints. The research journal retains the bounded prior snapshot needed for comparison.
- Drift classification distinguishes `stable`, `data_drift`, `code_change`, `code_regression`, `mixed_drift`, and unexplained deterministic-repeat degradation. Data changes alone are not mislabeled as code regressions.
- A detected code or nondeterministic regression blocks Level-2 promotion pending review; baseline and stable runs remain non-promoting unless all independent evidence and human approval gates also pass.

### [Next]

- Repeat drift classification across independently collected revisions and reviewed implementation changes before assigning production thresholds.

### Acceptance Gate

- The system never self-promotes an unverified hypothesis to durable knowledge.
- Negative results remain queryable and alter future experiment priority.
- Every suggested action is reproducible from a managed command and artifact path.
- Human approval remains required for roadmap and production-default changes.

## Phase 27: Portable Sparse Runtime

**Goal:** make the verified SARA runtime portable across Python, Rust, and constrained edge targets without changing semantics.

### Adopted Design Reference

- Use [Gigatoken](https://github.com/marcelroed/gigatoken/) as an external implementation reference for semantics-preserving CPU tokenization, not as a production dependency or an inherited performance claim.
- Adopt its relevant principles: direct Rust byte-stream processing, reduced Python crossings, safe-boundary parallelism, branch-minimized pretokenization, and reuse of frequent `pretoken -> token IDs` results.
- Preserve Gigatoken's negative engineering lesson that extra classification buffers and multiple full passes may lose to one-pass processing through increased memory traffic. Every optimization must be measured end to end rather than accepted from operation counts.
- Do not transfer published GB/s or speedup values to SARA. Hardware, tokenizer family, corpus, input size, thread count, cache state, and Python/API boundary must be frozen locally.

### [Done]

- Added the canonical sparse IR v1 representation, deterministic event ordering, replay digest, and same-version state migration checks.
- Hardened canonical IR ingestion with finite/range validation, unique event identities, unknown-field rejection, bounded event/tag/text sizes, version-bound idempotent migration, and canonical JSON serialization.
- Added managed positive and negative conformance vectors with a frozen replay digest so a future Rust implementation must match the Python contract rather than merely match another mutable run.
- Added `eval-phase27-portable-runtime` as an observed-only readiness gate; it explicitly does not claim Python/Rust equivalence.
- The gate surfaces existing Rust sparse-primitive output-equivalence evidence separately while keeping canonical IR replay equivalence unresolved.
- Added a frozen Python tokenizer snapshot contract, tokenizer fingerprint, and bounded exact pretoken cache with entry, byte, and per-entry token ceilings. Source-tokenizer mutation cannot alter an existing adapter snapshot.
- Added multilingual conformance fixtures covering English, Japanese, Simplified Chinese, mixed scripts, whitespace, punctuation, emoji/combining marks, and long-entry bypass.
- Added `eval-phase27-tokenizer-acceleration`; it verifies token IDs, decode output, canonical sparse-spike replay digests, repeated-input equivalence, malformed UTF-8 rejection, cache reuse, eviction, and state ceilings without changing production defaults.
- Added a deterministic Rust scalar BPE merge reference that consumes Python-defined pretokens, preserves the frozen vocabulary/merge/unknown-token contract, rejects duplicate merge pairs, and matches Python token IDs across all eight multilingual conformance cases.
- The initial cache timing remains diagnostic and preserves slower first-pass latency as a negative result rather than promoting acceleration. Rust scalar equivalence is observed, but no Rust performance or production-path claim is made; Gigatoken remains explicitly unobserved.

### [Later]

- Add Python/Rust replay equivalence for Event Memory, RISA proposals, and predictive feedback.
- Extend the Rust scalar correctness reference into an optional accelerated candidate only after equal-budget cold/warm timing, boundary-call, RSS, and downstream replay measurements are implemented.
- Freeze separate tokenizer identities for the custom `SaraTokenizer` BPE format and the standard `tokenizers` JSON used by `SpikeTokenizer`; never silently reinterpret one format as the other.
- Add TTL/generation invalidation to the existing fingerprint-keyed bounded pretoken cache; retain its entry, logical-byte, token-count, eviction, and long-tail bypass ceilings.
- Preserve input order under safe-boundary parallelism and reject boundaries that split UTF-8 sequences, special tokens, normalization units, or pretokenizer-dependent spans.
- Permit CPU SWAR/SIMD byte scanning and ARM64/x86 specialization because these are sparse preprocessing operations, not matrix computation. A deterministic scalar path remains the correctness reference and fallback.
- Keep direct-file acceleration restricted to managed source paths and deterministic document separators. Interactive short-text inference must retain a low-overhead path rather than paying batch setup cost.
- Add low-memory, ARM64, and optional neuromorphic capability profiles.
- Measure latency, state bytes, event count, and deterministic replay across targets.
- Add immutable canonical sparse-event checkpoints at semantic boundaries. Decouple chained-hash lookup granularity from physical checkpoint storage, restore hits into private copy-on-write state, and require all declared state groups to agree on one boundary.
- Atomically invalidate sibling checkpoints when any required group, source revision, tokenizer/runtime fingerprint, or state schema becomes incompatible. A partial multi-group hit must never be replayable.
- Compare full-state-per-step rollback with compact canonical-input replay. Compact replay is acceptable only when it reconstructs the accepted state and pre-action digest exactly under rejected, truncated, reordered, and duplicated-event controls.
- Keep hardware energy claims separate from software portability evidence.

### Tokenizer Acceleration Minimum Experiment

Use a frozen four-arm comparison:

1. current Python `SaraTokenizer`;
2. Python tokenizer with the same bounded pretoken cache policy;
3. Rust implementation using the exact same vocabulary, merges, and pretokenization contract;
4. optional Gigatoken compatibility mode only for tokenizer formats it can validate exactly.

- Require exact token IDs, token boundaries/byte offsets where available, ordering, special-token placement, unknown-token behavior, and decode round trips before recording any performance result.
- Cover English, Japanese, Simplified Chinese, mixed-script text, whitespace variants, punctuation, emoji, combining marks, malformed UTF-8 rejection, very long tokens, repeated hot pretokens, and long-tail cache pollution.
- Measure cold and warm cache separately for single short requests, small batches, and managed multi-document streams.
- Freeze source bytes, tokenizer fingerprint, normalization, document separators, core/thread count, affinity when available, environment fingerprint, cache budget, warm-up count, repetitions, and run order.
- Report input MB/s, tokens/s, p50/p95 latency, Python-boundary calls, cache hit/useful-reuse/eviction rates, peak cache bytes, peak RSS, and end-to-end `text -> token IDs -> sparse spike events` latency and event count.
- Keep Semantic Echo token-boundary accuracy, multilingual endpoint coverage, abstention, and downstream sparse-event decisions in the equivalence gate. Throughput cannot compensate for a semantic regression.

### Tokenizer Acceleration Failure Conditions

- Reject an optimization after any token ID, boundary, special-token, normalization, ordering, or decode mismatch.
- Reject it if improvement appears only after using more threads, more cache bytes, different input splitting, different tokenizer semantics, or a different measured boundary.
- Reject the cache if construction, lookup, hashing, or eviction cost harms preregistered short-text latency, or if long-tail input exceeds its state ceiling.
- Reject parallel splitting if output depends on chunk size, worker count, document order, or Unicode boundary placement.
- Keep an unsupported tokenizer family on the deterministic reference path. Do not silently approximate Janome, SentencePiece, WordPiece, or custom SARA behavior.
- Do not treat tokenizer throughput as SNN reasoning accuracy, model-token throughput, end-to-end agent latency, or physical-energy evidence.

### Acceptance Gate

- Canonical event traces replay to equivalent decisions across supported runtimes.
- State migrations are explicit, reversible, and reject incompatible versions.
- Unsupported hardware capabilities fail clearly without corrupting state.
- Accelerated tokenization is byte-for-byte and token-for-token equivalent on all frozen multilingual conformance cases.
- The selected accelerated path improves preregistered end-to-end tokenization and spike-event latency under equal thread/cache/input budgets without exceeding bounded state or RSS ceilings.
- Scalar fallback, cache invalidation, and safe-boundary replay remain deterministic across cold/warm runs and supported CPU profiles.
- Gigatoken remains optional until its compatibility mode passes the same local conformance and downstream Semantic Echo checks; no headline benchmark is adopted as SARA evidence.

## Phase 28: Level-2 Promotion Review

**Goal:** decide whether SARA has become a stronger general-purpose research prototype rather than a collection of passing mechanisms.

### [Done]

- Added `eval-level2-capability-matrix` to compare structural, continual, multimodal, causal, and agent capabilities with their evidence scope.
- The matrix explicitly reports unresolved independent-data gaps and keeps promotion blocked until human review.
- Independent horizon and multimodal gates are now separate mandatory checks; fixture-only Phase 23 evidence cannot satisfy the multimodal requirement.
- Human approval is bound to both external reports and becomes stale whenever either evidence set changes.

### [Later]

- Run the complete promotion ladder on Phases 21-27.
- Require at least one independent long-horizon workload and one multimodal workload.
- Compare against frozen controls, not only against prior SARA versions.
- Publish a capability matrix with accuracy, abstention, revision, state, event cost, latency, and provenance quality.
- Keep physical joule claims explicitly unresolved while Phase 6 remains pending.

## Phase 29: Scale-Up Experimental Validation

**Goal:** after ROADMAP_NEXT_LEVEL evidence and human review are complete, test whether the bounded mechanisms remain useful at a larger but controlled scale.

### [Done]

- Added an immutable managed preregistration contract for the scale-up protocol. It requires at least four uniquely fingerprinted domains, frozen fixture and environment fingerprints, the three fixed comparison profiles, 1,000/10,000 episode buckets, five fixed unique replicate seeds, equal source/fixture/episode ordering and state/event budgets, all eight metric thresholds, and a CPU-only non-energy policy.
- Added `register-scale-up-preregistration`; identical registration is idempotent, while any attempt to replace an existing protocol is rejected and requires a new experiment identity.
- Upgraded `eval-scale-up-readiness` so all three evidence gates, a managed preregistration path, and an exact canonical protocol fingerprint must pass before `ready_to_execute` can become true. The command remains planning-only and does not execute the experiment.

### [Next]

- Run 1,000 and 10,000 episode buckets across at least four domains.
- Compare frozen control, Event Memory, and structural-feedback Event Memory with five replicates per condition.
- Freeze source, fixture, environment, state, and event budgets before execution.
- Measure revision uptake, useful recall, catastrophic interference, abstention, state growth, event cost, latency, and provenance completeness.
- Keep the scale-up command planning-only until the Phase 28 promotion gate and both independent horizon and multimodal gates pass.

### Acceptance Gate

- Scale-up runs are reproducible from managed manifests and pre-registered thresholds.
- All controls use equal data, state, event, and replicate budgets.
- No metric improvement is promoted without independent source coverage and human review.
- Physical energy remains out of scope.

### Level-2 Promotion Criteria

- Structural reasoning improves on held-out compositional tasks.
- Continual revision improves without catastrophic interference.
- Cross-modal binding is useful and source-auditable.
- Causal and counterfactual outputs remain conservative and traceable.
- Agent action loops are verifiable and rollback-safe.
- Runtime remains sparse, bounded, CPU-first, and backpropagation-free.
- All negative results and unresolved evidence gaps are visible.

## Phase 30: Temporal Effective Interaction

**Goal:** test whether SARA can improve temporal-task accuracy by deriving short-lived local effective interactions from recent sparse activity without discarding spike timing or introducing a dense ANN runtime.

### Research Hypothesis

- Treat an effective interaction as an observable, temporary cache value over an already active sparse edge:

```text
g_ij(t) = f(
  bounded pre/post spike history,
  firing order and interval,
  delay and phase relation,
  short-term excitation/fatigue state,
  verified structural context
)
```

- `g_ij(t)` is not durable knowledge and is not assumed to be a biological synaptic weight. It expires, is recomputed only for locally active edges, and cannot bypass source-aware verification or Event Memory admission.
- The intended benefit is preservation and reuse of temporal evidence, not conversion of the runtime into an ANN. ANN-like accuracy, wall-clock latency, event cost, temporal representation quality, and physical energy are separate claims and must be measured separately.
- RISA may provide verified context for selecting eligible sparse relations, but unverified correlation, synchrony, or repeated co-activation cannot become a durable RISA relation through this cache.

### [Later]

- Reuse Phase 19 fixed/multi-timescale/liquid dynamics as the state-only controls instead of duplicating them; Phase 30 must prove that materializing a cache adds value beyond those temporal states.
- Add a bounded recent-event state containing timestamp, order, interval, delay, phase bucket, excitation/fatigue, expiry, and provenance reference.
- Materialize an effective interaction only after a local active-edge reuse threshold is met. Otherwise compute directly from temporal state so cache-construction cost cannot dominate one-shot events.
- Invalidate or freeze cached interactions after context revision, contradiction, temporal-distribution shift, expiry, or unstable oscillation.
- Keep fixed sparse synapses as the production control. No dense matrix calculation, backpropagation, GPU dependency, or ANN inference layer is permitted in the candidate runtime.
- Add deterministic replay and exact cache invalidation traces to the canonical sparse IR portability contract before any cross-runtime claim.
- Replace unbounded decay parameterizations in the candidate with an explicitly finite scalar range and compare smooth saturation with hard clipping. The bounds must be preregistered and cannot be chosen after observing task scores.

### Minimum Experiment

- Use frozen tasks where timing is necessary: firing-order reversal, equal-count/different-interval sequences, delayed response, phase/synchrony discrimination, irregular event gaps, and context revision.

The frozen ablation compares four equal-budget arms:

1. fixed sparse SNN;
2. history-averaged static interaction that intentionally removes fine timing;
3. temporal state without materialized effective-interaction cache;
4. temporal state with bounded local effective-interaction cache.

- Include shuffled-time, phase-shifted, duplicate-event, stale-cache, contradiction, unseen-context, and no-reuse negative controls.
- Freeze data, active-edge ceiling, event/state/cache-byte budgets, timestamp resolution, replicate seeds, environment fingerprint, and thresholds before execution.
- Report accuracy or F1, calibration and abstention, timing-sensitivity delta, revision recovery, stale-cache harm, cache hit/useful-reuse rate, construction and invalidation event cost, state/cache bytes, deterministic replay, and wall-clock latency.
- Compare with an offline ANN reference only as a labeled accuracy/latency control under the existing fairness rules. It must not become a runtime dependency or evidence of physical-energy efficiency.

### Failure Conditions

- Reject the hypothesis if improvement disappears against the temporal-state-only arm; this means the cache added no value beyond state dynamics.
- Reject it if gains come from additional data, active edges, state bytes, events, latency allowance, or replicate selection rather than the mechanism.
- Reject it if history averaging performs equally on timing-required cases, because the fixture did not demonstrate a temporal requirement.
- Reject it if cache construction/invalidation costs exceed useful reuse, stale cache increases harmful decisions, or revision recovery exceeds its preregistered bound.
- Reject it if state or active interactions become dense or unbounded, replay becomes nondeterministic, or a transient interaction leaks into durable RISA/Event Memory state without verification.
- Do not convert software event counts or latency into a physical-energy claim.

### Acceptance Gate

- The cache arm improves preregistered timing-required metrics over fixed, history-averaged, and temporal-state-only controls under equal budgets and at least five fixed replicates.
- Shuffled-time and phase-shift controls cause the expected bounded degradation or abstention, proving that the mechanism uses temporal structure rather than event counts alone.
- Context revision invalidates stale interactions within a bounded number of events without erasing unrelated verified state.
- Construction cost is amortized by useful local reuse, while cache size, active-edge count, event cost, state bytes, and latency remain under frozen ceilings.
- Independent temporal workloads and human review are required before changing production defaults. Physical energy and general ANN-parity claims remain unresolved.

## Phase 31: Repetition-Dependent Memory Consolidation

**Goal:** reproduce the bounded phenomenon that repeated, successfully recalled, and appropriately spaced sparse activity becomes easier to retrieve, while preventing repetition from being mistaken for truth.

### Research Hypothesis

- Maintain a sparse local state per eligible memory containing retrieval strength, consolidation stability, last activation time, repetition count, bounded source evidence, and contradiction history.
- Apply saturating local potentiation to repeated support, a larger consolidation gain for spaced retrieval, slow stability-dependent forgetting, and local depression after contradiction.
- Keep retrieval strength separate from verification strength. Repeating one source may improve accessibility, but only newly verified independent sources may increase verification strength.
- This is an observed-only memory mechanism. It uses no backpropagation, dense matrix operation, GPU path, or automatic durable-knowledge admission.

### [Done]

- Added a deterministic bounded sparse consolidation trace with explicit memory, source, identifier-byte, and event ceilings. Updates are local to one memory and use no backpropagation, dense matrix operation, or GPU path.
- Added saturating retrieval potentiation, spaced successful-recall consolidation, stability-dependent projected forgetting, contradiction-driven depression, and deterministic weakest-trace eviction.
- Retrieval strength and verification strength are separate. Exact source references are stored only as bounded SHA-256 identities; repeating one verified source cannot increase verification, while newly verified distinct sources can.
- Added eight frozen controls and `eval-phase31-repetition-consolidation`. The observed-only report covers one-shot, massed, spaced, delayed forgetting, contradiction, duplicate-source, distinct-source, saturation, long-tail capacity, event-budget, local-isolation, and deterministic-replay checks.
- Added an explicit-default-off `CandidateRepetitionReranker` over the verified Event State Cache retrieval-result contract. Only traces with nonzero verified-source evidence are eligible; the adapter cannot admit entries or mutate durable cache fields.
- Added six frozen delayed-recall and interference controls plus `eval-phase31-repetition-reranking`. The disabled and candidate arms use identical source events/state and are charged for the same bounded candidate scan; spaced verified recall can reorder close verified candidates, while unverified repetition and interference receive zero boost.
- Production Event Memory, RISA admission, and retrieval ranking remain unchanged.

### [Later]

- Run independent delayed-recall and interference workloads with equal event/state budgets before choosing thresholds.
- Test the default-off adapter with actual independently collected Event Memory and sleep-replay outputs without allowing retrieval strength to bypass verification receipts.
- Add persistence and cross-runtime replay only after the state schema and invalidation policy are reviewed.

### Minimum Experiment

- Compare equal-count one-shot, massed-repetition, and spaced-retrieval schedules under the same state and event budgets.
- Verify that spaced successful recall produces higher final stability than massed repetition, repeated support saturates, and an unused trace weakens after a delayed clock advance.
- Verify that contradiction lowers retrieval strength, one repeatedly verified source cannot inflate verification strength, and distinct verified sources can increase it only up to a fixed ceiling.
- Report retrieval strength, stability, verification strength, source count, event count, evictions, deterministic snapshots, and state-budget integrity.

### Failure Conditions

- Reject the mechanism if massed repetition grows without saturation, spaced retrieval has no advantage, unused memories do not forget, or contradiction strengthens a trace.
- Reject it if duplicate source evidence increases verification strength, if retrieval strength directly grants durable knowledge, or if state/source/event counts exceed their ceilings.
- Reject it if results depend on dictionary iteration order, wall-clock time, global dense updates, backpropagation, or a GPU.
- Do not treat synthetic repetition fixtures as evidence of human-equivalent memory, improved reasoning accuracy, or physical-energy efficiency.

### Acceptance Gate

- All frozen positive and negative controls pass deterministically under equal budgets.
- Retrieval and verification remain separate in the public state and evaluation report.
- Sparse state stays bounded during long-tail pollution and unrelated memories are not globally rewritten by a local repetition.
- Production integration remains blocked pending independent delayed-recall workloads, interference testing, provenance review, and explicit human approval.

## Phase 32: Sparse Depth Retrieval and Quantile-Calibrated Routing

**Goal:** test whether SARA can improve sparse information flow across processing depth and prevent expert collapse without importing dense Transformer attention or gradient-based MoE training.

### Research Hypothesis

- Partition a processing trace into a small fixed number of depth blocks. Each block emits a bounded sparse summary containing active event IDs, source/verification references, residual error type, and local state digest.
- At a later block, select at most `k` prior summaries using sparse overlap, resonance, recency, and verified-context compatibility. The selected summaries become read-only candidate inputs; they do not mutate durable knowledge.
- Maintain an expert-specific dispatch threshold separate from expert output strength. Estimate threshold corrections from bounded histograms of observed sparse routing margins and apply them only to the next routing window.
- Histogram counts are additive and deterministic, but calibration remains local and CPU-first. No dense token-by-expert score matrix, auxiliary loss, gradient, GPU collective, or dynamic expert replication is allowed.

### [Later]

- Add a block-bounded sparse depth-state store with explicit block, summary-width, selected-summary, event, and serialized-byte ceilings.
- Add a scalar deterministic routing calibrator over sparse candidate margins. Freeze bin range, bin count, target load, update interval, tie policy, and next-window-only activation before evaluation.
- Keep the existing fixed WTA plus homeostasis router as the production control. Depth retrieval and quantile calibration begin as independent default-off candidates.
- Reuse canonical sparse IR digests for each block summary and reject stale source revisions, incompatible schemas, missing state groups, or checkpoint identity mismatches.

### Minimum Experiment

Compare four equal-budget arms:

1. sequential sparse residual accumulation with existing homeostasis;
2. block summaries without selective retrieval;
3. selective sparse depth retrieval with existing homeostasis;
4. selective sparse depth retrieval plus histogram-calibrated routing.

- Include delayed dependency, irrelevant intermediate blocks, contradictory prior state, stale checkpoint, repeated dominant expert, dying expert, sparse candidate omission, histogram overflow, tie, and distribution-shift controls.
- Report task accuracy or abstention, retained dependency recall, contradictory-state rejection, expert load range/Gini, dead-expert count, route churn, calibration lag, histogram/state bytes, event cost, deterministic replay, and CPU latency.
- Test block counts and histogram bins selected before execution. A gain obtained only by increasing selected summaries, active experts, state bytes, or event cost is invalid.

### Failure Conditions

- Reject sparse depth retrieval if it does not beat the block-summary control, retrieves unsupported or stale state, or collapses into selecting every prior block.
- Reject quantile calibration if load improves by changing expert output weights, current-window routes, Top-k count, or available candidate edges rather than the next-window dispatch threshold.
- Reject it if a sparse candidate that never appears is treated as observed, histogram bounds silently clip material mass, route ordering becomes nondeterministic, or calibration oscillates after distribution shift.
- Do not transfer Kimi K3's MoE balance, scaling efficiency, throughput, or accuracy claims to SARA.

### Acceptance Gate

- Selective depth retrieval improves a preregistered delayed-dependency metric over sequential and block-only controls while preserving contradiction abstention.
- Histogram calibration reduces expert-load imbalance and dead-expert incidence over existing homeostasis without degrading held-out task quality beyond the frozen tolerance.
- State, histogram, selected-summary, active-expert, event, and latency ceilings all pass across at least five fixed replicates.
- Independent workloads and human review are required before either candidate can alter production routing.

## Phase 33: Structured Edge Microcircuits

**Goal:** test whether a bounded microstructure inside each active sparse relation can represent timing, polarity, context, and local cooperation more efficiently than adding outer nodes, layers, and routes.

### Research Hypothesis

- Replace the assumption that one active pair has exactly one scalar connection with a bounded contact bundle:

```text
outer relation A -> B
  contact 0: excitatory, branch 1, delay bucket 0, fast plasticity
  contact 1: inhibitory, branch 1, delay bucket 2, fatigue state
  contact 2: excitatory, branch 3, verified role/context, slow plasticity
```

- Each contact remains a local scalar-state event processor. A fixed small number of contacts may share a branch compartment, where coincidence, order, inhibition, and fatigue are combined before one bounded signal reaches the postsynaptic node.
- Contact-to-contact influence is an explicit sparse micrograph with hard contact, branch, internal-interaction, event, and byte ceilings. No dense vector-valued edge, per-edge MLP, HyperNetwork, backpropagation, matrix multiplication, GPU dependency, or recursively nested unbounded graph is permitted.
- Separate three kinds of change: contact-state plasticity, contact add/prune within an existing outer relation, and outer-graph rewiring. A local contact may not create a new durable outer relation or semantic role without the existing provenance, contradiction, and Event Memory/RISA admission boundaries.
- The central claim is an efficiency hypothesis, not a biological claim: richer bounded edges may allow fewer outer nodes/routes at the same held-out quality. Total contacts and internal interactions count as graph complexity and may not be hidden by reporting only the smaller outer graph.

### [Done]

- Added an immutable Phase 33 preregistration contract that freezes all five mechanism arms, three outer-graph simplification levels, 17 positive/negative case families, five unique replicate seeds, total-resource accounting, thresholds, and CPU-only/default-off execution policy.
- Added canonical protocol fingerprinting, managed-workspace enforcement, idempotent identical registration, immutable conflict rejection, and the `register-phase33-structured-edge-preregistration` CLI. This registers protocols only; it does not execute the candidate runtime.
- Added the frozen 17-case observed-only Phase 33 fixture and managed draft builder. The registered fixture fingerprint is `b8ef6e09c3da69aaa3e0dc626f8f4b630ad4b0a53acf62ad255647b9e8e0c230`; the current CPU/Python environment fingerprint is `20f6d07671c8afb52b54ecf1d617fc46ebdafb37965d4e00981c59e7481bc193`.
- Registered immutable experiment `phase33-structured-edge-observed-v1` with protocol fingerprint `63168395ac7f5235d4173072fb52823712b89895e16610856ced77adf70d64ff`. A repeated registration preserves the identical manifest. This synthetic fixture registration is a protocol milestone, not accuracy, simplification, biological, independent-data, or energy evidence.

### [Later]

- Preregister the state schema for contact identity, polarity, branch slot, delay bucket, short-term state, plasticity class, typed role/context, source revision, expiry, and contradiction state.
- Build a deterministic bounded contact-bundle runtime. Replace random branch assignment in the legacy dendritic prototype with a frozen tie rule and canonical replay order.
- Reuse Phase 30 temporal state for delay/phase controls, Phase 31 consolidation for repetition controls, and the existing structural-plasticity controller for outer-route controls rather than duplicating their claims.
- Add separate contact and outer-route budgets. Contact growth must require observed local reuse plus verified prediction-gain support; contradiction, inactivity, resource pressure, or unstable oscillation must prune or freeze locally.
- Keep the existing single-contact sparse route as the production default. Structured edges remain default-off until both the mechanism ablation and the outer-graph simplification test pass.
- Preserve the immutable `phase33-structured-edge-observed-v1` protocol. Register TwinProp-derived controls only as a new follow-up experiment identity after the existing five-arm run; do not alter its frozen fixture, thresholds, seeds, or fingerprint.
- Add a TwinProp-inspired follow-up with intact bounded branches, passive/linear branches, topology-collapsed aggregation, no slow coincidence state, and point-neuron controls. Use identical input events, contact/state budgets, fixed readout, replicate seeds, and tuning allowance in every arm.
- Compare deterministic contact placement with a preregistered location-shuffle control, and sweep task interaction order. Measure branch participation, branch-event overlap, local-state saturation, event cost, and serialized bytes using sparse scalar counters; do not import dense voltage PCA into the runtime.

### Minimum Experiment

Run two preregistered experiment families.

**Mechanism ablation under one equal total-resource envelope:**

1. one scalar contact per outer relation;
2. multiple contacts with linear summation;
3. typed contacts with independent delay, polarity, and short-term state but no contact interaction;
4. typed contacts grouped into bounded branch compartments with local interaction;
5. branch compartments plus bounded contact add/prune.

**Outer-graph simplification sweep:**

- Compare the best structured-edge candidate with the scalar control while reducing outer nodes, routes, and processing depth in fixed steps.
- Match total source events, contact plus route state bytes, executed local interactions, latency ceiling, and replicate seeds. A smaller outer graph is not simpler if hidden contact/internal-interaction cost increases beyond the frozen total budget.

Use tasks where one scalar edge is deliberately ambiguous: same pre/post pair with different delay-dependent meanings, excitatory/inhibitory context switching, same spike count with different order, branch-local coincidence, partial contact failure, repeated support, delayed contradiction, and outer-route deletion. Include shuffled contact identity, shuffled branch placement, duplicated contacts, missing contacts, stale source revision, all-linear, all-same-delay, no-reuse, and random-cluster controls.

Report held-out accuracy/F1, calibration and abstention, timing/order sensitivity, contradiction recovery, contact-failure tolerance, outer nodes/routes/depth, contacts per pair, branch slots, active internal interactions, total state bytes, event cost, latency, add/prune churn, deterministic replay, and an iso-quality total-complexity frontier.

**TwinProp-inspired follow-up after the immutable Phase 33 experiment:**

- Freeze a new experiment identity rather than extending `phase33-structured-edge-observed-v1`.
- Use the same minimal SARA readout in all arms. A deep decoder may be reported only as an offline reference and cannot supply state, features, tuning feedback, or acceptance evidence.
- Test whether the intact bounded-branch arm increasingly recruits distinct branch slots as interaction order rises, while remaining within fixed contact, local-state, event, byte, and latency ceilings.
- Treat a gain that depends on gradient-selected contact locations, extra restarts, unequal tuning, a dense analysis path, or larger hidden input expansion as a failed transfer.

### Failure Conditions

- Reject structured edges if branch interaction does not beat typed independent contacts, or if multiple linear contacts perform equally; the extra microstructure is then unnecessary.
- Reject the simplification claim if quality is preserved only by increasing total contacts, internal interactions, state bytes, events, latency, data, or tuning trials.
- Reject any accounting result that relabels intermediate outer nodes as hidden edge contacts without reducing the preregistered total-complexity measure.
- Reject the candidate if contacts become dense or unbounded, duplicate contacts collapse into an implicit larger scalar weight, source/role provenance cannot be isolated per contact, or local contact plasticity creates durable semantic structure directly.
- Reject nondeterministic branch assignment, unstable contact churn, failure to recover after contradiction, or dependence on dictionary order, wall-clock timing, gradients, matrices, or GPU execution.
- Reject the TwinProp-inspired transfer if passive, topology-collapsed, no-slow-state, and intact arms cannot be separated under the same fixed readout, or if shuffled placement performs equivalently to the claimed structured placement.
- Do not infer human-like dendritic computation, general parameter efficiency, ANN parity, or physical-energy savings from synthetic fixtures.

### Acceptance Gate

- Branch-local structured contacts improve the preregistered ambiguous-relation metric over scalar, linear-multi-contact, and typed-independent-contact controls under the same total resource envelope and at least five fixed replicates.
- At least one preregistered outer-graph reduction point preserves held-out quality and abstention within tolerance while reducing the total, fully counted complexity measure rather than only outer node count.
- Missing, duplicated, reordered, stale, contradictory, and capacity-exceeded contacts fail safely; replay and add/prune decisions are deterministic.
- Gains transfer to independent temporal and structural workloads. Human review is required before changing outer topology or production routing defaults.

## Immediate Execution Order

1. [Done] Implement bounded RISA subgraph composition and structural analogy.
2. [Done] Build the observed-only 10/30/100-episode continual horizon benchmark.
3. [Done] Connect the horizon benchmark to Event Memory retention profiles.
4. [Done] Add multimodal structural contradiction and missing-modality cases.
5. [Next] Collect independent 10/30/100 horizon and multimodal evidence.
6. [Later] Complete Python/Rust canonical replay equivalence.
7. [Later] Run the exact-tokenization four-arm conformance and bounded-cache ablation before selecting any accelerated tokenizer path.
8. [Later] Prototype Phase 30 temporal effective interactions only after preregistering the four-arm equal-budget ablation.
9. [Done] Implement the observed-only Phase 31 repetition-dependent consolidation contract without connecting it to production recall.
10. [Later] Preregister the Phase 32 four-arm sparse depth-routing experiment before implementing either candidate.
11. [Done] Implement the immutable Phase 33 structured-edge preregistration contract and managed CLI.
12. [Done] Freeze the Phase 33 observed-only fixture and CPU environment fingerprints, then register the immutable protocol before candidate execution.
13. [Later] Run the immutable Phase 33 experiment before registering the TwinProp-inspired follow-up as a separate protocol.
14. [Later] Reopen physical-energy evidence only by explicit operator decision.

## Required Managed Outputs

- `data/processed/benchmark_fixtures/next_level_structural_cases.jsonl`
- `data/processed/benchmark_fixtures/continual_horizon_cases.jsonl`
- `workspace/evaluation/next_level_structural_benchmark.json`
- `workspace/evaluation/continual_horizon_benchmark.json`
- `workspace/evaluation/phase23_structural_fusion_benchmark.json`
- `data/processed/autobot/phase23_independent_multimodal_manifest.jsonl`
- `workspace/evaluation/phase23_external_multimodal_gate.json`
- `workspace/autobot/phase23_multimodal_collection_targets.json`
- `workspace/evaluation/phase24_causal_benchmark.json`
- `workspace/evaluation/phase25_agent_loop_benchmark.json`
- `workspace/evaluation/next_level_research_journal.jsonl`
- `workspace/evaluation/next_level_promotion_gate.json`
- `workspace/evaluation/next_level_human_approval.json`
- `workspace/evaluation/scale_up_experiment_readiness.json`
- `data/processed/benchmark_fixtures/phase27_tokenizer_conformance_cases.jsonl`
- `workspace/evaluation/phase27_tokenizer_acceleration_benchmark.json`
- `data/processed/benchmark_fixtures/phase30_temporal_effective_interaction_cases.jsonl`
- `workspace/evaluation/phase30_temporal_effective_interaction_benchmark.json`
- `data/processed/benchmark_fixtures/phase31_repetition_consolidation_cases.jsonl`
- `workspace/evaluation/phase31_repetition_consolidation_benchmark.json`
- `data/processed/benchmark_fixtures/phase31_repetition_reranking_cases.jsonl`
- `workspace/evaluation/phase31_repetition_reranking_benchmark.json`
- `data/processed/benchmark_fixtures/phase32_sparse_depth_routing_cases.jsonl`
- `workspace/evaluation/phase32_sparse_depth_routing_benchmark.json`
- `data/processed/benchmark_fixtures/phase33_structured_edge_cases.jsonl`
- `workspace/evaluation/phase33_structured_edge_environment.json`
- `workspace/evaluation/phase33_structured_edge_preregistration_draft.json`
- `workspace/evaluation/phase33_structured_edge_preregistration.json`
- `workspace/evaluation/phase33_structured_edge_benchmark.json`
- `workspace/evaluation/phase33_edge_simplification_ablation.json`
- `data/processed/benchmark_fixtures/phase33_twinprop_ablation_cases.jsonl`
- `workspace/evaluation/phase33_twinprop_ablation_preregistration.json`
- `workspace/evaluation/phase33_twinprop_ablation_benchmark.json`

## Review Rule

新しいアイデアは、まずこのロードマップの対象Phase、仮説、最小実験、失敗条件、昇格条件を明記してから実装します。実装が先行しても、独立データ・否定例・再現可能な評価がない限り、知能の向上や本番昇格とは扱いません。
