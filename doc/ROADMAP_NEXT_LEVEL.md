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

- Correctness and normal runtime learning must not require global gradient backpropagation, end-to-end differentiation, or a stored whole-network backward graph. Runtime learning remains sparse-event-first, CPU-first, and bounded-state.
- Bounded backward information is allowed: bAP-like branch events, recent eligibility/activity traces, prediction error, outcome, reward or neuromodulatory scalars, and replayed consequences may flow to explicit local structures. These signals must not be called gradients unless they are mathematical derivatives.
- Credit assignment should be tested through local temporal eligibility and explicit hierarchical structure before any global-gradient exception: backward information selects or modulates eligible branches, while each synapse/contact/branch updates from locally available state under fixed event and byte budgets.
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

## Adopted Memory Caching Design Reference

The *Memory Caching: RNNs with Growing Memory* preprint is adopted as a long-context memory research reference ([arXiv, 2026](https://arxiv.org/abs/2602.24281)). Its central observation is relevant to SARA: one fixed recurrent state must increasingly compress the past, while caching selected segment-level memory checkpoints can preserve direct access to older information. The paper also makes the cost explicit: retrieving from every cached segment costs `O(NL)`, and segmentation trades recall resolution against compute and storage.

SARA adopts only the bounded, auditable translation:

- form immutable sparse checkpoint summaries at verified semantic boundaries rather than caching every token or a dense hidden-state matrix;
- compare equal-size segments with logarithmic multi-resolution retention instead of assuming either is universally better;
- retrieve at most fixed `k` checkpoints by deterministic sparse overlap, resonance, recency, source revision, and contradiction compatibility;
- preserve checkpoint identity, provenance, source revision, state-schema/runtime fingerprint, event range, and canonical digest through selection and replay;
- enforce hard checkpoint-count, summary-width, byte, selected-checkpoint, event-cost, age, and latency ceilings, with deterministic eviction or merge rules.

The paper's Residual/Gated Residual Memory, Memory Soup, learned projection/softmax router, dense Top-k scoring, matrix-valued memories, MLP memory modules, gradient-based test-time updates, AdamW training, GPU/TPU execution, model sizes, benchmark gains, and throughput claims remain comparison-only. SARA must not average or interpolate checkpoint parameters because that would obscure source isolation and contradictions. A cache hit supplies read-only candidate evidence; it cannot create durable knowledge or bypass RISA/Event Memory verification.

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
- [Done] Full Python test suite: `1656 passed` in the managed Python 3.10 environment after the Phase 34 semantic delayed-recall adapter and evaluator were added.
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
- Audited the current analogy boundary: `StructuralAnalogyEngine` compares only the Jaccard overlap of verified relation-type sets. It does not learn reusable role mappings, share local changes across structures, propose an unseen target/relation, or establish emergent knowledge.
- Audited the current structural-delta boundary: `BoundedStructuralEditTransaction` safely stages `create_provisional_node`, `strengthen_relation`, and `cut_relation` against a snapshot digest, but it does not persist a canonical replayable delta, reconstruct a target from `base + delta`, compare edit sequences, or learn reusable transformations.
- Phase 21 fixture and CLI benchmark for supported composition, unsupported abstention, analogy, and durable-mutation blocking.
- Added source-backed `create_provisional_node` feedback proposals; unknown nodes remain review-only and cannot enter the durable graph directly.
- Added bounded multi-edit staging with deterministic graph snapshots, exact-digest rollback validation, edit/node/edge budgets, and byte-equivalent restoration after any late edit failure.
- The Phase 21 benchmark now verifies provisional-node isolation, successful two-edit staging, and atomic rollback after a partially staged batch.
- Independent held-out structural benchmark over seven cases derived from six human-reviewed IETF/Python documentation records. The three supported multi-hop cases reached `1.0` accuracy versus `0.0` for direct single-edge retrieval; two unsupported cases abstained and two relation-signature analogy decisions were correct.
- The independent gate validates source record/hash bindings, rejects legacy-fixture entity overlap, caps hops/paths/edge work, and keeps every proposal provisional. Its evidence is explicitly limited: the external statements are independent, while their edge decomposition is benchmark-authored rather than autonomously learned.

### [Next]

- Preregister the Phase 37 structural-invariant sharing experiment now that the independent Phase 21 baseline is available.

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
- Treat this phase as evidence for safe binding of a declared cross-modal claim, not evidence that a modality-independent structure has been learned. Phase 42 separately tests whether structure transfers prospectively across modalities without label or pairing leakage.

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
- Added independent Rust parsing, validation, ordering, canonical JSON serialization, Python-compatible Unicode escaping, and SHA-256 replay digest for canonical sparse IR without calling the Python reference implementation.
- Rebuilt the PyO3 extension and observed exact Python/Rust canonical bytes and replay digests across all six frozen valid/invalid conformance cases. This closes canonical IR encoding equivalence only; semantic subsystem decisions remain unresolved.
- Added a bounded portable decision kernel for the shared Event Memory, RISA proposal, and predictive-feedback safety boundary. Python and Rust independently replay verification, contradiction, freshness, capacity, support, and prediction-match signals into canonical admit/reject/freeze/correction decisions.
- Added 18 managed multilingual decision cases and observed exact Python/Rust canonical decision bytes and digest across admission, retrieval, eviction, revision, RISA proposal, and predictive-feedback subsystem labels. This proves only the shared boundary kernel.
- Added frozen Python adapters for actual `CacheAdmissionResult`, `StructuralInterpolationProposal`/`StructuralEditProposal`, and predictive structural-feedback outputs. An end-to-end fixture now exercises real cache admission and contradiction rejection, independently supported RISA proposal generation, and predictive retain/correction generation before obtaining identical Rust decision bytes and digest.
- The initial adapter evidence is repository-generated and narrow. The following independent run broadens source coverage, while retrieval decisions, revisions, eviction decisions, and repeated feedback cycles remain unresolved.
- Added `phase27_independent_decision_replay.py` over six previously collected, provenance-bound external documentation records from `docs.python.org` and IETF RFC 9110. The observed-only run produced six real Event Memory admissions, six post-pressure retrieval decisions, two explicit capacity-eviction decisions, two RISA structural proposals, and five predictive-feedback decisions.
- All 21 independent-source-derived decisions matched Python/Rust canonical bytes and digest (`0b88d1fdcf29c46e231c30c4266a6dde83c6593bd4497e4625e63e1877ac7809`). The four-entry cache retained four records, evicted two, retrieved retained evidence, and abstained for evicted evidence. Structural grouping and prediction transitions remain benchmark-declared, so this does not establish semantic accuracy or complete subsystem equivalence.
- Fixed Event State Cache revision integrity: a newer verified revision replaces `source_revision` and `time_segment` only when it comes from the same source and is not older than the retained entry. Different-source duplicates keep the existing provenance rather than being mislabeled as a revision.
- Added a separately reported three-decision control over one external base record: verified `r1 -> r2` replacement, contradictory `r3` freeze, and oscillating strengthen/cut feedback freeze. Python/Rust digests matched (`54f867488b6b8056d09aeff5c88f52efa1c89e418dce40d408fe86d1cd565261`), but the report binds `independent_evidence=false` because the revision and contradiction are controlled perturbations rather than independently collected facts.
- Added a provenance-bound genuine version-history manifest for CPython `Lib/argparse.py` at official tags `v3.13.11` and `v3.14.6`. The two verified tag commits have distinct content hashes and retain one stable logical source identity.
- Added `phase27_revision_history_replay.py`. The newer official revision replaced the older Event Memory entry without state growth, and the resulting portable `replace_revision` decision matched Python/Rust canonical bytes and digest (`b40bc6ff90909bfafdaa27cb7421860052394f84b874003a9c2528c457f80186`). The report binds `independent_evidence=true` while explicitly limiting the result to one genuine source revision.
- Added a four-revision observed feedback history from official CPython tags `v3.14.3` through `v3.14.6`. Exact `Lib/argparse.py` hashes produce the observed transition path changed, changed, unchanged without interpreting byte changes as semantic quality.
- Added `phase27_observed_feedback_cycle.py`. The three-step predictive path emitted `strengthen_relation`, `request_more_evidence`, and `cut_relation`; all portable decisions matched Python/Rust canonical bytes and digest (`efa0a06dc11e75669d27d56435efefbb79815c9eef182792dc8e62120f8cf6e5`). The report binds `independent_evidence=true`, remains observed-only, and forbids contradiction or semantic-accuracy claims.
- Added a separately sourced explicit conflict from RFC 6350 Section 4 and Verified Technical Erratum 3484. The published ABNF permits a zone suffix in two truncated-time alternatives, while the verified correction removes that suffix from the same alternatives; the manifest freezes the proposition, opposite Boolean polarities, source roles, exact locators, and material hashes.
- Added `phase27_verified_contradiction_replay.py`. Event Memory admitted the published claim, blocked the verified contradictory correction from mutating the retained state, and emitted portable `freeze_revision`. Python/Rust canonical bytes and digest matched (`7ce91c62cbb0eb9e6271adfaf45c25f94af4c0fac2ac4b881c318c32e22f0cea`) with `independent_evidence=true`. This proves one explicit ABNF contradiction only, not general semantic contradiction detection.
- Upgraded `phase27_portable_runtime_readiness.py` to schema v3 and bound all four independent reports into one fail-closed evidence bundle. Every report must have its exact schema, pass its own frozen checks, remain observed-only and production-unchanged, declare independent evidence, preserve a non-empty claim boundary, and match its Python/Rust decision digest.
- Added CLI paths and negative tests for missing reports, synthetic evidence, and digest mismatch. The integrated observed run passed all four evidence entries plus the six canonical-IR and 18 portable-decision conformance cases. Individual claim boundaries remain visible and tokenizer acceleration remains unpromoted.
- Upgraded the tokenizer acceleration benchmark to v2 with one equal 30-input trace for Python and the optional Rust scalar candidate. It records end-to-end cold/warm process time, explicit Python/Rust boundary-call counts, process peak-RSS high-water growth, canonical snapshot state bytes, exact token outputs, and downstream sparse-spike replay digests.
- The observed candidate preserved token and downstream replay identity, used a 2,143-byte snapshot, and added 114,688 peak-RSS bytes in the recorded process. It made 30 Rust calls per pass and was slower than Python (`0.72x` cold and `0.59x` warm in the recorded run). This negative result blocks promotion and supports reducing boundary crossings before further optimization.
- Corrected the conformance interpretation for the frozen Janome pretokenizer: acceleration must reproduce reference decode exactly, while source-text round trip is reported separately because the reference itself normalizes leading/trailing boundary whitespace. Cache bypass is now tested with an independent fixed 64-byte probe rather than depending on language-specific pretoken segmentation.
- Added bounded `batch_tokenize_sara_bpe_pretokens` to the Rust extension. One call accepts at most 1,024 sequences, 65,536 pretokens, and 1,048,576 characters, rejects duplicate merge pairs, and returns one exact token-ID list per input without matrices, gradients, GPU execution, or production routing changes.
- The equal 30-input trace reduced Rust boundary calls from 30 to one per pass while preserving exact tokens and downstream spike replay. A larger 300-input trace with seven timed repetitions produced median speedups of `0.72x` for the scalar boundary and `1.02x` for the batched boundary versus Python in the recorded run.
- Frozen performance promotion requires more than `1.05x` repeated-median speedup. The batched result did not pass, so `rust_batch_performance_promotion_ready=false`; readiness surfaces equivalence separately from performance promotion.
- Added immutable `FrozenSaraBpeTokenizer` in Rust. Vocabulary, ranked merge pairs, and unknown-token identity are validated once at construction; subsequent bounded batch calls transfer only Python-defined pretokens. The object exposes no mutation path and preserves the same sequence, pretoken, and character ceilings.
- Added a Rust-reported build profile so debug extensions cannot satisfy performance readiness. Under an optimized release build, the snapshot candidate remained exact and improved over stateless batching, but independent executions varied around the threshold (`1.0519x`, `1.0544x`, `1.0379x`, and a later integrated `1.0404x`). Promotion therefore remains false despite isolated passing runs.
- Added `phase27_tokenizer_performance_stability.py`, which retains five fresh release-profile process runs without post-observation exclusion. It freezes the fixture digest, tokenizer fingerprint, 300-input/seven-repetition trace, exact replay/resource checks, and exclusive `>1.05x` threshold, then reports median, best, and worst speedup.
- The five-run stability report passed execution integrity but rejected promotion: observed speedups were `1.0616x`, `1.0682x`, `1.0755x`, `1.0710x`, and `0.9200x`; median was `1.0682x` and worst-run speedup was `0.9200x`. All trials remain recorded, production routing is unchanged, and readiness reports performance evidence separately from correctness readiness.

### [Next]

- Preregister the Phase 37 structural-invariant sharing experiment. Phase 21 independent held-out structural baselines and Phase 27 canonical/runtime equivalence are complete; Phase 27 optional tokenizer acceleration remains a retained negative result unless a separately preregistered optimization experiment is approved.

### [Later]

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

### [Done]

- Registered immutable experiment `phase30-temporal-effective-interaction-v1` before implementing the temporal cache candidate. Protocol fingerprint: `564a1b3d9092ebb50ce18540efb6ca420b5793d76a4cb8d46fa252a345d4b37a`.
- The registration freezes four equal-budget arms, thirteen timing/control families, five seeds, timestamp resolution, bounded recent-event and cache state, finite interaction/excitation/fatigue/phase ranges, local reuse threshold, canonical replay order, invalidation rules, resource ceilings, one tuning attempt, and non-promotion boundaries.
- Froze 130 deterministic source-/generator-disjoint histories covering all thirteen families, five replicate seeds, and separate train/evaluation partitions at exactly 256 events per case (33,280 events total). Candidate inputs and evaluator labels are stored separately; input digest is `39d28eb9…e36b`, evaluator-key digest is `a4f77c26…72d0`, and freeze fingerprint is `eed41072…0f17`.
- Added fail-closed fixture validation for exact case/event coverage, source/generator isolation, evaluator-label absence, case identity, canonical digests, and tampering. The freeze generator and registration suite pass 15 focused tests.
- Implemented all four default-off control runtimes over one shared bounded event contract: fixed sparse accumulation, timing-erased history averaging, finite temporal state, and temporal state with a local effective-interaction cache. The candidate uses deterministic LRU edge/cache eviction, direct computation below the frozen reuse threshold, finite excitation/fatigue/phase calculations, exact provenance-linked state, and no production or durable-knowledge mutation.
- Added deterministic context-revision, contradiction, explicit-expiry, cache-expiry, cache-capacity, and active-edge-eviction traces. Across all 520 fixture/arm executions, maximum observed event cost was `960/4096`, state was `13,921/65,536` bytes, cache was `275/16,384` bytes, active edges were `64/64`, cached interactions were `2/32`, cache-arm hits totaled `26,880`, and non-cache controls built zero entries. These are conformance/resource observations, not accuracy or promotion evidence.
- Executed all 520 frozen decisions before loading the evaluator key, then joined labels only through the hash-bound evaluator. Report digest is `bf560bb9…e2a5`; deterministic replay and every resource budget passed, with maximum recorded candidate CPU latency `1.089/50 ms`.
- Retained the registered result as negative evidence: all four arms reached identical timing-required accuracy `0.8` and overall accuracy `0.846154`, so timing-sensitivity delta was `0.0` and neither cache superiority nor history-average degradation was demonstrated. The cache reused work (`cache_hit_rate=0.807692`, `useful_reuse_rate=0.833333`) but failed calibration `0.243084`, justified abstention `0.6`, revision recovery `18/8` events, stale-cache harm `0.333333/0.05`, and timing-perturbation abstention `0.5`.
- Fixed `threshold_gate_passed=false`, `comparative_gate_passed=false`, `mechanism_gate_passed=false`, and `promotion_ready=false`. The one registered tuning attempt is consumed; the fixture, candidate, thresholds, and failed result must not be rewritten. Any repair requires a new experiment identity and must first address why timing-erased controls matched the candidate.

### [Next]

- Use the completed negative Phase 30 controls as the frozen temporal baseline for Phase 39. Do not promote or retune the temporal cache under this experiment identity.
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

**Spectral-neuron design translation:**

- *The Spectral Neuron* ([arXiv:2608.08003v2](https://arxiv.org/abs/2608.08003)) is relevant as a design-and-evaluation reference because it keeps response shape and feature influence mathematically inspectable while model capacity grows. Its actual mechanism is not a SARA runtime candidate: it learns dense symmetric matrices with gradient optimization and solves an eigenvalue problem for each input, with the paper itself reporting cubic eigensolver cost.
- Transfer only the matrix-free principle of **capacity with explicit behavioral guarantees**. For every structured-edge candidate, declare which ordered scalar axes have a justified monotone response within a fixed context, such as verified support, contradiction evidence, staleness, or fatigue. Do not impose monotonicity on timing, phase, context switches, or other axes where a non-monotone response is part of the hypothesis.
- Give each contact and branch-local scalar rule a preregistered maximum output change per unit input/event perturbation. Compose those local limits into an auditable relation-level influence envelope, and record the actually active contacts/branches and their signed contributions. This is the sparse-event analogue of coefficient transparency, not a spectral or eigenvalue claim.
- Treat a larger contact/branch count as useful scaling only if held-out quality improves without weakening the frozen response-shape contract, making the influence envelope vacuous, or exceeding the equal total state/event/latency budget. Matrix dimension, eigenspaces, spectral norms, semidefinite constraints, autograd, Adam, and the paper's benchmark gains remain comparison-only.

### [Done]

- Added an immutable Phase 33 preregistration contract that freezes all five mechanism arms, three outer-graph simplification levels, 17 positive/negative case families, five unique replicate seeds, total-resource accounting, thresholds, and CPU-only/default-off execution policy.
- Added canonical protocol fingerprinting, managed-workspace enforcement, idempotent identical registration, immutable conflict rejection, and the `register-phase33-structured-edge-preregistration` CLI. This registers protocols only; it does not execute the candidate runtime.
- Added the frozen 17-case observed-only Phase 33 fixture and managed draft builder. The registered fixture fingerprint is `b8ef6e09c3da69aaa3e0dc626f8f4b630ad4b0a53acf62ad255647b9e8e0c230`; the current CPU/Python environment fingerprint is `20f6d07671c8afb52b54ecf1d617fc46ebdafb37965d4e00981c59e7481bc193`.
- Registered immutable experiment `phase33-structured-edge-observed-v1` with protocol fingerprint `63168395ac7f5235d4173072fb52823712b89895e16610856ced77adf70d64ff`. A repeated registration preserves the identical manifest. This synthetic fixture registration is a protocol milestone, not accuracy, simplification, biological, independent-data, or energy evidence.
- Added a deterministic evaluation-only structured-edge runtime and `eval-phase33-structured-edge` CLI. The registered five arms, three simplification labels, five seeds, and 17 cases now execute as 1,275 conditions without durable or production mutation.
- Executed the immutable observed-only benchmark with matching fixture/environment/protocol fingerprints. All execution-integrity, replay, state, event, latency, CPU-only, no-matrix, no-backpropagation, and no-GPU checks passed. Branch-local fixture conformance exceeded typed-independent conformance, while linear multi-contact did not exceed the scalar control. These are designed fixture-mechanism observations, not independent accuracy or outer-graph simplification evidence; `promotion_ready` remains false.
- Added a separate 14-case TwinProp-inspired follow-up fixture and immutable preregistration contract. It freezes intact bounded branches, passive linear branches, topology-collapsed aggregation, no slow coincidence state, and point-neuron control under one fixed non-trainable spike-count readout, five seeds, equal tuning/resource budgets, structured-versus-shuffled placement, and interaction orders 2/3/4.
- Registered `phase33-twinprop-ablation-observed-v1` with protocol fingerprint `faa897ae9d52f0315aa2f9261e365591407197145b823a342306c314c93d39e4`, fixture fingerprint `5848945a009898fe1b5dc0a8985c636c575e664e5ddf29697b86fd1e5d96323f`, and environment fingerprint `8ebd79239d50c3645572e89fe710683d08b01e9c669fe3c875b82e6d2c0df821`. Re-registration preserved the identical manifest, and the parent Phase 33 fingerprint remains unchanged. Registration itself did not carry a result or accuracy claim.
- Added a bounded sparse TwinProp-inspired evaluation runtime and `eval-phase33-twinprop-ablation` CLI, then executed all 350 registered conditions. Execution integrity, fingerprints, fixed-readout equality, deterministic replay, abstention, state/event/latency ceilings, and CPU/no-gradient/no-matrix/no-GPU/no-digital-twin boundaries passed.
- On the designed mechanism fixture, the intact arm reached `1.0` conformance, while passive, topology-collapsed, and no-slow-state arms each reached `0.8571`, and the point-neuron control reached `0.7857`. Intact-minus-each-primary-ablation was `0.1429`; active branches increased from `1, 1, 2` across interaction orders `2, 3, 4`; structured-minus-shuffled readout was `1.0`. The preregistered mechanism gate passed, but these are synthetic causal controls rather than independent task accuracy, biological learning, simplification, or energy evidence, so `promotion_ready` remains false.

### [Later]

- Preregister the state schema for contact identity, polarity, branch slot, delay bucket, short-term state, plasticity class, typed role/context, source revision, expiry, and contradiction state.
- Extend the deterministic evaluation runtime to independent workloads before considering any production integration. Replace random branch assignment in the legacy dendritic prototype only through a separately reviewed migration with a frozen tie rule and canonical replay order.
- Reuse Phase 30 temporal state for delay/phase controls, Phase 31 consolidation for repetition controls, and the existing structural-plasticity controller for outer-route controls rather than duplicating their claims.
- Add separate contact and outer-route budgets. Contact growth must require observed local reuse plus verified prediction-gain support; contradiction, inactivity, resource pressure, or unstable oscillation must prune or freeze locally.
- Keep the existing single-contact sparse route as the production default. Structured edges remain default-off until both the mechanism ablation and the outer-graph simplification test pass.
- Transfer the registered mechanism, unchanged and default-off, to independent temporal and structural workloads with provenance and near-duplicate controls before any production review.
- Keep branch participation, branch-event overlap, local-state saturation, event cost, and serialized-byte measurement as sparse scalar counters; do not import dense voltage PCA into the runtime.
- Before the independent transfer, preregister a matrix-free response-shape and influence audit: ordered axes and contexts, per-contact local limits, relation-level bound composition, perturbation sizes, bound-tightness floor, capacity points, and failure thresholds. Constraints must be encoded directly in bounded scalar update rules rather than checked by a dense post-hoc model.

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

For the matrix-free response-shape and influence audit, sweep one eligible scalar axis at a time while freezing context and all other inputs. Report monotonicity violations, worst observed output change, declared influence envelope, observed-change/envelope ratio, active-contact attribution completeness, and how all five values change as contact/branch capacity grows. Include deliberately ineligible non-monotone timing/context cases so that the audit cannot pass by flattening every response.

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
- Reject the shape/influence translation if an eligible axis violates its frozen direction, an ineligible temporal/context case is flattened to satisfy the audit, any observed perturbation exceeds its declared envelope, active contributions cannot reconstruct the emitted local signal within tolerance, or the envelope becomes progressively less informative as capacity grows.
- Reject the TwinProp-inspired transfer if passive, topology-collapsed, no-slow-state, and intact arms cannot be separated under the same fixed readout, or if shuffled placement performs equivalently to the claimed structured placement.
- Do not infer human-like dendritic computation, general parameter efficiency, ANN parity, or physical-energy savings from synthetic fixtures.

### Acceptance Gate

- Branch-local structured contacts improve the preregistered ambiguous-relation metric over scalar, linear-multi-contact, and typed-independent-contact controls under the same total resource envelope and at least five fixed replicates.
- At least one preregistered outer-graph reduction point preserves held-out quality and abstention within tolerance while reducing the total, fully counted complexity measure rather than only outer node count.
- Missing, duplicated, reordered, stale, contradictory, and capacity-exceeded contacts fail safely; replay and add/prune decisions are deterministic.
- Gains transfer to independent temporal and structural workloads. Human review is required before changing outer topology or production routing defaults.

## Phase 34: Bounded Sparse Memory Checkpoint Caching

**Goal:** test whether bounded, selectively retrieved sparse state checkpoints improve long-delay recall beyond one continuously compressed recurrent state without approaching an unbounded token cache.

### Research Hypothesis

- At verified semantic boundaries, emit a read-only sparse checkpoint containing the event range, active structural IDs, source and verification references, contradiction state, schema/runtime identity, and canonical digest.
- Keep online recurrent state as the primary path. Cached checkpoints are optional retrieval candidates and cannot overwrite current state, admit durable knowledge, or restore a mutable historical graph directly.
- Compare fixed-size segmentation with logarithmic multi-resolution retention under the same total checkpoint and byte budget. Recent history may retain finer resolution, but older segments may be merged only by a deterministic provenance-preserving rule.
- Select at most fixed `k` checkpoints using scalar sparse overlap, resonance, recency, verified-context compatibility, and source-revision validity. No learned projection, softmax, dense query-by-checkpoint matrix, gradient, GPU, or scan of an unbounded cache is allowed.
- Reuse Phase 27 canonical checkpoint identity and Phase 32 sparse-selection primitives, but keep time-segment caching separate from depth-block retrieval so each mechanism can be ablated independently.

### [Done]

- Preregistered the immutable four-arm protocol as `phase34-memory-checkpoint-cache-observed-v1` before implementing any cache candidate. The 16-case observed-only fixture fingerprint is `b2065c2c1371bb7d073a80b2ba285ef50cdb140004864d3b04c1c5b696c9ca27`, the CPU environment fingerprint is `efb0c0f4b3ac0a2aaa5b6718abb7c3dfe85572dbd21f0a5f396a0f35ad755b25`, and the protocol fingerprint is `1e3d73dadd5d5ed49daf97617fc99403c8f6d2104143789afb9be142fe2b548e`. A second registration returned `identical_registration_preserved`.
- Frozen semantic boundaries, equal four-event segments, logarithmic retention tiers `[1, 2, 4, 8]`, oldest-first provenance-preserving merges, fixed `k=2`, deterministic scalar overlap/verification/recency scoring, stale and contradiction exclusion, and checkpoint/count/byte/event/latency ceilings. Parameter averaging, learned routing, softmax, dense summaries, unbounded scans, backpropagation, matrices, GPU execution, durable admission, and production mutation are forbidden by validation.
- Implemented a standalone, default-off bounded checkpoint store. Admission requires an observed and verified semantic boundary plus exact event interval, source references and revision, state-group identity, parent digest, runtime/schema identities, and optional expiry. Contradicted, future, malformed, over-width, or provenance-free candidates fail closed.
- Implemented deterministic equal-retention eviction and logarithmic oldest-compatible merging without parameter averaging. Merges preserve the full bounded union of source references and parent digests; an incompatible or over-width pair is not merged.
- Implemented fixed `k=2` sparse retrieval with registered overlap/verification/recency weights and deterministic tie ordering. Retrieval excludes stale, expired, and contradictory evidence and returns immutable evidence references with `restores_mutable_state=false`; it cannot restore historical mutable state or mutate durable knowledge.
- Added exact checkpoint identity validation, deterministic expiry/revision invalidation, byte/event ceilings, transactional rollback on byte-budget failure, canonical state round trips, and tamper rejection. The component remains disconnected from production Event Memory ranking and durable admission.
- Executed all 64 registered conditions and wrote `phase34_memory_checkpoint_cache_benchmark.json` with matching protocol, fixture, and environment fingerprints. All execution, deterministic replay, CPU-only, state/checkpoint/selection/event ceilings, stale/contradiction safety, non-durable mutation, and non-production-path checks passed.
- On the frozen synthetic fixture, the best checkpoint arm improved delayed-recall conformance from `0.5` for the recurrent/Event Memory control to `1.0`; revision uptake, contradiction/abstention integrity, selection precision/recall, and deterministic replay were `1.0`. Maximum observed checkpoint state was `3,367` bytes, event cost was `9`, and process CPU latency was `0.236 ms` in the recorded run.
- Retained `mechanism_gate_passed=false` and `promotion_ready=false`: equal-size retrieve-all, logarithmic retrieve-all, and equal-size sparse Top-k produced identical fixture quality/resource summaries, so no segmentation or selector trade-off was demonstrated. The immutable registration also omitted replicate seeds despite the stated five-replicate acceptance gate; correcting that requires a new experiment identity rather than altering this registration. Independent evidence remains absent.
- Registered the immutable follow-up `phase34-memory-cache-separation-observed-v1` without modifying the completed parent. It freezes five seeds (`107`, `223`, `311`, `419`, `521`), 12 capacity-pressure families, all five expected pairwise/safety relations, 240 future conditions, and the original `8`-checkpoint, `k=2`, byte/event/latency envelope. The fixture fingerprint is `28ae6c656a2b42ad7fe494b4b68a90061ccd7ac4ce99b8f2c9b4b142e0229dd1`, environment fingerprint is `b7e07483ea4b3ca97007efc4ea6049ff32ad1f08b4274e258235af0dd8c4995b`, and protocol fingerprint is `6b5f8394120936cab157c93c2917ea7bc3b75b98da74321e9be19f63aeaf8e06`. A second registration returned `identical_registration_preserved`.
- The follow-up is bound to parent protocol `1e3d73dadd5d5ed49daf97617fc99403c8f6d2104143789afb9be142fe2b548e` and parent report `996b974dc534b3c3ad8e4a68e5f7fd907f1840341d0b3f4a02d38351baed0429`. It freezes old-target overflow, multi-resolution retention, recent fine resolution, boundary bursts, relevance pollution, recency traps, deterministic ties, post-merge revision/contradiction/staleness, incompatible state groups, and missing targets before implementing the follow-up runtime.
- Executed all 240 registered follow-up conditions across the five fixed seeds. Protocol/fixture/environment identity, deterministic replay, all resource ceilings, CPU-only execution, safety integrity, non-durable mutation, and production isolation passed. The recorded maxima were `1,252` state bytes, `136` event cost, and `0.05 ms` process CPU latency.
- Preserved the negative mechanism result: logarithmic old-target recall beat equal retention by `1.0`, and the pairwise relation rate was `0.833333`, but recent-resolution delta was only `0.088542` against the frozen `0.1` threshold and Top-k pollution precision delta was `0.0`. The pollution cases evicted the relevant checkpoint before retrieval, so they confound retention with selection and cannot support a Top-k claim.
- Retained `threshold_gate_passed=false`, `mechanism_gate_passed=false`, and `promotion_ready=false`. The registration and failed relations were not rewritten. A future selector experiment must keep the relevant checkpoint inside an identical retained set for retrieve-all and Top-k arms, or separately cross retention policy with selection policy; it requires another explicit preregistration and review.
- Following explicit review, registered the new immutable factorial experiment `phase34-memory-cache-factorial-observed-v1`. It compares the recurrent control plus `equal/logarithmic retention × retrieve-all/Top-k`, uses five new fixed seeds and 12 selection/retention/safety families, and freezes 300 future conditions. The fixture fingerprint is `59bfab2a7022f1fa73900f57cae1e19c92f69649237018774dea1899636abc0f`, environment fingerprint is `885be0b0831909024bbe349907210a33d0714fd6b3b744683e032c4216a87807`, and protocol fingerprint is `3eca1c4aec95c374be3c0ba9637df93e0df852bcd75114b3253a9d5494eb47ed`; idempotent re-registration was preserved.
- The factorial contract requires identical retained-set digests within each retention pair, freezes retention state before selection, reports retention and selection bytes separately, and forbids query-visible admission. Selection fixtures are rejected during draft validation if the target is outside the equal-retention set, directly preventing the confound found in the failed separation follow-up.
- Executed all 300 registered factorial conditions across five seeds. Protocol/fixture/environment identity, deterministic replay, resource ceilings, query-blind retention, retained-set equality within both selection pairs, safety integrity, non-durable mutation, and production isolation passed. Maximum observed total state was `1,696` bytes, event cost was `144`, and process CPU latency was `0.202 ms` in the managed run.
- Under byte-identical retained sets, sparse Top-k improved selection precision by `0.875` with recall difference `0.0`; the selection-by-retention interaction was `0.0`. Logarithmic retention improved old-target recall by `1.0`, while equal retention improved recent temporal resolution by `0.1`. All preregistered synthetic mechanism thresholds passed.
- Retained `promotion_ready=false`: the factorial identifies a Top-k selection effect on frozen synthetic fixtures but provides no independent workload evidence, production provenance review, or human approval for integration. The standalone cache remains default-off and disconnected from production Event Memory ranking and durable admission.
- Added a managed independent-execution gate bound to the immutable factorial protocol and report. It refuses to count the synthetic 10/30/100 fixture as external evidence, forbids selector retuning and query-aware retention, and requires every external domain to reach horizons 10, 30, and 100 before execution.
- Before external collection, the six-record manifest passed source-quality checks but used global ranges `0-2` and `3-5`; the readiness gate correctly kept independent execution and promotion blocked rather than fabricating the missing horizon evidence.
- Corrected `migration_horizon_index` from a misleading global sequence to a contiguous per-domain sequence. The external gate now rejects skipped, duplicated, or cross-domain global indices instead of treating a sorted copy as evidence of an ordered horizon.
- Added a bounded transactional external collector and CLI for reviewed first-party sources. It restricts collection to HTTPS Python documentation and canonical RFC Editor documents, revalidates redirects, caps each response at 2 MB and stored normalized content at 12,000 characters, records retrieval time/revision/content and response hashes, rejects duplicate references or content, and does not replace the current manifest unless every requested domain reaches the target.
- Collected 16 new authoritative documents without changing the original six content hashes. The corrected manifest now contains 22 unique observed records: `docs.python.org` and `www.rfc-editor.org` each cover contiguous horizons `0-10`. The source-quality gate remains PASS, horizon-10 coverage is now complete, and the independent factorial blocker decreased from six domain/horizon targets to four (`30` and `100` for each domain). `independent_execution_ready=false` and `promotion_ready=false` remain unchanged.
- Revalidated the expanded manifest against bounded structural interpolation. The runtime still accepts at most eight items per relation: 16 records entered the two proposals and the six over-capacity records were explicitly accounted as rejected, so external collection did not silently widen structural state.
- Expanded the reviewed first-party catalog by 20 Python 3.14 library documents and 20 stable RFC Editor documents, then transactionally collected all 40 additions. The independent manifest now contains 62 unique records and both domains cover contiguous horizons `0-30`. Horizon-30 coverage passes; only horizon 100 remains missing for each domain.
- Revalidated structural capacity after the horizon-30 expansion without widening runtime limits. The two proposals still accept exactly 16 records under the eight-items-per-relation ceiling, and all 46 overflow records are explicitly reported as rejected. The external quality and structural interpolation gates remain PASS, while independent factorial execution and promotion remain blocked.
- Added an explicit horizon-100 catalog stage with 70 further Python 3.14 documents and 70 further RFC Editor documents. The catalog now contains 196 unique reviewed URLs across immutable stages `16/40/140`; a request beyond horizon 100 fails closed.
- Collected all 140 horizon-100 additions transactionally after replacing one unavailable Python documentation URL detected by the fail-closed fetch report. The independent manifest now contains 202 unique observed records, with both domains covering every per-domain index from `0` through `100`.
- The external horizon gate now passes all 10/30/100 buckets and the Phase 34 independent readiness gate reports `independent_execution_ready=true`, zero missing collection targets, and no blockers. This authorizes only the next independent evaluation step; `promotion_ready=false`, production isolation, no selector retuning, and no query-aware retention remain fixed.
- Revalidated bounded structural interpolation on all 202 records. Exactly 16 records remain admitted to the two bounded proposals and all 186 overflow records are explicitly rejected; no runtime capacity was widened to accommodate the new evidence.
- Built a deterministic provenance-bound independent adapter plan before executing any independent arm. The frozen plan contains 42 source-identity cases (`2 domains × 3 horizons × 7 families`), five unchanged parent arms, five unchanged seeds, and 1,050 future conditions. Stream positions use a fixed integer rule, signature decoys use scalar sparse Jaccard with a material-hash tie break, and every case is bound to the completed source-manifest and case-plan fingerprints.
- Registered adapter v1, then rejected it before execution during registration audit because its declaration to reuse parent budgets and thresholds did not bind every concrete value in the immutable validator. The v1 registration was not modified or executed.
- Registered the corrected immutable `phase34-memory-cache-factorial-independent-adapter-v2` with protocol fingerprint `7e4ce13ff7e0aded273a657133263ebf9c52e7d5285c3d2a341a87233bd44ec1`, source-manifest fingerprint `b5e3f13a527fa39ef647944744a52908fc73f83614a95ef92d2b179d6006f801`, and case-plan fingerprint `b0f72e3bd963ba851d341e9b2b2ac5e60846ef3052e536bf18fbfeb971a18f9f`. Identical re-registration was preserved.
- The v2 contract fixes every parent budget and threshold by value, limits claims to exact source-hash identity recall, and explicitly forbids semantic/language-understanding/ANN-parity/energy claims, selector retuning, query-aware retention, matrices, gradients, GPU dependence, durable admission, and production mutation. Registration is not an independent result; execution remains pending.
- Implemented the registered v2 adapter without changing its case plan, arms, seeds, budgets, or thresholds. Independent cases pass their real source URLs into retained and selected checkpoint state, so provenance contributes to state bytes and retained-set digests; legacy synthetic cases retain their prior `stream:index` behavior.
- Executed all 1,050 registered conditions with every source, case-plan, environment, parent-report, external-gate, readiness-gate, and protocol fingerprint matching. Execution integrity, deterministic replay, retained-set identity, source-ref containment, query-blind retention, resource ceilings, CPU/no-gradient/no-matrix/no-GPU boundaries, synthetic fail-closed controls, and production isolation all passed.
- The preregistered exact source-identity gate passed: Top-k selection precision improved by `0.875` with recall difference `0.0`; logarithmic old-identity recall improved by `1.0`; equal retention's recent-resolution advantage was `0.145833`; interaction was `0.0`; safety and retained-set identity were `1.0`. Maximum state was `5,432` bytes, event cost was `116`, and process CPU latency was `0.694 ms` in the recorded run.
- Positive exact-identity recall was `0.75` for recurrent/equal-retention arms and `1.0` for both logarithmic arms at each domain and horizon 10/30/100. The frozen plan sampled 66 unique materials and source refs from the 202-record qualification corpus; it did not test every document.
- Retained `promotion_ready=false`. These results establish bounded exact source-hash identity retrieval under real external provenance, not semantic delayed recall, language understanding, general memory accuracy, ANN parity, or energy savings. Missing/stale/contradiction controls remain synthetic and are reported separately from independent evidence.
- Added a read-only provenance review that rebinds the 66 sampled identities to raw content, processed manifest, case plan, adapter registration, and benchmark fingerprints before performing live source checks. All offline hashes, refs, revisions, collection metadata, observed/allow flags, and benchmark bindings passed.
- Re-fetched all 60 automatically collected sampled URLs. All 30 RFC Editor documents reproduced their normalized content hash, response-body hash, and revision. All 30 Python `/3.14/` documentation pages were reachable but had changed normalized content, response body, and revision, demonstrating that the version-series URLs are mutable rather than immutable release snapshots.
- Preserved the provenance failure: `automated_provenance_passed=false` and `provenance_review_complete=false`. The six historical transcribed excerpts also remain explicit manual-review targets. The executed v2 source fingerprint and result were not rewritten after observing drift, and semantic-workload preregistration remains blocked.
- Verified that the official `docs.python.org/3.14/archives/` artifacts are also mutable series archives rather than release-numbered immutable snapshots. They cannot repair the recorded drift claim.
- Identified a new preregistration candidate without changing v2: official CPython tag `v3.14.6` peels to commit `c63aec69bd59c55314c06c23f4c22c03de76fe45`, and commit-addressed `Doc/library/*.rst` sources are retrievable. The commit plus the exact 30-file list must be frozen under a new source-snapshot identity before collection.
- Registered the exact 30-file Raw HTTP snapshot contract before collection with protocol fingerprint `566b4232402240d8a1fd08793fc820d41698e27a52d4c79d08007a239642ddb5`. Both collection attempts failed closed on GitHub HTTP `429`; no partial raw or manifest output was written and the failed transport result was not used to rewrite the contract.
- Registered a separate shallow-Git fallback before execution with protocol fingerprint `277e12a24c1e9d2270e8085efffc488ef5280b92ed1e8b18659a83fce3e2efc8`. It preserves the same CPython repository, commit, case-plan-derived 30-file allowlist, byte limits, and claim boundaries while making the acquisition transport explicit.
- Collected all 30 preregistered Git blobs from commit `c63aec69bd59c55314c06c23f4c22c03de76fe45`. All records are commit-pinned, unique, UTF-8, and untruncated; the raw snapshot fingerprint is `d33c78a975c47a366a60b8e0cbb857f8a25ba4505472540ddf3ca6e0b5b04b17` and the manifest fingerprint is `75c9e11df10a091a37025cdf0e0c3afd3d4c7314b18f728662626f2a14eb08a3`.
- Generated an evidence-bound human review request for the six historical transcribed excerpts. Every target is bound to its stored content hash and cited source, but `review_complete=false`; no excerpt was silently replaced, reclassified, or approved by automation.
- Added a separate hash-bound human decision ledger and fail-closed review gate. Each decision must bind the immutable request fingerprint, record ID, stored excerpt hash, cited source, authoritative section locator/text hash, alignment result, distortion result, reviewer identity, timezone-aware review time, notes, and explicit human attestation. Conflicting replacement requires an explicit flag, while request/raw/v2 evidence mutation remains forbidden.
- Added `review-phase34-transcribed-excerpts` for read-only gate evaluation or one-at-a-time human decision recording. The initial managed gate correctly reports zero decisions, six pending targets, `review_gate_passed=false`, and `promotion_ready=false`; automation did not create a decision ledger or impersonate human approval.
- Preregistered the separate three-document human-review support snapshot before collection with protocol fingerprint `16fb51aec626a39cd5f584d9050ec32c87baa84734691a14d1bbe47c3d416f1e`. It binds the immutable review request to CPython commit `c63aec69bd59c55314c06c23f4c22c03de76fe45` copies of `argparse.rst` and `pathlib.rst`, plus RFC 9110, without altering the executed adapter, historical excerpts, or review request.
- Collected all three registered official documents and built a six-target comparison packet. The source snapshot fingerprint is `32b115d179bd495717ebabd139cfa5d1eb0dca5dd7fe2928c7f1b26dd224e495`; the packet fingerprint is `ae6fe43432f09e019d70ce12323c23dc9063c49ac6820ce16ee15b8bb81d8b3e`.
- The packet reports exact-substring status, the three highest sparse token-overlap paragraphs, paragraph hashes, fixed source revisions, and human decision placeholders. All six historical excerpts were non-exact strings against the frozen sources and top paragraph Jaccard ranged from `0.522727` to `0.970588`; these are navigation aids, not automated semantic-alignment or distortion decisions.
- The project owner explicitly approved all six hash-bound comparisons. The separate decision ledger records six `aligned` decisions, six `semantic_omission_or_distortion_found=false` results, explicit human attestations, authoritative locators/text hashes, and timezone-aware review time without mutating the request, raw evidence, or executed v2 fingerprints.
- The human-review gate now reports `review_complete=true`, `review_gate_passed=true`, `semantic_delayed_recall_preregistration_ready=true`, and `promotion_ready=false`. Its ledger fingerprint is `ec3ffe44777f332bcc07da428a0385a9904bc98eb403ff627db8dc89711b124f` and report fingerprint is `687bb140373fc0d2e695b6b2a1aff403df2dc505b292f7063fb48e7a60a58aac`.
- Registered the separate immutable semantic workload as `phase34-semantic-delayed-recall-v1` before implementing a semantic adapter. It freezes six human-aligned source propositions, English/Japanese/Simplified-Chinese queries, horizons `10/30/100`, five semantic/safety families, five unseen replicate seeds, the unchanged five parent arms, 270 cases, and 6,750 future conditions. The fixture fingerprint is `b32f5160b312d999383c1f61d96cee278fd95e55b5c42e8546b97ad9d164ccb9`, environment fingerprint is `b8ffdacebf471fbcb9ea344aa81be573154cc2f912e6b264760abd6b62d5f077`, and protocol fingerprint is `8cbeeebfe0297e7343ce0c243b16ac7c9027a70872d53d5df39651a91beafb23`; identical re-registration was preserved.
- The semantic contract hides expected decisions, proposition IDs, and source bindings from the candidate; exact hash identity and token overlap are explicitly invalid as semantic scores. Only paraphrase recall of the six approved propositions is independent semantic evidence. Lexical-overlap, revision, contradiction, and missing-evidence cases are synthetic controls and cannot expand the claim beyond this source-bound workload.
- Frozen semantic thresholds are distinct from the executed exact-identity adapter: macro paraphrase accuracy, checkpoint-minus-control delta, lexical-overlap abstention, revision uptake, contradiction/missing abstention, worst-language recall, source traceability, retained-set identity, deterministic replay, and existing byte/event/latency ceilings. General language understanding, general semantic memory, ANN parity, and physical-energy claims remain forbidden.
- Verification passed for all 95 targeted review-support, human-review, review-request, and CLI dispatch tests.
- Verification passed for all 92 targeted snapshot, review-request, and CLI tests. The full suite recorded `1634 passed` plus one pre-existing Phase 33 process-latency gate failure under aggregate test load; the unchanged Phase 33 test passed immediately in isolation with `max_latency_ms_observed=0.582`, so its threshold was not relaxed and it is not counted as a Phase 34 failure.
- Verification after semantic preregistration passed all 152 Phase 34 and CLI regression tests and the complete managed Python 3.10 `tests/` suite with `1651 passed`. The full run used the frozen Python version so existing environment-fingerprint gates remained meaningful.
- Implemented a default-off sparse multilingual semantic adapter that maps English, Japanese, and Simplified-Chinese source/query text into bounded typed subject and relation axes. It uses no proposition-ID lookup, exact source identity score, token-overlap score, matrix calculation, gradient, external model, GPU, production retrieval, or durable admission.
- Kept retention query-blind and preserved the parent recurrent/equal/logarithmic and retrieve-all/Top-k arms. Equal-retention and logarithmic-retention paired arms produce identical retained-set digests before selection; Top-k uses a fixed recency tie rule and stores sparse checkpoint references instead of duplicating retained evidence.
- Added `eval-phase34-semantic-delayed-recall` with exact candidate/evaluator field separation. Expected decisions and proposition IDs are absent from candidate inputs, retained state, selection state, and candidate traces; evidence-side source references are scored only after the candidate result is frozen.
- Executed all 6,750 frozen conditions under the registered CPython `3.14.7` environment without changing the fixture, protocol, arms, seeds, budgets, or thresholds. Execution integrity and every semantic gate passed: best checkpoint paraphrase accuracy `1.0`, checkpoint-minus-control `1.0`, lexical/revision/contradiction/missing safety `1.0`, worst-language recall `1.0`, source traceability `1.0`, retained-set identity `1.0`, deterministic replay `1.0`, maximum state `5,996` bytes, event cost `116`, and recorded process latency `0.235 ms`.
- Retained `promotion_ready=false`. The independent result covers only six human-aligned source-bound propositions, while probe wording was not separately human reviewed and the safety families are synthetic. Perfect registered-workload scores do not establish general semantic memory, language understanding, ANN parity, production benefit, or physical-energy savings.
- Verification passed all 157 Phase 34 and CLI tests and the complete managed Python 3.10 `tests/` suite with `1656 passed`. A bare repository-root pytest invocation also collected dependency tests cached under `workspace/`; the canonical project run explicitly targeted `tests/` with `PYTHONPATH=.:src`.

### [Later]

- Preserve the failed live-provenance and Raw HTTP transport results. Do not replace sources inside executed adapter v2 or treat the successful Git snapshot as a rewrite of that experiment.
- Keep current Event Memory and recurrent state as production controls. Do not connect checkpoint retrieval to durable admission or production ranking before independent evidence and human review.

### Minimum Experiment

Compare four equal-budget arms:

1. current recurrent/Event Memory retrieval with no checkpoint cache;
2. bounded equal-size segment checkpoints with retrieval over all retained checkpoints;
3. bounded logarithmic multi-resolution checkpoints with retrieval over all retained checkpoints;
4. bounded equal-size checkpoints with deterministic sparse Top-k selection.

- Use delayed key/value recall, long irrelevant intervals, repeated keys with revised values, contradiction, source replacement, duplicate and near-duplicate segments, missing segment, stale runtime/schema digest, reordered replay, cache overflow, and long-tail pollution controls.
- Freeze source events, boundary schedule, maximum retained checkpoints, selected `k`, summary width, serialized bytes, event cost, latency, seeds, environment, and tuning allowance. Arms that retrieve all retained checkpoints must still obey the same fixed selection/event ceiling.
- Report delayed recall or justified abstention, revision uptake, contradiction rejection, selection precision/recall, useful checkpoint rate, cache hit/eviction/merge counts, retained temporal resolution, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject checkpoint caching if it does not beat the no-cache recurrent control on preregistered delayed recall, or if equal-size/logarithmic/Top-k differences disappear under the same total budget.
- Reject it if quality depends on checkpoint count or selected `k` growing with sequence length, scanning every historical checkpoint, hidden dense summaries, extra tuning, or increased source-event budget.
- Reject any merge, averaging, or retrieval that loses source/event identity, mixes contradictory revisions, restores stale state, mutates durable knowledge, or makes replay depend on dictionary order or wall-clock time.
- Reject the selector if it chooses unsupported segments, misses an exact verified checkpoint without abstaining, cannot recover after revision, or costs more than useful retrieval saves.
- Do not transfer the paper's language-model accuracy, parameter-scale, GPU throughput, subquadratic complexity, or long-context claims to SARA.

### Acceptance Gate

- One checkpoint arm improves independent delayed-recall quality over the current recurrent/Event Memory control while preserving revision, contradiction, and abstention integrity.
- The selected design remains within fixed checkpoint-count, selection, byte, event, latency, and replay ceilings across at least five preregistered replicates and the longest horizon.
- Fixed versus logarithmic segmentation and retrieve-all versus sparse Top-k have interpretable, reproducible trade-offs rather than gains caused by unequal retained information.
- Promotion remains blocked until independent workloads, provenance review, and explicit human approval confirm that cached evidence cannot bypass durable-knowledge boundaries.

## Phase 35: Emergent Overlapping Spatiotemporal Expert Fields

**Goal:** test whether expertise can be represented as a bounded, overlapping, time-varying sparse activation field rather than a fixed expert module selected by a router.

### Research Hypothesis

- Treat an expert as an observation over the active sparse subgraph at time `t`, not as a persistent model object:

```text
F(x, t) = active nodes + active relations + phase bucket + bounded history state
```

- The same node may participate in several fields, and one field may move across language, retrieval, temporal integration, reasoning, and response stages. Field identity is reconstructed from canonical event traces; it is not stored as a dense per-node membership vector.
- Start with a fixed total node/contact/route budget. Boundary-free specialization must first emerge through local spike timing, eligibility traces, homeostasis, excitation/inhibition balance, fatigue, and phase-dependent routing. Unbounded neuron creation, dynamic model replication, or hidden parameter growth is not part of the initial hypothesis.
- Separate three claims that must not be conflated: variable-cardinality routing among fixed experts, overlapping boundary-free activation fields, and structural rewiring after repeated field reuse. Structural growth/pruning is a later ablation and cannot supply the initial field result.
- Reuse Phase 30 temporal state, Phase 32 sparse routing/load controls, Phase 33 structured-edge accounting, and existing structural-plasticity safety gates. Do not create parallel mechanisms with different provenance or resource accounting.

### Evidence Boundary

- [NeurIPS 2024 SEMM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/137101016144540ed3191dc2b02f09a5-Abstract-Conference.html) shows that spiking expert/router sequences can implement dynamic sparse conditional computation, and [SpikingMoE](https://arxiv.org/abs/2605.23188) reports input-dependent spike-driven routing. Both retain explicit expert modules; neither demonstrates boundary-free emergent expert fields.
- [Expert Choice](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/) supports testing variable expert cardinality, while [Soft MoE](https://arxiv.org/abs/2308.00951) motivates a continuous-routing control. Their dense token-expert score matrices, soft assignments, gradient optimization, Transformer accuracy, GPU efficiency, and scaling claims are comparison-only and are not SARA runtime candidates.
- The term `expert field` is an operational evaluation label, not a biological assertion. A visualization of overlapping activity is not evidence of useful specialization.

### [Later]

- Begin only after the Phase 32 fixed-expert routing/control experiment is preregistered and executed. Phase 32 supplies the fixed-boundary, fixed-Top-k, variable-cardinality, load-collapse, and deterministic-budget controls.
- Preregister a fixed-budget, five-seed experiment before implementing a field candidate. Freeze node/contact/route counts, spike/event budget, phase buckets, active-set ceiling, state bytes, latency, homeostatic updates, EI bounds, tuning attempts, and canonical tie order.
- Keep production routing unchanged and the field mechanism default-off. Do not persist semantic expert labels or allow a field observation to bypass Event Memory, Concept Review, contradiction, expiry, or durable-admission boundaries.
- Add structural add/prune only in a separately registered follow-up after the fixed-structure field mechanism passes. Total created and removed contacts, churn, recovery, and outer-graph complexity must remain explicitly bounded.

### Minimum Experiment

Compare five equal-resource arms:

1. existing fixed experts with fixed deterministic Top-k routing;
2. fixed experts with bounded variable-cardinality routing but unchanged expert boundaries;
3. boundary-free sparse activation clusters with timing/phase removed;
4. overlapping spatiotemporal fields with phase, delay, fatigue, and bounded history;
5. the same field arm with node identity or phase assignment canonically shuffled.

- Use single-specialty tasks, cross-specialty composition, ambiguous shared features, staged language→retrieval→reasoning→response sequences, same spike count with different order/phase, repeated co-activation, abrupt context switch, sparse candidate omission, contradiction, missing evidence, stale source revision, irrelevant bursts, dominant-region collapse, dying-region recovery, all-active, and no-reuse controls.
- Freeze source events and task labels before execution. A field may be observed only from local event traces available at that time; offline task labels cannot influence routing, field formation, homeostasis, or rewiring.
- Report held-out task quality/abstention, fixed-control delta, variable-cardinality delta, active node/contact count, field overlap Jaccard, participation entropy, load range/Gini, always-active and dead-node rates, temporal field-transition consistency, phase/order sensitivity, contradiction recovery, churn, state bytes, event cost, CPU latency, and deterministic replay.
- Report field statistics from sparse scalar counters and canonical sets. No dense node-by-field membership matrix, softmax, learned router, backpropagation, GPU collective, or post-hoc cluster-count search is allowed.

### Failure Conditions

- Reject the hypothesis if the spatiotemporal field arm does not beat both fixed-boundary controls on preregistered cross-specialty or staged tasks, or if removing phase/history does not reduce the claimed temporal advantage.
- Reject it if gains come from more active nodes, contacts, events, state bytes, latency, tuning trials, or variable total capacity; if every task activates the same region; or if fields fragment into one field per example.
- Reject specialization inferred only from attractive activation plots, task-label leakage, post-hoc choice of field count, dense similarity computation, nondeterministic clustering, or an observer that costs more than the routed computation.
- Reject it if shared nodes destroy single-specialty accuracy, dominant activity suppresses reusable shared structure, dead/always-active nodes do not recover under frozen homeostasis, or contradiction and revision changes cannot dissolve stale fields.
- Reject the structural follow-up if rewiring merely recreates fixed experts, grows without saturation, changes durable semantic structure without review, or hides complexity inside contacts while reporting only outer nodes.
- Do not transfer SEMM, SpikingMoE, Expert Choice, or Soft MoE accuracy, energy, hardware, routing-balance, or scaling results to SARA.

### Acceptance Gate

- Under identical total resources and at least five preregistered seeds, the intact field arm improves a frozen cross-specialty/staged metric over fixed Top-k, variable-cardinality fixed experts, phase-removed fields, and shuffled controls while preserving single-specialty quality and abstention.
- Field overlap and temporal movement are reproducible under canonical replay, are input-dependent rather than globally dominant, and remain within active-set, EI, homeostasis, state, event, and latency ceilings.
- Collapse, dying-region, all-active, missing, stale, contradictory, order-shuffled, and distribution-shift controls fail safely and recover without unbounded activity or structural churn.
- A separately preregistered independent workload and human review are required before any production-routing or structural-plasticity integration.

## Phase 36: Evidence-Preserving Learning-System Evolution

**Goal:** allow a new learning mechanism to reinterpret previously observed evidence without requiring its internal state to remain representation-compatible with the retired mechanism.

### Research Hypothesis

- Treat learned runtime state as a versioned, replaceable interpretation rather than the durable source of knowledge. Preserve a canonical architecture-independent chain:

```text
observed event -> episode -> relation/concept/theory revision -> supporting and contradicting evidence
```

- Each interpretation records its learning-system ID, state/schema version, source-event range, evidence links, assumptions, verification receipts, expiry, and canonical digest. Higher-level concepts must retain bounded reverse links to representative support, counterexamples, and revision history so a successor can replay the basis of a claim.
- Use three explicit migration paths instead of pretending that all representations are directly compatible: schema migration for provably equivalent state, evidence replay/recompilation for changed learning rules, and a time-bounded read-only overlap period in which the successor may query the predecessor with its evidence and uncertainty.
- Preserve observations and provenance as immutable records, but allow semantic interpretations to be revised. A predecessor answer is candidate evidence about prior behavior, not ground truth, and may not bypass contradiction, verification, RISA, or Event Memory admission.
- Represent learning methods as reviewed, versioned strategy descriptors with declared inputs, local update rules, budgets, compatibility range, and rollback contract. The meta-level may propose and compare candidates in an isolated default-off sandbox; it may not rewrite production code, relax gates, promote itself, or delete the last replayable predecessor.
- A multi-method ecosystem is a separate follow-up claim. Retaining several learning methods is acceptable only when each has measurable complementary value and bounded routing/state cost; indefinite accumulation of obsolete systems is not self-evolution.

### [Later]

- Begin the base migration experiment only after Phase 27 canonical replay/state-migration equivalence and an independent Phase 22 continual-horizon workload are available. Reuse Phase 31 consolidation and Phase 34 checkpoint provenance rather than creating a second memory hierarchy.
- Freeze the architecture registry, strategy descriptors, knowledge tiers, evidence-link retention policy, migration budget, predecessor-query budget, overlap duration, retirement rules, rollback point, and human approval before executing a successor candidate.
- Test a multi-learning-method routing follow-up only after Phase 32 supplies fixed routing controls; use Phase 35 fields only if that phase independently passes its own acceptance gate.
- Keep all candidates sandboxed, default-off, CPU-only, sparse, bounded, and backpropagation-free. No autonomous production mutation or predecessor retirement is authorized by benchmark success alone.

### Minimum Experiment

Compare five frozen successor-migration arms over identical source histories and declared migration ceilings:

1. predecessor-only control with no learning-system change;
2. direct typed state/schema migration, allowed only for declared compatible versions;
3. a bounded translator from predecessor state to successor state;
4. successor reconstruction from canonical evidence and hierarchical reverse links;
5. evidence reconstruction plus a bounded read-only predecessor-query overlap period.

- Run unchanged knowledge, revised facts, contradictions, delayed evidence, missing raw episodes, lossy concept summaries, corrupted adapter output, incompatible dimensions/coding schemes, Vector-to-spike-style representation change, predecessor error, rollback, repeated migrations, partial migration, and retirement/recovery cases.
- Include shuffled evidence links, removed counterexamples, stale receipts, unsupported predecessor answers, and equal-content/different-representation controls. A separate ablation compares full replay, provenance-preserving tiered replay, and concept-only replay under fixed byte/event/latency ceilings.
- Report verified recall and abstention before/during/after migration, revision uptake, contradiction rejection, evidence-chain coverage, interpretation equivalence, newly verified relation discovery, inherited-error rate, adapter-only dependence, predecessor-query rate, rollback fidelity, unrecoverable-knowledge rate, state bytes, replay events, CPU latency, and deterministic replay.
- A new interpretation is accepted by evidence-grounded behavioral equivalence and revision integrity, not byte equality with the predecessor's hidden state. Newly discovered relations count only after independent verification and normal durable-admission review.

### Failure Conditions

- Reject the migration if any accepted claim cannot trace back to observed evidence or an explicitly marked predecessor-only assertion, or if a translator launders stale, contradictory, unsupported, or corrupted predecessor state into verified knowledge.
- Reject it if reconstruction quality depends on retaining every raw event without a declared bound, scanning all history, unbounded predecessor queries, dense cross-representation matrices, gradients, GPU execution, hidden external models, or increasing migration resources with every generation.
- Reject retirement if the successor loses protected knowledge, revision behavior, abstention, counterexamples, provenance, or rollback capability; if unresolved cases cannot fall back safely; or if the predecessor cannot be restored from its frozen checkpoint and strategy descriptor.
- Reject the multi-method follow-up if routing benefit disappears against the best single-method control, methods duplicate one another, obsolete systems accumulate without saturation, or resource accounting omits dormant state and migration maintenance.
- Do not call a system self-evolving merely because it changes state schemas, translates vectors, replays data, or produces a different activation pattern. The experiment must demonstrate a preregistered improvement in the learning process on future held-out experience.

### Acceptance Gate

- Across at least five preregistered seeds and independent histories, one successor path improves a frozen future-learning metric over the predecessor while preserving verified prior knowledge, contradiction/revision integrity, abstention, evidence traceability, and bounded deterministic replay.
- Architecture-incompatible cases select evidence reconstruction or safe abstention rather than an undeclared adapter; compatible migrations remain idempotent, reversible, and digest-bound.
- The overlap period has a measured exit condition: predecessor queries decrease below the frozen ceiling, unresolved protected cases are zero or explicitly retained, and retirement plus rollback both pass without deleting source evidence.
- Production adoption, strategy retirement, and any meta-level proposal loop each require separate human review. A successful migration benchmark cannot authorize self-modification or erase a prior generation.

## Phase 37: Structural Invariant Sharing and Generative Transfer

**Goal:** test whether bounded, evidence-linked structural patterns can provide the cross-example sharing needed to propose useful relations that were never stored as edges, without introducing dense shared weights or uncontrolled graph-wide propagation.

### Research Hypothesis

- Exact edges are episodic/relational memory; they are not by themselves evidence of knowledge-like generalization. Define a reusable knowledge candidate as a typed structural invariant with explicit role slots, topology, direction, temporal/causal constraints, context, applicability conditions, supporting exemplars, counterexamples, revision, and provenance.
- Canonically normalize a small verified subgraph into anonymous roles such as `role:source`, `role:mediator`, and `role:target`. Structurally equivalent examples may update one bounded pattern record even when their node labels differ. Node identity and task labels must not define the pattern fingerprint.
- A local observation may update only the matching pattern's bounded support, exception, recency, and reliability counters. Its influence reaches another case only when a bounded sparse role match activates that pattern; there is no unconditional graph-wide broadcast.
- Applying a shared pattern produces a provisional relation hypothesis with the matched roles, supporting and contradicting exemplars, context mismatch trace, confidence decomposition, and canonical digest. It cannot create a durable edge without the existing independent verification, contradiction, RISA review, and Event Memory admission path.
- Keep five capabilities distinct in reports: exact relation retrieval, verified path composition, similarity scoring, proposal of an unstored relation, and later verification of that proposal. Only the last two can support a claim of structural generative transfer; an attractive motif visualization or high analogy score cannot.
- Structural sharing is the sparse auditable analogue to parameter sharing only if one verified pattern change predictably affects several later eligible cases. This is a testable functional analogy, not a claim that RISA reproduces ANN representations or biological knowledge.

### [Done]

- The Phase 21 independent held-out composition/analogy prerequisite is complete, and its report and fixture SHA-256 identities are bound into the Phase 37 protocol.
- Registered the immutable `phase37-structural-invariant-sharing-v1` protocol before candidate implementation. It freezes six equal-resource arms, fourteen structural case families, six binding shuffles, five seeds, canonical anonymous roles, topology/direction/order identity, match/expiry/tie rules, sparse capacity and CPU ceilings, one tuning attempt, failure rules, and non-promotion boundaries.
- The registered protocol fingerprint is `e77d34460bfc2ae2440d765616a65ce7dad734d07ef6cca3b0d17b1532cfe704`. Node identity and task labels are excluded from motif fingerprints; withheld relations and endpoint roles remain evaluator-only until proposals are frozen.
- Human review approved all eight RFC excerpt-to-edge mappings. The immutable source manifest (`6971a874…72096`), four-source train base (`a4f77286…66d7c`), and source-/structural-family-disjoint four-source evaluation base (`705f134d…1873`) are frozen under review draft `d28f0159…c89ef`; candidate implementation is now allowed while production promotion remains closed.
- Expanded the approved bases into all fourteen preregistered execution families with evaluator-label isolation. The authoritative v2 candidate input is `ccfccd4d…695d`; the evaluator-only key is `30fe7274…a9ea`. Label-renamed visible and withheld edges share one anonymous-node mapping, source partitions remain unchanged, and no candidate row contains a case-family label, expected decision, or withheld edge.
- Implemented the default-off sparse `CanonicalTypedMotifStore` and consumed the single registered Phase 37 attempt across all six arms. The intact context/exception-aware arm preserved justified abstention, rare exceptions, direction/order sensitivity, revision retraction, deterministic replay, provisional-only proposals, and resource bounds, but novel-relation precision/recall and held-out-domain transfer were `0.0`; decision accuracy was `0.7143`, event cost `136`, and maximum state `452` bytes.
- Phase 37 is retained as a negative result: the intact arm did not beat the frozen baselines and cannot support structural generative-transfer, knowledge-emergence, production-integration, or graph-mutation claims. The fixture, threshold, and one-attempt tuning identities remain frozen; any follow-up requires a new preregistered experiment identity.

### [Next]

- Preregister Phase 38 canonical structural-delta and transformation memory using the now-frozen Phase 37 role/invariant schema. Do not implement a persistent delta codec or shared transformation store before that registration.
- Reuse Phase 22 revision/contradiction histories, Phase 31 consolidation limits, and Phase 33 structured-edge resource accounting during execution.
- Keep the candidate default-off and separate from the current `StructuralAnalogyEngine`. Do not silently upgrade an analogy score into a relation proposal or alter production RISA graph state.
- Start with fixed nodes and relations. Pattern-driven structural add/prune is a later follow-up and must not provide the initial transfer result.

### Minimum Experiment

Compare six equal-source and equal-resource arms:

1. exact verified-edge retrieval only;
2. existing bounded verified-path composition;
3. current relation-type-set Jaccard analogy;
4. canonical typed-motif sharing without context or exception state;
5. canonical typed-motif sharing with context, time/order, counterexamples, and revision state;
6. the intact candidate with role assignment, evidence links, or topology canonically shuffled.

- Include label-renamed isomorphic structures, identical relation types with different topology, unseen nodes, held-out domains, multi-edge role transfer, support/function analogies, temporal order reversal, causal-direction reversal, context change, bird/penguin-style exceptions, rare counterexamples, revised evidence, contradiction, missing roles, adversarial hubs, duplicated evidence, stale sources, and no-transfer controls.
- Freeze all examples and split by structural family and source before execution. Near-isomorphic variants, node aliases, source revisions, or one template rendered with different labels may not cross the train/evaluation boundary.
- Require the candidate to predict a withheld relation type and endpoint role from the visible partial structure. The answer key and withheld edge may be used only by the evaluator after the proposal is frozen.
- Report verified novel-relation precision/recall, justified abstention, cross-domain transfer, exception preservation, direction/order sensitivity, role-map consistency, evidence-chain completeness, inherited-error rate, revision recovery, pattern reuse distribution, participation entropy, pattern count/growth, propagation fan-out, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject structural sharing if it does not beat exact retrieval, path composition, and current Jaccard analogy on preregistered withheld-relation cases, or if the shuffled-role/topology control retains the gain.
- Reject it if it copies the most frequent relation, depends on node names or task labels, leaks withheld edges, scans every stored subgraph, performs dense all-pairs similarity, uses gradients/GPU/external models, or broadcasts an update to unrelated structures.
- Reject it if exceptions are erased, contradictions spread, causal/temporal direction is ignored, hub nodes dominate matches, or a stale pattern continues proposing after its support is revised or expires.
- Reject it if patterns collapse into one global template, fragment into one pattern per example, grow without saturation, hide unbounded exemplars, or cost more state/events/latency than the verified transfer benefit.
- Do not call a proposal knowledge merely because it was not an explicit edge. A deterministic rule hard-coded for the fixture, a composed stored path, or an unsupported guess is not emergent structural knowledge.

### Acceptance Gate

- Across at least five preregistered seeds and independent structurally held-out histories, the intact context/exception-aware arm improves verified withheld-relation quality over all three existing baselines and the context-free candidate while preserving abstention and rare exceptions.
- Removing role identity, topology, time/order, counterexamples, or evidence links causes the corresponding preregistered degradation, demonstrating that gains come from reusable structure rather than label leakage or frequency bias.
- One bounded pattern is reused across multiple independent node sets and at least one held-out domain, with every proposal tracing to supporting and contradicting evidence and no direct durable mutation.
- Revision, contradiction, expiry, and source replacement deterministically retract or recalibrate affected proposals without changing unrelated patterns; state, fan-out, event, and CPU-latency ceilings hold at the longest horizon.
- Production integration requires a separate independent workload, provenance review, and human approval. Until then the result remains an observed-only structural-transfer candidate, not a general intelligence or ANN-parity claim.

## Phase 38: Canonical Structural Delta and Transformation Memory

**Goal:** test whether RISA can represent experience as exact reconstructible `base structure + typed delta`, then reuse recurring deltas as evidence-linked transformation knowledge without losing exceptions, provenance, or rollback safety.

### Research Hypothesis

- Separate three layers and keep their identities explicit:

```text
Layer 3: transformation patterns over deltas
Layer 2: canonical invariant structures with anonymous roles
Layer 1: concrete observed episodes and entities
```

- A canonical structure record contains schema version, typed nodes/events/concepts, directed relations, anonymous role assignments, temporal/order constraints, context, confidence components, supporting/contradicting evidence links, revision, expiry, and a canonical digest. Entity labels remain in concrete bindings rather than defining the invariant fingerprint.
- A structural delta is an immutable ordered sequence of typed edit operations with a base digest, target digest, preconditions, affected role paths, source evidence, revision, inverse/rollback information, event cost, and its own digest. The initial operator vocabulary is frozen before implementation and may include `ADD_NODE`, `REMOVE_NODE`, `ADD_RELATION`, `REMOVE_RELATION`, `CHANGE_ROLE`, `CHANGE_VALUE`, `GENERALIZE`, `SPECIALIZE`, `REORDER_TIME`, `MERGE`, and `SPLIT`.
- `target = apply(base, delta)` must reconstruct the exact canonical target and digest. Removal creates an evidence-preserving revision/tombstone operation rather than erasing historical provenance. Failed preconditions, missing bases, stale revisions, cycles, or budget overflow fail closed.
- Compare deltas only after canonical role normalization. Repeated independently sourced deltas may form a provisional transformation pattern containing applicability conditions, support, counterexamples, temporal ordering, uncertainty, revision, and expiry. A transformation pattern proposes a future delta; it does not directly mutate the durable graph.
- Treat explicit canonical deltas as one candidate representation, not the definition of difference. If Phase 39 supplies a valid anonymous resource-reuse representation, compare hand-applied edit deltas with differences derived from shared and non-shared active resource sets; do not assume in advance that a separate delta object is superior.
- Treat a concept candidate as the invariant remaining after bounded concrete bindings and justified exceptions are removed, and treat learning as improving a frozen description account for future observations. This is an operational research definition, not proof that shortest encoding equals semantic truth.
- Translate the MDL idea into explicit scalar accounting rather than gradient optimization:

```text
total_description_cost = structure_cost + delta_cost + exception_cost + codebook_cost
```

  The operator codebook, scalar costs, decoder/runtime bytes, maximum base candidates, chain depth, and tie order are preregistered. A smaller score is invalid if exact reconstruction, evidence traceability, revision integrity, or held-out reasoning quality decreases.

### [Done]

- Phase 37 canonical roles are frozen, and its negative result is explicitly bound as a prerequisite rather than reinterpreted as evidence for transformation sharing.
- Registered immutable experiment `phase38-canonical-structural-delta-v1` before codec, base-selector, materializer, MDL selector, or transformation-store implementation. Protocol fingerprint: `9dfafe9ed01d80c0eadf1d59620391332a24d40a7029bbccbc919f98e67080dd`.
- The registration freezes six equal-resource arms, eleven ordered typed operators and inverse requirements, twenty-six case families, five seeds, canonical ordering, bounded base selection, chain/branch/materialization limits, one tuning attempt, complete codebook/decoder/index/checkpoint/exception/materialization accounting, evaluator-label isolation, and non-promotion boundaries.
- Frozen Phase 38 execution identities from ten registered synthetic structural histories: eleven operator-complete train examples and twenty-six execution cases with source, structure, and transformation families disjoint across partitions. Candidate inputs (`7dee3c55…98c2`) are separated from evaluator-only exact targets and withheld deltas (`79b85c6d…ec1a`); the source manifest is `0dab9ac9…0870` and train fixture is `4a7ca69b…765f`.
- The fixture evidence scope is explicitly `registered_synthetic_control`. It can test codec correctness, rollback, accounting, and bounded transformation mechanics, but cannot support external-validity, semantic-learning, or production-promotion claims.
- Implemented the default-off `CanonicalStructuralDeltaCodec` and consumed the registered codec attempt. All valid cases achieved exact reconstruction, digest match, rollback fidelity, evidence traceability, and removal tombstone preservation at `1.0`; deterministic replay passed with event cost `99` and maximum state `1,440` bytes.
- Phase 38 is retained negative because malformed-control abstention was only `0.5`. Several frozen invalid families did not expose a candidate-observable contradiction/inverse/conflict marker, so accepting them cannot be repaired by reading evaluator labels. Transformation sharing was not executed, production remained unchanged, and the frozen fixture is not retuned.

### [Next]

- Preregister Phase 39 anonymous local-reuse as an independent mechanism arm using the frozen Phase 30 temporal contract and Phase 37 explicit-motif controls. Phase 38 remains a negative codec result and does not satisfy the Phase 41 transformation prerequisite.
- Reuse the current structural-edit transaction as the non-durable safety boundary, Phase 22 revision histories, Phase 31 bounded consolidation, Phase 33 contact/resource accounting, and Phase 36 schema-migration rules. Do not create an unversioned parallel graph format.
- Keep snapshot storage and production RISA behavior unchanged. The delta codec, transformation store, base selector, and materializer remain default-off until independent evaluation and human review.

### Minimum Experiment

Compare six equal-source arms under separately reported storage and inference budgets:

1. complete canonical structure snapshots with no deltas;
2. base snapshot plus an unshared chronological edit log;
3. base snapshot plus canonical typed deltas with no cross-case reuse;
4. shared invariant structures plus per-instance deltas and explicit exceptions;
5. invariant structures plus reusable transformation patterns over deltas;
6. the intact transformation arm with base identity, role mapping, edit order, or evidence links canonically shuffled.

- Include repeated THROW role substitutions; bird, penguin, ostrich, and emu exceptions; support/function transformations across domains; add/remove relation; role/value change; generalize/specialize; temporal reorder; merge/split; repeated independent transformation families; non-compressible random structures; ambiguous bases; equivalent-cost bases; long delta chains; branching and merge conflicts; duplicated evidence; stale revisions; contradiction; source replacement; missing base; corrupted delta; invalid inverse; cycle; and budget-exceeded controls.
- Freeze train/evaluation splits by source, structural family, and transformation family. A target structure, equivalent delta, renamed template, or descendant revision may not cross the isolation boundary.
- Evaluate both directions independently: exact materialization from `base + delta`, and prediction of a withheld delta/target-role change from visible context. The target and evaluator labels remain unavailable until the proposal is frozen.
- Report exact reconstruction and digest match, rollback fidelity, provenance/tombstone preservation, base-selection stability, delta-chain depth, materialization cost, total description cost including codebook/decoder, compression ratio, exception share, pattern reuse, withheld-transformation precision/recall, abstention, revision recovery, inherited-error rate, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject the codec if any valid target cannot be reconstructed exactly, operation order is ambiguous, an inverse fails to restore the base, a remove loses evidence, or replay depends on dictionary order, wall-clock time, or an undeclared schema migration.
- Reject compression if savings disappear after codebook, index, decoder, checkpoint, exception, and materialization state are counted; if the selected base requires an unbounded/global search; or if long chains merely defer snapshot cost and latency.
- Reject transformation learning if patterns collapse exceptions, confuse generalization with deletion, ignore causal/temporal direction, copy the most common delta, leak target structures, or retain stale transformations after contradiction/revision.
- Reject it if patterns grow one per example, all deltas collapse into one transformation, graph edits become dense, or implementation uses gradients, matrices, GPU execution, hidden external models, post-hoc operator costs, or target-aware base selection.
- Do not claim learning from a lower description cost alone. A compact but semantically wrong representation, a hard-coded fixture codec, or loss of rare verified knowledge is a failure.

### Acceptance Gate

- Across at least five preregistered seeds and independent structurally held-out histories, canonical deltas achieve exact digest reconstruction and rollback for every valid case while malformed, stale, contradictory, cyclic, and over-budget cases fail safely.
- At least one shared structure-plus-delta arm reduces total accounted description cost versus complete snapshots without reducing verified recall, abstention, exception preservation, revision integrity, evidence traceability, or deterministic replay.
- The transformation-pattern arm improves withheld transformation/target-role quality over unshared typed deltas and static invariant sharing, and loses that advantage when role, order, base, or evidence bindings are shuffled.
- Repeated transformations reuse bounded patterns across independent entities and at least one held-out domain; pattern count, chain depth, exceptions, state, events, and CPU latency saturate within frozen ceilings.
- Production integration requires independent provenance review, explicit human approval, and a separate migration plan proving that existing graph snapshots remain readable and recoverable.

## Phase 39: Usage-Driven Anonymous Latent Structure Reuse

**Goal:** test whether concept-like reusable structures can arise from repeated local sparse resource reuse, without asking the learner to classify examples, name a concept, perform global graph-isomorphism search, or receive the evaluator's latent labels.

### Current Evidence Boundary

- The existing `SparseOwnLatentPredictor` is sparse and locally counted, but `update(...)` receives an explicit `label` and `latent_terms`; its benchmark therefore demonstrates bounded supervised latent recovery, not anonymous concept emergence.
- The current RISA Kernel creates `pattern:{action}->{effect}` and `concept:shared_{action}_{effect}` from normalized human-readable action/effect fields. Reuse is present, but the candidate structure and abstraction boundary are substantially specified by the input schema.
- Existing sparse signatures, resonance, structural plasticity, Phase 35 fields, and Phase 37 motifs are useful controls and implementation references. None currently proves that an unnamed high-order structure self-organized from reuse.

### Research Hypothesis

- Encode each observed episode as a bounded set of local typed event/relation fragments. For each fragment, route only among a bounded candidate neighborhood of existing anonymous units using locally available type, timing, phase, context, and recent-activity state. Reuse a unit when it passes the frozen local gate; allocate a new unit only when no candidate passes.
- Do not run a separate command that says two complete experiences are isomorphic or belong to one class. Experiences become related only because some of their fragments activate the same anonymous units. Their shared and differing portions are then observable from overlapping and non-overlapping active resource sets.
- Allow units to participate in multiple overlapping assemblies. A concept candidate is an observer-visible, repeatedly reused assembly or transition pattern, not a stored class label and not a permanently disjoint cluster.
- Apply bounded homeostasis, fatigue, excitation/inhibition balance, capacity pressure, inactivity decay, and deterministic tie rules so frequent inputs cannot capture every unit and rare structures are not forced into dominant assemblies.
- Permit a bounded higher level to consume canonical traces of lower-level assembly reuse as events. Higher-order structures may form only when reuse and held-out predictive/compressive value pass frozen thresholds; hierarchy depth, width, promotion count, and state are capped rather than chosen after seeing the evaluation.
- Anonymous pattern IDs such as `latent:3817` need not receive a human semantic name. They must still be auditable through supporting/counterexample event references, activation and allocation traces, revision history, ablation effect, uncertainty, expiry, and resource cost.
- Human-designed primitives remain a possible inductive bias and must be reported. A system using typed time, causal, role, or spatial events may discover unnamed combinations of those primitives, but it may not claim completely assumption-free structure discovery.

### [Done]

- Registered immutable experiment `phase39-anonymous-local-reuse-v1` before implementing any anonymous unit, assembly, hierarchy, or reuse candidate. Protocol fingerprint: `5dfecedc4dfa239bb3a37c12bfc59069c01a7b5cf450149842db4dc6a3abf57f`.
- The protocol freezes six arms, twenty-one positive/negative case families, four canonical shuffles, five seeds, one tuning attempt, learner-visible and evaluator-only fields, bounded local-neighborhood construction, overlapping assemblies, homeostasis/fatigue/EI constraints, hierarchy/resource ceilings, fifteen thresholds, targeted ablation, and non-promotion boundaries.
- Bound the completed Phase 30 and Phase 37 negative identities as controls rather than supporting evidence. Phase 30 report digest is `bf560bb9…e2a5`; Phase 37 report SHA-256 is `af1b5d4a…de6b9`. Their failed mechanism/promotion status cannot be reinterpreted as proof for Phase 39.
- Added fail-closed validation for label leakage, global all-pairs search, predeclared hierarchy, prerequisite drift, resource/threshold omission, execution-policy drift, immutable replacement, and unmanaged outputs. Registration is idempotent and ten focused contract tests pass.
- Froze 210 deterministic execution histories across all twenty-one families, five seeds, and separate train/evaluation partitions at exactly 256 events per case (53,760 events total). Candidate inputs contain only opaque case/event identities and registered local fields; partition, case family, seed, generator/source identity, hidden factors, expected outcome, human names, task labels, and offline-cluster IDs are physically confined to the evaluator key.
- Input digest is `15a93686…8d953`, evaluator-key digest is `821670a3…6f08`, and freeze fingerprint is `7eb5ba8b…8c7d4`. Source and hidden-generator identities are disjoint across partitions, every registered evaluator field is absent from candidate inputs, and deterministic generation/tamper/leakage checks pass sixteen focused fixture/preregistration tests.

### [Later]

- Implement the bounded local-neighborhood, reuse/allocation, homeostasis/fatigue/EI, overlap, evidence, expiry, and deterministic replay controls exactly against the frozen candidate inputs.
- Use Phase 35 spatiotemporal fields only as a reusable control if Phase 35 passes independently. Do not transfer its expert-field result into a concept-emergence claim.
- Keep anonymous units, assemblies, and optional human labels default-off and outside durable RISA state. Naming or interpreting a pattern is a read-only review action and cannot change its routing or evaluation result.

### Minimum Experiment

Compare six equal-source and equal-resource arms:

1. the existing label/latent-term-supervised `SparseOwnLatentPredictor` reference;
2. Phase 37 explicit canonical-role motif matching;
3. an unlabeled offline/global clustering reference, excluded from runtime candidacy;
4. local anonymous resource reuse without homeostasis, fatigue, or EI balance;
5. local anonymous spatiotemporal reuse with bounded homeostasis, fatigue, EI balance, overlap, and optional capped hierarchy;
6. the intact local-reuse arm with event order, phase, unit identity, or neighborhood assignment canonically shuffled.

- Include different surface forms with the same hidden generator, similar words with different generators, repeated local fragments, unseen combinations, overlapping factors, known human concepts, evaluator-hidden synthetic factors, unnamed multi-timescale combinations, temporal order and interval changes, causal-direction reversal, spatial/role interactions, rare exceptions, abrupt context shifts, irrelevant bursts, forced hash collisions, dominant-frequency pressure, all-new/no-reuse streams, capacity saturation, dead-unit recovery, revision, contradiction, expiry, and source replacement.
- Hidden-factor IDs may be used only by the post-hoc evaluator and never by allocation, reuse, local updates, hierarchy formation, stopping, pruning, or hyperparameter choice. Real-data results without known ground truth are reported as predictive/reuse observations, not proof that a human-unknown concept was discovered.
- Freeze train/evaluation histories by source and hidden generator. Surface paraphrases, renamed copies, descendant revisions, or the same generator seed may not cross the boundary.
- Report held-out next-state/relation quality and abstention, cross-context transfer, post-hoc hidden-factor recovery, reuse selectivity, assembly overlap and stability, exception preservation, pattern participation entropy, dominant/dead/always-active unit rates, allocation/reuse/eviction counts, hierarchy depth/width, revision recovery, evidence-chain completeness, ablation delta, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject emergence if gains require labels, latent terms, task IDs, evaluator-hidden factors, human concept names, offline cluster assignments, global all-pairs comparison, dense embeddings/matrices, gradients, GPU execution, or an external model in the runtime path.
- Reject it if apparent structures are explained by token/hash collisions, raw frequency, source identity, fixed input slots, a predeclared hierarchy, or post-hoc selection of unit count, threshold, pattern count, or depth.
- Reject the candidate if every experience allocates a new assembly, one assembly captures nearly everything, units become permanently dead/active, overlaps are nondiscriminative, hierarchy grows without saturation, or resource cost scales as a scan over all prior experiences.
- Reject a latent pattern if it lacks prospective held-out utility, disappears under exact replay, cannot be causally connected to output by ablation, erases rare exceptions, or continues influencing proposals after its supporting evidence is contradicted, revised, expired, or removed.
- Do not call an opaque ID knowledge merely because humans cannot name it. Uninterpretable noise, collisions, or evaluator-selected clusters are not human-unknown concepts.

### Acceptance Gate

- Across at least five preregistered seeds and structurally/source-held-out histories, the intact anonymous-reuse arm improves a frozen future prediction or withheld-relation metric over the no-homeostasis local arm and shuffled controls while remaining competitive with explicit-motif and supervised references under equal runtime budgets.
- At least one anonymous assembly is reused across independent entities and contexts, predicts held-out outcomes, survives deterministic replay, and loses the corresponding advantage under targeted unit/connection ablation; no training-time semantic label identifies it.
- Known hidden generators are recovered post hoc above frozen chance/control levels without being visible to learning, while negative random/non-compressible streams do not produce equally strong assemblies.
- Collapse, dead-unit, all-active, collision, rare-exception, contradiction, revision, expiry, capacity, and distribution-shift controls recover or abstain within fixed state, event, hierarchy, and CPU-latency ceilings.
- Production integration or semantic naming requires separate independent evidence, provenance review, and explicit human approval. The accepted claim remains bounded anonymous structural reuse, not biological equivalence or discovery of an objectively new human concept.

## Phase 40: Dynamical Structural Validation

**Goal:** test whether bounded replay, local prediction error, resonance, competition, inhibition, and homeostasis can cheaply rank or quarantine structural candidates by their prospective stability, while retaining explicit evidence and safety verification wherever dynamical stability is not equivalent to truth.

### Adopted Research References And Evidence Boundary

- Rao and Ballard's predictive-coding model uses feedback predictions and feedforward residual errors as a computational account of visual cortical responses: <https://doi.org/10.1038/4580>.
- Turrigiano et al. reported activity-dependent synaptic scaling in cultured neocortical neurons and proposed a stabilizing role during Hebbian modification: <https://doi.org/10.1038/36103>.
- Wilson and McNaughton reported increased reactivation during post-task sleep of hippocampal cells that co-activated during behavior: <https://doi.org/10.1126/science.8036517>.
- These references motivate local error, bounded competition/homeostasis, and replay controls. They do not establish that biological stability is a general truth test or that SARA can remove its verifier.
- Existing SARA predictive feedback, resonance credit, contradiction freeze, relation-stability scoring, repetition consolidation, idle replay, and homeostatic mechanisms are useful components and controls. They currently execute as separate explicit mechanisms and do not prove that a coupled dynamical circuit validates a structure.

### Research Hypothesis

- Represent each eligible structure candidate as a bounded replayable sparse activation trace with context, temporal order, expected next events, alternatives, source identities, revision, and expiry. Replaying it against an observation produces local match, residual-error, latency, and branch-competition events rather than a semantic true or false label.
- Shared prefixes may resonate while incompatible continuations compete locally. Bounded inhibitory selection may suppress a poorly supported continuation for the current context, but losing branches remain recoverable as exceptions, alternatives, or dormant candidates rather than being physically deleted.
- Update only locally active candidate/transition state using frozen prediction-error and timing rules. Apply fatigue, synaptic-style scaling, source-diversity caps, and total-activity ceilings so raw repetition or one dominant source cannot capture all capacity.
- Treat prospective dynamical stability as one candidate-quality signal: reproducible low residual error, calibrated prediction, context-appropriate branch selection, bounded recovery after revision, and resistance to exact-replay perturbation. High activation, frequency, synchrony, or survival alone is insufficient.
- Preserve a two-layer validation contract:
  1. a low-cost dynamical layer ranks, abstains, sleeps, or quarantines candidates;
  2. the existing explicit layer checks provenance, source independence, contradiction, causal/intervention requirements, policy, rollback, and external side effects before durable admission or action.
- A dynamical candidate cannot emit causes_verified, enter durable Event Memory, retire evidence, or authorize a tool action by stability alone. The explicit verifier remains mandatory for irreversible or safety-relevant boundaries.
- If Phase 39 later produces anonymous assemblies, Phase 40 may replay their canonical activation traces without naming them. This is an optional downstream arm; Phase 40 must first work with explicit frozen structures so anonymous emergence and dynamical validation are not confounded.

### [Later]

- Preregister after the Phase 30 temporal-state contract is frozen and the independent Phase 31 replay/consolidation workload exists. Reuse Phase 24/25 explicit verification as the fixed safety control.
- Freeze trace schema, replay schedule, prediction horizon, error calculation, resonance and competition gates, inhibitory capacity, fatigue/scaling bounds, branch-retention rule, source cap, revision/expiry behavior, state/event/latency ceilings, seeds, tuning attempts, and deterministic tie order before candidate implementation.
- Keep all dynamical decisions default-off, observed-only, non-durable, and unable to change production RISA, Event Memory, causal status, or tool state.
- Add the Phase 39 anonymous-assembly arm only after Phase 39 passes independently. Do not use a positive Phase 40 result to claim anonymous structure discovery.

### Minimum Experiment

Compare six equal-source, equal-replay, and equal-resource arms:

1. frequency/support ranking without prediction error;
2. the current explicit verifier/stability controls without dynamical replay;
3. local replay with prediction error but no competition or homeostasis;
4. replay plus bounded continuation competition/inhibition;
5. replay plus competition, fatigue, homeostasis, source-diversity caps, and recoverable minority branches;
6. the intact dynamical arm followed by the unchanged explicit evidence/safety verifier.

- Include stable recurring transitions, equal-prefix context-dependent branches, genuinely stochastic alternatives, rare but verified exceptions, delayed outcomes, one-source duplication, repeated misinformation, coordinated synchronized false patterns, source revisions, abrupt environment reversal, missing observations, irrelevant bursts, causal correlation without intervention, independent counterevidence, capacity pressure, and all-novel streams.
- Add canonical shuffles of timing, continuation identity, context, source identity, prediction/observation pairing, inhibition targets, and replay order. Include an explicit no-replay control and a control in which the same events occur with the wrong temporal order.
- Freeze train/evaluation histories by source, revision, and hidden generator. Duplicate text, paraphrased copies, descendant revisions, or replayed copies of one source cannot count as independent support.
- Report held-out next-event/relation quality, calibration and abstention, residual error, stable false-positive rate, rare-branch retention, branch diversity, source-independence sensitivity, contradiction quarantine, reversal/revision recovery, replay gain, replay-induced harm, oscillation/collapse rate, active/dormant candidate counts, targeted ablation delta, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject dynamical validation if raw frequency performs equally, shuffled timing/context/source controls retain the gain, or synchronized repeated misinformation becomes more stable than independently supported evidence.
- Reject it if competition erases valid context branches or rare verified exceptions, if homeostasis prevents learning or merely rescales scores without changing prospective behavior, or if replay amplifies stale/contradicted candidates.
- Reject it if the candidate requires semantic truth labels, global all-pairs structure comparison, dense matrices, gradients, GPU execution, an external model, unbounded replay, or a scan over all stored structures.
- Reject it if state becomes oscillatory, winner-take-all collapse is permanent, revision recovery exceeds the frozen horizon, deterministic replay fails, or events/state/latency exceed ceilings.
- Reject any proposal to remove the explicit verifier when provenance, causal intervention, policy, rollback, durable mutation, or external side effects are involved.
- Do not interpret stability, synchrony, resonance, prediction success, or biological inspiration as proof of truth, understanding, intelligence, or brain equivalence.

### Acceptance Gate

- Across at least five preregistered seeds and source-held-out histories, the intact local-dynamics arm improves a frozen prospective prediction/calibration metric over frequency, prediction-error-only, and shuffled controls under equal budgets.
- The hybrid arm preserves or improves explicit-verifier precision and abstention while reducing its candidate workload or total event cost; it must never admit a candidate rejected by the unchanged explicit safety boundary.
- Context-dependent alternatives and rare verified exceptions remain recoverable, repeated single-source misinformation does not gain verification strength, and revisions/environment reversals recover within frozen horizons.
- Targeted removal of the responsible replay/competition path removes the corresponding predictive advantage, while unrelated structures remain stable and total state/event/CPU cost stays bounded.
- Production integration requires independent temporal workloads, adversarial misinformation evidence, provenance review, and explicit human approval. The accepted claim is bounded dynamical candidate screening, not truth without verification.

## Phase 41: Structural Factorization and Bounded Compositional Search

**Goal:** test whether an unseen problem can be decomposed into a small set of reusable structural factors, solved by bounded typed composition, and grounded back into a verified answer more effectively than exact retrieval, path search, analogy, or flat transformation search.

### Research Hypothesis

- `Structural factor` is an operational SARA term, not a claim that one mathematically unique set of structural primes exists. A factor is a bounded typed relation/subgraph or transformation with anonymous role slots, input/output interfaces, preconditions, postconditions, temporal/context constraints, provenance, revision, and a canonical digest.
- A candidate qualifies as a reusable factor only when it satisfies all three preregistered properties: it recurs across independent structures, its composition reconstructs or predicts held-out structures within tolerance, and removing it causes a prospective loss that cannot be explained by labels, node identity, or source duplication. Smaller fragments that lose relational or predictive information are not better factors merely because they are shorter.
- Keep four operations distinct and auditable:

```text
problem grounding
  -> bounded structural-factorization hypotheses
  -> typed factor composition / state transition search
  -> concrete answer grounding
  -> evidence and consistency verification
```

- A problem may admit several valid factorizations. Retain a fixed small beam of alternatives and score them using frozen reconstruction, interface compatibility, evidence, exception, predictive-error, and description-cost terms. Minimum description length is one signal; the shortest decomposition does not override failed prediction, contradiction, or missing evidence.
- Candidate retrieval must be a bounded sparse cascade: local role/interface signatures and resonance select a fixed candidate set, typed preconditions reduce it further, and only the surviving compositions execute. A future scale claim such as `10^8 stored factors -> 10^3 candidates -> 10 compositions` is valid only after measured sublinear retrieval; it cannot be inferred from a small fixture or an unbounded offline index scan.
- Separate factor discovery from factor use. The initial experiment uses a frozen dictionary learned or selected only from training histories. Evaluation problems, answers, hidden generators, and evaluator labels cannot create, rename, merge, split, or rank factors. Phase 39 anonymous assemblies may later supply factor candidates, but the base factorization result must first work with Phase 37/38 explicit invariant and transformation controls.
- Treat structural tokenization as a learned codebook research question: useful units may be unnamed and larger than one edge, but must retain typed interfaces and evidence. Do not replace one human ontology with a fixture-specific ID per example, a dense embedding tokenizer, or an opaque external model.
- Every proposed solution carries the ordered factors, role bindings, intermediate structures, rejected alternatives, source evidence, uncertainty, and verifier result. A valid composition remains provisional until current RISA contradiction, causal, policy, and durable-admission boundaries pass.

### [Later]

- Begin the explicit-factor experiment only after Phase 37 freezes canonical role/invariant identity and Phase 38 proves exact `base + delta` reconstruction. Reuse Phase 21 composition/path controls and Phase 24/25 verification rather than creating a parallel solver or verifier.
- Preregister the factor schema, interface/type rules, dictionary construction split, maximum factors, maximum candidates per stage, factorization depth, composition length, alternative-beam width, cycle handling, score terms, tie order, state/event/latency ceilings, seeds, tuning attempts, and source/structural-family split before implementing the solver.
- Add self-discovered anonymous factors only after Phase 39 passes independently. Compare them against the same explicit dictionary and shuffled-interface controls; do not use semantic labels to name, select, or repair anonymous factors during evaluation.
- Keep the solver default-off, CPU-only, sparse, bounded, backpropagation-free, matrix-free, and unable to mutate durable structures. Large factor stores require a separately measured sparse-index experiment before any million- or billion-scale claim.

### Minimum Experiment

Compare seven equal-source and equal-resource arms:

1. exact verified-edge and bounded path retrieval;
2. Phase 37 whole-motif matching without factorization;
3. Phase 38 flat transformation search without an explicit factor hierarchy;
4. factorization with a fixed human-declared primitive dictionary;
5. factorization with a training-only reuse/reconstruction-derived dictionary;
6. the same learned dictionary with factor interfaces or composition order canonically shuffled;
7. the intact bounded factorization, sparse candidate cascade, composition, grounding, and unchanged explicit verifier.

- Use problems generated from hidden relation, topology, transformation, temporal, constraint, and exception factors. Hold out entire factor combinations and at least one domain while allowing the individual factors to appear separately in training.
- Include support-loss/fall, containment/absorption, finite-resource competition, delayed positive feedback with saturation and inhibition, multi-step tool-like state transitions, alternate valid decompositions, irrelevant high-overlap factors, incompatible interfaces, missing factors, cyclic compositions, contradictory evidence, revised transformations, rare exceptions, duplicated sources, adversarial hubs, and random/non-compressible structures.
- Freeze the hidden generator and answer before dictionary construction, but expose them only to the evaluator. Post-hoc recovery of a hidden factor is diagnostic; evaluation labels may not influence allocation, decomposition, retrieval, composition, or stopping.
- Report verified held-out solution accuracy/abstention, novel-combination and cross-domain transfer, exact reconstruction, factor reuse distribution, factor ablation delta, decomposition stability, alternate-solution coverage, interface/type rejection, exception retention, revision recovery, evidence-chain completeness, candidate counts at every cascade stage, explored compositions, branching factor, factor/codebook bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject structural factorization if it does not beat exact/path, whole-motif, and flat-transformation controls on held-out combinations, or if the shuffled-interface/order arm retains the gain.
- Reject it if the dictionary becomes one factor per experience, one universal factor, a copy of evaluator labels, a hidden answer table, or an unbounded hierarchy; if rare exceptions disappear; or if factors cannot reconstruct their supported structures.
- Reject it if decomposition quality is selected post hoc, the beam/candidate count grows with memory size, all factors are scanned, node names or task labels leak across splits, evaluation examples update the dictionary, or one source is counted as independent reuse.
- Reject a composition if interfaces, direction, order, context, preconditions, revisions, or evidence do not match; if shortest-description preference overrides a predictive or contradiction failure; or if concrete grounding invents entities or relations absent from the factor trace.
- Reject the scale claim if sparse retrieval is not measured against indexed-store growth, or if state, codebook, index, event, and latency costs exclude dormant factors. Dense all-pairs similarity, matrices, gradients, GPUs, and external semantic models are forbidden runtime shortcuts.
- Do not infer rediscovery of scientific laws, general problem solving, or human-like abstraction from reconstruction of fixture generators. A useful unnamed factor is evidence only of bounded prospective reuse under the frozen workload.

### Acceptance Gate

- Across at least five preregistered seeds and structurally/source-held-out histories, the intact arm improves verified unseen-combination quality over all three non-factorized baselines and both declared/learned dictionary controls under the same total resource envelope.
- Multiple independent structures reuse at least one factor, and targeted removal of that factor selectively breaks its held-out compositions while unrelated solutions remain stable. Interface/order shuffling causes the preregistered degradation.
- Missing, incompatible, cyclic, contradictory, revised, non-compressible, and capacity-exceeded problems abstain or recover deterministically without unbounded search, factor proliferation, or durable mutation.
- The learned dictionary reaches a frozen saturation/turnover bound, candidate cascade width remains fixed as the test store grows, and every accepted answer reconstructs an evidence-linked factor trace within byte, event, and CPU-latency ceilings.
- Anonymous-factor and large-store follow-ups require separate independent evidence and human review. Production reasoning integration remains blocked until the explicit-factor experiment, verifier preservation, and end-to-end provenance review all pass.

## Phase 42: Predictive Cross-Modal Structure Boundary

**Goal:** test whether evidence-linked structures learned in one sensory or symbolic channel can improve prediction in another, while learning a bounded continuum between broadly reusable, modality-family, and modality-specific structure instead of forcing all observations into one common representation.

### Research Hypothesis

- Keep modality-local values and shared relational structure distinct:

```text
experience
  = modality-local coordinates and bindings
  + typed relational/temporal structure
  + evidence, uncertainty, and revision state
```

- Raw values such as image position, color, optical flow, audio frequency/phase, tactile pressure/temperature/shear, and text token/syntax identity are not directly interchangeable. Candidate sharing begins only after bounded typed operations such as state, difference, order, boundary, approach/separation, increase/decrease, repetition, phase shift, transition, prediction, and deviation are extracted. This initial vocabulary is a declared control, not a claim that the universal primitives are complete or human-defined names are necessary.
- Do not assign a structure permanently to `universal` or `modality-specific`. Maintain a sparse evidence-linked applicability map over observed modalities or modality families. A candidate becomes more broadly applicable only when using it improves frozen held-out prediction in independently sourced target channels without exceeding a negative-transfer ceiling. It remains family-level or local when transfer is partial, harmful, or unsupported, and it may be demoted after contradiction or environmental revision.
- Cross-modal transfer means preservation of a useful relation or trajectory, not reconstruction of identical raw values or subjective equivalence. Examples include repetition followed by deviation, rise/peak/release, approach/contact/rebound, containment, temporal order, periodicity, and prediction-error correction. A text, video, audio, or tactile realization may bind different values while sharing only the tested transformation.
- Separate three levels operationally: modality-local coordinates, modality-family structure, and cross-family relational structure. The levels form an evidence-weighted abstraction gradient; they are not a fixed ontology. Similar surface rhythm, timestamps, captions, class labels, or paired-example IDs cannot establish sharing.
- Reuse Phase 23 for evidence-safe binding, Phase 37 for canonical role invariants, and Phase 41 for explicit factor traces. Phase 42 asks a different prospective question: whether a source-derived candidate improves a target modality after the candidate and target adapter are frozen. Successful same-modality reuse or post-hoc structural similarity is insufficient.
- Keep target adapters local, sparse, bounded, CPU-only, backpropagation-free, and matrix-free. A candidate may route through typed event interfaces, but it cannot use a dense common embedding, an opaque pretrained cross-modal model, all-pairs matching, or a shared answer table as the runtime explanation.

### [Later]

- Begin the base experiment after Phase 23 has independent text/vision/audio evidence and Phase 37 freezes canonical role and order identity. Phase 41 factorization is required only for the later learned-factor follow-up, not for the declared-relation baseline.
- Preregister source/target modalities, modality families, local coordinate schemas, candidate relation vocabulary, extraction and matching rules, applicability update/demotion rules, uncertainty and abstention thresholds, negative-transfer ceiling, candidate fan-out, state/event/latency budgets, seeds, tuning attempts, independent-source split, and every allowed transfer direction.
- Stage A uses independently sourced text, visual, and audio sequences. Add tactile transfer only after an independent licensed or consented tactile sensor dataset, typed tactile adapter, provenance review, and tactile-specific negative controls exist. Synthetic pressure traces may test mechanics but cannot support a claim about human touch.
- Keep all cross-modal candidates provisional and default-off. Phase 23 verification, contradiction freeze, Event Memory admission, and human review remain unchanged; predictive transfer cannot itself authorize durable knowledge.

### Minimum Experiment

Compare seven equal-source and equal-resource arms:

1. independent modality-local predictors with no shared structure;
2. declared cross-modal claim binding through the current Phase 23 path;
3. sharing by class label, paired-example ID, or aligned timestamp as an explicit leakage control;
4. a frozen human-declared relation vocabulary shared across all modalities;
5. the same vocabulary with one forced universal applicability level;
6. training-only predictive applicability learning over universal, modality-family, and local levels;
7. the intact hierarchical candidate with role, order, evidence, or source-to-target mapping canonically shuffled.

- Include same-structure/different-surface and same-surface/different-structure pairs for repetition/deviation, rise/peak/release, approach/contact/rebound, containment, temporal reversal, periodicity/phase slip, delayed response, and uncertainty. Include modality-local controls such as color contrast, lexical syntax, harmonic interval, and tactile texture/pressure responses so universalization has a measurable cost.
- Split by source, structural family, realization generator, and recording session before extraction. Base evaluation is unpaired: a target example cannot share a caption, soundtrack, object identity, timestamp, template revision, or example ID with the source evidence. Paired examples may appear only in a separately reported diagnostic arm.
- Freeze source candidates and target adapters before revealing held-out target outcomes. Evaluate every preregistered source-to-target direction separately; do not average away one-way transfer or negative transfer. A relation that transfers among only a subset of channels should remain a modality-family candidate.
- Report held-out predictive gain per transfer direction, negative-transfer rate and severity, calibrated abstention, applicability-map sparsity and revision, structural-family holdout, order/direction sensitivity, value and modality-identity leakage, shuffled-control delta, round-trip structural consistency, evidence-chain completeness, candidate fan-out, state bytes, event cost, CPU latency, and deterministic replay.

### Failure Conditions

- Reject a universal-structure claim if gains disappear on unpaired or structurally held-out examples, if the label/pairing control matches the candidate, or if captions, filenames, timestamps, class IDs, source identity, evaluator labels, or generator seeds leak across the boundary.
- Reject sharing if raw coordinates from unlike modalities are compared as though they had the same meaning, local values cannot be recovered separately, all candidates become universal, every example receives its own structure, or a modality-local control is incorrectly promoted without prospective benefit.
- Reject the candidate if aggregate improvement hides harmful source-to-target directions, rare exceptions or direction/order information disappear, contradiction and revision cannot demote applicability, or missing modalities are imputed with unjustified certainty.
- Reject a transfer mechanism that scans every stored candidate, performs dense all-pairs or matrix operations, uses gradients, GPUs, external semantic models, unbounded adapters, or increases candidate fan-out with memory size.
- Do not claim shared qualia, human sensory equivalence, a complete universal grammar, or modality-independent intelligence from structural prediction. Text/vision/audio evidence does not imply tactile capability, and synthetic tactile fixtures do not establish human-touch transfer.

### Acceptance Gate

- Across at least five preregistered seeds and independent source/structural-family splits, the hierarchical applicability arm improves frozen held-out prediction over the independent, Phase 23 binding, label/pairing, declared-vocabulary, and forced-universal controls under the same total resource envelope.
- At least one relation transfers prospectively across three independently sourced modalities, at least one transfers only within a preregistered modality family, and at least one modality-local control remains local; the learned applicability and calibrated abstention must reflect all three outcomes without evaluator-label access.
- Order/role/mapping shuffling removes the corresponding benefit, while targeted removal of a transferred structure selectively removes its target prediction gain and leaves unrelated local predictions stable. No accepted transfer may depend on an exact paired example.
- Contradiction, source replacement, and environmental reversal deterministically recalibrate or demote affected applicability without altering unrelated structures; negative transfer, state, event, fan-out, and CPU-latency ceilings hold at the longest registered horizon.
- Tactile claims require the separate independent tactile gate. Production routing or durable cross-modal knowledge requires a further external workload, provenance review, unchanged explicit verification, and human approval.

## Phase 43: Canonical One-Step State Learning and Rollout Drift Recovery

**Goal:** test whether SARA can learn bounded local state-update rules from canonical replay targets, then remain stable when those rules consume their own generated states, without backpropagation through time, dense latent vectors, a Transformer teacher, matrix operations, or GPU execution.

### Adopted Design Reference

- [Pretraining Recurrent Networks without Recurrence (SMT/DMT)](https://arxiv.org/html/2606.06479v2) separates *what a recurrent state should retain* from *how one step updates that state*. Its transferable idea is supervised one-step transition fitting from state targets, followed by training on learner-generated states to expose rollout drift.
- SARA adopts that experimental decomposition, not the paper's Transformer predictive-state encoder, dense learned memory vector, gradient training, or reported language/pixel results. Canonical evidence replay supplies frozen target states; sparse local transition counters and plasticity rules remain the candidate learner.
- The attached design discussion contributes four compatible refinements: explicit working/episodic/semantic/procedural state groups with different update horizons; novelty-, contradiction-, and utility-gated event updates; bounded external episodic recall instead of unbounded active state; and homeostatic correction of drift, saturation, collapse, overwrite, and contradiction.
- RISA remains the representation layer and Phase 43 is an update-law experiment. A positive transition result cannot establish anonymous concept emergence, semantic truth, architecture self-evolution, or physical-energy advantage.

### Research Hypothesis

- Freeze a canonical state-target generator after Phase 27 replay equivalence. From an accepted history, emit only adjacent training records:

```text
(canonical grouped state at t, accepted event at t + 1)
  -> canonical grouped state at t + 1
```

- The target record contains separate bounded digests and sparse active identifiers for working state, episodic cues, semantic/RISA revision state, and procedural policy state. It must not contain the withheld answer, future events, evaluator labels, or a full-history shortcut. Each group declares its own update horizon, capacity, provenance range, and absence state.
- Fit the one-step candidate through local event-driven observations only. Reuse the current `SparseInternalPredictor` as the count-based baseline; a new candidate must justify any additional state, contact, timing, novelty, contradiction, utility, or homeostatic channel under an equal total budget.
- Teacher-forced accuracy is insufficient because inference consumes candidate-generated states. During a training-only correction stage, roll the frozen candidate forward, compare its generated grouped state with the canonical target, and retain a bounded, deduplicated counterexample only when a frozen drift threshold is crossed. Correction labels may update local transitions but may not query the canonical generator during held-out inference.
- Treat prediction, reconstruction, causal/revision integrity, novelty retention, and downstream utility as separately reported constraints. Do not collapse them into a tunable scalar whose weights can hide a failed rare-event, contradiction, or provenance gate.
- Update work may be skipped for low-information repeats only when a frozen event gate preserves rare, delayed, contradictory, and revision-bearing evidence. Active recurrent state remains fixed-capacity; external episodic recall is sparse, cue-addressed, provenance-linked, and charged to the event/state/latency budget.

### [Later]

- Begin only after Phase 27 proves Python/Rust canonical replay equivalence for Event Memory, RISA proposals, and predictive feedback. Use Phase 19 multi-timescale state, Phase 22 revision histories, Phase 31 consolidation, and Phase 34 semantic checkpoints rather than creating a parallel memory hierarchy.
- Preregister state-group schemas, target construction, split boundaries, update horizons, event-gate inputs, local learning rules, counterexample capacity and eviction, correction threshold, rollout length, perturbations, budgets, seeds, tuning attempts, and all stop conditions before implementing a new learner.
- Keep the candidate sandboxed and default-off. No result may replace canonical replay, enable autonomous self-teaching, retire a prior learning system, or mutate production RISA/Event Memory without the separate Phase 36 migration and human-review gates.

### Minimum Experiment

Compare seven frozen equal-history and equal-resource arms:

1. canonical replay with no learned transition approximation;
2. the current `SparseInternalPredictor` count-based local baseline;
3. one-step teacher-forced sparse transition learning from canonical adjacent pairs;
4. the same learner with explicit working/episodic/semantic/procedural state groups and frozen per-group horizons;
5. grouped learning rolled on its own states with no correction collection;
6. grouped learning with bounded training-only rollout-error collection and local correction;
7. the intact arm with state-group identity, event/target pairing, temporal order, correction targets, or episodic cues canonically shuffled.

- Split histories by source and time before target generation. Include long clean sequences, rare delayed evidence, repetition, revisions, contradictions, omissions, duplicates, reordering, truncation, context switches, unknown states, state saturation, collapse, corrupted cues, and cases where the declared bounded state is intentionally insufficiently Markovian.
- Compare always-update and frozen novelty/contradiction/utility-gated update schedules inside the intact arm without changing retained evidence or total capacity. Compare no external recall, bounded associative episodic recall, and an over-budget diagnostic; the latter can expose a capacity limitation but cannot satisfy acceptance.
- Report exact next-state digest agreement per group, accepted-decision equivalence, rollout divergence rate and time, recovery steps, revision uptake, contradiction rejection, rare/delayed retention, false corrections, episodic-query precision/abstention, update/event counts, state and counterexample bytes, canonical-oracle queries during training, CPU latency, and deterministic replay.

### Failure Conditions

- Reject the candidate if it succeeds only under teacher forcing, reads canonical targets or the oracle at held-out inference, stores one answer/state per training example, carries hidden full history, leaks future events/evaluator labels, or uses unbounded counterexamples or episodic scans.
- Reject it if correction hides rather than repairs divergence; changes an unaffected state group; launders contradiction into semantic acceptance; overwrites rare protected evidence; collapses all contexts into one state; fragments into one state per example; or fails safely only by copying canonical state wholesale.
- Reject event gating if it saves common updates by dropping rare, delayed, contradictory, revision-bearing, or provenance-critical events. Reject external recall if cost or candidate fan-out grows with the entire store, cue corruption silently returns confident unrelated memory, or the result depends on source identity leakage.
- Reject any implementation requiring backpropagation, BPTT, gradients through rollout, dense matrices/embeddings, GPU execution, an online Transformer teacher, all-pairs search, or more state/events/tuning trials than its controls.
- Do not transfer SMT/DMT accuracy, scaling, training-time, or memory claims to SARA. A successful local transition benchmark is not evidence of language modeling parity, semantic understanding, self-evolution, or measured energy efficiency.

### Acceptance Gate

- Across at least five preregistered seeds and independent source/time splits, the bounded correction arm improves held-out self-rollout decision equivalence and time-to-divergence over teacher-forced-only, uncorrected self-rollout, current local predictor, and shuffled controls under the same total budget.
- Every accepted grouped state and downstream decision remains reproducible from canonical evidence; correction is training-only, deterministic, bounded, and selectively restores perturbed groups without changing unrelated groups or bypassing contradiction and revision policy.
- The event-gated arm reduces update/event work versus always-update while meeting frozen rare/delayed evidence, revision, contradiction, provenance, and abstention floors. Bounded episodic recall must improve a preregistered long-horizon case over no recall without exceeding fixed query, fan-out, state, and latency ceilings.
- Non-Markov, missing, corrupt, capacity-exceeded, and distribution-shift cases abstain or degrade visibly rather than fabricating stable state. Production integration requires independent workloads, Phase 36 migration review where applicable, and explicit human approval.

## Phase 44: Recursive Multi-Scale Sparse Canopy Routing

**Goal:** test whether the same bounded local routing-and-processing rule can be reused recursively across multiple structural scales, so easy inputs stop on shallow paths, difficult or novel inputs recruit deeper or cross-linked paths, and new reusable branches can be proposed without converting SARA into a fixed taxonomy, dense MoE, or matrix-based Transformer.

### Evidence Boundary

- [FractalNet](https://arxiv.org/abs/1605.07648) provides evidence that a self-similar macro-architecture can contain interacting subpaths of different lengths and support shallow-to-deep behavior. It does not establish language-model, SNN, RISA, structural-growth, or energy-efficiency gains.
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) provides evidence that token-level routing can allocate different compute depths under a fixed compute budget. Its learned dense router and capacity-constrained Top-k mechanism are comparison baselines, not SARA's presumed optimum.
- SARA adopts only the testable abstraction: repeat one local event-driven rule across scales, allow bounded early exit and sparse cross-links, and evaluate whether useful route depth and branch reuse emerge. Backpropagation, dense matrices, GPU execution, Transformer blocks, and published benchmark gains remain out of scope.

### Research Hypothesis

- Represent a route as a bounded sparse graph of reusable local resources rather than a human-authored subject tree. Each active resource applies the same scale-independent cycle: `observe -> compete/route -> update local state -> emit or stop`.
- The desired fractal property is algorithmic, not geometric. The same bounded `local processing -> integration -> feedback -> plasticity` contract should be reusable at contact, branch, neuron, circuit, and module scales; no acceptance metric may reward visual tree similarity or a topology authored to resemble biology.
- Start the growth arm from the smallest preregistered seed that can solve the control task. Canopy depth, branching, and cross-links must arise from local credited use under a global resource envelope, not from initializing the target hierarchy and merely tuning it.
- A parent may activate zero or more children under a frozen event/state budget. Variable-cardinality routing is the candidate mechanism; fixed Top-k is retained as a controlled baseline because it can confound useful specialization with forced capacity use.
- Depth is an observed consequence of local uncertainty, prediction error, novelty, conflict, or unresolved structure, not an evaluator label or a post-hoc difficulty annotation. Early exit must preserve calibrated abstention and cannot simply truncate unresolved cases.
- Cross-links may connect structurally reusable resources across branches, but their creation, reinforcement, pruning, and invalidation must be local, bounded, provenance-linked, deterministic, and reversible. A proposed branch remains sandboxed until replay and contradiction checks pass.
- Structural growth is tested as provisional resource allocation, not durable knowledge creation. Phase 39 must first show anonymous local reuse; Phase 32/35 supply fixed-boundary, depth, cardinality, and emergent-field controls; Phase 30 supplies temporal-state controls.

### [Later]

- Begin only after Phase 30 temporal controls and Phase 32 fixed/variable-cardinality depth-routing controls execute, and after Phase 39 demonstrates anonymous reuse independently of explicit motif labels. Use Phase 35 results when available to distinguish recursive scale reuse from merely overlapping expert fields.
- Preregister the local rule, maximum depth, branch and cross-link budgets, early-exit signal, growth/pruning thresholds, protection for rare evidence, deterministic tie rules, histories, splits, seeds, tuning attempts, and stop conditions before implementing the canopy candidate.
- Keep all canopy routing, growth, and pruning default-off. It may not mutate production RISA/Event Memory, replace canonical verification, or claim self-evolution without Phase 36 migration and explicit human approval.

### Minimum Experiment

Compare seven equal-history, equal-total-state, equal-event, and equal-tuning-budget arms:

1. flat sparse local routing with no hierarchy;
2. a fixed human-authored hierarchy with the same resources;
3. fixed-capacity Mixture-of-Depths-style Top-k depth routing;
4. recursive variable-cardinality canopy routing with no cross-links or growth;
5. a minimal-seed network using scale-specific local rules and the same growth budget;
6. the intact minimal-seed bounded canopy using one scale-reused rule with sparse cross-links and provisional growth/pruning;
7. the intact arm with scale identity, parent-child links, cross-links, temporal order, exit signals, or reuse assignments canonically shuffled.

- Use source-, time-, and structural-family-disjoint histories containing easy and difficult cases, staged specialization, cross-domain structural reuse, novel compositions, contradictions, reversals, rare knowledge, noisy cues, distribution shifts, and deliberately insufficient budgets. Difficulty labels are evaluator-only.
- Report held-out decision accuracy and abstention, early-exit calibration, route depth and event cost by case family, compute-depth correlation, path overlap, cross-domain reuse, branch/cross-link growth and pruning, dead-branch and dominant-branch rates, catastrophic interference, rare-evidence retention, revision recovery, state bytes, fan-out, CPU latency, and deterministic replay.
- Include matched no-growth and no-cross-link ablations plus an over-budget diagnostic. The latter can reveal a capacity limit but cannot satisfy acceptance.

### Failure Conditions

- Reject the candidate if gains disappear under equal total resources, arise from extra depth/capacity/tuning, depend on task/domain/difficulty labels, store one branch per example, require a fixed human taxonomy or biologically shaped initialization, or use global/all-pairs route search.
- Reject it if all inputs take effectively the same path, shallow exit confidently truncates unresolved evidence, fixed Top-k alone explains the result, branches collapse into one dominant route, dead branches accumulate, or uncontrolled cross-links make provenance and invalidation non-local.
- Reject growth/pruning if revisions cannot demote obsolete paths, rare protected knowledge is erased, contradictory branches are silently merged, resource use grows with the full memory store, or replay cannot reconstruct each routing and structural decision.
- Reject any implementation requiring backpropagation, dense matrices or embeddings, GPU execution, an online Transformer router, evaluator labels at inference, or more state/events/tuning trials than its controls. A positive result is not evidence of LLM parity, autonomous self-evolution, human-like concepts, or measured energy efficiency.

### Acceptance Gate

- Across at least five preregistered seeds and independent source/time/structural-family splits, the intact canopy improves a frozen held-out quality-versus-event metric over flat, fixed-hierarchy, fixed Top-k, no-growth/no-cross-link, and shuffled controls under the same total resource envelope.
- Route depth responds prospectively to frozen uncertainty/error/novelty signals, shallow cases exit earlier without losing calibrated accuracy or abstention, and difficult/novel cases receive additional bounded work without evaluator-label access.
- At least one anonymous local resource is reused across independently sourced domains and scales; targeted removal must selectively remove its predicted benefit while leaving unrelated paths stable. Cross-links and new branches must add value beyond capacity alone.
- Growth and pruning remain bounded, deterministic, provenance-linked, replayable, revision-sensitive, and safe for rare evidence at the longest registered horizon. Production use requires independent workloads, migration review, and explicit human approval.

## Phase 45: Hierarchical Local Credit Assignment with Backward Information

**Goal:** test whether bounded backward information can assign outcome credit through recently active dendritic and canopy structure with substantially better delayed and deep credit assignment than STDP/Hebbian learning alone, without requiring global differential gradients.

### Claim Boundary

- SNN learning difficulty is not considered solved. The testable hypothesis is that part of the difficulty attributed to spikes or local learning instead comes from point-neuron simplification and the removal of dendritic subunits, hierarchical eligibility, backward local feedback, recurrence, structural plasticity, and multi-timescale credit memory.
- The candidate does not “replace Backprop” merely by sending reward backward through a tree. It must show correct causal attribution, delayed learning, joint-cause handling, reversal, and held-out outcome improvement beyond point-neuron and non-hierarchical local controls.
- Strong surrogate-gradient SNN and ANN/Transformer results may be included as offline diagnostic references with matched task definitions and transparent resource differences. They are not runtime dependencies, and failing to approach them must be reported rather than hidden behind biological plausibility or event-cost claims.

### Policy Boundary

- “Backward” describes information direction, not a derivative. Permitted signals include soma outcome events, bAP-like branch notifications, prediction-error category and magnitude, success/failure, bounded global reward or neuromodulation, novelty, and replayed consequence.
- No candidate may require differentiating an end-to-end loss through the network, maintaining an autograd tape, transporting exact downstream derivatives, symmetric forward/backward weights, BPTT, surrogate gradients, or GPU execution.
- Global reward/modulation may be broadcast, but synaptic change must require a locally stored eligibility trace and explicit structural membership. A reward alone cannot update every active connection.

### Research Hypothesis

- During forward activity, each contact records a bounded decaying eligibility trace containing activation time, order, branch identity, coincidence, local prediction contribution, novelty, and provenance. Parent branches retain only bounded aggregate child evidence.
- When the soma or a downstream verified outcome emits backward information, credit is routed through the explicit hierarchy from recently contributing branches to sub-branches and contacts. Routing uses activity and structural traces, not a numerical chain-rule derivative.
- Freeze an initial local rule of the form `credit = eligibility × temporal proximity × branch contribution × outcome modulation × novelty`, with signed success/failure modulation, normalization, saturation, and abstention when contribution is ambiguous. Do not tune a hidden scalar blend after observing held-out results.
- Apply credit to bounded local weight, delay, short-term state, contact selection, and synaptic tags first. Branch growth, cross-link formation, and pruning are separate slow-timescale actions that require repeated credited evidence, homeostatic budgets, replay validation, contradiction handling, and reversible provenance.
- The hierarchy may accelerate attribution but must not force one winner. Competing or jointly necessary branches may share calibrated credit; counterfactual branch suppression and targeted replay must test whether assigned credit is causal rather than merely correlated.

### [Later]

- Begin after Phase 33 provides branch/contact controls and Phase 44 freezes canopy topology, path, growth, and pruning controls. Use Phase 31 consolidation tags and Phase 40 dynamical validation when available rather than creating parallel stability systems.
- Preregister all trace fields, decay windows, backward-event types, credit formula, ambiguity/abstention rule, normalization, structural update schedule, budgets, seeds, tuning attempts, causal interventions, and stop conditions before implementing a new learner.
- Keep backward events and structural updates sandboxed, default-off, deterministic, and auditable. Passing an observed-only benchmark cannot authorize production topology mutation or claim biological equivalence.

### Minimum Experiment

Compare eight equal-forward-history, equal-state, equal-event, and equal-tuning-budget local arms:

1. fixed network with no learning;
2. point-neuron local Hebbian/STDP learning only;
3. dendritic local Hebbian/STDP with identical total contacts but no hierarchical credit;
4. point-neuron reward-modulated STDP with a global scalar;
5. flat eligibility-trace credit without branch structure;
6. hierarchical branch-to-contact credit with temporal eligibility;
7. the intact hierarchical arm plus slow bounded growth/pruning;
8. intact arms with branch identity, activity history, event time, outcome sign, eligibility, or parent-child paths canonically shuffled.

- Include delayed rewards, deep paths, jointly necessary branches, distractor activity, branch competition, repeated success, explicit failure, novelty, reversals, contradictions, rare causal paths, misleading correlations, variable delays, missing outcome, and structurally identical cases with different causal interventions.
- Where feasible, include two separately reported offline diagnostic references: a surrogate-gradient/BPTT SNN and an ANN or Transformer appropriate to the frozen task. Match forward capacity where meaningful, disclose optimizer and hardware advantages, and never use either model to generate candidate-runtime labels, routes, traces, or topology. These references cannot satisfy acceptance for the local mechanism.
- Report causal-credit precision/recall, signed update accuracy, delayed-credit retention, joint-cause allocation, distractor suppression, reversal recovery, topology precision, false growth/pruning, rare-path survival, held-out task quality, updates/events/state bytes, fan-out, CPU latency, and deterministic replay.

### Failure Conditions

- Reject the mechanism if it merely reinforces the most recent or strongest branch, broadcasts reward updates indiscriminately, cannot represent joint causes, silently assigns credit when evidence is ambiguous, or depends on evaluator labels or future events at inference.
- Reject apparent improvement caused by extra state, events, depth, capacity, tuning, replay passes, or one trace/branch per example. Reject topology learning that grows without a fixed budget, prunes rare valid paths, cannot undo obsolete credit after reversal, or loses source-linked reconstruction.
- Reject any path that computes or approximates an end-to-end chain-rule gradient as a mandatory runtime operation, stores a whole-network backward graph, uses BPTT/surrogate gradients, dense matrices, or GPUs. Local scalar prediction-error or derivative-like signals are allowed only when computed from locally available variables and reported precisely.
- Do not claim “Backprop-level” credit assignment from correlation with an offline gradient. The candidate must improve causal interventions and held-out outcomes over local controls within its own bounded runtime.

### Acceptance Gate

- Across at least five preregistered seeds and independent source/time/causal-structure splits, hierarchical credit improves frozen causal-credit and held-out outcome metrics over point-neuron STDP, dendritic STDP without hierarchy, reward-modulated STDP, flat eligibility, and shuffled controls under identical local budgets.
- Credit must follow targeted causal interventions: removing or delaying a contributing branch changes its assigned credit predictably, while inactive distractors and unrelated branches remain stable. Joint causes receive non-zero bounded shares and ambiguous cases abstain.
- Delayed success, failure, contradiction, and reversal update the correct eligible structures without overwriting rare protected paths. Growth/pruning must add benefit beyond weight/delay updates alone and remain bounded, reversible, provenance-linked, and deterministically replayable.
- The offline global-gradient reference, when used, is reported only as a diagnostic gap. Promotion requires independent workloads, unchanged durable-memory verification, migration review for topology changes, and explicit human approval.

## Phase 46: Multi-Timescale Delayed Structural Credit

**Goal:** test whether short local eligibility can be connected to minutes-, sessions-, and long-horizon consequences through bounded hierarchical summaries and targeted replay, without retaining an unbounded computation graph or scanning all past activity.

### Research Hypothesis

- Separate credit state by timescale: fast contact eligibility, branch contribution summaries, episodic causal anchors, and consolidated structural tags. Each slower level stores fewer provenance-linked records and never fabricates detail discarded by a faster level.
- A delayed verified outcome first updates still-live local eligibility. If that has expired, it may retrieve only a bounded set of causally plausible episodic anchors, replay their canonical event paths, and regenerate local eligibility before applying Phase 45 credit.
- Event Memory stores compressed route/module/branch participation and evidence ranges, not a hidden full activation tape. Credit retrieval proceeds `outcome -> episodic anchors -> relevant modules -> relevant branches -> regenerated local eligibility`, with explicit uncertainty at every narrowing step.
- Credit may cross neuron, circuit, and module boundaries only through typed sparse events carrying outcome, time range, route identity, provenance, uncertainty, and bounded modulation—not through an end-to-end derivative or an unrestricted broadcast update.
- Repeated delayed evidence may consolidate a structural tag or protected path; contradiction, reversal, source invalidation, and counterfactual replay must weaken or quarantine it. Frequency alone is insufficient.

### [Later]

- Begin only after Phase 31 consolidation, Phase 43 bounded rollout/replay state, and Phase 45 local causal-credit controls are frozen. Reuse Event Memory and canonical replay rather than creating an unbounded training-history store.
- Preregister timescale boundaries, decay and promotion rules, episodic anchor schema, retrieval fan-out, replay count, causal filters, ambiguity/abstention behavior, resource ceilings, seeds, tuning attempts, and maximum credit delay before implementation.
- Keep delayed credit observed-only and default-off. It may not rewrite durable memory, autonomously reactivate arbitrary history, or trigger external actions.

### Minimum Experiment

Compare no learning, immediate-only Phase 45 credit, one flat long eligibility trace, bounded multi-timescale traces without replay, intact targeted replay, an unlimited-history diagnostic, and time/route/provenance/outcome-shuffled controls under equal accepted evidence and declared budgets.

- Evaluate seconds-, minutes-, session-, and cross-session delays; distractor-rich intervals; multiple plausible causes; repeated and one-shot rare causes; delayed failure; reversal; contradiction; missing outcome; corrupted cues; source invalidation; and consequences requiring jointly separated events.
- Report causal-credit precision/recall by delay, credit survival curve, false remote attribution, replay precision/abstention, joint-cause allocation, reversal recovery, rare-path retention, held-out outcome gain, retrieved anchors, replayed events, state bytes, CPU latency, and deterministic reconstruction.

### Failure Conditions

- Reject the mechanism if long-delay gains require a trace whose state grows with elapsed history, full-store scans, hidden future labels, exact episode IDs supplied by the evaluator, unlimited replay, or one persistent trace per training example.
- Reject it if replay reinforces correlation without causal-intervention support, delayed reward contaminates unrelated recent activity, frequent paths erase rare valid causes, expired evidence is reconstructed with false certainty, or invalidated sources continue receiving credit.
- Reject mandatory global-gradient backpropagation, BPTT, dense matrices, GPUs, or an online external teacher. The unlimited-history arm is diagnostic only and cannot pass acceptance.

### Acceptance Gate

- Across at least five preregistered seeds and independent source/time/causal-family splits, bounded targeted replay improves delayed causal-credit and held-out outcomes over immediate-only, flat-trace, no-replay, and shuffled controls while remaining below fixed state, fan-out, replay, event, and latency ceilings.
- Targeted causal interventions selectively change the credited historical path; unrelated activity stays stable, ambiguous retrieval abstains, and contradiction/reversal/source invalidation deterministically demote affected credit.
- The bounded mechanism must retain a preregistered fraction of the unlimited-history diagnostic's benefit at every accepted delay without cost growing with total history. Production consideration requires independent long-horizon workloads and explicit human approval.

## Immediate Execution Order

1. [Done] Implement bounded RISA subgraph composition and structural analogy.
2. [Done] Build the observed-only 10/30/100-episode continual horizon benchmark.
3. [Done] Connect the horizon benchmark to Event Memory retention profiles.
4. [Done] Add multimodal structural contradiction and missing-modality cases.
5. [Done] Implement and execute the registered Phase 34 independent adapter v2 across all 1,050 conditions; retain `promotion_ready=false` because the result is exact source identity rather than semantic recall. Continue multimodal evidence collection separately.
6. [Done] Complete Python/Rust canonical replay equivalence, bind four independent subsystem reports fail-closed, and retain their separate claim boundaries.
7. [Done] Run exact-tokenization conformance, scalar/batch/immutable-snapshot ablations, and the five-process stability gate; retain the optional accelerated path as unpromoted negative evidence because worst-run speedup failed.
8. [Later] Prototype Phase 30 temporal effective interactions only after preregistering the four-arm equal-budget ablation.
9. [Done] Implement the observed-only Phase 31 repetition-dependent consolidation contract without connecting it to production recall.
10. [Later] Preregister the Phase 32 four-arm sparse depth-routing experiment before implementing either candidate.
11. [Done] Implement the immutable Phase 33 structured-edge preregistration contract and managed CLI.
12. [Done] Freeze the Phase 33 observed-only fixture and CPU environment fingerprints, then register the immutable protocol before candidate execution.
13. [Done] Execute all 1,275 immutable Phase 33 observed-only conditions with matching fingerprints and bounded deterministic replay; retain `promotion_ready=false` because simplification and independent evidence are absent.
14. [Done] Register the TwinProp-inspired follow-up as immutable experiment `phase33-twinprop-ablation-observed-v1` without modifying the completed parent registration.
15. [Done] Execute all 350 registered TwinProp-inspired conditions with the fixed readout and equal resource/tuning budgets; retain `promotion_ready=false` because evidence is synthetic and observed-only.
16. [Later] Evaluate the frozen structured-edge mechanism on independent temporal and structural workloads, including the preregistered matrix-free response-shape and influence audit, before production review.
17. [Done] Preregister the Phase 34 four-arm bounded checkpoint-caching experiment before implementing any cache candidate.
18. [Done] Implement and execute all 64 registered Phase 34 conditions without changing the frozen protocol or production defaults; retain `promotion_ready=false` because cache-arm trade-offs, five replicates, and independent evidence are absent.
19. [Done] Register the separate five-seed, 12-family Phase 34 cache-arm separation follow-up before implementing its runtime.
20. [Done] Execute all 240 registered separation conditions; preserve the failed recent-resolution and Top-k gates as negative evidence and keep production promotion closed.
21. [Done] Register the explicitly reviewed 300-condition retention-by-selection factorial with identical retained sets inside each selection comparison.
22. [Done] Execute all 300 registered factorial conditions; identify the synthetic Top-k main effect while retaining `promotion_ready=false` and production isolation.
23. [Done] Add the independent factorial readiness gate and freeze six missing domain/horizon collection targets without fabricating evidence.
24. [Done] Execute registered adapter v2 across all 1,050 conditions with retained-set identity and no selector retuning; all integrity and identity thresholds passed under the frozen exact-identity scope.
25. [Done] Register and collect the exact 30-file CPython `v3.14.6` snapshot at commit `c63aec69bd59c55314c06c23f4c22c03de76fe45`; preserve the Raw HTTP 429 failure and the successful shallow-Git acquisition as separate immutable contracts.
26. [Done] Complete human source-alignment review for the six hash-bound transcribed excerpts before preregistering semantic delayed recall.
27. [Later] Reopen physical-energy evidence only by explicit operator decision.
28. [Later] After Phase 32 fixed-expert controls execute, preregister the Phase 35 emergent overlapping spatiotemporal expert-field experiment before implementing any boundary-free routing candidate.
29. [Later] After Phase 27 replay/migration equivalence and independent Phase 22 histories exist, preregister the Phase 36 evidence-preserving learning-system evolution experiment before implementing any representation translator or predecessor-retirement path.
30. [Done] Preregistered the Phase 37 structural-invariant sharing experiment before implementing any shared-pattern store or unstored-relation proposal path; protocol fingerprint `e77d34460bfc2ae2440d765616a65ce7dad734d07ef6cca3b0d17b1532cfe704`.
31. [Done] Implemented and executed the default-off observed-only Phase 37 candidate once against the frozen identities. The result is retained negative: abstention/safety/resource gates passed, but novel-relation precision/recall and held-out-domain transfer were zero and the intact arm did not beat the baselines.
32. [Done] Preregistered Phase 38 canonical structural-delta and transformation memory before implementing a codec or shared transformation store; protocol fingerprint `9dfafe9e…80dd`.
33. [Done] Froze Phase 38 source/split and evaluator-isolated execution identities across all eleven operators and twenty-six registered case families; evidence remains registered synthetic control and production promotion is closed.
34. [Done] Implemented and evaluated the default-off Phase 38 codec. Valid exact reconstruction, digest, rollback, tombstone, evidence, replay, and resource gates passed, but malformed-control abstention was `0.5`; the result is retained negative and transformation sharing was not executed.
35. [Done] Freeze 130 Phase 30 source-/temporal-generator-disjoint histories and evaluator-only labels with exact canonical digests, 33,280 total events, and fail-closed tamper/isolation tests.
36. [Done] Implement Phase 30 fixed, history-averaged, temporal-state-only, and bounded-cache controls with deterministic replay, exact invalidation traces, and full 520-run resource conformance.
37. [Done] Execute all 520 evaluator-isolated Phase 30 decisions; retain the passed resource/replay gate and failed threshold/comparative/mechanism gates as immutable negative evidence with `promotion_ready=false`.
38. [Done] Preregister Phase 39 anonymous local reuse against the completed negative Phase 30 and Phase 37 controls; protocol fingerprint `5dfecedc…bf57f` freezes labels, six arms, twenty-one families, four shuffles, five seeds, budgets, and one tuning attempt.
39. [Done] Freeze 210 Phase 39 source-/hidden-generator-disjoint histories and evaluator-only keys with 53,760 events, zero registered evaluator fields in candidate inputs, and freeze fingerprint `7eb5ba8b…8c7d4`.
40. [Next] Implement Phase 39 bounded anonymous units, local candidate neighborhoods, reuse/allocation, homeostasis/fatigue/EI, overlapping assemblies, and deterministic evidence/revision traces against the frozen inputs.
41. [Later] After the Phase 30 temporal-state contract and independent Phase 31 replay/consolidation evidence are frozen, preregister the Phase 40 dynamical structural-validation experiment before coupling replay, competition, inhibition, or homeostasis into a structural admission signal.
42. [Done] Register the separate 270-case Phase 34 semantic delayed-recall workload against the passed hash-bound human-review gate before implementing any semantic adapter or selecting semantic thresholds.
43. [Done] Implement the registered semantic adapter/evaluator with strict evaluator-label isolation and execute all 6,750 frozen conditions without retuning selectors or changing production paths; all registered gates passed within the six-proposition scope and `promotion_ready=false` remains fixed.
44. [Later] After Phase 37 canonical invariants and Phase 38 reconstructible transformations pass, preregister the Phase 41 explicit structural-factorization experiment before implementing any factor dictionary or compositional solver; add anonymous factors only after Phase 39 passes independently.
45. [Later] After independent Phase 23 text/vision/audio evidence and the Phase 37 canonical role schema are frozen, preregister the Phase 42 predictive cross-modal boundary experiment; do not add tactile claims until an independent tactile dataset and adapter pass their own gate.
46. [Later] After Phase 27 cross-runtime canonical replay equivalence and independent Phase 22 revision histories are frozen, preregister the Phase 43 canonical one-step state-learning experiment; do not implement rollout correction or self-teaching before its state targets, leakage boundary, and bounded counterexample policy are immutable.
47. [Later] After Phase 30 and Phase 32 controls execute and Phase 39 demonstrates anonymous local reuse, preregister the Phase 44 recursive multi-scale sparse canopy experiment; treat fixed Top-k as a comparison baseline and keep branch growth, pruning, and cross-links provisional and default-off.
48. [Later] After Phase 33 branch/contact controls and Phase 44 canopy controls are frozen, preregister Phase 45 hierarchical local credit assignment; permit bounded backward information but do not require global differential gradients, BPTT, or a whole-network backward graph.
49. [Later] After Phase 31 consolidation, Phase 43 bounded rollout/replay state, and Phase 45 local causal-credit controls are frozen, preregister Phase 46 multi-timescale delayed structural credit; keep retrieval/replay bounded and reject history-sized eligibility or global backward graphs.

## Required Managed Outputs

- `data/processed/benchmark_fixtures/next_level_structural_cases.jsonl`
- `data/processed/benchmark_fixtures/continual_horizon_cases.jsonl`
- `workspace/evaluation/next_level_structural_benchmark.json`
- `workspace/evaluation/continual_horizon_benchmark.json`
- `workspace/evaluation/phase23_structural_fusion_benchmark.json`
- `data/processed/benchmark_fixtures/phase34_semantic_delayed_recall_cases.jsonl`
- `workspace/evaluation/phase34_semantic_delayed_recall_preregistration.json`
- `workspace/evaluation/phase34_semantic_delayed_recall_benchmark.json`
- `data/processed/autobot/phase23_independent_multimodal_manifest.jsonl`
- `workspace/evaluation/phase23_external_multimodal_gate.json`
- `workspace/autobot/phase23_multimodal_collection_targets.json`
- `workspace/evaluation/phase24_causal_benchmark.json`
- `workspace/evaluation/phase25_agent_loop_benchmark.json`
- `workspace/evaluation/phase27_independent_decision_replay.json`
- `workspace/evaluation/next_level_research_journal.jsonl`
- `workspace/evaluation/next_level_promotion_gate.json`
- `workspace/evaluation/next_level_human_approval.json`
- `workspace/evaluation/scale_up_experiment_readiness.json`
- `data/processed/benchmark_fixtures/phase27_tokenizer_conformance_cases.jsonl`
- `workspace/evaluation/phase27_tokenizer_acceleration_benchmark.json`
- `data/processed/benchmark_fixtures/phase30_temporal_effective_interaction_cases.jsonl`
- `data/processed/benchmark_fixtures/phase30_temporal_effective_interaction_evaluator_key.jsonl`
- `workspace/evaluation/phase30_temporal_effective_interaction_fixture_freeze.json`
- `workspace/evaluation/phase30_temporal_effective_interaction_decisions.json`
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
- `workspace/evaluation/phase33_twinprop_ablation_environment.json`
- `workspace/evaluation/phase33_twinprop_ablation_preregistration_draft.json`
- `workspace/evaluation/phase33_twinprop_ablation_preregistration.json`
- `workspace/evaluation/phase33_twinprop_ablation_benchmark.json`
- `data/processed/benchmark_fixtures/phase34_memory_checkpoint_cache_cases.jsonl`
- `workspace/evaluation/phase34_memory_checkpoint_cache_environment.json`
- `workspace/evaluation/phase34_memory_checkpoint_cache_preregistration_draft.json`
- `workspace/evaluation/phase34_memory_checkpoint_cache_preregistration.json`
- `workspace/evaluation/phase34_memory_checkpoint_cache_benchmark.json`
- `data/processed/benchmark_fixtures/phase34_memory_cache_separation_cases.jsonl`
- `workspace/evaluation/phase34_memory_cache_separation_environment.json`
- `workspace/evaluation/phase34_memory_cache_separation_preregistration_draft.json`
- `workspace/evaluation/phase34_memory_cache_separation_preregistration.json`
- `workspace/evaluation/phase34_memory_cache_separation_benchmark.json`
- `data/processed/benchmark_fixtures/phase34_memory_cache_factorial_cases.jsonl`
- `workspace/evaluation/phase34_memory_cache_factorial_environment.json`
- `workspace/evaluation/phase34_memory_cache_factorial_preregistration_draft.json`
- `workspace/evaluation/phase34_memory_cache_factorial_preregistration.json`
- `workspace/evaluation/phase34_memory_cache_factorial_benchmark.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_gate.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_case_plan.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_adapter_v2_environment.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_adapter_v2_preregistration_draft.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_adapter_v2_preregistration.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_adapter_v2_benchmark.json`
- `workspace/evaluation/phase34_memory_cache_factorial_independent_provenance_review.json`
- `workspace/evaluation/phase34_cpython_v3_14_6_snapshot_preregistration.json`
- `workspace/evaluation/phase34_cpython_v3_14_6_git_snapshot_preregistration.json`
- `data/raw/phase34_cpython_git_snapshot/source_rows.jsonl`
- `data/processed/autobot/phase34_cpython_v3_14_6_git_snapshot_manifest.jsonl`
- `workspace/evaluation/phase34_cpython_v3_14_6_git_snapshot_collection.json`
- `workspace/evaluation/phase34_transcribed_excerpt_human_review_request.json`
- `workspace/evaluation/phase34_transcribed_excerpt_human_review_decisions.json`
- `workspace/evaluation/phase34_transcribed_excerpt_human_review_gate.json`
- `workspace/evaluation/phase34_transcribed_excerpt_review_support_preregistration.json`
- `data/raw/phase34_review_support/source_rows.jsonl`
- `workspace/evaluation/phase34_transcribed_excerpt_review_comparison_packet.json`
- `workspace/evaluation/phase34_transcribed_excerpt_review_support_collection.json`
- `data/raw/architecture_migration/source_rows.jsonl`
- `data/processed/autobot/architecture_migration_latent_manifest.jsonl`
- `workspace/evaluation/continual_horizon_external_collection.json`
- `data/processed/benchmark_fixtures/phase36_learning_system_evolution_cases.jsonl`
- `workspace/evaluation/phase36_learning_system_evolution_preregistration.json`
- `workspace/evaluation/phase36_learning_system_evolution_benchmark.json`
- `workspace/evaluation/phase36_learning_system_retirement_review.json`
- `data/processed/benchmark_fixtures/phase37_structural_invariant_sharing_cases.jsonl`
- `workspace/evaluation/phase37_structural_invariant_sharing_preregistration.json`
- `workspace/evaluation/phase37_structural_invariant_sharing_benchmark.json`
- `data/processed/benchmark_fixtures/phase38_structural_delta_cases.jsonl`
- `workspace/evaluation/phase38_structural_delta_preregistration.json`
- `workspace/evaluation/phase38_structural_delta_codec_conformance.json`
- `workspace/evaluation/phase38_transformation_memory_benchmark.json`
- `data/processed/benchmark_fixtures/phase39_anonymous_structure_reuse_cases.jsonl`
- `data/processed/benchmark_fixtures/phase39_anonymous_structure_reuse_evaluator_key.jsonl`
- `workspace/evaluation/phase39_anonymous_structure_reuse_fixture_freeze.json`
- `workspace/evaluation/phase39_anonymous_structure_reuse_preregistration.json`
- `workspace/evaluation/phase39_anonymous_structure_reuse_benchmark.json`
- `workspace/evaluation/phase39_anonymous_structure_ablation.json`
- `data/processed/benchmark_fixtures/phase40_dynamical_structural_validation_cases.jsonl`
- `workspace/evaluation/phase40_dynamical_structural_validation_preregistration.json`
- `workspace/evaluation/phase40_dynamical_structural_validation_benchmark.json`
- `workspace/evaluation/phase40_dynamical_structural_validation_ablation.json`
- `data/processed/benchmark_fixtures/phase41_structural_factorization_cases.jsonl`
- `workspace/evaluation/phase41_structural_factorization_preregistration.json`
- `workspace/evaluation/phase41_structural_factorization_benchmark.json`
- `workspace/evaluation/phase41_structural_factorization_ablation.json`
- `data/processed/benchmark_fixtures/phase42_cross_modal_structure_cases.jsonl`
- `workspace/evaluation/phase42_cross_modal_structure_preregistration.json`
- `workspace/evaluation/phase42_cross_modal_structure_benchmark.json`
- `workspace/evaluation/phase42_cross_modal_structure_ablation.json`
- `data/processed/benchmark_fixtures/phase44_recursive_canopy_routing_cases.jsonl`
- `workspace/evaluation/phase44_recursive_canopy_routing_preregistration.json`
- `workspace/evaluation/phase44_recursive_canopy_routing_benchmark.json`
- `workspace/evaluation/phase44_recursive_canopy_routing_ablation.json`
- `data/processed/benchmark_fixtures/phase45_hierarchical_local_credit_cases.jsonl`
- `workspace/evaluation/phase45_hierarchical_local_credit_preregistration.json`
- `workspace/evaluation/phase45_hierarchical_local_credit_benchmark.json`
- `workspace/evaluation/phase45_hierarchical_local_credit_ablation.json`
- `data/processed/benchmark_fixtures/phase46_delayed_structural_credit_cases.jsonl`
- `workspace/evaluation/phase46_delayed_structural_credit_preregistration.json`
- `workspace/evaluation/phase46_delayed_structural_credit_benchmark.json`
- `workspace/evaluation/phase46_delayed_structural_credit_ablation.json`

## Review Rule

新しいアイデアは、まずこのロードマップの対象Phase、仮説、最小実験、失敗条件、昇格条件を明記してから実装します。実装が先行しても、独立データ・否定例・再現可能な評価がない限り、知能の向上や本番昇格とは扱いません。
