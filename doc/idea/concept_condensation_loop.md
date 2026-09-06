# Concept Condensation Loop

## Purpose

This note translates a dialogue about language accumulation and concept formation into a bounded SARA research mechanism. It does not assume that more text automatically creates intelligence, that a newly coined word denotes a real structure, or that collective human progress measures biological intelligence.

日本語: 多様な表現の蓄積から、まだ命名されていない共通構造を候補化し、未知例で検証した後だけ再利用する仕組みを定義する。

简体中文: 从多样表达中形成尚未命名的共同结构候选，并且只在未见样本验证通过后复用。

## Operational Mapping

| Dialogue idea | SARA representation | Required evidence |
| --- | --- | --- |
| Undivided knowledge | Source-aware episodes, relation fragments, unresolved residuals, and counterexamples without a concept label | Provenance completeness and bounded retention |
| Conceptual pressure | A candidate trigger caused by cross-context local reuse plus prediction residual or accounted description cost | Not explained by raw frequency, source identity, or paraphrase duplication |
| Semantic condensation | An anonymous overlapping assembly with concrete bindings and explicit differences | Held-out reuse, prediction/compression gain, and deterministic replay |
| Semantic germination | A new prospective prediction or composition enabled by the candidate | Gain disappears under targeted ablation and shuffled bindings |
| Lexical horizon | Separate multilingual names and definitions bound to a validated anonymous ID | Naming does not alter routing or evaluation |
| Intelligence compounding | A validated candidate becomes a typed input for a capped higher-level cycle | Fixed depth, fan-out, candidate, replay, state, and event budgets |

## State Separation

```text
EvidenceEpisode
  source_id, revision, timestamp, typed_fragments, observations

AnonymousConceptCandidate
  candidate_id, invariant_signature, bindings, differences
  support_refs, counterexample_refs, prediction_contract
  description_cost, ablation_effect, uncertainty, revision, expiry

LexicalBinding
  candidate_id, language, preferred_name, definition, aliases
  reviewer, revision
```

The three records have different authority. Evidence is observed. A candidate is provisional. A lexical binding is an interface for people and cannot turn the candidate into verified knowledge.

## Bounded Cycle

1. Encode incoming episodes as sparse typed fragments while retaining source and revision identity.
2. Search only a fixed local neighborhood selected by fragment signatures, recent activity, context, and time.
3. Raise conceptual pressure when independently sourced contexts reuse related local resources and the current representation leaves a preregistered prediction residual or description-cost opportunity.
4. Allocate an anonymous candidate under a hard per-window and total-capacity budget. Store shared structure, concrete bindings, differences, and counterexamples.
5. Freeze the candidate before evaluation. Test it on untouched future episodes and compare against exact retrieval, frequency, explicit motif, shuffled binding/order, and equal-budget non-spiking controls.
6. Promote only when all conjunctive gates pass: prospective utility, accounted compression, exception preservation, ablation specificity, replay determinism, and resource bounds.
7. Optionally add English, Japanese, and Simplified Chinese lexical bindings after promotion. Names are not learner inputs for the frozen evaluation.
8. Retrieve a capped set of provenance-linked memories and test whether the new factor yields additional provisional relations. Derived statements preserve their parent candidate and do not count as independent evidence.

## Why Text Volume Still Matters

Volume supplies opportunities to observe invariance, variation, exceptions, and consequences. The useful quantity is not raw token count alone. The system records at least expression diversity, context diversity, source independence, temporal separation, contradiction coverage, and predictive outcomes. Thousands of paraphrases can help test surface invariance while contributing only one independent evidence lineage.

No combinatorial formula is used as a capacity claim. A new concept can create many possible relations, but SARA explores only locally retrieved, typed, budgeted candidates and requires prospective evidence for each promoted relation.

## Failure Boundaries

- Reject a candidate explained by exact wording, token/hash collision, author/source identity, repetition count, fixed input slots, or evaluator labels.
- Reject a candidate that compresses by erasing counterexamples, causal direction, temporal order, source revision, or provenance.
- Reject lexical novelty as evidence. A persuasive definition can describe a nonexistent or useless partition.
- Reject self-confirmation: generated definitions, restatements, and candidate-derived predictions cannot become independent support for their parent.
- Reject unbounded corpus rescans, all-pairs comparisons, one-candidate-per-example growth, universal-cluster collapse, or hierarchy growth without saturation.
- Reject claims of autonomous discovery when human-designed event types or ontology boundaries supplied the decisive structure; report those priors explicitly.

## Integration Order

- [Done] The idea is mapped to SARA's evidence, anonymous structure, verification, lexical binding, and bounded replay layers.
- [Next] Complete roadmap R0 and establish the selected prediction path and hard resource contracts.
- [Later] Complete R1 and show causal local temporal learning on unseen streams.
- [Later] Use the frozen Phase 39 experiment for anonymous local reuse without adding naming to that protocol.
- [Later] Preregister a separate lexicalization and bounded reinterpretation experiment after Phase 39 passes.
- [Later] Feed validated factors into Phase 41 composition only after its explicit-factor prerequisites pass.
