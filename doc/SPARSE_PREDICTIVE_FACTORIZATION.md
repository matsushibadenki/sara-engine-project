# Sparse Predictive Factorization — SARA Research Proposal

## Status

[Later] This document translates the factorized predictive principle in [JEPA-Anything](https://arxiv.org/abs/2609.20800) into a SARA-compatible research hypothesis. It is not a production change, a claim that SARA has a world model, or evidence for a spike-specific advantage. The paper reports orthogonal predictive factorization across heterogeneous domains, intervention prediction, out-of-distribution evaluation, and long-horizon dynamics; its abstract also makes clear that the reference training uses a conventional predictive architecture, so SARA must not import its gradient-based optimization as a runtime requirement. [Paper source](https://arxiv.org/abs/2609.20800)

## Adopted principle

Instead of learning one undifferentiated predictive state, SARA may maintain a bounded set of local predictive factors:

```text
shared sparse event state
        ├── factor route A + local predictor
        ├── factor route B + local predictor
        ├── factor route C + local predictor
        └── bounded synthesis / abstention
```

The factors are not assigned human meanings in advance. A factor is useful only when it repeatedly reduces a future-state prediction residual, survives a held-out intervention or nuisance shift, and remains within event and state budgets. A branch that merely fires often, or receives a semantic label after the fact, is not a discovered concept.

## SARA translation

The paper's orthogonality and variance objectives become sparse event signals rather than matrix losses:

| Predictive-factor idea | SARA-compatible signal |
| --- | --- |
| factor prediction error | factor-local `prediction_error` event bound to recent eligibility |
| cross-factor overlap penalty | bounded pairwise active-route overlap counter and lateral inhibition event |
| dead-factor/activity regularization | local underactivity / starvation event with homeostatic threshold |
| latent recombination | bounded factor packet synthesis with explicit missing-factor abstention |
| intervention prediction | source-addressed counterfactual outcome event and held-out intervention split |
| dynamic factor count | later structural growth/pruning proposal, default-off |

No dense projection matrix, global covariance, whole-network gradient, or GPU is required. Pairwise overlap must be computed only for the capped active-factor set in the current event window. The implementation must expose overlap, prediction residual, factor count, event count, retained state, and abstention decisions for audit.

## Minimum study before integration

Use the smallest synthetic delayed-outcome or timestamped event task that can distinguish factor specialization from a larger monolithic control. Freeze the generator, private target factors, nuisance factors, intervention actions, and causal split before candidate implementation. Compare:

1. scalar / single-factor local predictor;
2. multi-factor routing without overlap suppression;
3. multi-factor routing with bounded overlap inhibition;
4. (later) overlap inhibition plus structural growth/pruning.

All arms receive the same observations, local update budget, factor capacity, and delayed outcome. The fourth arm must not be scored until the first three establish that factor separation helps independently. The first study must not use BPI or Sepsis frozen checkpoints for tuning and must not reopen CartPole held-out ranges.

## Required measurements and gates

Record prediction accuracy and Brier or calibrated residual where applicable, but also:

- held-out intervention and nuisance-shift accuracy;
- long-horizon rollout error only when the task supplies a bounded future stream;
- active-factor count and per-factor usage entropy;
- pairwise route-overlap rate before and after suppression;
- events per prediction, peak retained state, and local update count;
- exact replay and scalar-equivalence traces;
- factor-reset and factor-permutation controls;
- missing-factor abstention and malformed-packet rejection.

Promotion requires a preregistered improvement over the single-factor control on at least one held-out causal metric without a material regression in the resource envelope, and a targeted overlap ablation must remove the claimed specialization effect. Accuracy gain alone is insufficient. A positive result would support sparse predictive factorization, not a general JEPA or backpropagation replacement.

## Explicit non-adoptions

- Do not port the paper's projector, EMA target encoder, orthogonality matrix loss, or gradient optimizer into SARA's normal runtime.
- Do not name factors `speed`, `position`, `goal`, or `cause` before a factor survives anonymous held-out validation; names may be attached afterward as metadata only.
- Do not equate low overlap with truth. Provenance, contradiction handling, verification, and revision remain separate.
- Do not add dynamic factor growth until fixed-factor overlap suppression has a positive, reproducible causal result.

日本語: JEPA-Anythingから採用するのは勾配学習ではなく、予測状態を複数の局所因子へ分解し、重複を抑えながら予測・再合成するという仮説です。因子の意味は事前に命名せず、予測残差・介入汎化・資源上限・再現性で検証します。まず固定因子数の小規模課題を行い、構造成長は後段に分離します。

简体中文：从 JEPA-Anything 采用的不是梯度训练，而是把预测状态分解为多个局部因子、抑制冗余并重新合成的研究假设。因子不得预先命名，必须通过预测残差、干预泛化、资源上限和精确重放验证。先研究固定因子数的小任务，再单独研究结构增长。
