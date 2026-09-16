# Revision-Specific Local Gain Policy

## Decision — 2026-09-16

[Done] The single frozen attempt passes every registered component gate. Adopt the bounded revision-gain policy for the next research stage. Do not connect it to production learning yet.

`BoundedRevisionGainReadout` normally uses signed three-factor updates at rate `0.15`. A supplied strictly increasing revision ID opens a window of exactly 1,152 successful local decisions. During that window it uses the route-local prediction residual at rate `0.40`, with each change capped at `0.50`; it then returns to the stable rule. Expired feedback does not consume the window. The revision event contains no label, answer or route.

## Frozen evidence

- Protocol: `data/processed/benchmark_fixtures/revision_gain_policy_v1.json`
- Protocol SHA-256: `435bc412332752ee3dbc9d8f1580d8dde97671e60aeac49a31452ac7f4acf93b`
- Attempt: `workspace/evaluation/revision_gain_policy_v1_attempt.json`
- Result: `workspace/evaluation/revision_gain_policy_v1_result.json`

The experiment used five fresh seeds, 192 explicit fixed-topology neurons, 1,536 training episodes, 768 pre-revision episodes and 2,304 post-revision episodes per scenario. The true-revision scenario flipped every route's label; the false-revision scenario changed no label.

| Registered gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| Early accuracy gain over always-residual | +25.00 pp | ≥ +10.00 pp | Pass |
| Full post-revision Brier gain over always-residual | +0.092357 | ≥ +0.020000 | Pass |
| Full post-revision Brier gain over previous policy | +0.123416 | ≥ +0.020000 | Pass |
| Maximum pre-revision Brier regression | 0.000000 | ≤ 0.002000 | Pass |
| Maximum false-revision Brier harm | 0.000000 | ≤ 0.002000 | Pass |
| Worst adaptation latency | 735 episodes | ≤ 768 | Pass |

Every seed had the same positive early accuracy gain: candidate `58.33%` versus always-residual `33.33%`. The candidate's full post-revision Brier was `0.162795`, versus `0.255152` for always-residual and `0.286211` for the previous revision policy. SNN and equally informed scalar prediction traces matched exactly, and all frozen state, event-work and CPU-latency limits passed.

## Evidence boundary and next step

This establishes only that a larger bounded residual step repairs inherited-confidence delay in this synthetic fixed-address task. It does not establish autonomous change detection, learned topology, unseen-structure transfer, an SNN advantage, R2, production readiness or physical energy efficiency. The exact scalar match shows the measured gain comes from the local update policy rather than spiking dynamics.

[Done] The structurally held-out follow-up was executed once and retained as a negative adoption result because it failed the atomic-control superiority gate. See [structural transfer result](STRUCTURAL_REVISION_TRANSFER.md).

[Next] Return to the R2 entry gate and select one real timestamped prediction stream with independently observable delayed outcomes.

[Later] Evaluate a source-backed real revision stream and pair this policy with an independently validated revision detector. Durable knowledge mutation must continue through normal provenance and contradiction review.

日本語: 改訂期間だけ局所残差更新率を `0.40` に上げる方式は、5つの新規シードですべての登録条件を通過しました。常時残差則より初期正解率が25ポイント高く、変更後全体のBrierも0.092357改善し、誤通知の害はゼロでした。次は学習時に未観測の構造を含む改訂課題で再検証します。

简体中文：仅在修订窗口把局部残差更新率提高到 `0.40` 的方案，在五个全新种子上通过了全部预注册门槛。相对持续残差规则，早期准确率提高25个百分点，完整修订阶段Brier改善0.092357，误通知损害为零。下一步将在训练中未见过的结构修订任务上复验。
