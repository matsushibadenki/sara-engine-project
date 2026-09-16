# Observable Revision Learning Policy

## Decision — 2026-09-16

[Done] The single frozen attempt is a retained negative adoption result. A supplied revision notification improved early recovery without using hidden labels, but the candidate did not beat the strongest fixed policy over the complete post-revision interval. Keep the policy isolated from normal chat, verified evidence, durable memory and production learning.

The experiment answers a narrow question: can a strictly increasing external revision ID safely select residual learning for a fixed window and then return to stable three-factor updates? It does not detect revisions. It treats the notification as a routing event, not proof that the underlying facts changed.

## Contract

`BoundedObservableRevisionReadout` starts with signed constant three-factor updates. `notify_revision(new_id)` requires a strictly increasing integer ID, no pending prediction, and monotonic logical time. It selects residual updates for exactly `adaptation_horizon` successfully updated decisions. Expired feedback does not consume the horizon. The policy then returns to three-factor updates.

The notification contains no target, route, weight, prediction error or answer. One immutable prediction receipt binds the local routes, score, revision and selected mode before feedback. Copied, foreign, stale and repeated receipts are rejected. Defaults cap active routes, retained routes, feedback age, weight range and update magnitude. The component is single-owner and in-memory.

## Frozen protocol and evidence

- Protocol: `data/processed/benchmark_fixtures/observable_revision_policy_v1.json`
- Protocol SHA-256: `74aeea09837d97304c12c2d777b97f275914b4cd905c5136d9fe4a12dc193182`
- Attempt/provenance: `workspace/evaluation/observable_revision_policy_v1_attempt.json`
- Result: `workspace/evaluation/observable_revision_policy_v1_result.json`

Five fresh seeds each use 1,536 training episodes, 768 pre-revision evaluation episodes and 2,304 post-revision episodes. A visible revision ID changes from 1 to 2 immediately before the post-revision segment. In the true-revision scenario all 192 fixed address labels flip. In the false-revision scenario labels remain unchanged. The candidate uses residual updates for 1,152 decisions after the notification and then returns to three-factor updates.

| True-revision arm | Pre accuracy / Brier | Early post accuracy / Brier | Late post accuracy / Brier | Full post accuracy / Brier |
| --- | ---: | ---: | ---: | ---: |
| Observable policy SNN | 100.00% / 0.000000 | 16.67% / 0.518109 | 100.00% / 0.054314 | 58.33% / 0.286211 |
| Always three-factor SNN | 100.00% / 0.000000 | 0.00% / 0.679286 | 83.33% / 0.152149 | 41.67% / 0.415717 |
| Always residual SNN | 100.00% / 0.012563 | 33.33% / 0.445570 | 100.00% / 0.064734 | 66.67% / 0.255152 |
| Ignored notification SNN | 100.00% / 0.000000 | 0.00% / 0.679286 | 83.33% / 0.152149 | 41.67% / 0.415717 |
| Shuffled-feedback SNN | 51.69% / 0.284970 | 48.32% / 0.302864 | 48.85% / 0.298635 | 48.59% / 0.300750 |

The candidate improves early post-revision accuracy over always-three-factor by 16.67 percentage points, with a positive gain for every seed. Its worst adaptation latency is 1,133 episodes, within the frozen 1,152 limit. Pre-revision regression and false-notification harm are both zero. Candidate and equally informed scalar prediction traces match exactly. All resource contracts pass; maximum candidate state is 146,390 recursively measured Python bytes and maximum event work is 7.

Validation: 166 focused local-learning tests and the complete 2,134-test repository suite pass. The three saved experiment reports match their recorded source hashes.

The decisive failure is full post-revision Brier. The candidate is `0.031059` worse than the best fixed arm, while the preregistered gate requires at least `0.005` improvement. Always-residual also reaches 33.33% early accuracy, twice the candidate's 16.67%.

## Interpretation

Before revision, the candidate uses three-factor updates and starts the change at fully saturated ±1 weights. Always-residual retains slightly less confidence before revision (pre-revision Brier `0.012563`), so its weights have a shorter distance to travel after every label flips. Switching only the update formula does not remove this inherited-confidence delay.

The result supports one constrained follow-up: temporarily increase the residual step after an explicit revision notification. A false notification at a correct ±1 prediction still has zero residual, so this may preserve false-notification safety. This is a hypothesis, not a result. Resetting weights, using labels to choose a rule, or tuning the step on these consumed seeds is not justified.

## Next

[Done] The revision-specific learning-rate follow-up was frozen on new seeds and passed every component gate. See [revision gain decision](REVISION_GAIN_POLICY.md).

[Next] External or structurally held-out revision usefulness remains necessary. The synthetic fixed-address success does not establish autonomous revision detection, general learning, SNN superiority or physical energy efficiency.

日本語: 改訂通知による切替は、変更直後の正解率を三因子則より16.67ポイント改善し、誤通知の害もありませんでした。しかし変更後全体のBrierは常時予測誤差則より0.031059悪く、採用条件を満たしません。飽和した古い重みからの移動が遅いため、次は改訂期間だけ更新量を大きくする仮説を新しい条件で検証します。

简体中文：修订通知策略使变化初期准确率比三因子规则提高16.67个百分点，且误通知没有造成损害。但完整修订阶段的Brier比持续预测误差规则差0.031059，未达到采用条件。原因是旧权重已饱和，下一步将用全新条件检验仅在修订窗口提高更新幅度的假设。
