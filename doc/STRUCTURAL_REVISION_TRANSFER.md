# Structurally Held-Out Revision Transfer

## Decision — 2026-09-16

[Done] The single frozen attempt is retained as a negative adoption result. The shared sparse representation generalized well and the selected revision-gain rule improved post-revision adaptation, but it did not meet the registered advantage over an online-learning atomic control. The failure policy stops this synthetic shared-route transfer line.

## Contract

Each event activates three explicit sparse feature routes: left symbol, right symbol and interval. Training contains 144 address compositions; 48 different compositions are held out. Every one of the 19 individual feature values occurs on both sides of the split, so the shared arm can reuse local state without seeing the held-out combinations. The atomic control instead assigns one independent route to each complete address.

A supplied label-free revision ID flips the synthetic feature signs. The selected candidate uses local residual updates at rate `0.40` for 864 successful decisions and then returns to signed three-factor updates at rate `0.15`. No matrix, global gradient or GPU is used.

## Frozen evidence

- Protocol: `data/processed/benchmark_fixtures/structural_revision_transfer_v1.json`
- Protocol SHA-256: `d0767227fb327a2f31d202fee29c8aeebf0cffda88ef074be520987d29cfe522`
- Attempt: `workspace/evaluation/structural_revision_transfer_v1_attempt.json`
- Result: `workspace/evaluation/structural_revision_transfer_v1_result.json`

| Registered gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| Candidate held-out pre-revision accuracy | 99.90% | ≥ 80.00% | Pass |
| Held-out pre-revision gain over atomic control | +4.48 pp | ≥ +20.00 pp | **Fail** |
| Early post-revision gain over always-residual | +11.62 pp | ≥ +10.00 pp | Pass |
| Full post-revision Brier gain over always-residual | +0.022455 | ≥ +0.020000 | Pass |
| Full post-revision Brier gain over previous policy | +0.024583 | ≥ +0.020000 | Pass |
| Maximum seen-to-held-out accuracy drop | 0.52 pp | ≤ 10.00 pp | Pass |
| Worst adaptation latency | 237 episodes | ≤ 576 | Pass |

All five seeds had positive early gains. SNN and equally informed scalar traces matched exactly. The candidate used 19 explicit neurons, at most 29,242 recursively measured Python bytes, 19 event-work units and 0.121 ms per episode in the recorded runs, all within the frozen limits.

The atomic control had no reusable state at its first held-out encounter, but labels remained available after every prediction. Across twelve passes over each held-out composition it learned online and reached mean aggregate accuracy near 95.42%. The shared candidate's near-perfect transfer is real within the fixture, but the registered aggregate comparator gate correctly prevents promotion when a simpler bounded learner catches up within the same evaluation interval.

## Roadmap effect

[Done] Keep the revision-gain component as the selected synthetic revision rule; do not promote the shared feature representation or claim structural generalization superiority.

[Next] Return to the existing R2 entry requirement: select one timestamped, independently observable sensor or log prediction task and freeze its acquisition, chronological split, outcome timing, strongest bounded baseline and complete resource budgets. Do not run another synthetic shared-route retry.

[Later] A future independent task may report zero-shot transfer and online adaptation as separate metrics, but it must be justified by the real task contract rather than by tuning this consumed fixture.

日本語: 共有特徴方式は未観測の組み合わせで平均99.90%に達し、改訂後の初期正解率も常時残差則より11.62ポイント改善しました。しかし、評価中にオンライン学習できる原子的比較群との差は4.48ポイントで、登録条件の20ポイントを満たしませんでした。この合成研究線は停止し、次は実測可能な時系列ログ課題へ戻ります。

简体中文：共享特征方案在未见组合上达到平均99.90%，修订后的早期准确率也比持续残差规则高11.62个百分点。但可在评估期间在线学习的原子对照组仅落后4.48个百分点，未达到预注册的20个百分点门槛。因此停止这条合成研究线，下一步回到可独立观测的真实时间序列日志任务。
