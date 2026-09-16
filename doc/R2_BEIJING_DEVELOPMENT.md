# R2 Beijing Development Gate

## Decision — 2026-09-16

[Done] The registered training/development attempt is retained as a negative R2 gate result. The sparse local residual learner substantially improved real-data prediction over the strongest bounded transition baseline, but temporal-order causality and the frozen maximum-latency contract failed. The frozen test period remains unevaluated and unauthorized.

## Observed development result

The evaluator processed 290,312 chronological training pairs and 50,825 chronological development pairs from 12 stations. It did not load the July 2016–February 2017 frozen-test period into the model evaluator.

| Arm | Balanced accuracy | Brier |
| --- | ---: | ---: |
| Sparse local residual SNN path | 62.07% | 0.232606 |
| Equal-information scalar path | 62.07% | 0.232606 |
| Three-factor SNN path | 56.72% | 0.246432 |
| Temporal-order-destroyed SNN path | 61.84% | 0.233608 |
| Shuffled outcomes | 49.58% | 0.251308 |
| Frozen plasticity | 50.00% | 0.250000 |
| Online station×hour×direction table | 55.40% | 0.245976 |

The candidate beat the online transition table by 6.67 balanced-accuracy points and improved Brier by 0.013370. Its worst-station balanced accuracy was 58.84%, versus 53.46% for the transition baseline. Shuffled outcomes and frozen plasticity confirm that correctly paired delayed outcomes caused the useful local learning.

Two registered gates failed:

- Destroying the assignment of 1-hour and 24-hour changes reduced balanced accuracy by only 0.23 points, below the required 0.50 points. This hourly task therefore does not establish a meaningful spike-timing or temporal-order advantage.
- The frozen 5 ms maximum per-prediction CPU bound was exceeded by recorded outliers. Candidate maximum was 5.924 ms; other arms also had larger scheduling-sensitive maxima. State, event work and peak RSS remained within their individual ceilings, but the all-resource gate correctly fails as registered.

Candidate and equal-information scalar prediction traces matched exactly. The result supports sparse local residual learning on real delayed outcomes, but it does not support an SNN-specific advantage, R2 completion, production promotion or an energy claim.

## Execution incident

The first invocation stopped before emitting any result because the three-factor adapter read the wrong update-count field. No frozen-test records were loaded and no decision artifact existed. The error and recovery are retained under `workspace/evaluation/`; a focused regression test was added, and the identical frozen protocol was rerun without changing features, learning rates or gates.

## Roadmap effect

[Done] Preserve the Beijing workload as real-data evidence for bounded local outcome learning. Do not execute its frozen test because the development gate failed.

[Done] BPI Challenge 2012 was selected and hash-pinned as the irregular real-event follow-up. See [task contract](R2_IRREGULAR_EVENT_TASK.md).

[Next] Build its case manifest and training/development-only baselines before implementing the multiclass local learner.

日本語: 実測データでは局所残差学習が最良の遷移表をbalanced accuracyで6.67ポイント上回りました。しかし時間順序破壊との差は0.23ポイントに留まり、最大CPU時間の条件も失敗しました。スカラー経路と完全一致したため、SNN固有の優位性やR2達成とは扱いません。封印テストは実行しません。

简体中文：在真实测量数据上，局部残差学习的平衡准确率比最强转移表高6.67个百分点。但破坏时间顺序只降低0.23个百分点，最大CPU时间门槛也失败。由于标量路径完全一致，不能视为SNN特有优势或R2完成，冻结测试不会执行。
