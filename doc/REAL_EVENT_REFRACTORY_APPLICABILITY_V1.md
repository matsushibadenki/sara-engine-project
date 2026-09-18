# Real-Event Refractory Applicability V1

[Done] This is a descriptive, training-only screen for transferring the CartPole synthetic task's two-logical-step same-route refractory rule to irregular real-event streams. The [canonical protocol](../data/processed/benchmark_fixtures/real_event_refractory_applicability_v1.json) is SHA-256 `53e3e50697c45d92e41dda6bc5c2093f286ff428b0486505b931af26e9f9de94`; the [one-shot result](../workspace/evaluation/real_event_refractory_applicability_v1.json) is SHA-256 `422eb61da9d27a9da9a5a84ccdd0c882dbbac2fa021eb208d4de2d658c4fd64c`. The source-pinned audit uses the existing chronological case splits and only materializes activity labels for training cases. It does not train or score a candidate, inspect development/test outcome labels, or reopen a CartPole held-out split. XML bytes for nontraining cases are necessarily parsed to find trace boundaries and case start times; they are not converted to activity sequences or statistics.

For each training case, an emitted activity suppresses the next two *logical event positions* of the same activity, with no real-time cutoff. For each next-activity prediction position, the audit counts whether its current event would be masked. Within identical recent-three-activity contexts, it compares empirical next-activity distributions between emitted and masked positions. A context needs at least 20 predictions in each status. The frozen applicability screen requires a 2–50% masked-prediction rate, at least three eligible contexts and 200 supported predictions, weighted conditional total variation ≥0.10, and at most 5% of masked recurrences more than one day after the last emitted activity. These are feasibility thresholds, not a model-performance target.

| Training-only measure | BPI 2012 | Sepsis |
| --- | ---: | ---: |
| Cases / next-activity positions | 9,160 / 177,026 | 735 / 10,083 |
| Masked prediction positions | 61,964 (35.00%) | 1,303 (12.92%) |
| Median masked recurrence gap | 6,336.715 s | 86,400 s |
| Masked recurrences over one day | 25.77% | 45.21% |
| Eligible contexts / prediction support | 11 / 69,097 | 4 / 1,053 |
| Weighted conditional total variation | 0.1114 | 0.1439 |
| Frozen screen | Fail: long-gap criterion | Fail: long-gap criterion |

The association statistic is in-sample and observational: even conditional on three activities, it does not establish that the refractory status causes an improved prediction. Different longer histories or elapsed time could explain the association. More importantly, the same fixed logical-step mask suppresses many events separated by over a day. This fails the preregistered transfer screen on both datasets. Do not spend a new real-event candidate budget on this unchanged refractory rule or reinterpret the positive synthetic control as real-event benefit. A future physical-time-gated rule would be a new hypothesis with a fresh protocol, matched compact-time and label-blind thinning controls, development identities, and its own one-shot gate; no such candidate is authorized here.

日本語: 学習分割のみで、直近2論理ステップの同一route不応期を実イベントへ移す適用性を調べました。BPIで予測位置の35.00%、Sepsisで12.92%が抑制対象ですが、その再出現の25.77%／45.21%は前回発火から1日超です。事前固定した長間隔の基準に両方とも不合格で、現行ルールを実データ候補として評価する根拠にはなりません。条件付き関連は因果効果や精度改善の証拠ではありません。開発・テストの結果ラベルとCartPoleのheld-outは使っていません。

简体中文：只用训练分区检查了“同一路由在接下来两个逻辑事件位置内不应期”的真实事件适用性。BPI和Sepsis分别会屏蔽35.00%和12.92%的预测位置，但被屏蔽的重复事件中有25.77%和45.21%距离上次发放超过一天。两者均未通过预先固定的长间隔门槛，因此不能据此评估原样迁移的真实事件候选。条件分布相关性不等于因果增益或准确率提升；未读取开发／测试结果标签，也未打开CartPole留出集。
