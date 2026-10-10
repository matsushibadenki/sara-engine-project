# Matched temporal boundary learning v1

## English

🟢 [Done] Preregistered the middle-order/gap comparison before learner scoring, then implemented bounded readout-visible encoding, deterministic input generation and input-only preflight. Protocol SHA-256: `390ab21e3310e4d9253dc8cc5d1b9925a2f445c3fd50a575bf5d9018dcea7aa9`.

Five fresh training/evaluation seed pairs use disjoint source route pools (0–11 / 12–23) and suffix lengths (0/2 / 4/6 events). Each family has 1,024 training pairs and 256 evaluation pairs per seed. Both modes have identical histograms and duration. All 12,800 pairs pass the registered information boundary; input digest is `25296622241292401f5c53681b4e91ec0bbb9eb639bb6167093ef85548704809`. These are controlled variants of known synthetic templates, not independent task discovery or real-world confirmation.

| Readout encoding | Middle-order modes distinguishable | Gap modes distinguishable |
| --- | --- | --- |
| Histogram | No | No |
| First/last/count | No | No |
| Ordinal position × canonical route | Yes | No |
| Ordinal position × gap | No | Yes |
| Joint ordinal and gap | Yes | Yes |

Route ranks come from first appearance in the input history, never mode, identity, seed or target. Rank canonicalization is a supplied inductive bias. A sixth joint arm receives shuffled training feedback only. The sufficient ordinal and gap references are non-spiking controls; this protocol does not hypothesize or establish a spike-specific advantage.

All six arms will use the same bounded residual readout, signed targets, affine probabilities, four-interval receipt-bound feedback, and frozen evaluation weights. Each seed/family must reach joint accuracy ≥0.90 and Brier ≤0.20, plus ≥0.01 Brier improvement over the registered information-removal and shuffled-feedback controls. Comparison against the sufficient non-spiking reference is reported separately; superiority requires ≥0.01 in every seed. No tuning after scoring is allowed.

The encoder caps histories at 95 events and 24 routes, gap values at 255, active channels at 191, and channel IDs at 26,672. Maximum wire size is 194 bytes. The executed inputs peak at 12 events, 25 active channels, 28 wire bytes and 5,060 deep-counted transient Python bytes. These measurements exclude the future learner and are not complete runtime cost claims. Each arm must additionally account for weights, pending receipts, prediction/update visits, replay and scratch within the frozen 1 MiB total state cap. CPU measurements include instrumentation; no physical-energy claim is authorized.

Artifacts:

- Frozen protocol: `data/processed/benchmark_fixtures/temporal_boundary_learning_v1.json`.
- Exclusive audit report: `workspace/evaluation/temporal_boundary_learning_input_audit_v1.json`.
- Audit runner: `scripts/eval/temporal_boundary_learning_audit.py`.
- Encoder/generator: `src/sara_engine/evaluation/temporal_boundary_learning.py`.

The runner checks protocol and predecessor fingerprints and records implementation/Python identities. It refuses to overwrite its report. Regression coverage includes frozen-artifact tampering, route-renaming invariance, wrong information expectations, split overlap, empty/malformed input, maximum capacity, channel uniqueness and budget failure. Related suite: 33 tests passed on Python 3.10.18.

🟢 [Done] Implemented the six delayed learners, SHA-ranked balanced feedback shuffle, resource accounting, durable exclusive attempt receipts and saved-checkpoint replay. The original protocol remains byte-for-byte unchanged: its `learning_execution_ready=false` describes the preregistration stage; the execution receipt records readiness after 92 regression tests passed. The readout configuration ceiling was explicitly expanded from 4,096 to 65,536 to accept the registered sparse capacity of 26,673. Defaults, score normalization and local updates remain unchanged.

### One-shot result — 2026-10-10

The complete 5-seed × 2-family × 6-arm experiment trained 122,880 episodes and scored 30,720 frozen predictions. All 60 serialized research readouts exactly reproduced those 30,720 predictions on disk reload, without retraining. Integrity, resource and mechanism gates passed. The overall component gate **failed**: every middle-order joint Brier exceeded the frozen 0.20 ceiling. Gap joint quality passed every seed. Exit code 1 records the quality rejection, not an execution exception; the durable completion receipt exists and no failed-execution receipt was created.

| Family | Arm | Mean accuracy | Mean Brier |
| --- | --- | ---: | ---: |
| Middle order | Joint | 100% | 0.208092 |
| Middle order | Ordinal reference | 100% | 0.173120 |
| Middle order | Gap removed-information control | 50% | 0.250000 |
| Middle order | First/last/count | 50% | 0.250005 |
| Middle order | Shuffled-feedback joint | 40% | 0.250570 |
| Gap order | Joint | 100% | 0.199259 |
| Gap order | Gap reference | 100% | 0.158678 |
| Gap order | Ordinal removed-information control | 50% | 0.250001 |
| Gap order | First/last/count | 50% | 0.250004 |
| Gap order | Shuffled-feedback joint | 60% | 0.252380 |

Middle-order joint Brier ranged 0.207744–0.208369; gap joint ranged 0.198939–0.199956. Joint did not beat the sufficient non-spiking reference in either family. The mean shuffled accuracies do not imply a useful anti-learner or alternate candidate: feedback was independently shuffled, metrics were fixed beforehand, and no arm is retuned or promoted. The sufficient references retain fewer irrelevant active channels under the same normalized readout; this comparison does not isolate a biological mechanism.

| Phase | Episodes | Observation visits | Projection visits | Prediction route visits | Update route visits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 122,880 | 737,520 | 983,440 | 1,106,320 | 1,106,320 |
| Evaluation | 30,720 | 305,880 | 387,800 | 418,520 | 0 |
| Saved-weight replay | 30,720 | 305,880 | 387,800 | 418,520 | 0 |

Per-arm runtime state bounds peaked at 30,937 bytes. The separately charged offline shuffled-feedback setup scratch bound was 338,085 bytes, also below the 1 MiB cap with readout setup. The complete experiment used 39.97 process CPU seconds, including instrumentation, input generation, serialization and replay. These are conservative deep-object bounds plus explicit scratch allowances, not OS resident-memory measurements. The report also counts validation, canonicalization, wire projection, sorting inputs, normalization and instrumentation weight scans. Sorting-item counts are not comparison counts; counters are source-level work, not CPU instructions. Runtime state excludes shared evaluator result aggregation and interpreter overhead, both included in complete experiment CPU. No throughput comparison or physical-energy claim follows.

Execution artifacts:

- Result: `workspace/evaluation/temporal_boundary_learning_v1.json`, SHA-256 `bafc1953dd29fdfac4acce2e3daf3ee6e6ef1ac96aee9e013e59ebccb33bf1b8`.
- Durable reservation/completion: `workspace/evaluation/temporal_boundary_learning_v1_started.json` and `_completed.json`.
- Final research weights: `models/research/temporal_boundary_learning_v1.json`, SHA-256 `dbdf9328c6de80a0cf4a7db6e6f9280515bde7385a8497c8616aef6b260e209d`.
- Immutable portable receipt: `data/processed/benchmark_fixtures/temporal_boundary_learning_result_receipt_v1.json`.
- Runner: `scripts/eval/temporal_boundary_learning.py`; existing artifacts permanently block another attempt for this identity.

🟠 [Next] Select and preregister a fresh independently observable prospective event task with sufficient ordinal/gap controls and a source/time/outcome leakage audit before more learner scoring. Retain this consumed joint hypothesis as negative; do not tune its features, rates, seeds or Brier threshold.

🔴 [Later] Independent real-event confirmation and deep credit. Completed electricity/shared-observer hypotheses and sealed real datasets remain closed; production promotion is unauthorized.

⭕️ [Pending] Physical-energy measurement requires a suitable measurement environment.

## 日本語

🟢 [Done] 途中順序／時間間隔の局所学習比較を事前登録し、決定的な入力生成・有界特徴化・採点前監査を実装しました。5組の新しいseedで学習／評価のイベントID範囲と末尾長を分離し、12,800組すべてで情報の識別条件を確認しました。関連33テストが通過しました。

🟢 [Done] 6群の遅延学習・全処理の計測・一回実行・保存重みの再現まで完了しました。関連92テストが通過し、122,880件を学習、30,720予測を評価、保存した60学習器から全予測を完全再現しました。

joint群の精度は両課題とも100%ですが、途中順序の平均Brierは0.208092で登録上限0.20を全seedで超過しました。時間間隔は0.199259で合格、全体は不合格として保存しました。順序専用／gap専用対照はそれぞれ0.173120／0.158678でjointより良く、joint優位性はありません。合成課題の結果であり、実データ確認やスパイク優位性ではありません。

実行状態の保守的上限は30,937byte、別計上したshuffle準備の上限は338,085byte、全実験CPU時間は39.97秒でした。OS実メモリや電力実測ではありません。元プロトコルと完了済み実験は変更せず保持しています。

🟠 [Next] 新しい独立した将来結果予測課題と十分な非スパイク対照を事前登録し、情報漏洩を監査します。今回の条件や閾値を再調整しません。

🔴 [Later] 独立実イベント確認と深いcredit。⭕️ [Pending] 物理エネルギー実測には対応環境が必要です。本番昇格は行いません。

## 简体中文

🟢 [Done] 已预注册中间顺序／时间间隔的局部学习比较，实现确定性输入生成、有界特征编码和评分前审计。五组新seed使用互不重叠的路由范围与后缀长度，12,800对输入全部通过信息边界检查；相关33项测试通过。

🟢 [Done] 已实现六组延迟学习、资源计量、单次评估及保存权重重放。相关92项测试通过；训练122,880次，评估30,720次，从60个保存的学习器完全复现全部预测。

joint在两个任务均为100%准确率，但中间顺序平均Brier为0.208092，所有seed均超过0.20上限；间隔任务为0.199259，通过。整体结果保留为失败。顺序／间隔专用对照分别为0.173120／0.158678，均优于joint。运行状态保守上限30,937byte，单独计量的shuffle准备上限338,085byte，完整实验CPU时间39.97秒；这不是OS内存或物理能耗测量，也不证明真实数据或脉冲优势。

🟠 [Next] 为独立的新前瞻事件任务预注册充分的非脉冲对照，并审核信息泄漏。不重新调整本次条件或阈值。

🔴 [Later] 独立真实事件确认与深层credit。⭕️ [Pending] 物理能耗测量需要适当环境。不进行生产升级。
