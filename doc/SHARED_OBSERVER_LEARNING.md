# Shared observer local-learning milestone v1

## English

[Done] Completed preregistration, implementation, one-shot five-seed training/evaluation, bounded research checkpoint publication and final-weight replay. Protocol SHA256 `7e999640…fbfa`; result SHA256 `8c8b5093…3646`; checkpoint SHA256 `c3314eb8…f3a1`. No electricity values or sealed 2014 rows were opened.

The saved-file verifier reproduces all 12,800 predictions across 25 readouts without retraining. Verification SHA256 `3757be49…ae3f`; 56 related tests pass. The immutable outcome receipt is `data/processed/benchmark_fixtures/shared_observer_learning_result_receipt_v1.json`. This closes the research milestone, while keeping component success separate from failed backend increment and forbidding production promotion.

The task is explicitly synthetic: an exogenous mode changes the first two input events, while paired histograms, terminal order and distractor suffixes are identical. Every arm pays for the same first/last/count observer. Training uses 4,096 episodes per seed; evaluation uses 512 fresh seeded episodes per seed, with frozen weights and no outcome updates. Signed local targets and four-interval delayed receipts update only active routes, without global gradients, matrices or GPUs.

| Arm | Accuracy, all five seeds | Brier range |
| --- | --- | --- |
| Observer only | 1.000 | 0.239733–0.239755 |
| Observer + compact pulses | 1.000 | 0.244796–0.244801 |
| Observer + LIF output | 1.000 | 0.243630–0.243664 |
| First-position information removed | 0.500 | approximately 0.250000 |
| LIF output only | 0.500 | approximately 0.250000 |

**Component gate passes on every seed:** observer-only accuracy exceeds 0.9 and Brier improvement over the order-blind control exceeds the fixed 0.001 minimum. Opposite-target order-blind/backend-only prediction equality passes. **Backend increment gate fails on every seed:** LIF does not exceed the strongest observer/compact reference by the registered 0.01 Brier improvement; observer-only is better in all runs. Close this exact backend-increment hypothesis without retuning.

This establishes a narrow useful minimum: input events + explicit bounded ordinal memory + a local readout can learn this controlled order task. It does not establish general temporal reasoning, calibrated confidence (Brier remains near 0.24), natural-data transfer, deep credit assignment, or a general disadvantage of SNNs. Added backend channels also change readout normalization, so the increment comparison is about the registered complete configuration, not isolated spike biology. All observer-bearing arms have perfect classification, but confidence remains weak.

Maximum reported training state, including observer/transient features, is 27,354 bytes, below the 65,536-byte cap. State/event costs include shared observation and projection, not merely neuron costs. CPU timings are diagnostic single-run measurements and include instrumentation; no energy advantage or independent deployment speed claim follows. The final checkpoint contains 25 bounded readouts and explicitly forbids production loading.

Artifacts:

- Result and durable started/completed receipts: `workspace/evaluation/shared_observer_learning_v1*`.
- Final research weights: `models/research/shared_observer_learning_v1.json`.
- Saved-artifact replay: `workspace/evaluation/shared_observer_checkpoint_verification_v1.json`.
- Verification command: `PYTHONPATH=src python scripts/eval/verify_shared_observer_checkpoint.py` (the v1 command exclusively reserves its verification report; do not overwrite consumed artifacts).

[Next] Consolidate this component as a research baseline and preregister an independent task that separates first-visit memory from full sequence/gap requirements, with strong non-spiking controls and honest information accounting. Do not add deeper neuron mechanisms or rerun this hypothesis to chase calibration. Deep credit, real confirmation and production promotion remain [Later].

## 日本語

[Done] 事前登録・実装・5seedの一回実験・最終重みの研究checkpoint保存まで完了しました。各seedで学習4,096件、評価512件。共通observer単独は全seedで精度100%、順序情報除去／LIF単独は50%でした。observer単独の局所学習gateは合格、LIF追加効果gateは全seedで不合格です。この追加効果仮説は再調整せず終了します。

得られたのは「入力event＋有界の順序memory＋局所readout」がこの合成順序taskを学習できるという限定的な成果です。Brierは約0.24であり、確信度の良さ、自然データへの転移、深いcredit assignment、SNN全般の優劣は証明していません。backend追加によるreadout正規化の変化も比較に含まれます。observer等を含む最大stateは27,354byte。研究重み25個は本番読み込み禁止です。

[Next] この最小構成を研究baselineとして固定し、first-visitだけでは足りない順序／gap要件を分離する独立taskを別途事前登録します。電力実験を再開せず、2014年は封印維持。[Later] 深いcredit・実データ確認・本番採用。

## 简体中文

[Done] 已完成预注册、实现、五seed单次实验及最终研究checkpoint保存。每seed训练4,096次、评估512次。共享observer单独在所有seed准确率100%，去除顺序信息／LIF单独为50%。observer局部学习gate通过，LIF增量gate全部失败；不再调参追逐该假设。

限定结论是“输入event＋有界顺序memory＋局部readout”能够学习这一合成顺序task。Brier约0.24，不证明高质量置信度、自然数据迁移、深层credit assignment或SNN普遍优劣。backend增加带来的readout归一化变化也包含在比较中。含observer等的最大state为27,354byte；25个研究readout禁止生产加载。

[Next] 固定最小研究baseline，另行预注册能区分first-visit与完整顺序／gap需求的独立task。不重新打开电力实验，2014年保持封存。[Later] 深层credit、真实数据确认与生产采用。
