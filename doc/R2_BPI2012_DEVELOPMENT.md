# R2 BPI 2012 Sparse Multiclass Development Gate

## Decision — 2026-09-16

[Done] The single registered development attempt was executed under protocol SHA-256 `3d8f87797fdf33c772720b5706cc8d9d4c8de48d8c6dc72b61ed308c7849f940`. The candidate did not pass the complete gate, so the frozen test remains sealed and the sparse mistake-driven model is not adopted as the R2 architecture.

The candidate reached 79.93% top-1 accuracy and 60.49% macro-F1. The online second-order transition control reached 80.06% and 55.49%, respectively. The candidate therefore improved macro-F1 by 5.00 percentage points, but lost 0.13 points of accuracy instead of achieving the registered 1.00-point gain. It also missed the absolute 81.07% accuracy floor.

## What the result establishes

- Local outcome updates are functional: shuffled outcomes reduced accuracy from 79.93% to 12.05%, and the frozen learner reached only 2.06%.
- Observed event timing contributes to this candidate: replacing positive gaps with a constant reduced accuracy from 79.93% to 78.33%, a 1.60-point registered loss.
- Rare-activity mean recall improved from 44.98% for the second-order control to 47.65% for the candidate.
- The explicit-neuron and scalar arms produced exactly the same prediction trace. This prevents an SNN-specific quality claim.
- All registered resource contracts passed: 1,018 neurons/routes, 5,181 stored weights, 1.53 MB measured state, 248 maximum counted operations, 0.061 ms p99 latency, 4.17 ms watchdog latency and 187.5 MB peak RSS.

The Brier score also shows a material weakness that was not an acceptance criterion: 0.8839 for the candidate versus 0.2803 for the second-order control. The local mistake rule improves class coverage but produces poorly calibrated probabilities.

## Architecture decision

[Done] Stop tuning this consumed protocol. Retain `BoundedSparseMulticlassReadout` as an isolated bounded research component because its receipt binding, sparse local updates and resource contracts are valid. Do not promote it into the production prediction path.

[Next] Build a bounded confidence-routed hybrid in a fresh protocol. The second-order transition table supplies the default prediction and calibrated counts. A local temporal residual may override it only when its evidence margin passes a fixed threshold learned exclusively from training-period folds. This directly targets the observed tradeoff: preserve the transition model's frequent-class accuracy while using sparse temporal routes for rare activities and timing-sensitive cases.

The next protocol must use fresh evaluation identities. Existing development outcomes have now been inspected and cannot be reused for model selection. The currently sealed test will remain unopened until a new protocol has fixed the hybrid rule from training-only folds, or an independent event log is selected for the final external gate. It must register accuracy, macro-F1, Brier/calibration, rare-class recall, constant-gap and shuffled-outcome controls, scalar equivalence and the existing CPU/resource limits.

[Later] Test whether explicit spiking state adds anything beyond matched scalar routing. If prediction traces remain equal, retain the mechanism as sparse local learning rather than claiming SNN superiority. Physical energy claims require external measurement.

## Artifacts

- Protocol: `data/processed/benchmark_fixtures/r2_bpi2012_model_v1.json`
- Attempt marker: `workspace/evaluation/r2_bpi2012_development_v1_attempt.json`
- Immutable result: `workspace/evaluation/r2_bpi2012_development_v1_result.json`
- Result SHA-256: `83b0f531dc68a8061998ef41f67fd48761cd6f87f37341d35f368ac63921da00`

日本語: 局所学習と時間情報の効果は確認できましたが、正解率は二次遷移対照を0.13ポイント下回り、採用条件を満たしませんでした。凍結テストは開きません。次は、二次遷移を既定予測にし、訓練期間内で固定した十分な局所証拠がある場合だけ時間残差が上書きする方式を、新しい事前登録で検証します。

简体中文：局部学习和时间信息的作用得到确认，但准确率比二阶转移对照低0.13个百分点，因此未达到采用门槛。冻结测试集保持封存。下一步将在新的预注册中验证受限混合方案：默认使用二阶转移预测，只有在训练期折叠中预先确定的局部时间证据足够强时才允许覆盖。
