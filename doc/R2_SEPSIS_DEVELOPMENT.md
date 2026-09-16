# R2 Sepsis Confidence-Hybrid Development Gate

## Decision — 2026-09-16

[Done] The single preregistered Sepsis development attempt is a negative result. The frozen test remains sealed. Protocol SHA-256 is `f6cb79e10ca590c062f4f0f49e4ff3a26cd5dfc1173de55c44134fe68b0b5f77`; result SHA-256 is `d9209c674d94656260319846fc45255467a8887a486f6b5714ef5914e0244678`.

The online second-order base reached 55.87% top-1, 37.79% macro-F1 and 0.5706 Brier over 2,046 development predictions. The inherited hybrid reached 56.06%, 41.26% and 0.5756. Accuracy and class coverage improved, but calibration worsened by 0.00502 Brier.

The candidate overrode 21.41% of predictions, compared with 10.22% in the BPI final test. Constant-gap macro-F1 was 41.33%, slightly higher than the real-gap candidate, so the timing gate failed. Shuffled-outcome macro-F1 was 33.62%; the 7.65-point loss did not reach the registered 10-point requirement. Scalar equivalence and every resource contract passed: 995 neurons/routes, 3,820 weights, 1.42 MB state, 177 maximum counted operations and 0.049 ms p99 latency.

## Architectural consequence

[Done] The BPI confidence threshold does not transfer across sources. Its absolute probability cap admits about twice as many overrides on Sepsis, exposing the poorly calibrated local probabilities. Do not open the Sepsis frozen test and do not claim cross-source R2 replication.

[Next] Replace the absolute confidence cap with a source-normalized, outcome-blind routing score fixed from training inputs and model scores only. Register an approximately 10% override budget before implementation, keep the local learner and route vocabulary unchanged, and prohibit Sepsis development labels from threshold selection. Treat this as a new hypothesis; the current result remains immutable.

[Later] If the normalized router passes a training-internal confirmation gate, preregister one final Sepsis test attempt. Independent usefulness and SNN-specific advantage remain unestablished.

日本語: 正解率とmacro-F1は改善しましたが、Brier、時間除去、結果シャッフルの3条件が不合格でした。BPIから継承した絶対確率閾値では上書き率が約10%から21%へ増え、データ源を越えて校正できませんでした。凍結testは開きません。

简体中文：准确率和macro-F1有所改善，但Brier、时间消融和结果打乱三项门槛未通过。继承自BPI的绝对概率阈值使覆盖率从约10%升至21%，无法跨数据源校准。冻结测试集保持封存。
