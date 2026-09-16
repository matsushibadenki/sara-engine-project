# R2 BPI 2012 Confidence-Hybrid Diagnostic

## Decision — 2026-09-16

[Done] The training-period-only diagnostic passed all registered gates without reading the existing development or frozen-test partitions. Protocol SHA-256 is `82a5a7a411b1932b1b3de9523466938753ae9ae0a3b6fb70f4a1c1b7e26257a5`; result SHA-256 is `6d50e6aded38fa4ad2bda2c535d899ace66fa8e26bd808f57b59f12e98ec2ad0`.

The selection fold fixed one of 48 preregistered rules: override the second-order transition prediction only when its maximum probability is at most 0.75, the local model's top-two score margin is at least 0.10, and the transition state has at least 16 observations.

On the untouched training-internal confirmation fold, the base reached 80.06% top-1, 53.06% macro-F1 and 0.2802 Brier. The hybrid reached 82.01%, 56.10% and 0.2728. It overrode 2,559 of 27,400 predictions, or 9.34%. Constant-gap macro-F1 was 55.35%, and shuffled-outcome macro-F1 was 45.06%. Explicit-neuron and scalar traces were identical.

[Done] This resolves the prior tradeoff within training-period data: selective routing preserves frequent transition structure while recovering local temporal and minority-class information. It remains diagnostic evidence because the confirmation cases come from the same source and historical training interval.

[Next] Freeze the selected rule in a separate final protocol before implementing or running a final evaluator. The final gate may use the never-opened chronological test partition once, with the exposed development period used only for online fitting under the already fixed rule. It must compare the hybrid with its simultaneously updated second-order base, retain timing/shuffled/equivalence controls, require non-inferior accuracy and Brier plus at least one point macro-F1 gain, and preserve resource limits.

[Later] Seek a second independent event log before claiming general usefulness. Exact scalar equivalence still blocks an SNN-specific quality claim, and energy remains unmeasured.

日本語: 訓練期間内の未使用確認foldで、ハイブリッドは正解率、macro-F1、Brierを同時に改善しました。約9.34%だけを局所時間モデルへ切り替えています。これは内部診断の成功であり、独立した汎化証拠ではありません。

简体中文：在训练期内未使用的确认折中，混合模型同时改善了准确率、macro-F1和Brier，只覆盖约9.34%的预测。这是内部诊断成功，不是独立泛化证据。
