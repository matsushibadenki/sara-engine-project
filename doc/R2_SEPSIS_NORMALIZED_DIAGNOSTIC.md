# R2 Sepsis Source-Normalized Router Diagnostic

## Decision — 2026-09-16

[Done] The registered training-only diagnostic is retained as a negative result because seven of eight gates passed. Development and frozen-test partitions remained unopened by this run. Protocol SHA-256 is `77f11279defb04befdd877315be3288c08dc9fd089e3c60e78e3ff5e67e8f83e`; result SHA-256 is `312012a19188a9e2c1096a0b592d0a93d4cfc4fc3019d6a08a5e3fe3a6bb4d9d`.

Outcome-blind calibration selected normalized score threshold `0.6223091976516634` and produced a 10.00% calibration override rate. On the untouched training-internal confirmation fold, override rate was 10.05%. Top-1 improved 56.42%→57.22%, macro-F1 38.06%→40.66%, and Brier 0.5818→0.5779. Constant-gap macro-F1 was 40.21%, so the timing gate passed. Scalar equivalence and resources passed.

Shuffled-outcome macro-F1 was 37.37%, below the 38.06% base and therefore eliminating the candidate's 2.60-point uplift. However, its absolute 3.29-point gap from the candidate did not meet the preregistered 10-point threshold. The run therefore remains failed.

[Done] The absolute shuffled-gap gate is structurally mismatched to a router capped near 10% overrides. Correct it prospectively for the final protocol: shuffled local learning must provide no positive macro-F1 gain over the simultaneously updated base. Do not alter the model, score threshold, learning rule, features or probability rule.

[Next] Preregister and implement one final Sepsis test attempt with threshold `0.6223091976516634`, maximum 12% overrides and the relative shuffled-uplift gate. This correction is post-diagnostic and must be disclosed in the final result.

日本語: 正規化ルーターは正解率、macro-F1、Brier、時間条件を改善し、上書き率を10%に抑えました。ただし絶対10ポイントのシャッフル差だけが不合格です。結果は不合格として保存し、最終プロトコルではモデルを変えず、シャッフル時の改善が基準以下へ消えることを要求します。

简体中文：归一化路由器改善了准确率、macro-F1、Brier和时间条件，并把覆盖率限制在10%左右。唯一失败的是绝对10个百分点的打乱差距。该结果保持失败；最终协议不修改模型，只要求打乱后的提升降至基线以下。
