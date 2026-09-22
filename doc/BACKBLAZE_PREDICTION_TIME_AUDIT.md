# Backblaze Prediction-time Audit

## Result

The metadata- and source-only audit (`750026ed…c7b3`) reviewed the frozen 2013/2014 manifests and experiment source code. It did not read SMART values, rerun any candidate, modify any prior result, or open held-out features.

It found three research-contract defects:

1. All evaluated feature windows admit observations on the positive drive's first `failure=1` day. There is no positive lead time between the final feature and the outcome.
2. A negative case is defined using the drive's last observed day with no failure row. This is a retrospective, outcome-dependent endpoint choice rather than an at-risk control observed through a prespecified future horizon.
3. The cross-year script caps 2013 training reads at August 31, while two selected training cases have September endpoints. Its 30-row check tests length but not whether the final feature date equals the manifest endpoint.

These defects do not change the recorded numbers or the fact that the registered gates failed. They change their interpretation: Backblaze V1 and cross-year results are retrospective terminal-state classifications, not validated delayed-outcome predictions. In particular, the signed-delta cross-year result cannot establish prospective transfer.

## Corrected forward contract

`admit_prospective_failure_case` provides a small pure eligibility gate for future tasks:

- fix the feature cutoff before the outcome window;
- admit a positive only when first failure occurs strictly after that cutoff and within a declared horizon;
- admit a negative only when failure-free observation covers the entire horizon;
- abstain on censored or already-failed cases;
- require chronological, unique feature dates ending exactly at the cutoff.

This function does not repair the consumed Backblaze evaluations. Any new prospective experiment needs a newly frozen task identity, horizon, at-risk sampling plan, exact observation cutoff checks, independent development/held-out cases, and unchanged baselines before feature extraction.

## 日本語要約

既存Backblaze評価は故障当日のSMART値を含み、非故障例も観測終了日から事後選択していました。これは早期故障予測ではなく回顧的な終端分類です。cross-yearでは2013年9月終端のtraining 2件が8月31日の読み取り上限とも食い違いました。凍結結果は維持し、予測性能という解釈を取り消します。新しい実験では予測時点と将来の結果窓を明確に分離します。

## 简体中文摘要

现有Backblaze评估包含故障当天的SMART值，未故障对照也按最后观测日事后选择。因此它是回顾性终点分类，而非提前故障预测。跨年度脚本还在8月31日停止读取2013训练数据，影响两个九月终点案例。冻结结果保持不变，但撤回预测性能的解释。新实验必须明确分离特征截止日与未来结果窗口。
