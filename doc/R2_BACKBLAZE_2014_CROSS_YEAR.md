# Backblaze 2014 Independent Cross-year Compact Event Study

## Source and Identity Gate

The official 2014 Backblaze archive is hash-pinned as `a16ae31a…9ef5`. Its 365 daily files contain 12,582,414 rows. The source audit reads only date, anonymous serial, model, capacity, and failure.

Every serial seen in the 2013 source is excluded. The remaining source contains 20,035 independent drive identities, 251 failed drives, and 185 failed drives with at least 30 observations through failure. Failure support is 133 training, 46 development, and 72 held-out cases. SMART values were not read during this audit.

## Manifest Result

The first 2014-only manifest is retained as a negative result. Exact-model one-to-one matching produced 16/16 training, 35/35 development, and 60/60 held-out cases. Training failed its frozen 50-per-class minimum; the threshold was not relaxed.

The 35/35 development cases remain valid for a stronger cross-year question: train the already frozen Compact Event hypothesis on the 498 matched 2013 training cases and evaluate once on 70 drives that never appeared in 2013. The 2014 held-out set remains sealed.

## Cross-year Result

The registered cross-year development gate is negative (`eec95da9…c413`). Route prevalence is computed from 2013 training only. Route 36 exceeded 0.95 prevalence and was removed before 2014 development access.

| Arm | Balanced accuracy | Dominant route | Median events |
| --- | ---: | ---: | ---: |
| V1 event encoding | 0.4714 | 1.0000 | 60 |
| Final log snapshot | 0.5000 | 0.9286 | 6 |
| Signed delta, unsuppressed | 0.5571 | 1.0000 | 30 |
| Signed delta, training-only suppression | 0.5857 | 0.4000 | 2 |
| Suppressed encoding, shuffled outcomes | 0.4857 | 0.4000 | 2 |

The candidate passes history, case support, selectivity, event sparsity, gain-over-snapshot, and shuffled-outcome checks. It fails only the frozen minimum balanced accuracy of 0.60. This is evidence that generic training-only route suppression transfers some useful signal across years, but it is not an accepted predictor and does not authorize held-out evaluation or additional neuron complexity.

## Prediction-time correction (2026-09-22)

The [read-only audit](BACKBLAZE_PREDICTION_TIME_AUDIT.md) found that the 2014 positive feature window includes the failure day, and negative endpoints depend on knowing the drive's final nonfailure observation. This makes the comparison retrospective endpoint classification, not prospective early warning. It also found that the cross-year script stopped reading 2013 training at August 31 although the frozen manifest includes two September-endpoint training cases. Those two received incomplete or earlier windows despite the script's 30-row history check. The prior score remains frozen but cannot support a complete-training or predictive-transfer claim. The 2014 held-out set remains sealed.

## 日本語要約

2013年に存在した全ドライブを除外し、2014年の独立driveだけで検証しました。route抑制で0.5857でしたが事前下限0.60に届かず陰性です。さらに故障当日の特徴を含むため事前予測ではなく、2013 trainingの9月終端2件も読み取り上限の影響を受けています。凍結スコアは変更せず、予測転移の主張は撤回します。2014 heldoutは未開封です。

## 简体中文摘要

研究排除了所有在2013年出现过的硬盘，只使用独立的2014硬盘进行验证。路由抑制达到0.5857，但未达到预先冻结的0.60门槛。后来发现特征包含故障当天，而且2013训练清单中两个九月终点案例受到读取截止日期影响。因此保留原始分数，但撤回提前预测和完整训练迁移的解释。2014保留集未开放。
