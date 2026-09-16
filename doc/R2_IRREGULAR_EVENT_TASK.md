# R2 Irregular Real-Event Task

## Selection — 2026-09-16

[Done] The next R2 task is BPI Challenge 2012, a real loan-application event log published by Eindhoven University of Technology through 4TU.ResearchData. The hash-pinned source contains 13,087 cases, 262,200 events and 24 activity types between October 2011 and March 2012.

Source: [BPI Challenge 2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1), DOI `10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f`, distributed under the 4TU General Terms of Use.

The file MD5 matches the publisher value `74c7ba9aba85bfcb181a22c9d565e5b5`; SHA-256 is `5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`.

## Why this task

The task predicts the next activity immediately after each non-terminal event. Inputs are limited to the bounded observed activity/lifecycle prefix, elapsed times, hour, weekday and prefix length. Case identifiers, case attributes and future events are forbidden as features.

Intervals are genuinely irregular: median 47.875 seconds, 90th percentile about 20.1 hours, 99th percentile about 7.9 days, and 148,911 distinct observed gap values. The primary timing ablation replaces positive gaps with one constant bucket while preserving every case, activity order, lifecycle and event count. This directly tests whether elapsed time adds information beyond sequence counts.

Cases are split by start time, never by individual event:

- Training: first 70% of case starts
- Development: next 15%
- Frozen test: final 15%

[Done] The deterministic manifest contains 9,160 training cases, 1,963 development cases and 1,964 sealed test cases, with zero case overlap. These provide 177,026 training, 39,393 development and 32,694 sealed next-activity predictions.

Development top-1 accuracy is 20.86% for the global majority, 65.66% for first-order activity transition, 80.07% for second-order activity transition, 75.35% for activity+lifecycle+observed-gap transition and 70.11% for its constant-gap control. Real gaps therefore add 5.24 points within the timing-aware baseline, while second-order activity remains the strongest overall bounded baseline.

[Done] The sparse multiclass protocol was frozen and its single development attempt was executed. The candidate reached 79.93% top-1 and 60.49% macro-F1 versus 80.06% and 55.49% for the second-order control. Timing ablation, shuffled outcomes, rare recall, scalar equivalence and resources passed, but both accuracy gates failed. The frozen test remains sealed. See [the decision](R2_BPI2012_DEVELOPMENT.md).

[Done] A training-only diagnostic fixed the confidence router, and a separate preregistered run passed all gates on the never-opened chronological test. Top-1 improved from 80.62% to 82.13%, macro-F1 from 53.60% to 56.67%, and Brier from 0.2728 to 0.2659. See [the final result](R2_BPI2012_HYBRID_FINAL.md).

[Next] Replicate the architecture on a second independent irregular event log before making a general usefulness claim.

日本語: BPI Challenge 2012の局所多クラス候補は時間情報と希少活動で効果を示しましたが、二次遷移対照の正解率を超えず不採用です。凍結テストは封印したまま、次は訓練期間内で固定する信頼度付きハイブリッドを検証します。

简体中文：BPI Challenge 2012的局部多分类候选在时间信息和稀有活动上有效，但准确率未超过二阶转移对照，因此不采用。冻结测试集继续封存；下一步验证仅用训练期确定的置信度混合方案。
