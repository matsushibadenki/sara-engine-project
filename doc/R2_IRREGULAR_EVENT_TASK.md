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

[Next] Build the deterministic case manifest and training/development-only majority, first-order, second-order and timing-aware transition baselines. Then freeze gap buckets, sparse routes, multiclass local updates, acceptance thresholds and percentile/watchdog resource limits before evaluating a candidate.

日本語: 次のR2課題には、実際の融資申請プロセス13,087件を含むBPI Challenge 2012を選びました。次活動を予測し、活動順と件数を保ったままイベント間隔だけを一定化することで、時間情報の因果的な有用性を検証します。

简体中文：下一项R2任务选用包含13,087个真实贷款申请流程的BPI Challenge 2012。任务预测下一活动，并在保留活动顺序和事件数量的同时把事件间隔固定，以检验时间信息是否真正有用。
