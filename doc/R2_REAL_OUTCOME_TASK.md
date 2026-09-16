# R2 Real Outcome Task

## Selection — 2026-09-16

[Done] R2 now has a real, independently recorded outcome source: the UCI Beijing Multi-Site Air Quality dataset. It contains 420,768 hourly rows from 12 monitoring sites between March 2013 and February 2017. The original files report air-quality measurements from the Beijing Municipal Environmental Monitoring Center and matched meteorological observations from the China Meteorological Administration. UCI distributes the dataset under CC BY 4.0.

Source: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), DOI `10.24432/C5RK5G`.

The downloaded outer archive SHA-256 is `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`. Its nested 12-station archive SHA-256 is `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.

## Prediction contract

At hour `t`, the learner sees only allowed measurements available at `t` and earlier. It predicts whether `PM2.5(t+1) > PM2.5(t)`. The outcome arrives one hour later. Pairs with missing PM2.5 or a non-hourly gap are excluded; future weather, another station's future row and test-derived preprocessing are forbidden.

The source audit found 409,180 valid consecutive PM2.5 pairs. Per-station rise rates range from 47.54% to 50.89%, so constant-class accuracy cannot look strong merely through severe imbalance.

Chronological boundaries are fixed before feature or model implementation:

- Training: 2013-03-01 through 2015-12-31
- Development: 2016-01-01 through 2016-06-30
- Frozen test: 2016-07-01 through 2017-02-28

Random row splitting is prohibited. The immutable archive necessarily contains every outcome, and the source audit computed whole-source aggregate completeness and rise rates before this contract was frozen. This is recorded as prior exposure. Model design, feature fitting and threshold selection may not inspect test-period records or period-specific metrics; only the single-use evaluator may compute them after the model protocol is frozen.

## Evidence requirements

The candidate must face a training-majority constant, current-change persistence, bounded transition table, equal-information sparse non-spiking learner, frozen SNN, shuffled outcomes and temporal-order destruction. Report balanced accuracy, Brier score, every station, seasonal and missingness slices, adaptation over time, state bytes, event work, CPU latency, peak RSS and deterministic replay.

Passing this task requires improvement over the strongest bounded baseline. A timing claim additionally requires a temporal ablation effect. Exact equivalence to the sparse scalar control would rule out an SNN-superiority claim. Physical energy remains unclaimed without paired measurement.

[Done] The deterministic source manifest contains 290,312 training pairs, 50,825 development pairs and 68,019 sealed test pairs. Development balanced accuracy is 50.00% for the training-majority constant, 53.05% for current-change persistence, 55.33% for the frozen bounded station×hour×direction table and 55.40% when the same table receives delayed online outcomes. No test-period metric was computed during this stage.

[Done] The model protocol was frozen and the development run completed. Sparse local residual learning reached 62.07% balanced accuracy versus 55.40% for the online transition table, but temporal-order and maximum-latency gates failed. The frozen test remains sealed. See [development decision](R2_BEIJING_DEVELOPMENT.md).

[Next] Move the R2 timing question to an irregularly timed real event stream where intervals are predictive. Keep this Beijing result as evidence for local delayed-outcome learning only.

日本語: R2の実課題として、北京12地点の実測大気品質データを選定しました。時刻`t`までの観測から次の1時間でPM2.5が上昇するかを予測します。学習・開発・凍結テストは時間で完全に分離し、テスト由来の前処理や未来情報を禁止します。

简体中文：R2真实任务选用北京12个监测站的实测空气质量数据。系统根据时刻`t`及之前的观测，预测下一小时PM2.5是否上升。训练、开发和冻结测试按时间严格分离，禁止使用测试期预处理统计或任何未来信息。
