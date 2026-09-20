# R2 MetroPT-3 Delayed-Outcome Pilot

## Selection — 2026-09-20

[Done] The next real delayed-outcome pilot uses the [UCI MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset), DOI `10.24432/C5VW3R`, under CC BY 4.0. It contains sensor readings from a metro train compressor operating between February and September 2020. The observations are unlabeled; four company failure-report intervals provide independently recorded delayed outcomes. This combination is useful for testing whether a bounded earlier event trace can receive later local credit without manufactured intermediate labels.

The official ZIP is stored at `data/raw/r2_metropt3/metropt3.zip`, SHA-256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`. It contains the 218,300,507-byte `MetroPT3(AirCompressor).csv` and the publisher's data-description PDF. The source has 1,516,948 rows and 15 sensor signals plus timestamp and source index.

## Source audit

[Done] The first audit is retained as a parser-contract negative: it expected pandas' inferred name `Unnamed: 0`, while the raw CSV header is an empty string. It did not train or score a model. Result SHA-256: `3b5688bebb72cf019b7829142440bec41512147e613c7a6063c620dff9a10b78`.

[Done] V2 uses the raw header and passes. The stream spans `2020-02-01 00:00:00` through `2020-09-01 03:59:50`, has no negative or duplicate timestamps, a median 10-second gap, and a maximum gap of 172,918 seconds. The four reported failure windows contain 8,663 / 2,360 / 17,315 / 1,622 rows, respectively. [Audit result](../workspace/evaluation/metropt3_source_audit_v2.json) SHA-256: `76c20434c2d0fae0b5af5b3806ae79aa71bf5db1bf158ee70724345ae9345568`.

## Evidence boundary

MetroPT-3 is a real sensor stream but has only four reported failure intervals and comes from one compressor. It is therefore suitable for a small pilot, not a strong generalization claim. Individual rows from the same failure must never be split across training, development, and held-out evaluation. The failure report identity—not row count—is the independent unit.

The future task contract must be frozen before extracting candidate features. It should use chronologically separated failure reports, complete pre-failure windows, and matched non-failure windows from disjoint calendar ranges. The delayed target must be defined from a report start time, with a fixed prediction horizon and exclusion buffer. Raw future measurements, report identity, absolute date, maintenance-after-the-fact information, and any statistic fitted on development or held-out periods are forbidden inputs.

The first candidate must compare a no-learning prior, a bounded recent-state control, a single-factor local predictor, and Local Credit Packet or Sparse Predictive Factorization only if the same outcome information and update budget can be maintained. Required controls include outcome shuffle, time-order destruction, eligibility expiry, route reset, and exact scalar equivalence. Report event encoding cost, retained state, update count, CPU latency, class balance, report-level accuracy/abstention, and deterministic replay. With only four positive reports, no result may be described as production-ready, cross-compressor generalization, physical energy saving, or general backpropagation replacement.

[Done] Froze the report-level [feasibility protocol](../data/processed/benchmark_fixtures/metropt3_report_feasibility_v1.json) at SHA-256 `2058feb93d760777f861348c30041b8f1da19e7689230ab5e90c5c0a8c017476` before inspecting window coverage. Reports 1–2 were assigned to training, report 3 to development, and report 4 to an untouched held-out identity. Each six-hour pre-failure window was paired with a same-clock six-hour control window exactly 14 days earlier. The gate required at least 1,500 rows, maximum gap 600 seconds, median gap 20 seconds, no window overlap, and a 24-hour exclusion buffer around every failure.

[Done] The one-shot timestamp-only feasibility result is a retained negative (SHA-256 `2b255dc08d1c2453aff4e8a8c6bd9c67fb005a7dc2efbca013e17b3c73bacaec`). Report split counts, matched controls, non-overlap, and failure-buffer isolation pass. Seven of eight windows pass coverage. The negative control paired with failure 1 has only 739 rows versus the frozen minimum of 1,500, although its maximum gap is 10 seconds; the source simply has no observations for much of that clock window. All four positive windows pass with 1,790/2,179/2,179/2,179 rows. No sensor value was read, no held-out sensor row was materialized, and no candidate was scored.

[Next] Stop this MetroPT-3 task identity before model implementation. Do not move or widen the failed control window after observing coverage. A future MetroPT-3 study would require a substantively new, outcome-blind control-selection hypothesis and cannot reuse this gate as a pass; given only four failures from one compressor, prefer a new source with more independent outcome episodes.

日本語: MetroPT-3の報告単位feasibility gateは、8窓中1つの対照窓が最低行数を満たさず不通過でした。結果を見て窓を移動せず、センサー値や候補を評価する前にこのtask identityを停止します。

简体中文：MetroPT-3按故障报告划分的可行性门槛未通过；八个窗口中有一个对照窗口未达到最低行数。不会在观察结果后移动窗口，并在读取传感器值或评估候选之前停止此任务身份。
