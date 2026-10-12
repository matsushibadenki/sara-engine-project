# Long-household qualification and delayed comparison — 2026-10-12

## Decision

🟢 [Done] A genuinely long, unconsumed household **passes every registered source condition**: 2,071 recorded days, recurring current notifications, **73/10 required history contrast groups**, and both classes on all 60 development days. Execute the frozen eight-arm comparison with **56,028 changed local writes per arm**. Saved-state and independent-process replay agree; old negative evidence and all seals remain intact.

The temporal joint readout **fails usefulness and gap-necessity gates**. Best current-state balanced accuracy is 62.5424%, joint 59.9764%; best sufficient non-spiking daily Brier is 0.069184, joint 0.069395. Constant-gap destruction improves the joint result rather than removing useful timing. Paired-outcome and scalar-equivalence gates pass. Keep this as a qualified-source **negative learner result**, not an SNN advantage or production adoption.

🟠 [Next] Preregister a **training-only audit and bounded confidence/backoff comparison for sparsely supported temporal branches**, then use fresh independent validation identities for generalization. Keep shared current inputs, sufficient non-spiking references, exactly matched changed writes and due-ordered feedback. Do not select a new gap encoding, gain, decision threshold or confidence rule from these development scores. HH103 development results are consumed; its long test tail stays sealed.

## Independent source, bounded acquisition and metadata

The [CASAS official index](https://casas.wsu.edu/datasets/) points to [Cook's longitudinal home archive](https://zenodo.org/records/15708568), DOI 10.5281/zenodo.15708568. Cite Cook, Crandall, Thomas & Krishnan (2013), *CASAS: A smart home in a box*, DOI 10.1109/MC.2012.328. The archive contains ambient PIR/door notifications from distinct households. **HH103** is the smallest unconsumed HH identity; HH101/HH102 and their sealed periods are excluded from selection. Household independence is not a claim of independent timestamps or cross-home transfer.

HTTP byte ranges avoid downloading the full 2,723,059,864-byte archive. Read its 128 KiB tail/central directory and a 1 KiB local header, then acquire only `data/hh103.csv`'s 41,980,989 compressed bytes at offsets **135–41,981,123**. Validate the exact Content-Range, compressed SHA-256, expanded length **214,673,501**, CRC32 **532,271,806**, and expanded SHA-256. The acquisition receipt pins the source URL and directory/member metadata. The full-archive MD5 is not independently recomputed; integrity is pinned for the acquired member. Raw member, range headers and source data remain ignored under `data/raw/r2_casas_180/`; raw records are not redistributed.

A bounded DEFLATE reader limits chunk/line size and checks length, CRC and SHA at EOF. Before task registration, inspect **date, time and column count only**. The first metadata attempt stopped because the raw member is whitespace-separated despite its `.csv` suffix, whereas earlier annotated extracts used commas. Record that metadata parser amendment and original code hash; no sensor/message or outcome cells were inspected, and selection/split rules did not change. Metadata finds 5,425,649 records, all four-column format, June 15, 2011–May 10, 2017, **2,071 distinct dates and no clock reversals**.

Freeze the selection rule before metadata interpretation: require >=180 valid recorded dates; exclude every date involved in a timestamp inversion; train on the first min(180, floor(2N/3)) valid recorded dates, develop on the next min(60, floor(N/6)), seal the remainder. Here:

| Period | Boundary | Recorded days |
| --- | --- | ---: |
| Training | 2011-06-15 <= time < 2011-12-13 | 180 |
| Development | 2011-12-13 <= time < 2012-02-11 | 60 |
| Sealed | 2012-02-11 onward | 1,831 |

No source input or future result values are interpreted until `casas_180_registration_v1.json` pins protocol, code, metadata, member integrity and old evidence hashes. The remaining tail is subsequently inspected **only by date**, before parsing time/sensor/message. Its bytes are streamed for integrity validation; this is not label or input inspection. No test examples or metrics are constructed.

## Unchanged task and strengthened recurrence qualification

Keep the HH102 observable task and controls unchanged. At most one cutoff per 60 seconds; input is the **current raw sensor ID and binary message**, weekday and quarter-hour clock, with eight prior/current binary events and adjacent gaps in [0,300] seconds. This is notification state, not complete household physical state. Continuous messages and activity annotation are never predictive inputs.

The result is the **first actual binary notification >=300 and <=600 seconds later**: Kitchen-prefix sensor `ON` is positive; any other observed binary notification is negative. Silence, missing or late follow-up is not a negative. Purge crossing histories/outcomes and interrupted histories. This task is conditional on observed follow-up, not occupancy or health assessment.

Keep every earlier source threshold: training >=500/class on >=20 days/class; development >=200/class on >=10 days/class; >=10 exact current/weekday/quarter-hour groups with distinct joint-history addresses and >=4 observations per opposite future label. Add explicit recurrence requirements: >=90% of training decisions use a repeated raw current state; >=10 rich-current contexts have >=8 observations. A source failure would stop before fitting or scoring; **no diagnostic bypass** is permitted in this identity.

| Qualification | Result |
| --- | ---: |
| Training decisions | 18,676: 2,576 positive / 16,100 negative |
| Usable training class-days | 179 for each class; one recorded date has no eligible decisions |
| Development decisions | 10,136: 940 positive / 9,196 negative |
| Development class-days | 60 for each class |
| Raw current states / repeated states | 16 / 16; 100% repeat-row coverage |
| Rich-current contexts / contexts with >=8 observations | 3,344 / 697 |
| Supported opposite-history/result groups | **73 / required 10** |
| Opened records / sealed date-only records | 593,285 / **4,832,364** |

Exclude 1,718 nonbinary rows, 11,359 late outcomes and 8,871 interrupted histories; four unresolved cutoffs are not scored. The rich-current contrast condition now passes without dropping weekday/clock, coarsening consumed addresses, changing label/horizon or reducing support counts.

## Eight-arm result with real delayed writes

Reuse the previously frozen address functions and delayed integer-count learner unchanged. Three distinct local counters per feedback: global, shared current/clock, arm detail, with hierarchical prior strength 8. Exactly **18,676 × 3 = 56,028 changed writes per arm**, 448,224 across eight arms. Receipt identity, exact due time, chronological order and capacity are checked before calling the label supplier. Pending outcomes are flushed before the next prediction; development updates are zero.

The controls retain current quarter-hour and five-minute clock, full unordered history histogram, lag-1/lag-4 transition, joint transition plus lag-1/lag-4 gap bins (<1,<10,<60,>=60 seconds), constant-gap bin 2, half-rotated training outcomes, and identical scalar-joint implementation. The shuffled-outcome arm is an explicitly offline mismatched-feedback control, not a deployable outcome source.

| Arm | Daily mean Brier | Balanced accuracy | Changed writes |
| --- | ---: | ---: | ---: |
| Current | 0.069495 | 62.5424% | 56,028 |
| Finer current | 0.069493 | 60.1523% | 56,028 |
| Histogram | 0.069205 | 61.9584% | 56,028 |
| Transition | **0.069184** | 62.2710% | 56,028 |
| Joint | 0.069395 | 59.9764% | 56,028 |
| Constant gap | 0.069184 | 62.2710% | 56,028 |
| Shuffled outcome | 0.096321 | 49.6237% | 56,028 |
| Scalar joint | 0.069395 | 59.9764% | 56,028 |

Keep the earlier performance thresholds exactly:

- Daily Brier gain >=0.01 over the best sufficient reference: **fails**, joint is 0.000211 worse.
- Balanced-accuracy gain >=2 percentage points over the best reference: **fails**, joint is 2.5660 points worse.
- Constant-gap daily Brier loss >=0.005: **fails**, removing gap information improves Brier by 0.000211 and balanced accuracy by 2.2946 points.
- Mismatched-outcome daily Brier loss >=0.01: **passes**, loss 0.026926.
- Scalar-joint prediction equality: **passes exactly**.

Constant-gap and transition prediction hashes match exactly; joint and scalar-joint match exactly. Final experimental count checkpoints are deterministic compressed JSON under `models/casas_hh103_180/`; restore reproduces every development prediction. They are research readouts, not production SNN artifacts.

## Limits and next hypothesis

The source gate establishes **available contrast support**, not that this timing representation predicts better or that timing is necessary. The trained temporal readout loses to sufficient current/history controls even though correct outcome pairing matters. A **post-score training-checkpoint-only** support audit (no tuning, no development/test inputs) finds 8,346 joint detail routes, 5,660 supported only once, mean 2.24 training outcomes per route. Transition has 5,787 detail routes and mean 3.23; current has 3,344 and mean 5.58. This motivates a future, separately preregistered confidence/backoff hypothesis; it does not causally establish why joint failed or justify tuning on this result.

All arms retain the 100,000-route / 32-pending-receipt ceilings; actual routes range 6,689–11,691 and peak pending is five. Extraction pending ceiling remains 16. CPU-only sparse local counters, no matrix operations, GPU or whole-network gradients. No physical-energy, latency superiority, factor-discovery, cross-household transfer or spike-specific usefulness is established. Anonymous factor growth and production integration remain deferred.

## Verification and replay

The original read-only full replay was sampled during its unexpectedly slow capacity check: `set(receipt.addresses) - cells.keys()` scans retained state, including during frozen deliveries. Stop only that verification process; preserve its original implementation, experiment result and published models. Separately register `SparseCapacityCounts` before using it for equivalence verification. Since each receipt already guarantees three distinct routes, count missing keys with exactly three membership probes during training and none during frozen delivery. Receipt/clock/capacity/supplier validation, prior, count writes, thresholds and checkpoint schema remain identical. Tests prohibit model-wide iteration with 10,000 unrelated retained routes and compare reference/faster state and error behavior. This is verification-backend work, not a new scientific arm, a retuned result or a measured energy advantage.

89 related tests cover metadata-only 180-day qualification, timestamp precision, bounded compressed/expanded bytes, corruption rejection, date-first seals with malformed fields, split purging, unchanged gates, receipt identity/clock/capacity checks before supplier access, overlapping horizons, exact real write counts, frozen updates, historical negative/code pins, model integrity and training-only support audit. A fresh process independently refits every arm with the registered equivalent sparse-capacity verification backend; all source conditions, metrics, prediction hashes and checkpoints match. The original scientific implementation remains pinned and unchanged.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_180_sparse_replay.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_180.py tests/test_sparse_count_replay.py tests/test_delayed_sparse_counts.py tests/test_casas_longitudinal.py tests/test_casas_notification.py tests/test_room_temporal.py tests/test_traffic_temporal.py tests/test_appliances_learning.py tests/test_appliances_prospective.py
```

For source recovery, acquire exactly bytes 135–41981123 of the official `data.zip` into `data/raw/r2_casas_180/hh103.deflate`; the committed acquisition receipt supplies length and SHA checks. The compressed member must match before replay. No need to download or inspect other homes. Metadata/registration/execution identities are consumed and must not be rerun. Protocols, metadata, acquisition, registration, execution and result receipts remain in managed processed locations; final checkpoints remain in `models/`.

日本語: 未使用HH103の2,071日から訓練180日・開発60日を固定し、現在状態の反復・73群の履歴対照・開発両クラスの成立を確認しました。8群各56,028回の実更新を揃え、遅延処理と再現性を検証しましたが、時間分岐は強い対照を上回らず、間隔固定対照の方が良好でした。条件を緩めず負の結果として保存し、旧結果と封印を維持します。

简体中文：未使用的HH103包含2,071天记录，固定前180天训练及后60天开发，当前状态重复、73组历史对照与开发期两类支持均成立。八组各执行56,028次真实局部写入并验证延迟反馈与精确重放，但时间分支未超越强对照，固定间隔对照更好。保持原条件，保留负结果与所有测试封存。
