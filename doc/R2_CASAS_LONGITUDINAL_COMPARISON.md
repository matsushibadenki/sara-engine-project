# Longitudinal notification-state comparison — 2026-10-12

## Decision

🟢 [Done] Selected a fresh, independent CASAS household with recurring **raw notification state** and adequate both-class chronological development support. Eight delayed sparse readouts execute exactly **18,192 changed local writes each**, with zero development writes and exact saved-state/fresh-process replay. The preregistered rich-current temporal contrast gate still fails (0/10), and a separately registered first-score diagnostic fails all four quality/mechanism thresholds. Preserve these negatives and every prior test seal; there is no production or SNN adoption.

🟠 [Next] Inspect timestamp/schema metadata for an **unconsumed household with at least 180 collection days**, then preregister complete current-notification inputs, observed future outcomes, both-class chronological support and same-current temporal contrasts before values. The 60-day annotated extracts support both classes but still provide inadequate support per current/clock/history branch. Use a genuinely longer raw stream rather than moving HH101/HH102 splits, weakening consumed contrast thresholds or opening their sealed tails. The official raw CASAS archive is a candidate; acquire a bounded member if practical, not an unreviewed multi-gigabyte workload.

## Official source and identities

[CASAS official dataset index](https://casas.wsu.edu/datasets/) now links to [Cook's longitudinal home archive](https://zenodo.org/records/15708568), DOI 10.5281/zenodo.15708568. Cite Cook, Crandall, Thomas & Krishnan (2013), *CASAS: A smart home in a box*, DOI 10.1109/MC.2012.328. The archive describes ambient PIR/door records and external activity annotation in some homes. This is distinct from the consumed UCI traffic, appliance and occupancy sources. Different homes within one CASAS archive are distinct source identities; this experiment does not test cross-home transfer.

Acquired `labeled_data.zip` from the official Zenodo record to `data/raw/r2_casas_longitudinal/source.zip` (SHA-256 `7d9e976308e1cb5bc91cb39937848bfe073fc7b44c8a33cf166de203014b0091`, approximately 225 MiB). No raw records are redistributed. The current bundle lacks Aruba, so the smallest numbered HH home, **HH101**, was selected by identity and date metadata. A first row mistakenly displayed as a header exposed one July 20 notification and annotation; exclude that entire day and record the exposure.

The initial HH101 identity failed before support counts or scores on variable fractional timestamps, missing trailing annotation columns and a single 0.22547-second clock inversion on July 21. Keep original registration, failed execution markers, parser amendments and failure receipt. A fresh **HH101 v2** excludes already-opened July 21 as well; July 22–August 19 training, August 20–September 3 development, September 4 onward sealed. Its five-minute future Sleep/Sleeping task has 82/5,135 positive/negative training and 91/3,018 development examples; both class-day counts pass, but example counts and 0/10 contrasts fail. No HH101 learner is trained or scored, and 65,019 sealed record messages/annotations remain uninterpreted.

Next select **HH102**, another household, prospectively with a different observable task. Metadata-only inspection shows 62 calendar days, June 15–August 15, 2011, and no timestamp inversion. No HH102 sensor, outcome or annotation distribution was inspected before its registration. An implementation amendment skips one record lacking sensor/message fields as a non-notification, without interpreting the remaining field or assigning it a negative outcome. The original failed marker and registration remain. Historical source strings/checksums corresponding to earlier code pins are retained in `casas_source_implementation_history_v1.json` for audit.

## HH102 task and preregistered support

Current state is the **current event's raw sensor ID and binary message**, not a fully observed physical state of all rooms. This distinction avoids imputing sensor silence as zero. All arms share these exact inputs, weekday and quarter-hour clock. Continuous measurements and activity annotations are excluded prospectively.

Take at most one cutoff per 60 seconds. History contains the last eight binary notifications, with every adjacent gap in [0,300] seconds. Future result is the first binary notification at least 300 seconds later, provided it arrives within 600 seconds. A sensor identifier beginning with `Kitchen` and message `ON` is positive; another actual binary notification is negative. No missing/no-event interval is a negative. This is a conditional future-notification prediction task, not occupancy, sleep, anomaly or health assessment. Purge crossing history/outcomes and interrupted histories.

Freeze June 15–July 14 training, July 15–29 development, July 30 onward sealed. Support thresholds remain >=500 examples/class on >=20 training days/class, >=200/class on >=10 development days/class, and >=10 exact current/clock groups containing distinct joint-history addresses with >=4 observations per opposite future label.

| Evidence | Training | Development |
| --- | ---: | ---: |
| Usable decisions | 6,064 | 3,213 |
| Future kitchen ON | 670 | 375 |
| Other observed notification | 5,394 | 2,838 |
| Days for each class | 30 | 15 |

All 18 raw current sensor/message states recur in training. Of 1,896 current/weekday/quarter-hour contexts, 1,118 recur. **Supported opposite-history/outcome groups remain 0/10**: recurrence and class support do not themselves establish temporal identifiability. This source gate fails before fitting comparison arms. No thresholds or split were relaxed. Extraction opens 218,137 pre-test rows; excludes 315 nonbinary and one non-notification row; counts 1,367 late outcomes and 1,941 interrupted histories. Four unresolved cutoffs are not scored. All 63,736 sealed HH102 records are checked only by date; their messages and annotations are not parsed.

## Separately registered first-score diagnostic

After the source gate failure is known, `casas_notification_diagnostic_registration_v1.json` pins the code, source, old evidence and unchanged comparison conditions **before any learner scores**. This is explicitly a post-support descriptive comparison on already inspected support data; source support is permanently failed and promotion eligibility is always false. It is not a fresh unseen source qualification or evidence that timing is necessary. Even a nominal metric pass would not overturn the source gate.

Eight arms use three local routes: global, shared current/clock, and arm detail. Integer negative/positive counts use hierarchical scalar backoff with strength 8. A receipt owns its exact address tuple, cutoff and due time. The chronological heap delivers all due outcomes before the next cutoff; identity, due time and route capacity are checked before calling the outcome supplier. No future-label access via the readout occurs before due. Each training outcome changes one count in each of three distinct routes: **6,064 × 3 = 18,192 real writes per arm**, 145,536 total. Development does not update counts.

Controls include current and finer five-minute current clock, full unordered history histogram, lag-1/lag-4 transition, joint transition+gap bins, constant-gap destruction, half-rotated training outcomes and an identical scalar-joint reference. The gap bins are <1, <10, <60 and >=60 seconds; constant gap uses bin 2 in both channels. Half-rotation is an offline mismatched-feedback control, not a deployable causal source of outcomes.

| Arm | Daily mean Brier | Balanced accuracy | Changed writes |
| --- | ---: | ---: | ---: |
| Current | 0.104570 | 50.6086% | 18,192 |
| Finer current | 0.103028 | 50.3748% | 18,192 |
| Histogram | 0.103486 | 50.5357% | 18,192 |
| Transition | 0.103438 | 50.5181% | 18,192 |
| Joint | 0.102301 | 50.5786% | 18,192 |
| Constant gap | 0.103438 | 50.5181% | 18,192 |
| Shuffled outcome | 0.107456 | 49.9471% | 18,192 |
| Scalar joint | 0.102301 | 50.5786% | 18,192 |

Joint improves daily Brier by **0.000726** versus the best sufficient reference, below 0.01. Balanced accuracy is 0.0300 percentage points below the best reference, rather than 2 points above. Constant-gap loss is 0.001137 vs required 0.005; shuffled-outcome loss 0.005154 vs required 0.01. All four quality/mechanism gates fail. Scalar equivalence passes exactly, providing no spike-specific evidence.

All saved states restore exactly and reproduce development probabilities. Fresh-process replay refits the same integer counts, reproduces all prediction hashes and compares published research checkpoints. Routes are bounded at 100,000; observed counts range 3,793–6,008. Peak pending predictions is five vs capacity 32; extraction cap 16. CPU-only sparse dictionaries, no matrix computation, GPU or global backpropagation. No energy or latency superiority is measured. Final experimental readout artifacts live only in `models/casas_hh102_diagnostic/`, not workspace; they are not production models.

## Verification and reproducibility

71 related tests cover overlapping horizons, earliest due order, counterfeit/stale receipt rejection before supplier access, pending/capacity bounds, frozen updates, integer write/checkpoint integrity, ignored annotations, observed future targets, malformed sealed fields, split purging, variable timestamp precision, old code/history pins, unchanged negative artifacts and final trace/write counts.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_longitudinal_comparison.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_notification_comparison.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_notification_diagnostic.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_delayed_sparse_counts.py tests/test_casas_longitudinal.py tests/test_casas_notification.py tests/test_room_temporal.py tests/test_traffic_temporal.py tests/test_appliances_learning.py tests/test_appliances_prospective.py
```

Registration/execution are consumed identities and must not be repeated. Reacquire the exact official archive for replay. All protocol, registration, source failure, amendment and result receipts use managed processed locations. No earlier appliance, traffic, room, synthetic or electricity evidence was altered; prior held-out periods remain unopened.

日本語: 長期HH102では開発15日すべてで両クラスが成立し、現在通知も反復しました。ただし強い現在状態・時計条件下の履歴対照群は0/10で不成立です。別登録の診断比較は8群各18,192回の実更新と遅延処理・再現性を検証しましたが、4つの性能・機序閾値はすべて未達です。SNN優位性や時間情報の必要性は主張せず、旧結果と封印を維持します。

简体中文：HH102开发期15天均包含两类结果，当前通知也重复出现。但强当前状态与时钟条件下的历史对照组仍为0/10。另行登记的诊断比较验证了八组各18,192次真实局部写入、延迟反馈和精确重放，四项性能与机制门槛均未达到。不主张SNN优势或时间信息必要性，保留既有负结果与测试封存。
