# Repeated-state source screening and local identification — 2026-10-11

## Decision

🟢 [Done] Registered and executed two fresh source screens before sensor-value inspection. Neither qualifies for a development comparison. Preserve both failures, all older negative receipts and every sealed period. The additional **training-only** room diagnostic shows that locally learned temporal branches can distinguish observed histories under an identical present-state address. It does not establish temporal necessity, generalization, factor discovery or an SNN advantage.

🟠 [Next] Select a longer, fully observed discrete sensor stream with independently annotated outcomes and positive/negative support in separate chronological development days. Verify timestamp/schema metadata first; preregister sensor availability, annotation availability, recurring current-context contrasts and both-class/day support before decoding labels. Do not move the December split or open January to rescue this dataset.

## Sources and frozen tasks

1. [UCI Single Elder Home Monitoring: Gas and Position](https://archive.ics.uci.edu/dataset/799/single+elder+home+monitoring+gas+and+position), Marín López et al. (2023), DOI 10.24432/C5762W, CC BY 4.0. Official archive SHA-256 `9f28c90ee7ea129d7748f4bb3dc73ecbd0bce0c3366b43baa9c927155e41a6d3`. The frozen next-observation living-room-bit task required complete five-room binary input and all adjacent gaps at most 30 minutes. November training / December development / January-onward sealed. All 17,352 opened rows fail completeness: unreported rooms are blank. Blank is not zero and is not silently forward-filled. Zero examples, no learner scoring; 9,717 sealed rows inspected only for timestamps. This is a source-contract failure, not evidence against temporal prediction. The first execution stopped at the very first nanosecond timestamp before accessing sensor cells; a retained, hash-pinned parser amendment truncates fractional seconds to Python 3.10 microseconds. Original registration and both execution markers remain.
2. [UCI Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation), Singh & Chaudhari (2018), DOI 10.24432/C5P605, CC BY 4.0. Official archive SHA-256 `a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd`. Two digital PIR sensors are available inputs; independently manually noted occupancy is evaluator-only. This prospectively defines a PIR-only interface; temperature, light, sound, CO2 and precomputed CO2 slope are excluded. Results cannot be claimed against full multisensor current-state controls. Official metadata describes four collection days, but timestamp-only inspection found seven dates and mainly 30/31-second intervals. Freeze December 22–24 training, December 25–26 development, January 10–11 sealed before reading numeric fields.

Room history contains 16 observations, every gap strictly positive and <=65 seconds. Target is occupancy >0 at the second following observation, 60–130 seconds later. Windows crossing splits or interrupted by gaps are purged. Current controls retain **both exact PIR bits and clock**, at quarter-hour and minute resolutions. The age branch retains that shared current address and bins each sensor's last active past observation into lag 1, 2–4, 5–8, or 9–16/absent. Histogram, whole-sequence and identical scalar-age references are also specified. No future occupancy label is an input.

Support requires >=100 examples per class and >=2 days in each opened split, plus >=5 current-PIR/quarter-hour groups containing distinct age histories with >=4 observations per opposite outcome. All thresholds were frozen before numeric inspection.

| Evidence | Result |
| --- | --- |
| Training | 5,200 usable decisions: 1,519 occupied / 3,681 unoccupied; 3 days |
| Development | 2,762 usable decisions, **all unoccupied**; 2 days |
| Qualifying repeated-current contrast groups | 3 / required 5 |
| Numeric rows opened / sealed timestamp-only rows | 8,084 / 2,045 |
| Development comparison | Stopped before fitting/scoring comparison arms |
| Test metrics and production models | Not computed / not published |

The registered prospective quality gate therefore **fails on source support**. The frozen module's comparison helper is unexecuted and is not a causal overlapping-receipt runtime: do not reuse its sequential fit loop for another source. The implemented causal queue below is the verified local-update path. The reversed-home comparison is likewise unexecuted; no temporal-destruction effect was measured.

## Separately registered training-only identification

After observing the failed support screen, `room_local_identification_registration_v1.json` freezes a descriptive diagnostic on the already-opened 5,200 training decisions. This is explicitly post-support registration, not fresh unseen quality evidence. No development scores, quality thresholds or source splits are changed.

Six arms (current, minute-current, age, histogram, full sequence, identical scalar-age) each receive the same **5,200 actual delayed outcomes and 10,400 changed integer counters**. A time-ordered receipt queue flushes outcomes only at their recorded due time, before prediction when times are equal. At 30-second cutoffs with 60-second feedback, the first three predictions see 0, 0, then 2 writes; the next-row future label cannot leak into an earlier decision. Two local count writes per feedback, bounded route capacity, scalar hierarchical posterior with prior strength 8, CPU-only Python; no matrix operations, GPU or network backpropagation.

Four current-address groups have learned age-branch probabilities on opposite sides of 0.5. For example, both PIR sensors are zero at 11:00–11:14, but different history branches produce **0.139699 vs 0.732252**. This establishes representability and training-data identification only. It includes weakly supported branches and is **not the five-group support gate**. Current-only addresses remain identical; histogram/sequence references show alternative non-spiking representations are available. Age and scalar-age counts/checkpoint hashes match exactly. Integer updates are real changes, not attempted/no-op calls. Route counts range from 376 to 2,138; no physical energy measurements were made.

## Verification and reproduction

53 related tests pass, covering premature outcome rejection, overlapping delayed receipts, frozen updates, shared current inputs, sealed malformed numeric fields, old artifact/code hashes, exact write counts and canonical source replay. Separate-process replays reproduce source counts and training-only state hashes.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/motion_temporal_preflight.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/replay_room_support.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/room_local_identification.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_room_temporal.py tests/test_traffic_temporal.py tests/test_appliances_learning.py tests/test_appliances_prospective.py
```

Acquire archives from the official dataset download links into `data/raw/r2_motion/source.zip` and `data/raw/r2_room/source.zip`. Do not re-register or re-execute consumed identities. Protocols, registrations, amendments, execution markers and result receipts remain under `data/processed/benchmark_fixtures/`. The original room runner's `replay` compares integer label keys with JSON string keys and rejects equal content; the independent `replay_room_support.py` verifies pinned hashes and canonical JSON equivalence without modifying that registered code or its result.

日本語: 二値の現在状態が反復するPIR課題を事前登録しましたが、開発期間に在室例がなく比較条件は不成立です。訓練限定の別登録診断では、6群各10,400回の実更新を揃え、同じ現在状態の異なる履歴を局所読み出しで4群識別しました。これは汎化やSNN優位性の証拠ではありません。既存の負の結果と封印を維持します。

简体中文：预注册的二值PIR任务存在重复当前状态，但开发期没有有人样本，比较条件未成立。另行登记的训练诊断让六组各执行10,400次真实局部写入，在四组相同当前状态下区分了不同历史。这不能证明泛化或SNN优势；保留既有负结果及封存测试期。
