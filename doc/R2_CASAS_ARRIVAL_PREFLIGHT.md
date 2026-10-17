# Fresh HH105 training-only arrival and state-growth preflight — 2026-10-17

The identity, split rule, target, budgets and whole-run stopping policy were fixed before reading HH105 sensor/message fields. This preflight measures input arrival, recurrence, vocabulary and route growth. It computes no task class labels or prediction scores and performs no learner updates. Earlier quality negatives remain in force.

## Preregistration and scope

HH105 is the smallest unused HH identity after HH101–HH104. The selection receipt records only its official ZIP directory name, offset, size and CRC before acquisition; no outcome-based identity selection or substitution is permitted. The [official CASAS archive](https://zenodo.org/records/15708568) is acquired by HTTP range for `data/hh105.csv` alone. Cook et al., *CASAS: A smart home in a box* (2013), [DOI: 10.1109/MC.2012.328](https://doi.org/10.1109/MC.2012.328), is the provider's requested citation. Full archive MD5 is not claimed as verified. Exact range length, compressed SHA, expanded SHA/size and ZIP-member CRC are verified locally.

Protocol `ff614f7562606f2eea0be4b6feb76efa47254054b792442c82e7b06c19df9a74` and code hashes are registered before timestamp metadata and sensor/message inspection. The registration pins 309 JSON/gzip artifacts, including 308 older evidence artifacts and the new immutable HH105 selection receipt. Prior negatives and conditional forecasts were known. This identity becomes consumed training input after measurement, not an independent generalization result.

A timestamp/schema-only pass must establish at least 180 chronological recorded dates. The unchanged outer splitter fixes the training prefix to `min(180, 2*N//3)` recorded dates. Independent development and test begin at the unchanged metadata boundaries. Only the training prefix is parsed for inputs during measurement. The metadata pass reads timestamps/schema across the member to establish dates and clock exclusions; predictive event fields remain sealed. Measurement discards independent rows by date before parsing their time, sensor or message. No internal confirmation, independent scores or labels are evaluated.

## Fixed target and budgets

| Resource | Frozen ceiling |
| --- | ---: |
| Compressed source member | 67,108,864 bytes |
| Expanded source member | 536,870,912 bytes |
| Parsed training rows | 5,000,000 |
| Qualified observed outcomes | 263,295 |
| Hypothetical future readout writes per arm | 789,885 |
| Audit route addresses per arm | 100,000 |
| Input event-vocabulary entries | 100,000 |
| Pending delayed extraction receipts | 16 |

The arrival target is **70,212 qualified outcomes**, fixed from the previous common-scale planning estimate for both arms. It is not a prediction-quality or class-support threshold. The first registered training prefix is the only allowed window; failure cannot extend it or trigger a household substitution.

Raw event types are `(sensor identifier, binary message)` pairs. The reference vocabulary is HH104's retained 20-type inventory. All events admitted by the original binary-message schema are included, including pairs absent from that reference; no pair is dropped or remapped for novelty. The reference comparison matches strings across households, not physical sensor identities. Original nonbinary rows remain outside the unchanged task and are counted separately. Vocabulary includes both all observed binary training inputs and the subset retained by current/two-lag readout addresses.

The exact current remains the raw event, weekday, hour and 15-minute clock bin. Pool and joint use their original address functions; the `log(80)` evidence rule is unchanged and not invoked for prediction. Histories contain eight observed binary events with adjacent gaps <=300 seconds and cutoffs at least 60 seconds apart. The first observed binary event at least 300 seconds later supplies a qualified arrival only if it is no later than 600 seconds. Late, missing and unresolved results are never converted into negative labels. No Kitchen/outcome class is computed. A dummy label is passed only because the existing address API uses the `Example` container; address generation ignores it and no learner consumes it.

At each due arrival, both bounded audit address sets are checked before either is committed. They describe would-be route occupancy, not fitted count models. The hypothetical readout count budget is three writes per qualified outcome in each arm; **actual learned count writes are zero**. Audit set insertions, recurrence counters and daily statistics are reporting work, not three-factor learning updates or energy measurements.

On any qualified/write, row, vocabulary, route or pending ceiling, the entire measurement terminates. It does not skip an unknown event and resume. All remaining training rows are date-only while the stream drains for integrity, and independent periods always remain date-only. Truncation, uncommitted arrivals and unresolved work are reported; a truncated run cannot pass the complete-window condition.

## Measurements and interpretation

The registered report includes due-date growth of rows, binary events, qualified arrivals, event types/new reference types, exact rich current contexts, retained vocabulary, per-arm route occupancy and hypothetical writes. It reports measured yield, rates per recorded training date and calendar day, delay bounds, late/unresolved outcomes, recurrence and final event inventory. These observations do not predict future stationarity, vocabulary closure, raw-row yield elsewhere, physical memory, CPU energy or independent predictive usefulness.

Preflight gates require the long source and complete bounded training prefix, the frozen arrival target, original raw-current recurrence >=0.9, at least ten rich current contexts with support eight, bounded observed routes and matched hypothetical writes. The original class/day/opposing-history and all quality gates remain **unassessed and required** for any later comparative study. Passing this input-only preflight would not satisfy them.

## Artifacts and reproduction

Selection, protocol, registration, acquisition, timestamp metadata, integrity receipt, consumed execution marker, growth trace, event inventory, result and replay receipt use `data/processed/benchmark_fixtures/casas_arrival_preflight_*_v1.json`. Raw range bytes and response headers live only in ignored `data/raw/r2_casas_arrival/`. No model is created or changed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_arrival_preflight.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_arrival_preflight.py
```

Replay needs the pinned HH105 compressed member. Exclusive output creation prevents silent replacement of a consumed run. Tests compare extraction and address counts against the original parser, preserve unknown binary events, verify late/unresolved behavior, check both arms' atomic route stop, exercise all measurement-budget boundaries and ensure date-only parsing after a stop or at the independent split. Daily-prefix tests prohibit future vocabulary from changing earlier growth observations.

## Retained result and growth

🟢 [Done] The complete source has **6,430,307 records on 1,859 chronological recorded dates**, with no clock regression. The registered outer training prefix contains 180 recorded dates, from 2011-06-15 through 2011-12-26 (195 calendar days). Independent development starts 2011-12-27; test starts 2012-02-29. The remaining **5,731,389 records** are date-only sealed during measurement, covering all independent development and test event fields. The source member is 49,022,804 compressed bytes and 254,553,158 expanded bytes, within the registered source-size ceilings.

All 180 training dates are measured without a budget stop: **698,918 parsed training rows**, including **695,102 binary input events** and **3,816 original-schema nonbinary rows**. The audit retains **18 binary event types**, including **two types absent from the HH104 reference**, accounting for **43,260 binary occurrences**. These novel pairs are included, not mapped away or filtered. The qualified-address vocabulary also contains 18 types, while only 17 occur as qualified current events; retained history is therefore not reducible to the current-only vocabulary.

**29,552 qualified delayed arrivals** occur on 171 training dates, only **42.09% of the frozen 70,212 target**. Measured rates are 164.18 per recorded training date and 151.55 per calendar day; qualified arrivals are 4.25% of binary input events. These are observed rates, not a forecast that extends this failed window. There are 6,650 late results, excluded rather than labeled negative, and four unresolved outcomes. Accepted delays range from 300.000526 to 599.955637 seconds, mean 338.706536 seconds; peak pending is five, below 16.

Current recurrence passes its unchanged input conditions: all qualified raw-current observations belong to repeated types (fraction 1.0), with 4,089 rich current contexts and **1,257 contexts at support eight or more**. This does not establish opposing future outcomes; class labels and the corresponding gates are not computed.

| Recorded date index | Date | Cumulative arrivals | Input types / novel reference types | Pool routes | Joint routes |
| --- | --- | ---: | ---: | ---: | ---: |
| 30 | 2011-07-29 | 3,713 | 18 / 2 | 3,326 | 4,284 |
| 60 | 2011-08-28 | 8,234 | 18 / 2 | 5,698 | 8,093 |
| 90 | 2011-09-27 | 12,908 | 18 / 2 | 7,625 | 11,544 |
| 120 | 2011-10-27 | 19,044 | 18 / 2 | 9,434 | 15,522 |
| 150 | 2011-11-26 | 24,143 | 18 / 2 | 10,411 | 18,213 |
| 180 | 2011-12-26 | 29,552 | 18 / 2 | **11,327** | **20,839** |

Both observed route counts remain below 100,000. The hypothetical readout update budget is **88,656 writes per arm**, below 789,885; actual learner writes are **zero in both arms**. No final model is created. The conditional full known-vocabulary envelopes at 18 retained types are 60,481 pooled routes and 62,717,761 joint routes. The measured joint occupancy is much smaller than its syntactic envelope; exceeding an envelope certificate is not a measured overflow. The stable vocabulary over this measured prefix is not an all-future closure guarantee.

**Preflight fails on arrival quantity.** Long-source, complete-window, recurrence, observed-route and matched hypothetical-write checks pass, but the predeclared arrival target fails. The window is not extended and HH105 is not replaced after the result. There are no class-support, temporal-identification, predictor-quality or SNN-advantage claims; their original requirements remain unchanged and unassessed. A capacity/recurrence result cannot override this planning failure or any prior learner negative.

145 related tests pass. Fresh-process summary, full 180-date growth trace and event inventory replay are exact. The 309 artifacts pinned by this registration remain hash-identical, including 308 older evidence artifacts and the immutable selection receipt; all older code/negative-result seals are preserved. The 5,731,389 HH105 independent rows are newly retained as sealed predictive fields, with no need to open earlier households' raw data.

🟠 [Next] Preregister a timestamp-only upper bound on achievable arrival counts before consuming another household's sensor/message fields. Validate it on existing training timestamps and adversarial synthetic cases so the bound cannot fall below actual qualified arrivals under the unchanged eight-event, gap, cutoff-spacing and delayed-result rules. Use it only as an early exclusion screen: an upper bound below 70,212 stops input inspection, while a higher bound still requires all original source and quality prerequisites. No target relaxation, window extension or post-score identity substitution.

🔴 [Later] A new comparative learner or independent validation needs a separate justified preregistration and unmet source/positive-component/quality gates. This failed preflight does not authorize predictive evaluation, factor growth or runtime promotion.

日本語: 未使用HH105の訓練180記録日を予算停止せず測定しました。未知2種類を含む18種類を保持し、現在状態の反復と状態数11,327／20,839は成立しましたが、到着29,552件は固定目標70,212件に届かず不合格です。仮の読み出し更新は両群88,656回、実学習は0回でした。既存の負の結果と独立期間5,731,389件の封印を維持し、期間延長や対象差替えはしません。次は新しい入力を消費する前に、日時だけの到着数上限を事前登録して検証します。

简体中文：在预算内完整测量了未使用HH105的180个训练记录日，保留含2类新事件的18类输入。当前状态重复和11,327／20,839个地址满足条件，但29,552次合格到达未达到固定目标70,212，预检未通过。两组假设读取更新均为88,656次，实际学习均为0。保留既有负结果与独立期间5,731,389条封存，不延长窗口或替换家庭。下一步在消耗新输入前预注册并验证仅时间戳的到达数量上界。
