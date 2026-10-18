# Timestamp-only safe arrival upper bound — 2026-10-18

🟢 [Done] A preregistered timestamp-only screen bounds qualified arrivals for **any binary subsequence** of valid raw training rows. The bound dominates all stored arrival counts for three existing training windows and excludes all three windows from the current 70,212-arrival planning target. It does not change the task, inspect sensor/message values, evaluate predictions or reopen independent periods.

## Registration and fixed scope

Protocol `81875efb6a762116856c0dc2c77eca0747b30882f68219209dd6343a67e90313` and code hashes were registered before the new real-data timestamp audit. The registration pins 319 prior JSON/gzip evidence artifacts. Earlier measured counts and negative results were known; these sources are consumed training windows used to validate a method, not fresh independent data or competing dataset candidates.

The fixed sources are HH103's original 180 training dates, HH104's 80 fitted dates from the pooling diagnostic, and HH105's 180 preflight training dates. No source is acquired. Each existing compressed member is checked against its original compressed SHA and expanded SHA/size/CRC, with the original bounded decoder. Each member is below the 512 MiB expanded-source ceiling. Date prefixes exclude all rows from each training end onward before parsing time; metadata clock exclusions remain unchanged. Only training date/time prefixes are interpreted. A delimiter is chosen at the ISO-date boundary, never by inspecting the event payload.

The actual task retains eight binary events, adjacent history gaps <=300 seconds, cutoffs at least 60 seconds apart, and the first binary result in the inclusive 300–600-second window. The current raw event, weekday, hour and 15-minute bin, original `log(80)` evidence rule, target, class/support/quality gates and all learner conditions are unchanged. The one-minute intervals below are a counting partition for an upper bound, not a replacement current-state feature or cutoff rule.

## Why a naive all-raw simulation is unsafe

Consider eight simultaneous rows at 0 seconds, followed by rows at 60, 119 and 700 seconds. The original greedy cutoff clock consumes the initial event before it has a full history. If every row is binary, it then creates a cutoff at 60, skips 119 because only 59 seconds have elapsed, and receives the next result at 700: the 640-second delay is too late, so there are **zero arrivals**.

If the row at 60 is nonbinary, the next cutoff is instead 119. Its result at 700 is only 581 seconds later: there is **one arrival**. Consequently, simulating all raw rows as binary can **undercount** arrivals after the original binary filter. That value must not be used as a safe exclusion bound.

The implemented bound is one for this witness, covering the filtered arrival. It never predicts which rows are actually binary or removes unknown event types.

## Bound and soundness argument

1. **Necessary history condition.** An actual valid eight-binary-event history implies that the raw timestamp streak ending at its cutoff contains at least eight rows with no adjacent raw gap greater than 300 seconds. Removing rows cannot bridge such a gap or supply missing events. Mark every raw timestamp meeting this necessary condition as a potential cutoff, ignoring the actual greedy clock phase.
2. **Injective minute mapping.** Two actual qualified cutoffs cannot lie in the same half-open clock-minute interval, because their separation would be less than 60 seconds. Map each actual qualified cutoff to its minute. Thus actual arrivals are at most the number of minutes that could contain a valid cutoff/result pair.
3. **Exact result-window union within a minute.** For potential cutoffs t in a minute, the raw-result intervals `[t+300,t+600]` overlap: candidate times span less than 60 seconds while each result interval has width 300 seconds. Their union is exactly `[first+300,last+600]`. Store just these two endpoints. The minute has closed before the earliest possible result, so its last endpoint is already known.
4. **Safe certification.** At the first later raw time u>=first+300, count the minute if u<=last+600; otherwise expire it. Any actual binary result is a raw timestamp in its cutoff's union. A counted minute can be spurious, but a minute containing an actual qualified arrival cannot be lost. At complete training EOF, unresolved minutes have no result inside that permitted prefix and contribute no arrivals.

This proves the bound for all binary/nonbinary assignments under the fixed timing conditions. It is deliberately an upper bound, not an attainable maximum or a predictor of actual yield. No independence, stationarity, p-value or learned model is assumed.

After mature buckets are popped, pending first times lie within 300 seconds of the current time. This intersects at most **six** clock-minute intervals. Dense duplicate rows update the last endpoint, rather than growing a timestamp queue. A guard of 16 pending buckets and a ceiling of 5,000,000 training timestamps remain explicit. If a guard stops a scan, the result is **inconclusive**, with no usable upper bound; the implementation never treats the observed prefix as a complete exclusion certificate. Remaining rows are drained by date only for integrity.

## Counterexample and boundary validation

Before real timestamp parsing, the registered validator exhausts **8,192 binary masks** across three frozen timestamp traces and compares each against the original preflight extractor. Undercounts are zero. Their maximum actual arrivals versus safe bounds are 1/1, 2/4 and 2/3; the latter examples demonstrate that the bound may be loose.

Tests also cover inclusive 300/600-second result boundaries, a 300 versus 301-second history gap, midnight, duplicate bursts, the last potential cutoff's result-window endpoint, clock-regression rejection, poisoned sensor/message payloads, CSV/whitespace prefixes, date-only seals and inconclusive stops. A synthetic bound of exactly **70,212** is `cannot_exclude`; **70,211** is excluded. The target comparison is unchanged and strict.

## Existing training results

All windows start on 2011-06-15. End dates below are exclusive. Different lengths are fixed by the earlier studies, so these are not source rankings or retrospective changes to those studies' targets.

| Existing window | Recorded training dates | Training end | Stored actual arrivals | Timestamp upper bound | Slack |
| --- | ---: | --- | ---: | ---: | ---: |
| HH103 original training | 180 | 2011-12-13 | 18,676 | **25,334** | 6,658 |
| HH104 pooling fit | 80 | 2011-09-03 | 17,553 | **22,411** | 4,858 |
| HH105 input preflight | 180 | 2011-12-27 | 29,552 | **37,748** | 8,196 |

Every bound exceeds its stored actual count and remains below the current target 70,212. For these exact windows, no selection of binary rows or outcome classes can supply that many qualified arrivals under the unchanged timing rules. In particular, HH105's earlier planning failure could be established before interpreting its event fields. This does not invalidate older protocols with different objectives, change their learner negatives, or imply anything about independent predictive usefulness.

At runtime, the screen uses only two valid decisions: exclude input inspection when a complete upper bound is below target, or `cannot_exclude` otherwise. `cannot_exclude` is not a source/class/quality pass. The optional budget-capped upper bound is `min(upper,263295,789885//3)`; it does not alter the arrival criterion or make an incomplete scan safe.

## Work, preservation and reproduction

The three sources require **16,797,370 streamed rows**, including **15,237,888 date-only sealed rows**. Only **1,559,482 training timestamps** are parsed: 417,027 for HH103, 443,537 for HH104 and 698,918 for HH105. No sensor/message fields, new labels or predictions are interpreted. Each source reaches a peak of six pending buckets. All actual learner writes are zero.

The sealed row counts are 5,008,622 / 4,497,877 / 5,731,389 respectively. HH104's internal confirmation is also excluded in this audit because the bound ends at its earlier fitted-window boundary. Full member checksumming, decompression, UTF-8 line transport and date-prefix handling still cost work; the screen is not zero-I/O or an energy-efficiency result. This accounting follows the user-provided circuit-optimization memo's emphasis on charging the cost of deciding what work to avoid.

157 related tests pass. Fresh-process synthetic validation, result and retrospective daily bucket totals replay exactly. All 319 earlier JSON/gzip artifacts remain hash-identical. No new identity is acquired, model created, runtime feature enabled or source condition relaxed. Existing negative results and test seals remain in force.

Protocol, registration, consumed execution marker, synthetic-validation receipt, daily audit, result and replay receipt live under `data/processed/benchmark_fixtures/casas_timestamp_bound_*_v1.json`. Replay requires the three existing pinned compressed members:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_timestamp_bound.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_timestamp_bound.py
```

Daily records are retrospective by candidate cutoff date; a result shortly after midnight can certify the preceding day's bucket. They are not early predictions. Exclusive artifact creation prevents silently replacing consumed execution evidence.

🟠 [Next] Preregister a finite roster of unused households, a fixed order and aggregate acquisition/timestamp budgets before applying this screen to new identities. Inspect only timestamps under the same split rules, preserve sensor/message fields and independent periods, and stop without expanding the roster if no window survives. A surviving upper bound only permits a separately registered input preflight; all original source, positive-component and quality prerequisites remain required. No target relaxation or post-score identity substitution.

🔴 [Later] Input preflight, comparative learning, independent validation and runtime/factor integration require their own justified preregistrations and unmet usefulness prerequisites. No such work is performed by this method-validation study.

日本語: 日時だけで実到着数を下回らない1分区間上限を事前登録しました。生の全行をバイナリとみなす単純な方法は、除外による時計位相の変化で過小評価する反例があり、安全ではありません。全8,192割当と既存訓練3期間で過小評価0を確認し、上限はHH103が25,334、HH104が22,411、HH105が37,748件でした。どれも固定目標70,212件未満なので、これらの期間は入力値を読む前に除外できます。条件・既存の負の結果・封印を維持し、学習は0回です。

简体中文：预注册了仅用日期时间、不会低于实际到达数的分钟区间上界。将全部原始行视为二值事件的朴素模拟会因过滤改变采样相位而低估，因此不安全。穷举8,192种赋值及三个既有训练窗口，未发现低估；HH103／HH104／HH105上界分别为25,334／22,411／37,748，均低于固定目标70,212，可在读取事件值前排除这些窗口。条件、既有负结果与封存保持，新增学习为0。
