# Finite unused-household timestamp screen — 2026-10-18

🟢 [Done] The finite roster **HH106 → HH107 → HH108** has no survivor: complete training upper bounds are **37,857 / 59,892 / 24,569**, all below the unchanged **70,212** target. Stop this roster without extending it or opening event values. This is an arrival-feasibility exclusion, not a new learner negative or a claim about every CASAS household.

Selection, order, aggregate budgets and screening code were registered before acquiring these members or inspecting their timestamps. HH101–HH105 were already consumed. The three smallest following HH identities were chosen from the previously pinned ZIP directory; duration, arrivals, event values and outcomes were not known when fixing the roster. Distinct file identities do not establish independent predictive usefulness.

## Preregistration and budgets

Selection SHA-256: `f03d5781a90437c8f8fc9d1badb298562d287f33bf6d350d8d241df2e2046e23`. Code registration SHA-256: `11009f32d2fe45f4fae04fb59823a827d70aff08708d99ada21152be1cf9b65f`, registered at `2026-10-18T02:47:23Z` before acquisition. It pins 326 earlier JSON/gzip artifacts plus this selection (327 total), the implementation and the original extractor/bound code. The original negative results cannot be overwritten or reclassified by this screen.

The source is the official [CASAS Zenodo v1 catalog](https://zenodo.org/records/15708568), with four raw fields: date, time, sensor identifier and message. Only the first two are interpreted. The pinned archive length is 2,723,059,864 bytes. Its published full-ZIP MD5 is not claimed as verified: only the selected ranges are acquired, checked against the original directory, and verified with compressed SHA-256 plus expanded size/CRC/SHA-256.

| Frozen capacity | Aggregate | Per household |
| --- | ---: | ---: |
| Household identities | 3 | No substitution |
| Requested HTTP body bytes | 201,326,592 (192 MiB) | 67,109,888 (64 MiB + 1 KiB) |
| Expanded bytes reserved across metadata and arrival passes | 2,147,483,648 (2 GiB) | 536,870,912 per pass |
| Metadata timestamp parses | 36,000,000 | 12,000,000 |
| Training timestamp parses | 15,000,000 | 5,000,000 |
| Pending minute buckets | Sequential processing | 16 |

Acquisition permits one 1,024-byte ZIP local-header request and one exact compressed-member request per household, in roster order, with no retries, replacement or full-archive download. Requested body bytes are reserved before each request and remain spent if the request fails. A request has a 1,800-second timeout and exact body-size ceiling; HTTP range and archive length must match. HTTP/TLS/header overhead is outside the body-byte accounting; at most six range requests are permitted. Raw bytes, request journals and private response headers stay in the ignored managed raw directory.

The fixed directory entries imply 101,722,176 requested body bytes including the three local-header ranges and 1,069,349,162 expanded bytes for two complete passes. All work runs sequentially on the CPU without matrices, GPU use or model learning. There is no energy-efficiency claim from this screening study.

## Unchanged conditions and seals

The metadata pass uses the original chronological policy: census dates/times, exclude both dates implicated in every clock regression, require at least 180 valid recorded dates, take `min(180,2*N//3)` training dates and `min(60,N//6)` development dates, then seal the remaining test dates. This pass reads timestamps across the source **only to establish the original splits and clock exclusions**, as in earlier source audits. It does not count payload fields or interpret sensor/message content. Missing/malformed timestamps or insufficient duration cannot produce an exclusion certificate.

The arrival pass uses only the fixed training dates. Independent and excluded rows are skipped by date before time parsing. Complete-stream checksumming/decompression and UTF-8 transport still traverse sealed payload bytes; this is not semantic access to their contents or predictive evaluation. No independent event values, class labels, delayed outcomes, predictions or learning are opened. The all-source metadata census and the training-only arrival pass are reported separately.

The [existing safe bound](R2_CASAS_TIMESTAMP_UPPER_BOUND.md) remains unchanged: necessary eight-event raw history, <=300-second adjacent gaps, at most one actual cutoff per half-open minute under >=60-second spacing, and overlapping inclusive 300–600-second result windows. The original current event/weekday/hour/15-minute context, evidence threshold `log(80)`, support and positive-component requirements, unknown-event policy, quality controls and actual-update comparison rules are unchanged. Every raw row is a possible binary event, so unknown event types are included without being interpreted. No binary filtering, event vocabulary or current-state recurrence is measured here; those original input/quality conditions remain unassessed rather than waived.

Before real timestamp screening, the original 8,192 binary-mask checks and naive all-raw greedy counterexample are rerun. Every real exclusion requires complete source integrity and a complete training scan with upper bound **strictly below 70,212**. Equality or a larger upper is `cannot_exclude`, never an input/source/class/quality pass. A source failure or exhausted budget is `inconclusive`, never a below-target result. Process all three fixed identities; do not add identities, change windows or lower the target after observing results.

## Results and continuation

All three acquisitions and complete member-integrity checks pass. There are no retries, inconclusive sources or capacity stops. All windows contain the original first 180 valid recorded training dates and 60 development dates; no calendar window is moved after observing arrivals.

| Identity | Valid recorded dates | Training start | Training end (exclusive) | Training timestamps | Upper bound | Date-only independent rows in arrival pass |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| HH106 | 1,123 | 2011-06-15 | 2011-12-12 | 604,178 | **37,857** | 3,540,266 |
| HH107 | 371 | 2012-07-20 | 2013-01-16 | 1,074,195 | **59,892** | 1,305,650 |
| HH108 | 1,840 | 2011-06-15 | 2011-12-12 | 575,399 | **24,569** | 6,355,356 |

Original clock exclusions are `2014-04-11` for HH106, none for HH107 and `2015-11-01` for HH108. Their test starts are 2012-02-10 / 2013-03-17 / 2012-02-10. The full-source metadata pass establishes these dates without interpreting event fields. Development and test prediction information remains sealed. Current-state recurrence, history contrasts, event vocabulary, both classes and predictive quality are **unassessed**, not failed or waived.

All bounds are below the target even though every raw row, including any unknown event, is admitted as potentially binary. No actual binary-arrival count is computed. HH107 and HH108 end with five and one unresolved minute buckets: complete training EOF provides no eligible result for them inside training. Independent periods are not borrowed to resolve these buckets.

| Charged work per execution | Actual / reserved | Registered aggregate ceiling |
| --- | ---: | ---: |
| Requested and received HTTP body bytes | 101,722,176 | 201,326,592 |
| Expanded bytes across both complete passes | 1,069,349,162 | 2,147,483,648 |
| All-source metadata timestamps | 13,455,044 | 36,000,000 |
| Training arrival timestamps | 2,253,772 | 15,000,000 |
| Peak pending buckets per household | 6 / 6 / 6 | 16 per household |
| Actual learning writes | 0 | No learner authorized |

The arrival passes stream 13,455,044 rows, including **11,201,272** date-only independent rows. The metadata and arrival passes together transport 26,910,088 rows; replay repeats the same local work without network acquisition. The work table is per execution, not a claim that verification is free. Compressed hashing, integrity checks, line transport and request overhead also cost work.

All 177 related tests pass, including the previously pending stored-result test. They cover original split/clock parity, payload poisoning, shared/per-household ceilings, exact-EOF versus truncated scans, fixed-roster mutation refusal, ZIP/range checks, original boundary/counterexample checks and hash preservation. The original 8,192 binary masks have zero undercounts. Fresh-process validation, metadata, expanded integrity, daily counters and final result replay exactly; all 326 earlier evidence artifacts and the selection remain hash-identical. Existing learner negatives and all original seals remain in force.

The consumed run, source acquisition receipts and exact fresh-process replay are stored in `data/processed/benchmark_fixtures/casas_roster_screen_*_v1.json`. Raw source members and private headers are excluded from Git. No checkpoint, comparative learner or runtime feature is created.

The one-time fresh-process replay from the locally pinned compressed members is:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/eval/casas_roster_screen.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_roster_screen.py tests/test_casas_timestamp_bound.py
```

Registration, acquisition, execution and the replay receipt use exclusive artifact creation and are not repeatable overwrite operations. The replay command writes its receipt once; after that, the test command audits the stored result and pins without replacing evidence. Replay performs no network acquisition.

The registered decision is `stop_roster_no_survivor`. No identity is added or replaced; no input-value preflight or class/quality test is authorized by these results. The high arrival demand remains an unmet prerequisite, not evidence that the local temporal readout works or cannot work.

🟠 [Next] Review acquisition feasibility for a different source strategy using public sampling-density/duration metadata and the unchanged target/current/evidence conditions before proposing another finite acquisition. Require an external rationale for sufficient arrivals; do not continue a sequence of CASAS households automatically or reuse these independent periods to rescue the roster.

🔴 [Later] Only an independently justified new preregistration can authorize acquisition or a surviving source's training-input preflight. All original recurrence, history-contrast, class/support, state/write-budget and quality prerequisites still precede any matched-update learner and independent validation.

日本語: 未使用HH106・HH107・HH108の順序・総予算・コードを取得前に固定し、日時だけで上限37,857／59,892／24,569件を確認しました。全候補が固定目標70,212件未満なので、候補追加・期間変更・入力値の検査をせず終了します。取得約102MB、訓練日時約225万件は予算内で、別プロセス再現と関連177テストが成功しました。センサー・メッセージ、独立期間の予測情報、既存の負の結果を保持し、学習は0回です。次は公開メタデータに基づく別の取得方針の実現性検討に限ります。

简体中文：采集前固定未使用的HH106、HH107、HH108、顺序、总预算和代码，仅用时间戳得到上界37,857／59,892／24,569，均低于固定目标70,212。终止该名单，不扩充候选、不改变窗口或读取事件值。采集约102MB、训练时间戳约225万次均在预算内；新进程精确重放和177项相关测试通过。保留传感器、消息、独立期间预测信息和既有负结果，新增学习为0。下一步仅根据公开元数据研究另一采集策略的可行性。
