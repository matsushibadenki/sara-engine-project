# Input-only event-work accounting — 2026-10-22

🟢 [Done] Preregistered and implemented a finite accounting contract for a shared temporal observer, delayed result notifications, idle intervals and receipt expiry. **All 48 cases pass**, with exact fresh-process replay and **356 related tests**. The existing neuron and event-driven scalar implementations produce identical diagnostic state/receipt traces. This establishes accounting feasibility, not predictive usefulness, CPU speed or energy superiority.

## Preregistration and fixed input

Selection SHA-256 `260b8095878a7c189f3e8fc09fa8f4852610be3fdd0d980c010dc1b7f1257c01` was saved at `2026-10-22T08:42:50Z`, **before implementation**. Code/tests and the existing input/observer/neuron sources were frozen at `08:55:33Z` in registration `cb78f2c8…47a2`, before the one-shot finite execution. The result hash is `f0c0ee7a425924860019a141c5c82e2ba52324ae2f6b62b5110e1a6efd1bcc68`.

Inputs are the six existing `first_order`, `middle_order` and `gap_order` templates from `temporal_capability_boundary_v1.json` (`7574d2d1…c789`), using the fixed route pair **0, 1**. No route/gap value is changed. Coverage is this pair and these six templates, not all 276 original route pairs or every possible neuron input. Four lifecycle schedules and two final horizons, **16 and 1,000,000,000 logical time units**, give 6 × 4 × 2 = **48 cases**, each with both arms. The horizon extension adds no native observation.

| Registered limit | Ceiling | Observed |
| --- | ---: | ---: |
| Cases | 48 | 48 |
| Native events / routes per case | 6 / 2 | 6 / 2 maximum |
| Pending receipts / retained timers per arm | 8 / 8 | 5 / 5 maximum |
| Accounting journal entries per case | 4,096 | 685 maximum |
| Sampled owned object graph per scope | 1,048,576 bytes | 12,219 bytes maximum physical graph |
| Serialized result | 1,048,576 bytes | 519,456 bytes |
| New source reads / learner writes / prediction scores | 0 / 0 / 0 | 0 / 0 / 0 |

All **358 prior artifacts** (357 JSON/gzip files and one metadata request journal) and inherited registered code remain hash-identical. The original Kitchen ON task, current-state/evidence/support/quality conditions, 70,212 planning target, source exclusions, negative results and independent-period seals are unchanged. Raw source rows, learned weights, held-out example JSONL and earlier prediction rows are not read. Hash checks are preservation checks, not new evaluation.

## Execution contract

The shared stage decodes each native route/gap once, updates the existing ordinal and bounded sequence observers, and creates an immutable causal prefix packet. Each arm receives the same packet. A diagnostic probe stores that packet and the backend state; it does not compute a learned prediction or use a target label. Future observations and result schedules are absent from the packet, checked against an independent prefix oracle.

The neuron arm calls the existing one-branch `Neuron.add_input_to_branch` and `Neuron.step`, with registered positive current **0.6**, only for the arriving route. The scalar arm independently implements the same dendritic transform, strict firing threshold, reset and two-step refractory rule. Both retain state between arrivals. These are the existing **arrival-indexed** semantics: an elapsed gap does not cause repeated neuron steps or introduce continuous-clock membrane decay. Gap information remains available in the shared history. No new neuron timing kernel or runtime optimization is introduced.

Each probe creates a bounded receipt and an expiry timer at cutoff + **4**. Result notifications carry only an opaque receipt identity, with no outcome value. At a timestamp, cancellations run before results, then inputs, then inclusive expiry. A result exactly at its deadline is valid. Timers before the next external timestamp execute at their own causal deadlines. The final horizon also drains timers when no further input arrives.

Resolved receipts leave their timer in the FIFO until its deadline. These stale timer pops, lookups and callbacks are charged. This keeps cancellation/cleanup costs visible. Foreign, duplicate, cancelled and expired result notifications are rejected. Every created receipt has exactly one terminal event: delivery, cancellation or expiry.

| Fixed schedule | Receipts per arm across both horizons | Delivered | Cancelled | Expired | Rejected notifications | Stale timer pops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| On time: result at +4 | 56 | 56 | 0 | 0 | 0 | 56 |
| Late: result at +5 | 56 | 0 | 0 | 56 | 56 | 0 |
| Missing: no result | 56 | 0 | 0 | 56 | 0 | 0 |
| Mixed: delivery, cancellation, duplicate and foreign ID | 56 | 40 | 16 | 0 | 44 | 56 |
| Total | **224** | **96** | **16** | **112** | **100** | **112** |

Both arms execute **224 backend steps, 96 spikes, 224 timer pushes and 224 timer pops**, including 112 stale pops, over the 48 cases. They handle 196 result notifications and 16 cancellation requests each. These are lifecycle/operation counts, not learned-update counts. **Actual changed learning writes are zero** because this input-only fixture has no learning interface; zero is not presented as a positive matched-update learning experiment.

## Fair accounting and limits

Each entry has an owner (`shared`, `neuron`, `scalar`, `harness`), stage, metric and unit. Stage ownership is checked; independent conservation rules detect omitted observer work, lost/double terminal events and unpopped timers. The journal is reconciled with counters. Costs remain a vector of calls, records, fields, timer operations, channels, logical elapsed time and other typed units; no arbitrary scalar “total cost” is formed.

For every metric `m`, each arm is charged `shared[m] + arm[m]`. Physical paired execution contains `shared[m] + neuron[m] + scalar[m]`, with the harness reported separately. The shared observer actually executes **224 updates and 448 snapshot calls**, projecting 1,200 channels. Each arm pays the complete shared vector once. Physically executing it once is not claimed as a per-arm deployment saving.

Counters are **inclusive source-level operations**: for example, an observer call owns its internal validation and Python/library execution, with snapshot sort inputs and copied prefix items additionally reported as volumes. Backend constructors, active dispatch, branch transform/refractory work, receipt insertion/deletion, timer-head checks, expiry, wait advances and teardown have owners. These counters cover the registered stages; they do not enumerate CPU instructions, every interpreter assignment or allocator action.

The scheduler visits external actions and due timers. It records logical elapsed units separately from advance calls and timer dispatches. Extending the idle tail from 16 to one billion changes **only elapsed units** in the operation vectors and leaves the semantic trace unchanged. Both arms benefit equally from event-driven scheduling. This is not evidence of an ANN-absent property or measured idle-energy saving.

Tape preparation/sorting, diagnostic trace retention, memory sampling, oracle/reconciliation checks and hashing belong to the harness. They are not quietly assigned zero deployment cost or used to claim a fast backend. The result reports 23,524 journal entries and 2,796 object-graph sampling calls across the cases. Inclusive finalization includes journal serialization/hash and charge checks; generated reports and file I/O are outside the measured deployment stage scope.

Sampled owned graphs include the shared observer and packet plus each arm's backend, pending receipts and timers, recursively accounting for dictionaries, objects, slots and deques. Shared references are deduplicated physically and charged to each arm. On registered **Python 3.10.20**, maximum samples are **9,787 bytes neuron**, **7,645 bytes scalar**, and **12,219 bytes physical paired graph**. These are sampled owned-data graphs, not full process memory or transient allocation peaks. Wrapper/control frames, diagnostic tape/journal/traces, measurement scratch, interpreter/library memory and RSS are outside that number. CPU seconds, energy joules and total transient peak bytes remain **null/unknown**. Replay requires the registered Python patch version; these object-size values are runtime-specific.

The evidence does not favor an SNN cost advantage: the registered semantic work counts match, and the explicit neuron's sampled owned graph is larger. Neither fact establishes a general runtime ranking. The user's circuit-efficiency direction is tested here as “activate only the arriving route and dispatch only due lifecycle work,” while giving the scalar control the same opportunity. A useful-system or physical-energy claim still needs its own independent evidence and the original R2 gates.

## Verification and next step

**356 related tests pass**: 115 new accounting/prior decision/observer/boundary/path tests plus 241 retained source/delayed-readout tests. Counterexamples check missing/doubled shared charges, omitted costs, wrong scalar state, timer callbacks at causal deadlines, deadline-inclusive delivery, cancellation ties, duplicates, future-input leakage, monotone time, bounded journal/state, memory deduplication and refusal to repeat consumed execution. Small synthetic unit tests do not rerun earlier learning experiments. Fresh-process result replay is exact and its receipt is immutable.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/event_work.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_event_work.py
```

🟢 [Done] The subsequent [instrumentation-separation contract](R2_EVENT_WORK_SEPARATION.md) now passes the same 48 cases: 1,232 state comparisons including teardown, prior receipt hashes, receipt-only execution, 381 related tests and exact replay. All 363 earlier artifacts/code pins, negatives and seals are preserved. Returned buffers/control are counted; intra-call peaks remain unknown.

🟢 [Done] The subsequent [isolated allocation diagnostic](R2_EVENT_ALLOCATION_DIAGNOSTIC.md) is implemented, but its 144-worker finite run failed the fixed 8 MiB result-storage gate. No raw values or replay are retained; preserve this failure without relaxing the limit or rerunning. 411 tests pass with one expected pre-execution skip; 368 earlier artifacts/code pins and all seals remain intact.

🟢 [Done] The subsequent [report-representation preflight](R2_ALLOCATION_REPORT_PREFLIGHT.md) proves a conditional 7,302,888-byte ceiling within fixed 8 MiB, with lossless ownership/unknown-site encoding and pre-launch saved-result validation. 41 tests and exact static replay pass, preserving 372 prior artifacts, failure and seals. No allocation measurement or diagnostic worker is run.

🟢 [Done] A separately registered [allocation diagnostic v2](R2_EVENT_ALLOCATION_V2.md) now retains primary/replica reports of 608,412 bytes each, with 3,840 matching windows and 51 tests. Preserve 378 earlier artifacts/code pins, failed v1 and seals. No neuron memory saving or CPU/energy advantage is established.

🟠 [Next] Preregister a finite retained-evidence decision gate combining original predictive/mechanism results with completed accounting/separation/allocation receipts. Identify a decision-changing SNN mechanism and independently qualifiable task prerequisite, or hold further resource work. No new measurement, scoring, search/acquisition, automatic v3, threshold change or R3 optimization.

⭕️ [Pending] Independent usefulness, full deployment CPU/transient memory and physical energy evidence remain unestablished. New acquisition, original test opening and production promotion retain their original prerequisites.

日本語: 既存6合成列と4種の結果配送、2種の終了時刻を実装前に登録し、48ケースで共有観測器・遅延結果・待機・期限切れの計上契約が成立しました。各方式224回のバックエンド更新、224件のタイマー消費を行い、112件の失効と112件の解決済みタイマー処理も計上しました。10億単位まで待機を延ばしても両方式の処理回数と状態列は変わりません。356テストと別プロセス再現が成功し、358件の既存証跡・条件・負の結果・封印を保持しました。学習・採点は0で、速度・電力優位は未確認です。続く計測分離は48ケース・381テストで成立しましたが、隔離割当診断は固定8 MiBの結果保存上限で不合格です。失敗を保持し、続く形式事前検証は上限7,302,888バイト・41テスト・完全再現で成功しました。続くv2も別登録し、各608,412バイトの結果保存と3,840窓の一致、51テストを確認しました。SNNのメモリ優位は示されず、次は既存証跡だけで機構検証へ進む根拠を有限に判断します。

简体中文：在实现前固定已有6条合成输入、4类结果配送和2个结束时刻，48个案例均通过共享观测器、延迟结果、空闲和过期处理的核算契约。每个实现执行224次后端更新并消耗224个定时器，包含112次过期和112次已完成回执的残留定时器处理。将空闲延长至十亿单位不会改变两种实现的操作次数或状态轨迹。356项测试与新进程精确重放通过，保留358项既有证据、条件、负结果和封存。学习与评分为0，尚未证明速度或能耗优势。后续测量分离通过48个案例和381项测试，但隔离分配诊断因结果超过固定8 MiB上限而失败。保留失败；后续格式预检以7,302,888字节上界、41项测试和精确重放通过，后续v2也已单独注册，各保存608,412字节，3,840个窗口一致且51项测试通过。未显示SNN内存优势；下一步仅用已有证据有限判断机制验证的依据。
