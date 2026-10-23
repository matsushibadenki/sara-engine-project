# R2 synthetic execution and instrumentation boundary — 2026-10-22

🟢 [Done] The preregistered separation contract passes on the **same 48 synthetic cases** used by [event-work accounting](R2_EVENT_WORK_ACCOUNTING.md). The new executor has no diagnostic counters, journal, retained trace, sampling callbacks, graph traversal or hashing. Its caller consumes bounded receipt batches and performs diagnostics outside mutating calls. All **1,184 operation boundaries plus 48 teardown boundaries** match the frozen reference's state. Receipt streams match the prior saved hashes, and removing boundary diagnostics preserves receipts and final state. Exact fresh-process replay succeeds.

This establishes a measurement boundary, not an SNN resource advantage or independent predictive usefulness. The event-driven scalar control receives the same shared history, notification schedule, expiry rules and ownership charges. Original negative results, Kitchen ON conditions, evidence thresholds, 70,212-arrival target and all seals remain unchanged. No acquisition, label/prediction scoring, learning writes or runtime promotion occurs.

## Registration and finite scope

The [selection](../data/processed/benchmark_fixtures/event_work_separation_selection_v1.json) was saved at **2026-10-22 11:09:42 UTC**, before implementation. It inherits the original contract and fixes three passes per case, at most 64 boundary calls, eight returned rows per arm/call, 256 diagnostic graph samples per case, 1 MiB sampled owned graphs and a 1 MiB result. Code/tests were [registered](../data/processed/benchmark_fixtures/event_work_separation_registration_v1.json) at **11:17:01 UTC** before the consumed finite execution. Existing results were already known; this is an engineering equivalence check, not a new blind learner experiment.

| Evidence | SHA-256 |
| --- | --- |
| Selection before implementation | `93b961a1dfabc468d172bc3d1f9c08df0566cd7d0ee51bac86ec6681407748cd` |
| Code registration before execution | `1c0de2cb2f042464a07fdf43948c58773e75eb8273c5b2fbeb1795703343734d` |
| Result, 212,416 bytes | `c52b951c2169d0ef852fd820f19354f1ee9ab9cc38a62c920ff15d88b73a8a2d` |

Only the existing route pair `[0,1]`, both members of first-order/middle-order/gap-order templates, four delivery scenarios and horizons **16 / 1,000,000,000** are used. TTL remains four logical units. Results at the deadline are valid; strictly earlier deadlines run before external work, and same-time cancellation, result, input and inclusive expiry retain their ordering. Missing results expire; late, duplicate, cancelled and foreign notifications are rejected. Stale timers are consumed rather than silently discarded. These notices contain identities, not outcome labels.

The selection preserves **363 prior artifacts**, including all five event-work receipts, and inherits the earlier code pins. Execution and result writes use exclusive creation in managed paths. A consumed execution cannot be overwritten. Verification hashes preserved artifacts without interpreting held-out payloads. Results are in the [result fixture](../data/processed/benchmark_fixtures/event_work_separation_result_v1.json), with an immutable [replay receipt](../data/processed/benchmark_fixtures/event_work_separation_replay_v1.json).

## Execution and diagnostic ownership

The research-only [executor](../src/sara_engine/evaluation/event_execution.py) uses the existing ordinal and sequence observers once per input, existing one-branch neuron calls only on arrival, and an independent scalar recurrence. It keeps the prior strict threshold, reset and arrival-indexed refractory semantics. Both arms use the same immutable shared packet. Returning the causal receipt batch is part of execution; retaining an audit history of batches is the caller's responsibility. No no-op instrumentation callback replaces the removed counters.

The [external harness](../src/sara_engine/evaluation/event_work_separation.py) runs the three registered paths:

1. Frozen instrumented reference components with their memory callback disabled. Their journal is reference-harness work, not a new deployment cost estimate.
2. The separated executor, followed by external state snapshots, logical occupancy checks and owned-graph samples after every operation and teardown.
3. The same separated executor with receipt consumption only. There are no per-operation state/memory probes; final and teardown snapshots are taken for comparison after execution.

State comparisons cover observer counts/records/history/latest packet, every neuron and branch field, scalar cells, pending receipt features, timer queues and logical clocks. The independent prior receipt oracle also checks causal prefix features and exactly one terminal result per receipt. A returned batch can contain several due timers; equality is checked at the public operation boundary, and each timer's causal deadline remains in the receipt. This does not claim observation at every internal instruction or allocation.

| Service volume, over 48 cases | Each arm |
| --- | ---: |
| Backend arrival steps / created receipts / popped timers | 224 each |
| Delivered / cancelled / expired receipts | 96 / 16 / 112 |
| Stale timer pops / rejected notifications | 112 / 100 |
| Result notices / cancellation requests | 196 / 16 |
| Logical advance calls | 748 |

Shared work retains 224 input/ordinal/sequence operations, 448 snapshots and 1,200 projected channels. Each arm is charged the whole shared service vector once; physical execution performs it once. These externally reconstructed **service volumes** reconcile to the saved result and reference components. They are not a new complete source-level operation vector: the separated wrapper's validation, returned buffers and interpreter work have changed. No speed improvement is inferred from fewer diagnostic calls or from reusing the old totals.

The long idle tail preserves receipt hashes, nonclock final state, operation counts and occupancy. It changes the clock and logical elapsed exposure. Sparse scheduling works for both arms; it does not establish a capability absent from ANN/non-spiking implementations.

## Memory boundary and unmeasured temporaries

The sampler includes shared observers/latest packet, backend/queue wrappers, cells, pending features, timer deques, executor/mapping shells and the **currently returned batch**. It recursively deduplicates shared references for the physical paired graph. Each arm's graph includes the entire shared dispatcher and observer ownership, conservatively including the paired dispatcher shells; these are not measured independently deployed single-arm processes. Temporary graph-root tuples and the graph walker's seen-set/child lists belong to the measurement harness.

| Maximum boundary sample, Python 3.10.20 | Bytes |
| --- | ---: |
| Shared roots and dispatcher shells | 3,147 |
| Shared plus neuron arm and output batch | 10,825 |
| Shared plus scalar arm and output batch | 8,683 |
| Physical paired graph with deduplication | 13,425 |

These values include a different scope and sampling boundary from the previous audit, so their difference is **not** an allocation regression or improvement measurement. Every sample is below the fixed 1 MiB ceiling. Peak queues hold five pending records and five timers per arm; peak returned batches contain four rows per arm. Controlled capacity tests exercise all eight allowed returned rows. Observed packet/channel/history bounds remain two routes, six channel pairs and six prefix items.

The executable allocation inventory identifies the following scope without pretending to measure a complete byte peak:

| Allocation/lifetime | Ownership and treatment |
| --- | --- |
| Returned batch dict/lists/rows and referenced packets | Execution output, included at the return boundary; must be consumed/released. Retaining a batch can keep packets alive after executor teardown. |
| Observer sorted records, frozen records and validation sets | Shared execution; at most two records/entries per named collection for these inputs. Objects freed inside the call are not visible to boundary samples. |
| Channel and prefix construction, gap-validation iteration | Shared execution; at most six pairs/items/visits per named collection/iteration. Construction overlap and iterator storage remain unmeasured. |
| Neuron active-branch temporary list | Neuron execution; at most one active branch index. Allocation capacity and arithmetic temporaries are unmeasured. |
| Scalar arithmetic and arriving-cell construction | Respective arm execution; one arriving cell per call. No claim of a total temporary-object or byte bound. |
| Expiry batch construction | Each arm; at most eight returned rows, including stale timers. Intermediate list-resize peaks are unmeasured. |
| Tape/schedule, reference/journal, accumulated receipts, snapshots, hashes/reports, sampler scratch | Harness ownership; excluded from deployment graph samples, never treated as free or zero-cost. |
| Frames, intermediate allocation overlap, allocator arenas, native/library/interpreter memory and RSS | Unknown; outside the sampled graph. |

There are **4,928 external graph probes / 370,656 visited objects**, including the separate shared-root probe. The frozen reference produces 14,328 journal rows in the harness. Neither quantity is hidden in the executor or treated as deployment savings. **CPU seconds, energy joules and total transient peak bytes remain null.** Structural item ceilings and boundary samples cannot prove the full-memory budget. The explicit neuron's sampled graph remains larger than the scalar's; neither a general runtime ranking nor an energy benefit follows.

## Verification and next step

**381 distinct related tests pass**: 25 separation tests, 115 retained accounting/observer/decision/path tests and 241 retained data-source/delayed-readout tests. They check poisoned diagnostic dependencies, sampler stack separation/purity, actual shared-observer dispatch, internal branch-state corruption despite unchanged returned values, dropped receipts, incorrect saved hashes/charges, deadline/cancellation ties, bounded output and input rejection, returned-packet lifetime, teardown, preservation hashes and refusal to repeat consumed execution. Exact result replay uses a fresh process under registered Python 3.10.20.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/event_work_separation.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_event_work_separation.py
```

🟢 [Done] The subsequent [isolated allocation diagnostic](R2_EVENT_ALLOCATION_DIAGNOSTIC.md) is implemented, but its 144-worker finite run failed the fixed 8 MiB result-storage gate. No raw values or replay are retained; preserve this failure without relaxing the limit or rerunning. 411 tests pass with one expected pre-execution skip; 368 earlier artifacts/code pins and all seals remain intact.

🟢 [Done] The subsequent [report-representation preflight](R2_ALLOCATION_REPORT_PREFLIGHT.md) proves a conditional 7,302,888-byte ceiling within fixed 8 MiB, with lossless ownership/unknown-site encoding and pre-launch saved-result validation. 41 tests and exact static replay pass, preserving 372 prior artifacts, failure and seals. No allocation measurement or diagnostic worker is run.

🟠 [Next] Separately preregister a fresh allocation-diagnostic v2 on the same fixed inputs with the bounded report encoder and saved-result gate. Retain a complete result before replication, fail explicitly on capacity overflow, and preserve failed v1. Equal shared/output ownership for neuron and event-driven scalar controls; no acquisition, learner scoring, threshold change or R3 optimization.

⭕️ [Pending] Independent usefulness, full deployment resource/physical-energy evidence, new acquisition and production promotion retain their original prerequisites. The circuit-efficiency idea remains useful as a design hypothesis: activate only necessary circuits while charging shared observation and delayed-result work equally to sufficient scalar controls. This audit supplies a cleaner measurement boundary, not the missing SNN-specific evidence.

日本語: 同じ合成48ケースで計測分離を実装前に登録し、1,184回の操作境界と48回の終了処理後の状態、既存配送ハッシュ、診断なし実行との一致を検証しました。返却バッファと制御オブジェクトをメモリ計上に含め、関数内の一時割当・フレーム・RSSは未測定として残しました。381テストと別プロセス完全再現が成功し、既存363証跡、条件、負の結果、封印を保持しています。学習・採点は0で、SNN固有の速度・電力優位は未確認です。続く隔離割当診断は固定8 MiBの結果保存上限で不合格となり、生値は未保存です。失敗を保持し、続く形式事前検証は上限7,302,888バイト・41テスト・完全再現で成功しました。割当計測は0で、次の診断には別の事前登録が必要です。

简体中文：在实现前注册相同48个合成案例的测量分离契约，验证1,184个操作边界与48个清理边界的状态、既有回执哈希及无诊断执行结果一致。内存范围包含返回缓冲区和控制对象；函数内临时分配、栈帧和RSS仍为未知。381项测试及新进程精确重放通过，保留363项既有证据、原条件、负结果和封存。学习与评分为0，尚未证明SNN独有的速度或能耗优势。后续隔离分配诊断因结果超过固定8 MiB上限而失败，原始数值未保存。保留失败；后续格式预检以7,302,888字节上界、41项测试和精确重放通过，新增分配测量为0；下一次诊断需要另行预注册。
