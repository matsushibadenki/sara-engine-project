# R2 isolated Python allocation diagnostic — 2026-10-22

🟢 [Done] The isolated diagnostic, standalone scalar/neuron adapters and tests are implemented. **The finite audit failed its registered result-storage gate.** All 144 worker processes reached completion, but the serialized report exceeded the fixed **8 MiB** limit before the result file was created. The execution remains consumed. No limit increase, rerun or replication was performed.

The [failure receipt](../data/processed/benchmark_fixtures/event_allocation_failure_v1.json) preserves this negative. Per-worker allocation values were held in the exited parent process and were not saved. Their exact sizes, distributions and replay equality are therefore **unknown**; progress messages are not a substitute for a retained measurement artifact. This result does not establish an SNN memory, speed or energy advantage.

## Registration and failure evidence

The [selection](../data/processed/benchmark_fixtures/event_allocation_selection_v1.json) was saved at **2026-10-22 11:42:17 UTC**, before implementation. It fixes the same six existing synthetic templates, route pair `[0,1]`, four delivery scenarios and horizons 16 / 1,000,000,000: **48 cases**. There is no new task, label, training input, source search or acquisition.

The three modes are the unchanged paired executor and two standalone diagnostic adapters, each of which owns and executes the complete shared observer. One fresh process per case/mode gives **144 workers**, with one process at a time, 30 seconds per worker, 600 seconds per execution, at most 66 recorded calls per worker, a 1 MiB newly traced call-peak ceiling, and an 8 MiB serialized-result ceiling. One replication was permitted only after a valid retained execution; the failure stops the audit before replication.

| Artifact | SHA-256 |
| --- | --- |
| Selection before implementation | `22a9bbfdc6feef6e2e87dd3a169cbf6f7730b02569401741ccedbd02ff6f5fa6` |
| Code registration, 11:50:09 UTC | `8b3d3c27e5c1a2fb2d3c203ea994c80db37bc58495db902e6c08fe2842856d8d` |
| Consumed execution, 11:50:12 UTC | `091fe09b3f8fdbd1680f9002380764e0198790d26aa3376a1eefa6c7037c3e3a` |
| Failure receipt, recorded after the error | `d839b8d73ebcd81a0af6a5ef33a33ad880a6a8079e1dcfedf3084fdcbaac901e` |

The command printed progress through `144/144`, then exited with `ValueError: Artifact byte budget exceeded` in `save('result', result)`. The registered serializer uses indented JSON and retains per-call allocation-site details. It checks the serialized byte count before exclusive file creation. The exact rejected size was not printed or saved; only **greater than 8,388,608 bytes** is established. There is no result, replication, replay receipt or replay-execution marker. Do not invoke the frozen execute/replay/worker entry points to recover missing values: that would restart measurements after the stop rule. The retained failure is checked without tracing or launching workers.

The failure receipt and four post-failure verification tests were added after the outcome and are explicitly not preregistered measurements. Registered implementation/test files remain unchanged. All **368 earlier artifacts and inherited code pins** verify; current-state/evidence thresholds, original Kitchen ON semantics, prior negative results and all independent-period seals remain intact. Learning writes, prediction scoring and new raw/learned-value inspection are zero.

## Implemented measurement boundary

The [allocation harness](../src/sara_engine/evaluation/event_allocation.py) measures constructor, scheduled operation and drained teardown calls. It leaves the [previous executor](../src/sara_engine/evaluation/event_execution.py) unchanged. The [standalone adapter](../src/sara_engine/evaluation/event_allocation_execution.py) uses the same `SharedInput`, `Backend` and `ReceiptQueue` components, bounded validation and ordering for either arm. Both standalone modes pay for the entire observer; shared work is not omitted from the scalar control or credited only to the neuron.

Each worker first executes the same untraced paired reference, checking its state/receipt hashes against the retained separation result. After every diagnosed call, shared state, the selected backend/queue state and causal receipts are compared to that reference outside the measurement window. Paired and standalone dispatcher layouts differ; their peaks cannot be subtracted to infer an arm's incremental cost. The original deadline-inclusive results, cancellations, duplicate/foreign rejection, stale timers, missing-result expiry and arrival-indexed neuron semantics remain unchanged.

| Measurement or ownership | Exact scope and limitation |
| --- | --- |
| Newly traced current / peak bytes | Start `tracemalloc` afresh at depth 1 immediately before one callable invocation; query current/peak immediately after return. Includes call/probe envelope allocations. Old live state predates tracing and is excluded. This is not total execution memory. |
| Peak minus current at return | Raw excess of the window peak above its return reading. Can reveal allocations freed before return, but is not the sum of all temporary allocations, exclusive executor bytes or freed-site attribution. |
| Live allocation-site snapshot | Taken after the peak query, then classified as execution, measurement or other. It includes post-query bookkeeping and cannot recover allocation sites of already freed objects. Its sum must conserve snapshot bytes, not equal the earlier current reading. |
| Tracer metadata | Queried after snapshot creation and reported separately. It is measurement overhead, not zero or a deployment saving. |
| Executor / output ownership | After tracing stops, traverse object identities and shallow sizes. Partition the union into executor-only, output-only and shared-reachable bytes. A packet reachable from both sides is counted once. Initializer return belongs to executor state; `close` has no receipt output. |
| Returned buffers | Receipt mappings/lists/rows remain execution outputs until consumed. Output-only bytes describe graph ownership at the boundary, not an independently measured allocation peak. |
| Harness | Reference prepass, retained oracle states/receipts, schedules, graph and snapshot analysis, hashes, reporting and subprocess management lie outside the target-call window. The small invocation/query envelope remains inside raw tracing figures and is not silently subtracted. |
| Unknown | Full intra-call live ownership, freed source sites, allocator arenas, untraced native/library memory, RSS, uninstrumented CPU and physical energy. Do not add traced bytes to graph bytes or sum call peaks into a process peak. |

The regime fixes CPython **3.10.20**, child `PYTHONHASHSEED=0`, GC collection before each recorded call, cyclic GC disabled only inside the tracing envelope, and cleanup/restoration even on exceptions. It rejects an already active allocation/profiling tracer. This diagnostic regime perturbs allocation behavior and is not a normal-runtime performance benchmark.

The contract permits raw traced values to differ on the single registered replication while requiring exact semantic state and ownership partitions. Both raw runs would be retained and differences reported without a tolerance-based advantage gate. **That replication did not occur**, so no byte stability or exact-replay claim is made here.

## Verification retained despite the storage failure

**411 tests passed, with one expected pre-execution skip**: 26 new diagnostic unit tests, 140 retained separation/accounting/observer/path tests, 241 retained data-source/delayed-readout tests, and four post-failure receipt checks. The skipped test requires a successfully saved finite result; it is not counted as passed. No allocation tests or experiments were rerun after the failure; the four additional checks only validate retained files and consumed-execution refusal.

The unit tests verify:

- A known 64 KiB internally allocated and freed buffer produces at least 64 KiB raw peak excess; retaining a new buffer leaves it in the return reading, while returning a pre-existing buffer does not count it as a new allocation. The implemented witness return structure includes the empty-call reading without subtraction; no finite-run witness values were saved.
- Standalone neuron and scalar modes each execute the shared observer on all six input arrivals, and receipt/state comparisons cover all four delivery scenarios in controlled unit cases.
- Identity partitioning avoids double charging shared packets, graph analysis is rejected during tracing, active tracers are rejected, and exceptions restore tracer/GC state.
- Wrong prior hashes and corrupted adapter state fail; the finite execution cannot be repeated after its result-save failure.

These are implementation checks, not retained quantitative findings from the 48-case audit. The previous accounting and separation results retain their own successful replay status; this audit's storage failure does not overwrite them.

```sh
# Read-only post-failure verification; does not repeat allocation measurements.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_event_allocation_failure.py
```

🟠 [Next] Preregister a **bounded report-representation preflight** using the frozen schema, source inventory and existing synthetic fixtures only. Prove a serialized-size ceiling within the unchanged 8 MiB budget before any further allocation run, preserve ownership/unknown-site accounting, and fail closed before worker launch when a valid execution result is absent. No fresh allocation measurements or retry of this consumed run. Any later experiment needs a separate preregistration retaining this failure.

⭕️ [Pending] Reproducible finite allocation statistics, complete deployment resource evidence and independent SNN usefulness remain unestablished. The circuit-efficiency idea remains a hypothesis to test against sufficient event-driven scalar controls. The failure justifies fixing report feasibility first, not loosening prediction/support/source conditions, adding CASAS candidates, opening seals, promoting runtime or starting R3 optimization.

日本語: 同じ48ケース・144隔離プロセスの診断を実装前に登録しました。実行部・返却バッファ・計測処理を分ける実装と対照テストは成立しましたが、結果保存時に固定8 MiB上限を超え、監査全体は不合格です。生の割当結果は保存されず、定量値や再現成功は主張しません。411テスト成功・1件の実行前スキップを記録し、368既存証跡、条件、負の結果、封印を保持しました。上限変更・再実行は行わず、次は新たな測定をせずに報告形式のサイズ上限を事前検証します。

简体中文：在实现前注册相同48个案例、144个隔离进程的诊断。执行部、返回缓冲区和测量工具的边界及对照测试已实现，但保存结果时超过固定8 MiB上限，整个审计判定失败。原始分配数据未保存，不声称定量结果或重放成功。记录411项测试通过、1项执行前跳过，并保留368项既有证据、原条件、负结果和封存。没有提高上限或重新运行；下一步在不新增测量的情况下，预注册并验证报告格式的大小上界。
