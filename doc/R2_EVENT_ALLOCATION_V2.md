# R2 retained isolated allocation diagnostic v2 — 2026-10-23

🟢 [Done] The freshly preregistered v2 diagnostic retains complete reports for the same **48 cases × three modes**, plus one independent replication. Each report is **608,412 bytes**, below the fixed 8 MiB limit. All **3,840 measured windows** match between runs, including raw allocation details; semantic state and graph ownership also match exactly. **51 related tests pass**. The failed [v1 run](R2_EVENT_ALLOCATION_DIAGNOSTIC.md) remains consumed and unchanged.

The observed standalone neuron's maximum newly traced call peak and boundary graph are larger than the scalar control's. This establishes reproducible diagnostic observations under the fixed Python/probe regime, without establishing SNN usefulness, speed or energy superiority. No learner scoring, actual learning writes, source acquisition, threshold changes, seal opening or runtime promotion occurs.

## Registration, retention and replication

The [selection](../data/processed/benchmark_fixtures/event_allocation_v2_selection_v1.json) was saved **2026-10-23 at 20:16:05 JST**, before implementation. It reuses the immutable v1 measurement functions, standalone adapter and paired executor. [Code/tests were frozen](../data/processed/benchmark_fixtures/event_allocation_v2_registration_v1.json) at **20:22:32 JST**, before primary execution. This is a new engineering diagnostic on known inputs; it does not retry or recover the missing v1 measurements.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `eda12263a670c122b46757bf0fcd79c5eceb2b4107017fb7fb2db60d508600a4` |
| Code registration | `9ab6673d605710b3087b465a3586c20b7fc4d7a72b02b7ed9ff4ec2f8a68e7af` |
| [Primary report](../data/processed/benchmark_fixtures/event_allocation_v2_result_v1.json) | `ca6f1b32455966c4c54e206fcaa942f0f43baf0e211c66ad3a82929212217d4b` |
| [Independent replication](../data/processed/benchmark_fixtures/event_allocation_v2_replication_v1.json) | `06b9df9890be40c142fbb3e7d729ff3ac9d7455093df0951746f44a006a15e52` |
| [Final descriptive/replication receipt](../data/processed/benchmark_fixtures/event_allocation_v2_replay_v1.json) | `bd13f44d03d9ea39d132a58753a2837cbfb4c6ed23006b606b32158fbfd7289c` |

The new [runner](../scripts/eval/event_allocation_v2.py) freezes phase markers, launches 144 fresh serial workers per phase, and saves the full canonical [bounded report](R2_ALLOCATION_REPORT_PREFLIGHT.md) before allowing replication. The saved-result gate checks hash, schema, registration/execution binding and the non-synthetic flag before creating the replication marker, running calibration or launching workers. Replication workers independently check that retained primary result too. Completed or failed phases cannot restart. Failures retain a receipt and stop without retry or capacity adjustment.

Both reports preserve all 61 site identities and 27,914 site incidences per phase, within the fixed 1,024-site / 75,000-incidence / 256-encoded-byte-name capacities. No site is clipped or aggregated away. The frozen classifier assigns 54 identities to execution, six to measurement and one to `other`. That last retained location is the runtime's `tracemalloc.py:558` inside the managed Python installation; its original category is preserved rather than reassigned after seeing results. Site histograms are post-query live allocations and can contain measurement bookkeeping.

All v2 artifacts total **1,303,231 bytes**, with **86,407 auxiliary bytes**, below the registered 17 MiB total / 1 MiB auxiliary ceilings. The report content is identical between phases except for its execution-marker binding, which explains the differing file hashes. **378 earlier artifacts and inherited source/test pins** verify. The original Kitchen ON/current/evidence conditions and all previous negative results and independent-period seals remain unchanged.

## Same workload and fair control

The existing first-order, middle-order and gap-order members use route pair `[0,1]`, four unchanged delivery scenarios and horizons 16 / 1,000,000,000. TTL remains four units; exact-deadline delivery, cancellation priority, stale timers, missing outcomes and late/duplicate/foreign rejection retain their original semantics. Every traced operation and teardown is checked against the untraced paired reference and prior saved state/receipt hashes.

Each mode records **1,280 calls**, including initialization and drained teardown. Each standalone arm actually executes the full shared observer for **224 inputs**. Physically paired execution runs that observer once per input, with both arm paths retained. Each arm/mode creates 224 receipts, delivers 96, cancels 16, expires 112, consumes 112 stale timers and rejects 100 result notifications. These are service volumes, not changed learning writes or CPU instructions. Both non-spiking and neuron paths receive the same causal shared history and scheduling opportunity.

## Allocation and ownership observations

| Per-mode maximum over primary calls; replication matches | Neuron standalone | Scalar standalone | Paired |
| --- | ---: | ---: | ---: |
| Newly traced live bytes at return | 3,048 | 2,280 | 3,424 |
| Newly traced peak bytes during call | 3,048 | 2,280 | 3,896 |
| Raw call peak minus return reading | 320 | 320 | 536 |
| Executor/output union graph at boundary | 10,778 | 8,636 | 13,425 |
| Output-only graph bytes | 716 | 716 | 1,060 |
| Shared-reachable graph bytes | 1,435 | 1,435 | 1,634 |
| Tracer metadata after snapshot | 5,904 | 5,136 | 6,320 |

Column maxima can occur at different calls. In particular, subtracting the maximum return value from the maximum peak does not recover the maximum call-local excess. Output-only, shared-reachable and executor-only partition bytes conserve the union at each individual boundary; their independently taken maxima must not be added.

The standalone input-stage peak is 3,048 bytes for the neuron and 2,280 for the scalar. Their maximum raw peak excess occurs during teardown: 320 bytes in each arm, with a positive excess in 48 of 1,280 windows. Paired execution has positive excess in all 1,280 windows, reaching 536 bytes on notification calls. Its dispatcher uses a different mapping/comprehension layout; subtracting paired and standalone peaks cannot identify either arm's incremental allocation cost. A zero excess at another call does not prove the absence of temporary allocations: the largest live point can coincide with retained output/state.

The known 64 KiB internally allocated/freed witness has a raw peak of 66,077 bytes and an excess of 65,593 bytes. A retained buffer leaves 66,041 newly traced bytes at return. Returning a pre-existing buffer leaves only a 392-byte envelope reading; the empty-call reading is 384 bytes. These witnesses validate scope sensitivity and are retained in both reports. Their values are not subtracted from the workload measurements.

Tracing starts afresh per call, after fixed GC collection, with cyclic GC disabled only during the envelope. Existing state predates tracing. Current/peak is queried immediately after return; snapshot classification and metadata queries happen afterward. Graph traversal runs after tracing stops and includes complete executor controls, receipt buffers and aliased packets. Prepass/reference states, retained audit receipts, reporting and process orchestration are outside the target-call window, while the invocation/query envelope remains in the raw figures. The original [scope contract](R2_EVENT_ALLOCATION_DIAGNOSTIC.md#implemented-measurement-boundary) remains applicable.

These observations do not measure all intra-call ownership, freed-source locations, allocator arenas, untraced native memory, process RSS, normal-runtime CPU or physical energy. **CPU, energy, RSS and total transient process peak remain null.** Tracer metadata alone is substantial relative to this tiny workload. Call peaks are not summed, graph bytes are not added to traced bytes, and the observed neuron/scalar difference is not generalized into an overall implementation ranking.

## Verification and research decision

There are **51 passing tests**: ten v2 orchestration/provenance checks and 41 retained report/failure/path tests. Mock-worker tests enforce the exact fresh roster, full shared-input charge, fixed new CLI, time/output/site limits and stopping on worker/identity failures. Missing or synthetic results launch no replication work. Actual saved reports round-trip losslessly, phase markers remain bound to source registration, and repeat execution/replication verification launches no new workers. The immutable measurement functions retain their earlier unit-test evidence; their allocation unit experiments were not rerun here.

```sh
# A completed replication receipt is verified without new allocation measurements.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/event_allocation_v2.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_event_allocation_v2.py tests/test_allocation_report.py tests/test_event_allocation_failure.py tests/test_project_paths.py
```

🟠 [Next] Preregister a finite **retained-evidence decision gate** combining the original predictive/mechanism results with completed accounting/separation/allocation receipts. Identify a decision-changing SNN-specific causal mechanism and independently qualifiable task prerequisite, or explicitly hold further resource work. Close this synthetic measurement-feasibility branch rather than adding diagnostic variants to seek a favorable spike result. No new measurement, scoring, source search/acquisition, automatic v3, threshold change or R3 optimization.

⭕️ [Pending] Independent predictive usefulness and full deployment/physical-energy evidence retain their original gates. Circuit sparsity/local learning remains a design hypothesis, but the scalar control shares sparse scheduling and the retained diagnostic does not favor a neuron memory saving. Future work needs evidence that can change the mechanism decision, not a relabeling of shared costs as an SNN-only benefit.

日本語: 診断v2を実装前に別登録し、固定48ケース・各144プロセスの本実行と独立再現を完了しました。詳細結果は各608,412バイトで保存され、3,840窓の割当値、状態・配送・所有権が一致しました。単独方式の最大新規割当ピークはニューロン3,048／スカラー2,280バイトで、SNNのメモリ優位は示されません。51テストが成功し、378既存証跡、v1失敗、条件・負の結果・封印を保持しました。次は測定の派生を増やさず、既存証跡だけで次の機構検証へ進む根拠を有限に判断します。速度・電力・独立有用性は未確認です。

简体中文：在实现前单独注册诊断v2，完成固定48个案例、每阶段144个进程的主执行及独立复现。详细结果各608,412字节，3,840个窗口的分配值、状态、回执和所有权一致。独立实现的新分配峰值为神经元3,048／标量2,280字节，未显示SNN内存优势。51项测试通过，保留378项既有证据、v1失败、原条件、负结果和封存。下一步不扩充分配诊断，仅依据已有证据有限判断继续机制验证的依据。速度、能耗和独立效用仍未确认。
