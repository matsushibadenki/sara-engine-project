# Fixed-condition numerical reproduction v2 — 2026-10-26

🟢 [Done] The separately preregistered v2 primary execution and fresh-process replication **match exactly** under the [validated report comparator](SPARSE_ROUTE_REACTIVATION_CODEC.md). They also reproduce the historical saved scientific report after the two explicitly registered provenance-field bindings. **The component remains negative**: all ten changed cases still fail 256-tick recovery. The old consumed execution/replay is untouched and its failure receipt is not reclassified.

## Registration and unchanged conditions

The owner explicitly requested this separate numerical reproduction. The [v2 protocol](../data/processed/benchmark_fixtures/route_numeric_v2_selection_v1.json) was saved at **13:18:52 JST**, before implementation. [Implementation/tests registration](../data/processed/benchmark_fixtures/route_numeric_v2_registration_v1.json) at **13:22:51 JST** preceded the primary execution. Replication started at **13:23:00 JST** in a different process. Both use the exact registered CPython **3.10.22** runtime, including matching build, architecture and executable identity.

The entire original scientific specification is copied and compared exactly, including seeds, tasks, conditions, generator, learning rule, controls, windows, gates and candidate bounds. The original learner, generator, route code, codec and old runner remain byte-for-byte pinned. No coefficient or implementation of the scientific mechanism is changed.

| Fixed element | Value |
| --- | --- |
| Study seeds | 1201, 1213, 1229, 1237, 1249 |
| Per phase | Two tasks × five seeds × stationary/changed = 20 cases; five arms each |
| Learning duration / change point | 1,024 / tick 512 |
| Warmup / exploration / cooldown | 180 / 180 / 64 ticks |
| Trigger | Brier EWMA rate 1/16; threshold 0.20 |
| Candidate bank / shortlist | 45 / nine; three distinct updates per output per tick |
| Per arm/case actual updates | 9,216 route updates; 36,864 changes of four registered learning fields |
| Recovery gate | Two consecutive 64-tick blocks ≥90%, ending within 256 post-change ticks |

The two new phases each execute 100 arm/case runs: **921,600 route updates and 3,686,400 registered learning-field changes per phase**. These are reproduction runs using the same synthetic observations, not fresh independent data or a new effectiveness trial. Scheduler/detector metadata and total service cost retain the original accounting limitations. There are no real-data reads, resource/energy probes, GPU requirements or global-gradient learning.

## Exact comparison and provenance

The [new runner](../scripts/eval/route_numeric_v2.py) uses only the `route_numeric_v2_*` artifact namespace. It calls the pinned learner functions directly through a lazy import after its gates; it never calls the old runner's execution/replay modes. The old execution markers and failure/result files remain immutable, and an old success receipt is required to remain absent.

The [comparison helper](../src/sara_engine/evaluation/route_numeric_v2.py) uses the unchanged `ReportCodec`. Only its previously validated `initial_pool` tuple/list locations may normalize; no float tolerance, type coercion, array sorting, dropped metrics or ignored scientific fields are introduced. All 20 cases, 100 state digests, 100 prediction/update ledger digests, learning counters, quality/recovery/retention gates and summary fields compare in full.

Each new report has a v2 provenance envelope and a legacy-layout comparison payload. The latter retains the original scientific selection ID and v1 report schema solely for the validated typed layout. It binds the two non-scientific fields below to actual new-run provenance. The historical reference is deterministically projected to those same values before comparison; every other field remains untouched. The mapping was registered before execution and is saved in both reports.

| Historical reference field | Old value | v2 value |
| --- | --- | --- |
| `registration_sha256` | `72447318…d1486` | `957c2aaa…86dff` |
| `preserved_artifacts` | 399 | 422 |

This mapping does not relabel the old registration as new execution evidence. The wrapper separately binds the new protocol, code registration, phase, exact runtime, historical raw-report hash and phase-marker hash. Primary and replication payloads include the same new provenance values and compare **without any further projection**. Their shared canonical payload SHA-256 is `b6ce4c49884631d77b717b0724994b238236fdb4f7c8134e22fc1fac29035a12`.

Exclusive markers precede learner imports. Replication cannot start without a retained, valid primary report bound to its marker, runtime, protocol and registration, with exact historical-reference agreement. It must use a different PID. Full reports are saved before the final success proof. A mismatch/exception retains a phase failure and stops; no consumed phase may restart. Read-only `verify` checks saved evidence without invoking a learner.

## Retained results and limits

The [numerical proof](../data/processed/benchmark_fixtures/route_numeric_v2_result_v1.json) passes both full primary/replication equality and the registered historical-reference comparison. Primary and replication reports are **500,106 / 500,110 bytes**; their differing phase/provenance envelopes explain their different raw hashes. Each is within the fixed 2 MiB report limit. Seven artifacts total **1,068,894 bytes**, including **68,678 auxiliary bytes**, below the 6 MiB total / 1 MiB auxiliary ceilings. All **422 prior file hashes** and the newly registered sources verify.

Scientific outcomes are unchanged. All ten stationary cases have zero false exploration starts and pass their gates. All ten changed cases fail recovery within 256 ticks; three fail exact dependency retention, two fail the required permanent-pruning gain, and one interacting case fails both tail and frozen accuracy. Nine changed cases reach the two-block recovery criterion only at 320 ticks, and one never reaches it. See the [original result tables](SPARSE_ROUTE_REACTIVATION.md) for the unchanged descriptive metrics.

**56 tests pass under Python 3.10.22**: 21 new pure saved-report/mock-gate tests and 35 retained codec/path tests. They reject single-ULP, gate, count, state/ledger-digest and array-order changes, metadata/type/envelope tampering, missing primary results, consumed phases and same-process replication. They verify the exact two-field mapping and preservation of negative gates. These tests execute no learners; only the two preregistered v2 phases do so. Saved-proof verification also passes without a new study run.

The old failure receipt remains `full_replay_verified=false`, with the discarded old replay's numerical equality unknown. V2 establishes numerical reproduction from two **new** complete executions and the saved historical reference; it cannot recover or certify the discarded old replay. No old success receipt is created. The prior loss, failed attempt, thresholds and sealed periods remain unchanged.

| Artifact | SHA-256 |
| --- | --- |
| v2 protocol | `3f818ccde2cd11d084c1d0b2904ab0eb8f384a139b93f41054db065582528c1c` |
| v2 code registration | `957c2aaacd18249f30a3ca82f6b34f846375d83d3c31bfcc561b1678d0e86dff` |
| [Primary report](../data/processed/benchmark_fixtures/route_numeric_v2_primary_v1.json) | `8574c01c05ef4f82e134abdf4fb75471b3961940ca94dc6040d7843eeb5e9994` |
| [Replication report](../data/processed/benchmark_fixtures/route_numeric_v2_replication_v1.json) | `5ae9bbc3003c2ca1b3fbd874b4db9b3486a782b6007950145ec48b7346442fce` |
| Numerical proof | `11f1d2e62c2d6dd6705503f73cfb3e712595e3cee8b6d58777ab1453ce7ed103` |

```sh
# Completed evidence verification only; no new learner execution.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_numeric_v2.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Numerical reproducibility does not rescue the failed mechanism or qualify an independent task. Further mechanism trials require separately justified causal/task evidence and a new protocol; no automatic tuning or third numerical run is queued. Nonstationary promotion, R2 independent usefulness, delayed branch-local credit, physical energy and production remain under their existing holds. This non-spiking comparison establishes no ANN-absent SNN capability or energy advantage.

日本語: 条件を固定した別バージョンv2を実装前に登録し、新実行と別プロセスの独立再実行で全比較ペイロードの完全一致を確認しました。事前登録した来歴2項目の対応付け後、旧保存報告の科学的結果とも一致しました。56テストが成功し、422既存ファイルのハッシュ・旧再現失敗・負の結果・封印を保持しました。変更10ケースすべての回復基準不合格も再現され、機構の成功には変更しません。旧実行は再開せず、破棄された旧再現報告の数値一致は未確認のままです。

简体中文：在实现前注册条件固定的独立v2版本，新执行与另一进程的独立重复执行中，完整比较载荷完全一致。按事前注册的两个来源字段进行绑定后，与历史保存报告的科学结果也一致。56项测试通过，保留422个既有文件哈希、旧重放失败、负结果和封存。全部10个变化案例的恢复门槛失败也被复现，不改判为机制成功。不恢复旧执行，已丢弃的旧重放报告的数值一致性仍未知。

🟢 [Done] Follow-up [finite saved-report failure audit](SPARSE_ROUTE_FAILURE_AUDIT.md) separates timing constraints, exact dependency membership, reported accuracy and missing causal evidence without another learner run. Original failures and thresholds remain unchanged.
