# R2 bounded allocation-report preflight — 2026-10-23

🟢 [Done] The preregistered **report-only** preflight proves an encoded-size ceiling of **7,302,888 bytes**, within the unchanged **8,388,608-byte** budget, for every report accepted by the new representation. Lossless round-trip checks, an independently constructed syntax witness and exact fresh-process replay pass. A new saved-result gate rejects missing, corrupt, incomplete, synthetic or incorrectly bound reports before invoking a supplied launcher.

This addresses the storage failure in the [isolated allocation diagnostic](R2_EVENT_ALLOCATION_DIAGNOSTIC.md). It does not recover that run's unsaved values or certify the capacity needed by a future tracer. There are **zero new allocation measurements and zero diagnostic-worker launches**. The consumed v1 execution, its failure receipt and all original task/evidence conditions, negative results and seals remain intact.

## Registration and retained evidence

The [selection](../data/processed/benchmark_fixtures/allocation_report_selection_v1.json) was saved on **2026-10-23 at 17:12:52 JST**, before implementation. It fixes the domain limits, size proof, round-trip checks and launcher-refusal contract. Engineering construction checks are known inputs; this is not a blind predictive experiment. [Code/tests were frozen](../data/processed/benchmark_fixtures/allocation_report_registration_v1.json) at **17:23:38 JST**, before the retained static execution.

| Evidence | SHA-256 |
| --- | --- |
| Selection | `1513e5a82c96b9e379eba5c50f88fa97a5a9c9b03917f4135c2422f42c4f3c8b` |
| Code registration | `1892fa911cff4e3c0b0a0536e1a24cb92d98d3a27f780a2618cd59293370a9bb` |
| Static result, 2,811 bytes | `1cd610868eab0e42a4437ea32450f2a666ae72c29411d5f686567607a106b595` |

The [result](../data/processed/benchmark_fixtures/allocation_report_result_v1.json) and [exact replay receipt](../data/processed/benchmark_fixtures/allocation_report_replay_v1.json) preserve **372 previous artifacts** and inherited code pins. The inventory reads seven frozen source files / 906 lines and the existing synthetic fixture/aggregate result. It does not execute their components or read new sensor, message, outcome or learned values. Old allocation result/replication files remain absent. The failed v1 code is unchanged; the new gate is a separate API for a future separately registered runner.

## Bounded representation and proof

The [codec](../src/sara_engine/evaluation/allocation_report.py) uses compact ASCII JSON with a final newline, fixed column tables and a shared site dictionary. It preserves all supported raw worker fields: case/mode identities, five state/input/receipt hashes, full shared-input counts, lifecycle receipt counts, each call's allocation scalars, owner byte/block totals, six graph-ownership fields and every accepted site byte/block entry. Owner and graph conservation are validated. Unknown allocation locations retain their full names and `other` ownership.

| Registered representation capacity | Limit |
| --- | ---: |
| Fixed case/mode roster | 48 × paired/neuron/scalar = 144 workers |
| Calls per worker | 66 |
| Allocation/ownership numbers per call | 16 unsigned 64-bit integers |
| Allocation-site identities | 1,024 |
| Encoded site-name length, including quotes/escapes | 256 bytes |
| Site incidences across calls and four witnesses | 75,000 |
| Numeric representation | 0 through 18,446,744,073,709,551,615 |

These are representation capacities. Overflow is rejected, never clipped or aggregated away. The previous 1 MiB per-call traced-peak condition and fixed input/service semantics remain enforced. The format does not add call peaks, subtract calibration or change shared-observer costs. CPU, energy, RSS and total transient peak remain explicit nulls.

The proof bounds fixed metadata/columns/roster, all worker headers, **9,504 call slots**, four witness rows, all dictionary entries and all 75,000 site incidences. Every unsigned integer has at most 20 decimal digits; escaped string lengths are checked after JSON encoding. Separators and the final newline are included conservatively. The independent construction fills all syntax capacities and yields **7,293,366 bytes**, below the analytic ceiling. It deliberately relaxes semantic consistency to exercise simultaneous width maxima; it is a syntax witness, not an accepted report or an allocation observation.

The proved ceiling leaves **1,085,720 bytes** of budget. It is conditional on domain validation. In particular, the source inventory does not prove that future tracer output uses at most 1,024 sites, 75,000 incidences or 256 encoded bytes per name. Any such excess must fail a future run explicitly. No capacity increase or new diagnostic run is authorized by this result.

The [constructed round-trip fixture](../src/sara_engine/evaluation/allocation_report_preflight.py) covers the 48 existing case identities and all three modes, using the maximum 66 report slots per worker. Its numbers and stages are deliberately constructed for schema testing; they are not the v1 run's missing measurements or an observed execution schedule. The fixture is marked synthetic, contains unknown Unicode site names, and serializes to **699,788 bytes**. Exact decoding restores the full payload. This size cannot be used as a compression ratio for the lost v1 report.

## Saved-result boundary and verification

`saved_replication_gate` checks an allowed managed path, expected result SHA-256, the complete canonical schema and matching registration/execution hashes before calling the supplied launcher. Synthetic reports are rejected. Tests use inert callbacks and workspace-only toy files; no subprocess launcher is supplied. The gate is necessary report/provenance validation, not a replacement for the future experiment's scientific and source gates.

**41 tests pass**: 35 codec/preflight tests, four retained allocation-failure checks and two managed-path tests. They cover byte and Unicode escaping boundaries, maximum integers, bool/float/null rejection, duplicate JSON keys and unused registry entries, site/incidence overflow, malformed indices, truncation, extra/omitted fields, graph/owner/receipt conservation, fixed case identities, synthetic and incorrect-execution refusal, and consumed-run refusal. Static import checks reject dependencies on tracing, subprocesses and execution components. The original allocation unit tests were not rerun.

```sh
# Static only; an existing replay receipt is verified without repeating work.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/allocation_report.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_allocation_report.py tests/test_event_allocation_failure.py tests/test_project_paths.py
```

🟢 [Done] A separately registered [allocation diagnostic v2](R2_EVENT_ALLOCATION_V2.md) now retains primary/replica reports of 608,412 bytes each, with 3,840 matching windows and 51 tests. Preserve 378 earlier artifacts/code pins, failed v1 and seals. No neuron memory saving or CPU/energy advantage is established.

🟠 [Next] Preregister a finite retained-evidence decision gate combining original predictive/mechanism results with completed accounting/separation/allocation receipts. Identify a decision-changing SNN mechanism and independently qualifiable task prerequisite, or hold further resource work. No new measurement, scoring, search/acquisition, automatic v3, threshold change or R3 optimization.

⭕️ [Pending] Actual allocation statistics, complete deployment resource/physical-energy evidence and independent SNN usefulness retain their prerequisites. A smaller report improves evidence retention for both implementations; it does not show an ANN-absent SNN benefit. The circuit-efficiency hypothesis still requires causal/useful behavior and sufficiently strong matched controls.

日本語: 報告形式を実装前に事前登録し、受理可能な報告の上限7,302,888バイトが固定8 MiB内に収まると証明しました。未知の割当位置と所有権を含む完全復元、保存結果を検証してから後続処理を許可するゲート、41テストと別プロセス完全再現が成功しました。372既存証跡と失敗・条件・封印を保持し、割当計測・診断ワーカー起動は0です。将来の実データが形式容量に収まるかは未確認で、次の測定には別の事前登録が必要です。SNN固有の優位は示していません。

简体中文：在实现前注册报告格式，证明被接受报告的大小上界为7,302,888字节，低于固定8 MiB预算。包含未知分配位置和所有权信息的无损恢复、验证保存结果后才调用后续处理的门控、41项测试及新进程精确重放均通过。保留372项既有证据、失败、原条件和封存；新增分配测量和诊断进程启动均为0。未来真实数据是否满足格式容量仍未确认，下一次测量须另行预注册，未证明SNN独有优势。
