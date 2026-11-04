# Hand-metadata commit/consume state machine v1 — 2026-11-04

🟢 [Done] Separately preregistered pure metadata reference implements repeatable preparation, full nine-route successor construction, duplicate refusal and recovery classification. Six engine/role fixture cases pass, with 24 read-only checks and 3,126 recovery classifications. Twenty-four faults, twelve tests, exact fresh-process replay and 581 historical hashes pass. This establishes hand-metadata behavior, not actual model/storage atomicity or independently executing engines.

## Frozen scope and inputs

Selection SHA-256: `7d896061c86d1661187ffee599c0a5df3b46c62cc6cd5230a0b15727b64ba5d4`. Hand fixtures, recovery expectations, mask enumeration and fault roster were fixed before implementation. New kernel/harness/runner/tests, path utility and CPython 3.10.22 were registered before the primary phase. Only `route_runtime_protocol_result_v1.json` and `route_independent_transition_selection_v1.json` are parsed: 93,455 bytes within the fixed 262,144-byte cap. No learner, old numeric fixture constructor, model updater or scorer is imported by the kernel/harness.

The literal predecessor revision is 10 and successor 11; receiving ID is 64 and delivery tick 65. Original nine-route IDs and shuffled source address are copied from the already frozen metadata plan, without new samples or altered delay conditions. Core/allocation/diagnostic tokens are hand-defined 64-character hexadecimal labels, not digests of observed model values. Roles and engine names parameterize six ownership namespaces of **one shared reference state machine**, not two independent numerical engines. The logical example is restricted to the registered cumulative/uniform metadata case.

## Preparation, commit and consumption

Immutable metadata state contains owner, revision, core/allocation/diagnostic tokens, consumed receiving IDs, last receipt and a nine-bit completion mask. Pending authority fixes the predecessor, expected route order, receiving/source IDs and successor tokens. Preparation checks current metadata against that entire trusted predecessor and a supplied minimum known revision, binds the own delayed delivery and returns an immutable ticket. The ticket includes the bound delivery metadata, ordered nine routes and predecessor fingerprint.

Preparation is repeatable and leaves the predecessor unconsumed. Commit rechecks the current predecessor, known revision floor and trusted pending/ticket binding, then returns one immutable successor containing all nine completed route bits, successor core/allocation tokens, unchanged diagnostic, incremented revision and consumed receiving ID. Partial, duplicate, extra or reordered route batches are rejected. A second delivery or commit against the returned current state is refused.

There is no writer callback, model mutation, journal, database transaction or filesystem state publication. Returning one Python object is the reference rule for a prospective atomic operation; it cannot demonstrate actual storage atomicity, durability or concurrent compare-and-swap. The mask indexes the nine registered credit positions, not 45 global route IDs or 36 observed learning-field writes. Opaque token changes cannot establish numeric changes or correct allocation contents.

## Recovery classifications and read-only revalidation

| Classification | Meaning in this hand-defined context |
|---|---|
| `predecessor_unconsumed` | Exact complete predecessor, no consumption, at or above the supplied known revision floor. |
| `successor_consumed` | Exact registered successor revision/core/allocation/diagnostic with receiving ID consumed and all nine bits set. |
| `reject_stale` | Observed revision is below the supplied known floor. |
| `reject_inconsistent` | Partial/mixed generations, state-only or consumption-only publication, unknown future revision, foreign owner or any other unmatched observation. |

Classification is read-only: it neither applies a transaction nor authorizes retry/replay. Each case inspects the predecessor twice and successor twice, yielding 24 repeat read-only checks. Two preparations also yield equal tickets without consuming the input. The original immutable states remain unchanged throughout.

Nine literal recovery rows per case cover valid predecessor/successor, known rollback, older version, eight-route commit, state without consumption, consumption without state, mixed allocation generation and unknown future revision. In addition, all 512 nine-bit masks are checked against the full successor metadata: only 511 classifies as complete. Six cases ×(9 literal rows +512 masks) =3,126 classifications, comprising 54 literal checks and 3,072 mask checks. These are counterfactual metadata observations, not actual crash/restart or partial model-write traces.

## Faults, preservation and evidence limits

All 24 faults reject without mutating the original fixture: stale revision/core/allocation/diagnostic; partial/duplicate/extra/reordered routes; duplicate commit/delivery; foreign owner; wrong receiver; source used as receiver; wrong source; early/late delivery; bool/float revision; bool route; forged predecessor fingerprint; altered successor ticket; consumption without state; rollback below the known revision floor; and bool watermark. Twelve tests additionally cover all classifications, repeat preparation, exact literal successor, all 511 incomplete masks, split-state recovery, read-only inspection without a commit call, immutability, delivery-bound tickets, byte bindings and no-promotion flags.

The minimum revision is itself trusted hand metadata. An external rollback accompanied by a forged/lowered authority, a fabricated consistent pending/ticket/delivery pair or cross-process concurrent application is not ruled out. Sequential duplicate refusal on the returned current state is not a claim of durable exactly-once application. Metadata fingerprints bind the supplied values but do not authenticate their real origin. All seven missing runtime/task/SNN evidence categories from the parent protocol remain missing.

Actual learner executions, numeric model-fixture executions, learning writes, task scores, acquisitions, resource measurements and storage crash tests are zero. All old failures, recovery/retention negatives, scientific thresholds/conditions and test/source seals are retained. No SNN advantage or production/runtime promotion follows from this shared metadata reference.

Result size is 12,181 bytes; the six artifacts total 102,288 bytes under the fixed 256 KiB/result and 2 MiB aggregate ceilings. Artifacts are `data/processed/benchmark_fixtures/route_commit_state_machine_*_v1.json`: selection, registration, execution, result, replay_execution and replay. Full canonical fresh-process replay and pinned-source verification pass. Consumed primary/replay markers refuse restart before computation; failures would remain terminal without tuning/retry.

🟠 [Next] Preregister a finite read-only source-to-evidence audit locating real receipt/predecessor authority, commit/storage boundaries and independent scalar after-state producers. Use source and saved reports only, without importing or executing models/calculators. Identify the minimum missing adapter/evidence or stop; do not add new arithmetic fixtures or resource measurements merely because metadata tests pass. Real provenance, storage atomicity, independent runtime equality and independent task/SNN causal evidence remain pending.

## 日本語

手作業のメタデータだけで、準備・確定・受信ID消費・復旧分類の状態機械を事前登録・実装しました。6ケース、読み取り専用確認24件、復旧分類3,126件、故障24件、テスト12件、別プロセス再検証が成功し、581件の既存ハッシュを維持しました。9経路の全512マスクを調べ、全経路がそろった場合だけを後状態・消費済みと分類します。与えられた既知の最低状態版による古い状態の拒否と、返却された現在状態での重複配送・重複確定拒否も確認しました。ただし実モデル、保存装置、並行実行の原子性・永続性ではありません。追加学習・採点・取得・計測・条件調整はゼロで、旧失敗・負の結果・封印を維持します。

## 简体中文

仅用手工元数据预注册并实现准备、提交、接收ID消耗及恢复分类状态机。6个案例、24次只读检查、3,126次恢复分类、24项故障、12项测试和独立进程复验通过，保留581项历史哈希。枚举九条路径的全部512个掩码，仅全部完成时才分类为后状态且已消耗。验证了基于给定可信最低版本拒绝旧状态，以及在返回的当前状态上拒绝重复投递/提交。本结果不证明实际模型、存储设备或并发执行的原子性与持久性。新增学习、评分、采集、测量和条件调整均为零，旧失败、负面结果及封存不变。
