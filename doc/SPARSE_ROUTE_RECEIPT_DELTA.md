# Frozen receipt-to-numeric-delta contract v1

🟢 [Done] A separately preregistered, read-only bridge binds frozen receipt probability/result errors to the existing exact numeric delta validator. All 66 literal transactions pass: 2,376 requested learning targets, 2,376 serialized changed learning fields, 26,334 unchanged learning fields; 66 allocation targets but only 33 changed allocation flags. Twenty-four faults, twelve tests, exact fresh-process replay and 569 historical hashes pass. These are fixture comparisons, not actual model writes or runtime engine equivalence.

## Registration, fixed sources and limits

Selection SHA-256: `45e8a58f0ccb1795c3ab252fbc530611cd46c55472eef5ef3a07411366997bbf`. The selection and target after literals were written before implementation. New source/tests, the unchanged receipt resolver and six transitive read-only/path dependencies were pinned before primary execution under CPython 3.10.22. No old protocol, source or failed run was modified or resumed.

Three saved inputs supply the eleven local probability/error records, twelve symmetric before/after templates and frozen allocation/source plans: `route_error_binding_preflight_selection_v1.json`, `route_numeric_delta_preflight_selection_v1.json`, `route_independent_transition_selection_v1.json`. Total bytes are 241,306, within the 262,144-byte aggregate cap. Bounds are 11 local rows ×3 roles ×2 delivery-view branches =66 baseline transactions, nine route credits each, 435 numeric learning fields and 150 allocation flags. Fault checks and unit/proof verification are additional pure fixture inspections, not extra independent observations or learning trials.

All target after values are literal entries in the new selection, fixed before the harness was written. Other after values come unchanged from the matching old symmetric template. No update function produces the after-state. These examples remain standalone algebraic snapshots, including observation counts 8/16→9/17 that need not describe feasible full learned prefixes. Side/key placeholders retain the earlier fixture meaning; no actual sampled task or joint outcome stream is introduced.

## Binding and separation of authority

The existing immutable receipt contract captures each route/key before an outcome-side argument is accepted. The bridge receives a trusted pending batch, delivered literal results and trusted numeric predecessor separately from the proposed after-state and error claims. For each of nine credits it resolves the frozen receipt against the supplied delivery-before view, derives its own error witness and compares the candidate's claim by strict finite float hex equality. Candidate after-values or candidate witnesses do not define the expected error. A self-consistent fictitious error/after pair is refused if it disagrees with the receipt.

The unchanged 435-field read-only validator then checks exact rounded increments to credited mass, observations, error sum and weight sum, required positive numeric changes, inactive/opposite-key preservation, provenance/address structure and conservation. It receives the errors derived from frozen receipts, rather than the prior externally supplied error witness. Existing scientific rules and numeric bounds remain unchanged. This deliberately reuses the existing verifier; it is not a new independent runtime engine. Its engine name is a descriptor only.

Both proposed views pass through canonical JSON encoding/decoding in memory before typed comparisons. The trusted predecessor's encoded fingerprint must match. Integer, float and boolean types remain distinct; original numeric finite/range/no-negative-zero checks apply. Rounding a required learning increment away, or merely requesting an unchanged learning field, cannot satisfy the change budget. This establishes representation-level roundtrip behavior, not disk durability, filesystem persistence or writes to a trained model.

## Requested targets versus persisted differences

| Namespace | Catalog per transaction | Requested targets per transaction | Serialized actual changes per transaction | Total requests | Total changes |
|---|---:|---:|---:|---:|---:|
| Learning fields | 435 | 36 | Exactly 36 | 2,376 | 2,376 |
| Allocation metadata | 150 bool flags | 1 | 0 or 1 | 66 | 33 |

Exactly 399 learning fields per transaction remain unchanged (26,334 total). Allocation metadata has 9,867 unchanged flags across the baseline transactions. The single metadata request explicitly ensures the target is allocated; an already-true flag is a permitted, separately reported idempotent metadata request. That exception **does not** permit a no-op in any of the 36 required learning fields. Metadata requests cannot be counted as learning targets or used to fill the learning-change budget. Counts are independently derived from typed before/after differences after roundtrip, not trusted from the proposal's claimed totals. Requested/touched manifests describe intended fixture targets; physical write attempts are not observed here.

The stable branch uses the receipt-time target allocation/masses as the delivery-before state. The changed branch toggles target allocation, with masses 0/0 when now absent or 0/2 when now allocated. Expected after metadata always admits the target. These are hand-defined counterfactual views, not executed state changes. All six transactions originating from the unallocated local record retain a frozen `None` probability/fallback witness. Allocated zero remains explicitly 0.5. Comparing numeric admission literals does not implement an unallocated-key learner.

## Verification and failure coverage

All 594 baseline route witnesses validate. Errors stay the same across the paired delivery-view branches even when numeric predecessors/allocations differ. Receiving/source IDs, fixed one-tick delay and diagnostic namespace stay bound; 1023→1024 uses the receiving clock, not the later public-weight bucket. The read-only bridge returns no modified learner or application authorization.

Twenty-four registered faults are refused without mutating trusted fixtures: one-ULP witness; delivery-state probability witness; mutually consistent false witness/after error; learning no-op; inactive-field change; extra/duplicate/missing learning targets; metadata in the learning roster; learning in the metadata roster; missing/extra metadata target; missing admission; inactive allocation change; metadata counted as learning; incorrect metadata change count; bool count; stale predecessor; wrong receiver; wrong source; same-tick delivery; diagnostic drift; float error rounded to int; and delivery-clock error weighting. Twelve tests additionally check full totals, all allocation cases, exact fresh roundtrip, independent witness authority, stable/changed-state binding, absence versus zero, no-op/count refusal, late boundary, strict address types, input hashes and nonpromotion.

The result is 54,481 bytes; all six artifacts total 142,396 bytes, below the frozen 256 KiB/result and 2 MiB aggregate ceilings. They are saved under `data/processed/benchmark_fixtures/route_receipt_delta_*_v1.json`: selection, registration, execution, result, replay_execution and replay. Result SHA-256: `479e8a99f996bc2aab7268e5f5e42d72ad05fbe16aded746c528fba2ed1fa153`. Fresh processes match complete canonical results; consumed primary/replay markers refuse restart before computation. Failures would remain terminal without tuning or retry.

## Limits and next step

Trusted predecessor, pending read views, source mapping and result labels are still literal fixture authorities. Mutually fabricated trusted inputs or rollback outside this adapter are not authenticated runtime history. Validating a batch repeatedly is a read-only inspection; no durable exactly-once model-commit protocol has been established. No actual learner, model capture/update, task prediction scoring, source/model payload acquisition, resource measurement, threshold/condition adjustment or seal opening occurred. Old replay failure, recovery/retention negatives, support/current-state gates and causal uncertainty remain preserved. SNN-specific advantage, independent task usefulness, same-role runtime equality and production readiness remain unestablished.

🟠 [Next] Preregister and statically validate the minimum same-role integration protocol for genuine pending/predecessor provenance, atomic nine-credit receipt consumption and observed persisted-write evidence. Distinguish read-only repeat validation from exactly-once application and independent scalar execution from an engine descriptor. No live run or further resource work without separately justified task/causal prerequisites and a frozen protocol; no tuning or seal opening.

## 日本語

受信時確率に結び付いた誤差を、既存の厳密な数値差分検証へ接続する読み取り専用契約を事前登録・実装しました。66件すべてで学習フィールドの要求2,376件・実変更2,376件・無変更26,334件が一致し、割り当てメタデータは要求66件と実変更33件を別計上しました。誤差申告と更新後状態を同時に偽装しても、受信時の証跡に合わなければ拒否します。24故障・12テスト・別プロセス再検証が成功し、569件の既存ハッシュを維持しました。ここでの保存後差分はメモリ内JSON往復後の固定記録の比較で、実モデルやディスクへの書き込みではありません。追加学習・採点・取得・計測・条件調整はゼロで、旧失敗・負の結果・封印を維持します。

## 简体中文

已预注册并实现只读契约，将绑定接收时概率的误差接入既有精确数值差分验证。66个案例中，学习字段请求2,376项、实际变化2,376项、不变26,334项全部匹配；分配元数据单独记录66项请求和33项实际变化。即使伪造相互一致的误差声明与后状态，只要不符合接收证据仍会拒绝。24项故障、12项测试及独立进程复验通过，保留569项历史哈希。所谓保存后差分仅指内存JSON往返后的固定记录比较，未写入实际模型或验证磁盘持久性。新增学习、评分、采集、测量及条件调整均为零，旧失败、负面结果和封存不变。
