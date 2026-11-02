# Frozen receipt allocation/probability contract v1

🟢 [Done] Separately preregistered nonlearning contract preserves explicit allocation and receipt-time probability through fixed delayed delivery. All 11 existing local records × three roles produce 33 immutable snapshots and 66 exact delivery branches. All 33 unchanged/changed-view pairs preserve the same frozen readout. Twenty-four faults, twelve tests, exact fresh-process replay and 563 historical hashes pass. No learner, task scoring, acquisition or resource measurement occurs.

## Immutable protocol and inputs

Selection SHA-256: `a865e9a4c10a15fdd70a38d44b7135ad8fb6e2805ad67c9d205715b22183e610`. The selection was recorded before implementation; kernel, harness, runner, tests, path utility and CPython 3.10.22 were frozen before primary execution. Only `route_error_binding_preflight_selection_v1.json` and `route_independent_transition_selection_v1.json` are parsed (161,706 bytes total). They supply the unchanged eleven local probability/error literals and the already frozen source-index metadata. No old run or saved failure is resumed.

Budgets remain two saved inputs ≤262,144 bytes aggregate; 33 baseline receipt snapshots; 66 baseline delivery branches; 24 fixed faults; ≤262,144 bytes/result and ≤2 MiB/all artifacts. Fault preparation creates additional pure metadata snapshots/resolutions for rejection checks, not additional baseline observations or learner updates. Test/verification calls inspect the same fixed records. The six artifacts total 128,470 bytes; result size is 43,445 bytes, within the frozen limits.

## Capture and delivery semantics

The kernel is a new standard-library-only implementation of immutable `View`, `Snapshot`, `State` and `Delivery` records. It does not import a learner, existing calculation helper or numeric table updater. `open_receipt(state, receiving_id, source_id, view)` has no result-side argument. It retains the typed allocation flag, integer class masses, route/key, owner, view revision, diagnostic handle and independently computed receipt-time probability. The snapshot validates the probability against its own frozen view and receipt clock.

| Receipt representation | Stored probability | Residual fallback | Meaning |
|---|---|---|---|
| Explicitly unallocated, masses 0/0 | `None` | 0.5, flagged as used | No allocated key |
| Explicitly allocated, masses 0/0 | Exact float 0.5 | Not used | Allocated zero-evidence key |
| Explicitly allocated, nonzero masses | Exact smoothed float | Not used | Allocated evidence |

All original eleven rows are included; the previously unsupported absent row is now supported **as a readout snapshot**, not as an unallocated-key learning transition. Positive masses on an absent key are rejected; bool-for-int and int-for-bool coercions are rejected. Equal numeric residuals for absent and allocated-zero views do not erase their distinct allocation/probability/fallback metadata.

`resolve(state, delivery, current_view)` requires the same owner, receiving ID, trusted source ID, training purpose and exactly one receiving-tick delay. Shuffled source ID is an outcome address within the same fixed phase, not a timestamp. Current view must refer to the same owner/route/key and cannot go backwards in revision. Its masses and allocation do not select the residual probability. The local error uses the stored receipt probability (explicit absent fallback only), delivered binary side and public weight from the **receiving tick**, never the delivery clock. This is pure fixture arithmetic, not task prediction scoring.

Resolution returns a fresh closed metadata state with a consumed receipt and unchanged diagnostic, plus a reporting dictionary. It performs no count/error/weight/observation/allocation writes to a model. Duplicate delivery is rejected on the returned state; opening another receipt while one is pending is rejected. The reporting dictionary is not a model-update authorization. Immutable inputs remain unchanged on both accepted and rejected operations.

## Counterfactual delivery views and exact checks

For each snapshot, resolve two separate fixture branches: an unchanged view at revision 0 and a changed view at revision 1. The changed view is explicitly absent with zero masses when the original was allocated; when the original was absent, it is explicitly allocated with masses 0/2. These are fixed hand-defined counterfactual views. No learner applies those changes, and the two branches are not duplicate accepted results in one chronological run.

Both branches match every original probability/error literal by exact type and float hexadecimal value. Their readouts are identical after excluding the explicitly reported delivery revision, and retain the original allocation/fallback metadata. No tolerance or new expected values are introduced. A dedicated test shows that different result sides produce identical captured snapshots but different delayed residuals: the side is unavailable to capture. The 1023→1024 case retains weight 32768.0, rather than recomputing the next public bucket at delivery.

Twenty-four faults are refused atomically: absent positive mass; bool allocation; bool mass; NaN probability; one-ULP probability drift; absent treated as 0.5; allocated zero treated as absent; allocation drift; source used as receiver; wrong source; early/late/duplicate delivery; foreign owner; wrong route/key; diagnostic drift; backward revision; evaluation-purpose result; delivery-view substitution; unsupported outcome side; receipt-clock drift; mutable view; and opening an unresolved second receipt. Twelve tests cover full roster, capture API, allocation distinction, changed live view, future-side isolation, public-clock boundary, immutability, duplicate and early refusal, shuffled identity, byte bindings and no-promotion flags.

## Provenance limits and retained results

Trusted fixture owner, source mapping, read view, diagnostic and binary result side are not authenticated real runtime evidence. A self-consistent fabricated snapshot/history/label, or rollback to an older pending state, cannot be ruled out by this in-memory contract. The snapshot hash retained in the report is an integrity binding, not authentication. A wrong but valid binary outcome cannot be detected without an independently trusted result source; the unsupported-side fault checks schema only.

The result establishes contract-level absence handling and receipt-time freezing, not full learner equivalence, genuine numeric learning writes or an independent useful temporal task. No learning is implemented for absent keys. Existing old replay failure, recovery/retention negatives, thresholds, conditions, causal unknowns and test/source seals remain unchanged. No independent-task, SNN-specific mechanism, physical-energy or production promotion follows.

The six JSON artifacts are under `data/processed/benchmark_fixtures/route_receipt_probability_*_v1.json`: selection, registration, execution, result, replay_execution and replay. Complete canonical reports match in fresh processes. Consumed primary/replay markers refuse restart before computation; failure receipts would remain terminal without adjustment or retry.

🟠 [Next] Preregister a finite nonlearning integration contract binding frozen receipt error witnesses to the existing exact numeric delta ledger. Explicitly separate allocation metadata changes from the four learning fields per credited route; preserve matched actual-write rules and diagnostic isolation. No model updates/capture, learner launch, scoring, acquisition, measurement, condition changes or seal opening until a separately justified protocol exists. Runtime provenance and same-role independent engine equality remain pending.

## 日本語

既存11件×3役割の33スナップショット・66配送分岐で、受信時の割り当て状態と確率、既存誤差期待値を完全一致で保持できました。未割り当ての `None` と割り当て済みゼロ件数の `0.5` は別状態として保持し、前者だけフォールバック使用を記録します。配送時の状態を反転しても受信時の値を参照し、1023→1024の境界でも受信時刻の重みを使います。24故障・12テスト・別プロセス再検証が成功、563件の既存ハッシュを維持しました。追加学習・採点・取得・計測はゼロです。実モデルの由来認証、未割り当てキーへの学習、SNN固有の効果は未確認として、旧失敗・負の結果・封印を保持しています。

## 简体中文

既有11条记录×3种角色产生33个快照和66个投递分支，接收时的分配状态、概率及既有误差期望值均精确保持。未分配的 `None` 与已分配零计数的 `0.5` 保持为不同状态，仅前者记录回退使用。投递时视图反转仍使用接收快照，1023→1024边界也使用接收时钟权重。24项故障、12项测试和独立进程复验通过，保留563项历史哈希。新增学习、任务评分、采集及测量均为零；实际运行来源认证、未分配键学习和SNN特有优势仍未确立，旧失败、负面结果及封存不变。
