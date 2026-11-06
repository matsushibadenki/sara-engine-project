# Model-free single-owner host control interface — 2026-11-06

🟢 [Done] Separately preregistered and implemented a control-metadata-only host with identity-bound owner capability, host-instance epoch label, monotonic lifecycle generations, shared exclusive mutation/pause gate and terminal exception isolation. Twenty-one registered scenarios, three cross-thread exclusion checks, nine unit tests and exact fresh-process replay pass. 623 historical artifact hashes are preserved. No model, learning rule, probability/error calculator, persistence adapter or live collector is accepted or executed.

Selection SHA-256: `d04ddd6f6d84a517ac902986208b22258fd5d6aa553b27682521b7564e1d6aa6`. Contract, scenarios, fixed exception handling and five-second thread-completion ceiling were saved before implementation. Kernel/harness/runner/tests/path utility and CPython 3.10.22 were registered before primary execution. Only `route_authority_design_result_v1.json` is parsed as historical input. Six create-only artifacts use `route_owner_host_*_v1.json`; ceilings are 64 KiB report input, 256 KiB/artifact and 2 MiB aggregate. Thread join timeout is a deterministic failure bound, not a resource-performance measurement.

## Interface and transitions

`OwnerHost.create(owner_label, epoch_label)` returns a host and minted immutable Owner capability. The constructor stores control metadata only, never a model or callable. Capability checks use object identity; constructing equal labels or creating another host with identical labels does not transfer access. Labels are hand metadata, not globally unique or authenticated run identities. Capabilities must be confined by a future authorized caller; Python private attributes are not a security sandbox.

| Operation | Control behavior |
|---|---|
| `inspect(owner)` | Acquire gate without waiting and return an immutable View of stable/closed/poisoned control metadata. Does not recover or authorize application. Busy during active mutation or pause. |
| `mutation(owner, expected_revision)` | Acquire exclusive gate without waiting, require exact nonnegative integer stable revision, mark odd in-progress generation, yield immutable View. On normal body return publish next even stable generation. No callback/payload execution by the host. |
| Mutation body exception | Catch BaseException (including the registered KeyboardInterrupt case), retain odd generation, mark poisoned and re-raise. Gate releases. No rollback, resume, restore or automatic retry API exists. |
| `paused(owner, expected_revision)` | Hold the same exclusive gate at the expected stable generation through the entire context. Issue an identity-bound immutable pause ticket. No revision advance. |
| `revalidate(owner, ticket)` | Active same-thread exact ticket only; check unchanged view and stable phase, return immutable View without changing state. No callback, numeric traversal or file operation. |
| Pause body exception | Release gate and invalidate ticket without poison or revision change: the interface allows no mutation under that ticket. A future model-bearing adapter would need its own verified write exclusion. |
| `close(owner, expected_revision)` | Stable-to-closed terminal transition; no further mutation/pause. Inspection remains possible. It is not a learning-state write or successful mutation revision. |

Initial generation is zero; successful metadata mutation goes 0→1(mutating)→2(stable), then 2→3→4. A failed body keeps its odd generation poisoned. Wrong owner, stale revision or Boolean/float revision rejects before generation change. Nested same-thread requests and cross-thread mutation/pause attempts fail immediately instead of waiting, avoiding a reentrant pause that permits mutation. Exiting a pause invalidates its ticket; another thread or a dataclass copy cannot revalidate it.

No data payload is changed. These generation changes are control transitions, not actual local updates. A pause is lifecycle exclusivity for requests through this object, not a paused real learner, stable OS process snapshot or genuine authoritative predecessor. A capability shared with untrusted code, direct private-attribute writes, external aliases and arbitrary out-of-band model mutations are outside the cooperative contract. Asynchronous interpreter termination or interruption during context setup/teardown is not proven crash-safe; tested exception isolation applies to exceptions raised through the mutation body. No process restart continuity, durable rollback protection, concurrent storage transaction or external authentication is claimed.

## Registered verification

The 21 scenarios cover initial state; successful mutation; repeated pause revalidation; RuntimeError and BaseException poison; pause exception release; close; foreign owner; stale/Boolean revision; foreign host with matching labels; nested mutation; mutation during pause; pause during mutation; expired/copied ticket; closed/poisoned refusal; cross-thread exclusion during pause and mutation; and foreign-thread revalidation. Three bounded worker threads are joined before result creation. Cross-thread identity/order details are not serialized, so exact replay compares deterministic classifications rather than OS thread IDs.

Nine unit tests additionally check strict rejection without generation changes, dataclass immutability, no model parameter, invalid labels, terminal poison/close, and same-label cross-host capability refusal. Actual kernel control paths execute; no learner/calculator/model-storage path executes. The primary and fresh-process replay pass, frozen-source verification succeeds, and consumed execute/replay refuse before computation. The four selected roadmap execution gates still pass their independent documentation check.

The host does not inspect a model, traverse numeric fields, prove true ownership origin, cover all model writers or exclude labels from a real runner. It does not unblock capture. All seven missing runtime/task/SNN evidence categories remain missing. Existing event-driven scalar control obligations, original acceptance gates and real-source support conditions remain unchanged. Learning/calculator/model-storage/live-capture executions, task scoring, acquisitions and resource measurements are zero. Old reproduction failure, negative recovery/retention/resource outcomes and source/test seals remain intact.

## Next step and holds

🟠 [Next] Preregister a bounded source-only integration map for complete writer/alias coverage and the input/outcome split before attaching this host. Enumerate each begin/observe/scheduler/diagnostic/restore access that would need the same gate, and identify where outcome-bearing Sample access must be removed. Specify required changes and blockers only; do not import/execute a model, attach callbacks, restart old runs or substitute hand labels as authority. This targets the newly uncovered ownership coverage obligation, not more arithmetic fixtures.

⭕️ [Pending] Source-backed host origin, actual exclusive model ownership/all-writer coverage, genuine monotonic epoch/revision, input-only pending capture, independent primitive readback, numerical producer independence, durable atomicity and independent task/SNN causal evidence. Reference lifecycle success is not evidence of any SNN-only benefit, energy advantage or deployment readiness.

## 日本語

モデルもコールバックも受け取らない単独所有ホストを事前登録・実装しました。所有者トークン、単調な制御版、更新と停止で共有する排他ゲート、例外時の隔離と再開拒否を扱います。21シナリオ、スレッド間確認3件、9テスト、別プロセス再検証が成功し、623件の既存証跡を維持しました。保証はこの制御オブジェクトを経由する協調的操作だけで、実モデルの権限・外部エイリアス・全更新経路・永続性は未立証です。学習器・計算器・モデル保存処理を実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册并实现不接收模型或回调的单一所有者主机。处理所有者令牌、单调控制版本、更新与暂停共用的排他门以及异常隔离和拒绝恢复。21个场景、3项跨线程检查、9项测试和独立进程复验通过，保留623项历史证据。保证仅限经此对象的协作操作，未证明实际模型权限、外部别名、全部写入路径或持久性。不执行学习器、计算器或模型存储，保留旧失败、负面结果与封存。
