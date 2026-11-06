# Read-only authoritative capture protocol v1 — 2026-11-06

🟢 [Done] Separately preregistered protocol and finite source-only feasibility audit: five pinned sources, one saved readiness report and eight named symbol excerpts. Seven tests and exact fresh-process replay pass; 611 historical artifact hashes remain unchanged. **Actual capture is blocked**, not executed or shown feasible. Four source-grounded blockers remain.

Selection SHA-256: `4d3a6befe0a79a4370a191916e21df18670faabd43511c0aed19ff4efb84bb21`. Call policy, authority, fields, independent readback, budgets, stops and source anchors were fixed before implementation. Checker/runner/tests/path utility and CPython 3.10.22 were registered before execution. Six create-only artifacts use `route_capture_protocol_*_v1.json`. Static input limit is 128 KiB across five source files and one report; eight anchors, 256 KiB per artifact and 2 MiB aggregate limits apply. Audit artifact writes are not model persistence execution.

## Frozen prospective protocol

**Permitted runtime model/helper calls: none.** Constructors, begin, observe, train, probability, snapshot, state_digest, open_receipt, resolve, scoring, loading, saving and commit are forbidden. This work parses source ASTs as data and reads saved reports only. It never instantiates a model, obtains a live handle or executes a getter. A prospective collector could copy primitive fields directly only from an externally paused authoritative owner or an immutable snapshot supplied through a separately established boundary. That boundary is not supplied by this protocol.

Authority requires external owner identity, trusted monotonic revision and revision floor, exclusive pause or immutable snapshot, outcome-unavailable attestation and receiving/source identity binding. Model clock/steps, a digest of caller fields or a hand revision is not that authority. Record already-existing pending receipt information under the same paused boundary; do not call begin to manufacture it. Copy already-recorded probabilities, retaining None and explicit allocation. Never recompute probabilities from counts.

Required fields: owner; receiving/source IDs; receipt/predecessor revisions and digest; trusted revision floor; route catalog; nine active routes and keys; receipt allocation/probabilities; complete 435 learning-field catalog and 150 allocation flags in the registered representation; pending record; consumed receiving ledger; diagnostic state; scheduler state; step/update counters. These catalog dimensions are prospective encoding requirements, not claims that the current model already exposes that encoding or a consumed ledger. Unsupported fields block capture rather than being synthesized, guessed or dropped.

Independent readback requires two separately traversed primitive copies, without aliasing or recycling the first serialized copy. Complete typed state equality and unchanged authoritative owner/revision/floor before and after are required; no learning, calculation or mutation between reads. Two traversals still do not prove scalar-engine independence or authentication. These checks were not executed against a model.

Prospective fixed ceiling: one owner boundary, two readbacks, nine route credits, 435 learning fields, 150 allocation flags, zero model/helper calls, 256 KiB output, one attempt. These are design limits only, not an authorized runtime trial. Actual runtime cannot proceed until a genuine authority/outcome-free boundary is evidenced under a separately registered integration protocol. No wall-time, energy or memory-performance measurement is taken.

Stop on missing trusted owner/revision, missing pause/immutable boundary, outcome already available or unknown ordering, no existing pending receipt, missing/extra field, incomplete/duplicate route roster, wrong receiver/source, changed revision/content between reads, required probability recalculation, any forbidden call, exceeded budget or need for sealed/consumed input. Do not retry by weakening conditions or manufacturing authority.

## Static feasibility findings

| Blocker | Selected source evidence | Consequence |
|---|---|---|
| Authority missing | `state_digest` serializes supplied model fields; `open_receipt` accepts supplied immutable state/view. | Neither authenticates an external owner, revision floor, pause boundary or pre-outcome order. A hash is binding, not authority. |
| Receipt creation changes state | `BudgetRouteMemory.begin` calls narrowing/prediction and writes pending/clock; `ReactivationMemory.begin` delegates and extends pending. | Calling begin is outside a read-only collector. No new receipt may be fabricated by this work. |
| Existing snapshot exceeds call scope | Factorial snapshot recomputes probability, score/rank and uses true dependency hints; probability methods calculate from counts. | Prior snapshot nonmutation checks do not qualify it as a no-calculator outcome-free raw receipt copy. |
| Independent readback not connected | Selected digest/observe paths do not provide a paused authoritative handle and two bound primitive traversals; observe updates learning. | Do not substitute live observe, model construction, two aliases or hand labels. |

The result pins symbol excerpts, hashes, line ranges, syntactically present calls and attribute/subscript store targets. Selected syntax confirms relevant call/writer boundaries; it is not full call-graph purity analysis or a proof that no integration exists anywhere else. The five-file scope is fixed. No new scan, fixture expansion or resource work follows automatically.

## Decision and validation

⭕️ [Pending] Actual capture remains blocked on trusted owner/revision/pause authority and an existing outcome-free receipt read boundary. To reopen, provide source-backed integration evidence satisfying those prerequisites, then separately preregister the exact capture scope. Do not implement a collector or invoke a learner merely because this static checker passed. A source-only authority-provider design is a possible future proposal, not evidence that a provider exists or permission to execute one.

Seven tests verify finite blocked output without runtime imports, repeat nonmutation, source/report drift rejection, roster/budget refusal, runtime permission refusal, missing/ambiguous anchor rejection and syntactic detection of the existing writer. Primary and separate-process replay match exactly. Consumed execute/replay refuse before computation. Seven missing runtime/task/SNN categories remain missing; all learner/calculator/model-storage/live-capture executions, scoring, acquisition, resource measurement and new hand fixtures are zero.

Old numerical reproduction failure, negative recovery/retention/resource results, original thresholds/current-state/outcome conditions and all test/source seals remain unchanged. No actual provenance, atomicity, independent scalar execution, SNN-specific benefit or production promotion is established. Generic read-only copying and sparse local arithmetic remain available to scalar controls; this protocol is not SNN causal evidence.

## 日本語

読み取り専用採取の許可呼び出し・権限・状態版・採取項目・独立再読出し・予算・停止条件を事前登録しました。5ソース・1保存済み報告・8シンボルを静的検証し、7テストと別プロセス再検証が成功、611件の既存証跡を維持しました。現時点で許可するモデル呼び出しはゼロです。既存beginは状態変更、snapshotは計算を行い、信頼できる所有者・状態版・停止境界の接続も未確認のため、実採取は保留です。権限や不足項目を手作業で捏造せず、学習器・計算器・モデル保存処理を実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册只读采集的允许调用、权限与版本、字段、独立重读、预算和停止条件。静态检查5个源文件、1份已保存报告和8个符号，7项测试与独立进程复验通过，保留611项历史证据。当前允许的模型调用为零。现有begin修改状态，snapshot执行计算，可信所有者、版本及暂停边界的连接仍未建立，因此实际采集继续暂停。不伪造权限或缺失字段，不执行学习器、计算器或模型存储操作，旧失败、负面结果与封存保持不变。
