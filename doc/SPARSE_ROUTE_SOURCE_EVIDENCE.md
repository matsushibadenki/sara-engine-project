# Read-only source-to-evidence audit v1 — 2026-11-04

🟢 [Done] Preregistered six source files and two saved reports before implementation. The audit parses source ASTs as data, binds ten named symbol excerpts to exact file hashes and line ranges, and carries forward seven missing runtime evidence categories. Six tests and exact fresh-process replay pass; 587 historical artifact hashes remain unchanged. No audited module is imported or executed.

Selection SHA-256: `45500071b4bb8e9c16067d7313acbe169ae926f21a29eb477c0fe7f27a3fa0a8`. Fixed questions cover receipt/predecessor authority, commit/storage boundary, independent scalar production and minimum missing connection. Limits: six source files, two reports, 256 KiB input and 64 KiB result. Implementation/runner/tests/path utility and CPython 3.10.22 were registered before primary execution. Six create-only artifacts use `route_source_evidence_*_v1.json`; execution and replay cannot be restarted after consumption.

## Source findings and evidence limits

| Boundary | Existing source evidence | What remains unproven |
|---|---|---|
| Receipt capture | `ReactivationMemory.begin` retains pending probabilities; `observe` consumes these after exactly one tick. `FactorialMemory.observe` also uses pending probabilities. | No connection in these selected methods to the trusted predecessor revision, allocation snapshot, receiving/source authority or sealed receipt contract. Retaining a tuple is not authenticating its origin. |
| Actual update | Both observe methods mutate counts, observations, errors and weights inside a route loop, then advance steps and clear pending. | Nine sequential updates do not establish all-or-nothing publication. An exception mid-loop could leave a prefix changed; no crash was injected here. |
| Independent checking | Factorial `train` computes expected deltas and checks mass/weight/update conservation in process. | This is valuable historical checking, not a separately captured independent same-role runtime engine or durable writer. No new training was run. |
| Scalar producer | `route_independent_transition.transition` independently constructs fresh fixture fields and a consumed ledger, using probabilities computed from supplied before-counts. | Its fixed fixture interface has no explicit absent/allocation state or frozen receipt-probability input. Delivery-before counts need not equal receipt counts. It cannot be substituted as a verified live delayed producer. |
| Frozen probability/error | `open_receipt` and `resolve` freeze supplied allocation/probability and bind delayed metadata without learning. | Supplied views and revisions remain trusted inputs, not captured/authenticated live model states. |
| Serialized delta | `route_receipt_delta.persisted` roundtrips JSON in memory. | No actual durable publication/readback or storage transaction is observed. |
| Commit/consumption | `route_commit_state_machine.commit` returns a full immutable metadata successor; `inspect_readonly` classifies without applying. | Hand tokens and shared engine namespaces do not establish numeric model changes, independent engines, durable exactly-once or concurrent atomicity. |

The result records exact symbol excerpts and their hashes/line ranges, not executed semantics. These findings are a bounded manual source interpretation corroborated by saved report flags; AST extraction is not a formal proof. Absence statements apply to the six selected files/method boundaries, not every repository component or hypothetical adapter. No model, calculator or old fixture helper is imported by the audit. The saved seven missing categories remain unchanged.

## Decision and minimum next step

Stop before runtime integration. The highest-value next verification is a separately preregistered **source-only adapter specification**, rather than more hand arithmetic or resource measurements. It must define: (1) the authoritative predecessor/readback and receipt-capture boundary before outcome availability; (2) an independent scalar producer interface with explicit allocation and frozen receipt-error inputs; (3) ownership of joint nine-route numeric/metadata/consumption publication and failure recovery. Specify mismatches and stop conditions without implementing a live writer or executing a learner. Existing references can validate proposals but cannot supply independent runtime provenance by relabeling themselves.

⭕️ [Pending] Live adapter correctness, durable atomic commit/recovery, independent runtime equality, an independent qualifying task and SNN-specific causal benefit. Event-driven scalar comparison remains mandatory. Local update arithmetic, immutable metadata and sparse execution are useful but are also available to scalar systems; this audit supplies no SNN accuracy, energy, memory or capability advantage.

Learner/calculator executions, scoring, acquisition and resource measurement are zero. No thresholds, conditions or test/source seals change; historical reproduction failure and all negative recovery/retention/resource outcomes remain. The absent old replay-success artifact stays absent. No live integration or scientific launch is authorized by this audit.

## Verification

Six unit tests cover bounded extraction without adding imported modules, altered source/report rejection, extra source rejection, missing/ambiguous symbol rejection and input budget refusal. Primary and separate-process replay agree exactly; verification rechecks registered source and all preserved hashes. Input bytes: 78272; result bytes: 20062. This measures artifact lengths for fixed output limits, not runtime resource performance.

## 日本語

6ソース・2保存済み報告だけを事前登録し、10か所のシンボルをハッシュと行位置に結び付けました。6テストと別プロセス再検証が成功し、587件の既存証跡を維持しました。既存モデルの受信時確率保持と逐次更新は確認できますが、実状態の由来、永続的な一括確定、独立スカラー実行は未確認です。次は実行を伴わない接続仕様を固定します。追加学習・採点・取得・資源計測・条件調整は行わず、旧失敗・負の結果・封印を維持しました。SNN固有の利点は立証していません。

## 简体中文

预注册仅审查6个源文件和2份已保存报告，将10个符号与哈希及行号绑定。6项测试与独立进程复验通过，保留587项历史证据。现有模型保留接收时概率并逐项更新，但真实状态来源、持久原子提交和独立标量运行仍未验证。下一步仅固定不执行学习器的适配接口规范。没有新增学习、评分、采集、资源测量或条件调整；旧失败、负面结果与封存保持不变，未证明SNN特有优势。
