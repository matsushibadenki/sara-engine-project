# Hand-reference adapter-envelope interface v1 — 2026-11-04

🟢 [Done] Preregistered six hand-authored engine/role records and 28 binding faults before implementation. Pure read-only `validate(envelope, authority)` checks receipt, scalar-input and commit declarations against a separately supplied hand reference. Nine tests, 12 repeat read-only checks and exact fresh-process replay pass; 599 historical artifact hashes remain unchanged. No learner, probability/error calculator, model writer or persistence adapter is imported or executed.

Selection SHA-256: `a9e7cbafc6a8ff440d3e262060886c91eb4683e09f8e7fc2136662e651d76820`. Implementation, runner, tests, path utility and CPython 3.10.22 were registered before primary execution. The only historical report parsed is `route_adapter_spec_result_v1.json`; all seven missing runtime categories are retained. Six create-only artifacts use `route_adapter_envelope_*_v1.json`, with 64 KiB report input, 256 KiB per artifact and 2 MiB aggregate ceilings. Hand records reside in the selection, not acquired/model data.

## Interface and trusted input boundary

`validate(envelope, authority)` returns an observation only: binding validity, nine routes, and false application/authority-authentication/atomicity flags. It does not return an executable commit ticket, mutate inputs or consume a receipt. The `authority` argument is caller-trusted hand data. Exact canonical JSON comparisons bind types and reject extra/missing record fields. This is a bounded interface for these six preregistered examples, not a general runtime schema or authentication service.

| Record | Bound contents |
|---|---|
| Authority reference | Owner, receiving/source IDs, receipt revision/digest, delivery-before revision/digest, successor revision/digest, diagnostic token, error reference, per-route hand values, pre-outcome assertion, known revision floor and prior consumption. |
| Receipt | Own authority reference and identity; receipt-time revision/digest; diagnostic token; exactly nine ordered route/key/allocation/probability records; pre-outcome assertion. |
| Scalar input | Same identity; both receipt and delivery-before revisions/digests; receiving-time clock and delayed delivery tick; producer namespace; outcome; error reference; nine receipt allocation/probability, delivery allocation/probability and fixed-error values. |
| Commit declaration | Same complete binding; successor revision/digest; complete ordered nine routes; receiving-ID consumption; pending removal and allocation in the publication unit; requested/declared learning counts; no-writer assertion. |

The six namespaces are two engine labels × candidate/permanent/shuffled-outcome roles. They do not execute independent engines. Receiving ID is 64, delayed delivery is 65; shuffled source address and nine routes are copied from the earlier frozen hand metadata. Source ID never supplies the clock. Receipt revision is 10, delivery-before revision 11, successor revision 12 and trusted minimum 11. Successor derives from the delivery-before revision, not receipt revision. Hand digest tokens are literal labels, not observed model-state digests.

## Allocation and fixed-error distinction

The first route is explicitly unallocated at receipt (probability None), then allocated at delivery (probability .75). Other receipt routes are allocated with probability .5; delivery probabilities remain .75. Thus replacing receipt probability with a delivery probability is rejected, and absence cannot be replaced with an allocated zero-mass interpretation just because both can have a one-half effective readout. No count masses or actual zero-count table are supplied here; the latter is an interpretation boundary, not new arithmetic evidence.

Outcome is the hand literal [0,0,0], fixed errors are .25. The checker does not derive .25 from probability/outcome, recompute .75, update masses or build numeric after-states. It compares errors to the separately supplied hand authority and error reference. Error provenance and independent derivation remain unproven. Producer names are namespaces; a validator renamed as a producer is rejected, but distinct labels cannot prove distinct implementations.

The commit record declares 36 requested and changed learning fields, 399 unchanged fields and all nine routes in the same state/consumption publication unit. These are checked declarations, not observations of actual field changes. Numeric values, allocation deltas, storage operations and crash behavior are outside this interface. Pending removal, receiving-ID consumption, successor revision and unchanged diagnostic binding must all agree. A consumed-before reference, old delivery version below the floor, incomplete/duplicate/reordered route list or altered diagnostic is refused.

## Verification and limits

All 28 registered faults reject: foreign authority; wrong receiver; source as receiver; wrong source; receipt/delivery revision mismatch; stale known floor; receipt/delivery digest mismatch; post-outcome capture; unallocated replaced as allocated; receipt probability overwritten; delivery allocation mismatch; altered fixed error/error reference; wrong producer; early delivery; source clock; partial/duplicate/reordered routes; split consumption; retained pending; changed diagnostic; wrong successor revision; wrong learning count; writer claim; Boolean revision. Shuffled IDs are deliberately distinct for source/receiver fault checks.

Nine tests cover all six cases without importing additional runtime modules, every mutation without input changes, valid differing receipt/delivery values, exact Boolean allocation type, extra-field rejection, already-consumed authority, repeat read-only inspection and report drift/budget refusal. A deliberate **consistent fabricated authority reference passes** and still reports unauthenticated: this documents the trust limit rather than concealing it. These are static binding tests, not simulated writes or runtime replay. The fresh-process replay repeats only the checker; consumed audit phases reject before recomputation.

All seven runtime evidence categories remain missing. Learning/calculator/model-storage executions, task scoring, acquisition and resource measurements are zero. Audit artifact JSON is written through managed project paths; this is distinct from executing model persistence. No conditions, old failures, negative results or test/source seals change. The old failed replay-success artifact remains absent. No SNN-specific advantage, independent runtime equality, production promotion or live launch follows. Sparse, local and event-driven metadata behavior is also available to scalar implementations.

🟠 [Next] Preregister a saved-evidence readiness decision identifying which minimum runtime boundary would need separate authorization/evidence and which holds remain. Do not add more hand fixtures without a specific uncovered invariant; no learner/calculator/storage execution, new scoring/acquisition/resource work or condition/seal change. Real authority capture, independent producers, durable transaction/recovery and independent task/SNN causality remain ⭕️ [Pending].

## 日本語

手作業の権限参照と受信・スカラー入力・確定宣言を結ぶ非学習インターフェースを事前登録しました。6ケース、読み取り専用確認12件、故障28件、9テスト、別プロセス再検証が成功し、599件の既存証跡を維持しました。ID、受信版10・配送前版11・後版12、割り当て、登録済み固定誤差、9経路の完全性を確認します。誤差や状態を計算せず、実モデルの由来・実変更・原子性・独立性は立証しません。一貫した偽の権限参照も通る限界をテストで明示しています。学習器・計算器・モデル保存処理は実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册连接手工权限引用、接收、标量输入和提交声明的非学习接口。6个案例、12次只读检查、28项故障、9项测试及独立进程复验通过，保留599项历史证据。检查ID、接收版本10、投递前版本11、后版本12、分配、固定误差和九条路径完整性。不计算误差或模型状态，不证明真实来源、实际修改、原子性或独立性；测试明确表明一致伪造的权限引用仍可通过。不执行学习器、计算器或模型存储操作，旧失败、负面结果与封存保持不变。
