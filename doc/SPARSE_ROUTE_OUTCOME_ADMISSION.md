# Model-free input/outcome admission reference — 2026-11-08

🟢 [Done] Separately preregistered model-free admission gate with immutable input-only envelopes, distinct producer/collector identity capabilities, capture acknowledgement binding and one-time delayed-result release. Six engine/role namespaces, 24 rejected faults, nine tests and exact fresh-process replay pass; 635 historical artifact hashes are preserved. No host/model attachment, real producer, learning, probability/error calculation or model persistence is executed.

Selection SHA-256: `5b57a8498026188e7c37d2d3796c5a8093c66713aef04ee828c94f9a877271d1`. Hand identity/routes, phase revisions/digests, fault roster, states and stop policy were saved before implementation. Kernel/harness/runner/tests/path utility and CPython 3.10.22 were registered before primary execution. Only `route_integration_map_result_v1.json` is parsed as historical input. Six create-only artifacts use `route_outcome_admission_*_v1.json`, with 128 KiB input, 256 KiB/artifact and 2 MiB aggregate limits.

## Input, acknowledgement and release

`Input` contains only owner, epoch, receiving ID, source ID and five binary events. It is an exact frozen/slotted record: no result, label or callback field; dictionaries or extra payloads are not accepted. `CaptureAck` contains the same identities, stable pre/receipt revisions, hand digests and exactly nine distinct ordered route IDs. Expected acknowledgement is supplied as a trusted **hand plan**, not observed capture evidence. Revisions/digests are checked declarations, not host reads.

`AdmissionGate.create(expected_ack)` returns a gate plus two distinct identity capabilities. Producer admits one Input and releases the own delayed result; collector acknowledges the capture. Capabilities cannot be exchanged or replaced by equal labels. They are same-process cooperative identities, not authenticated producer ownership or a sandbox against private-attribute access.

| State/request | Behavior |
|---|---|
| empty → admit | Require exact producer capability, immutable input type and complete expected identity. Retain input only; enter awaiting_ack. |
| awaiting_ack → acknowledge | Require collector capability and exact complete acknowledgement, including both revisions/digests and ordered nine routes. Enter ready. |
| ready → release | Require producer capability, exact receiving-ID-plus-one tick and immutable three-bit outcome; return immutable Delivery bound to input and capture. Enter released once. |
| Missing/incorrect phase or valid-capability bad request | Stop without retry; do not retain an outcome. A duplicate release leaves terminal released state unchanged. |
| Foreign capability | Refuse without changing state, so a foreign caller cannot poison the current admitted receipt. |
| input_view/phase | Expose immutable input/control metadata only. No result is retained by the gate before or after delivery. |

Before acknowledgement, release checks phase before inspecting its supplied payload; the trap-payload test confirms this. Its function argument can already exist in caller memory, so this is a **release boundary**, not proof that the entire process lacks the outcome. The test producer supplies hand result bits only at release, not at admission. Real input/outcome sources, cross-process secrecy and channel authority remain absent. The Delivery contains outcome only after successful release and can be retained by the caller; the gate itself stores no outcome.

All operations use one lock for atomic reference-state checks/transitions. This has not been concurrency-stress-tested, connected to a host pause or demonstrated as a durable model transaction. Invalid dataclass construction is refused before reaching the gate and therefore leaves it untouched; a valid constructed but mismatching acknowledgement submitted with the collector capability stops the gate. Malformed field constructors should not be described as submitted stopped transactions. A consistently fabricated hand plan and matching acknowledgement can pass: no live capture or digest authentication is established.

## Frozen examples and verification

Two engine labels × candidate/permanent/shuffled-outcome roles give six namespaces, not independent numerical engines. Receiving ID64 delivers at65; shuffled source and nine route IDs come from the previous frozen metadata. Stable pre revision10 and receipt revision12 are distinct; hand hexadecimal tokens do not authenticate model state. No probability, error, numeric update or score is computed.

The 24 faults cover foreign producer/collector; release before acknowledgement; acknowledgement before input; wrong owner/epoch/receiver/source and source-as-receiver; mismatching pre/receipt revisions or digests; partial/duplicate/reordered routes; duplicate input/ack/release; early/late delivery; invalid outcome; Boolean revision; and an input dictionary carrying a result. A receipt revision14 is a mismatching future revision, despite the legacy fault name `stale_receipt_revision`; all nonexact revisions reject.

Nine tests verify all frozen cases without runtime module imports, input schema/immutability, preack rejection without payload inspection, terminal mismatch behavior, foreign-capability nonmutation, exact one-time delivery, strict types/route completeness, capability role separation, and source-report drift/budget refusal. Primary and fresh-process replay agree exactly; registered sources and 635 historical hashes verify. Consumed primary/replay refuse before computation. The separate roadmap gate checker remains valid.

Actual model/producer authority, external outcome unavailability, host writer coverage, real capture, independent numerical producer execution, durable nine-route state/consumption and SNN-specific benefit remain unproven. The seven missing runtime/task/SNN categories are retained. Learner/calculator/model-storage executions, real task scoring, acquisition and resource measurement are zero. All old failures, negative recovery/retention/resource results, scientific conditions and test/source seals remain unchanged. This admission pattern also works for scalar systems and is not SNN-exclusive evidence.

🟠 [Next] Preregister a bounded model-free composition contract between owner lifecycle and admission acknowledgement: bind a still-active pause/epoch/revision to one acknowledgement, distinguish copied metadata from an authoritative capability, and specify refusal after pause release/poison/close. Do not attach a model, real producer, calculator or storage backend. Reference composition would not by itself establish genuine capture provenance or all-writer confinement.

⭕️ [Pending] Real ownership and outcome-only-after-capture channel, complete writer/alias coverage, independently traversed numeric snapshots, actual scalar comparisons, durable atomicity, independent task and SNN causal/resource evidence. No live attachment or old-run restart is authorized.

## 日本語

モデルを使わない入力・結果の受付を事前登録・実装しました。入力は結果を持たず、所有者・epoch・受信ID・元ID・前後版・ダイジェスト・9経路が一致する採取確認後だけ、1tick遅延の結果を一度配送します。6ケース、故障24件、9テスト、別プロセス再検証が成功し、635件の既存証跡を維持しました。実入力元やモデルには接続せず、呼び出し元が結果を別経路で知っていないことや採取確認の真正性は未立証です。学習器・計算器・モデル保存処理を実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册并实现无模型的输入/结果接收接口。输入不含结果，仅在所有者、epoch、接收/来源ID、前后版本、摘要及九条路径完全匹配的采集确认后，单次配送延迟1tick的结果。6个案例、24项故障、9项测试及独立进程复验通过，保留635项历史证据。不连接真实输入源或模型，未证明调用方无法通过其他路径获知结果，也未认证采集确认。不执行学习器、计算器或模型存储，保留旧失败、负面结果与封存。
