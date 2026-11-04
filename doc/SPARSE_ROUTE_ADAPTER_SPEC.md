# Static connection adapter specification v1 — 2026-11-04

🟢 [Done] Separately preregistered declarative connection specification; eight ordered stages, three same-role engine pairs and 18 deliberate drift cases. Seven tests and exact fresh-process replay pass, preserving 593 historical artifact hashes. No callable model adapter, scalar calculator or storage writer is implemented or executed.

Selection SHA-256: `f1f603f0f3e560d456f748578821cebd3f5375971a7dc57ebf8d253762c8cc29`. Contract, fields, stage order, fault roster and budgets were saved before checker implementation; checker/runner/tests/path utility and CPython 3.10.22 were registered before execution. The only parsed historical input is `route_source_evidence_result_v1.json`. All seven missing runtime categories remain missing. Six create-only artifacts are `route_adapter_spec_*_v1.json`; 64 KiB input, 256 KiB per artifact and 2 MiB aggregate limits apply.

## Trusted boundaries

The prospective adapter obtains a complete predecessor from trusted store readback, not caller-provided model tokens. An authoritative owner and monotonic revision/watermark bind its digest. Hashes bind values but do not authenticate origin. A compare-and-swap against the exact current predecessor and durable unconsumed receiving ID must guard commit. How authority is provisioned, authenticated and protected against rollback remains unimplemented.

Before outcome availability, capture owner, receiving ID, source ID, predecessor revision/digest, receipt revision, ordered nine route IDs/keys, explicit allocation flags, receipt probabilities and diagnostic digest. The schema uses plural route/key fields and per-route allocation/probability values: cardinality and actual capture provenance are prospective obligations, not exercised data here. Source ID is an outcome address, never the receiving clock. Delivery occurs at receiving ID plus one, including shuffled roles. Receiver/source identity must persist through the entire envelope.

## Independent scalar input and output

The independent producer receives the complete sealed receipt binding plus delivery tick, bound outcome, frozen error, count increment, error weight and trusted delivery-before learning fields. `allocated` refers to receipt allocation; delivery-time allocation belongs to the separately bound delivery-before state and allocation metadata. A future envelope must make both revisions explicit and must not overwrite receipt fields with delivery values. Unallocated receipt probability is None with explicit fallback one-half; allocated zero mass has allocation true and probability one-half. They are distinct states despite equal effective readouts.

Frozen error is derived only from the sealed receipt probability, bound delayed outcome and receiving-time weight, then frozen for both producers. This specification passes that error explicitly, not a probability recomputed from delivery-before counts. Independent validation must separately check error provenance; merely supplying an error is insufficient. Delivery-before fields supply the bases of actual increments and must be revision-bound. Requested learning edits require nine actual route updates, 36 changed learning fields and 399 unchanged fields under the existing bounded catalog. Allocation requests and actual allocation changes are counted separately, allowing idempotent metadata without accepting learning no-ops.

The sparse and event-driven scalar producers must be separate implementations, consume equivalent same-role authoritative inputs and return fresh proposals without mutating models. Reference validators are validation-only, never after-state producers. Three comparisons are candidate-to-candidate, permanent-to-permanent and shuffled-outcome-to-shuffled-outcome; engine names alone do not establish independence. Existing scalar fixture arithmetic cannot be relabeled as the required live producer.

## Ordered stages and transaction ownership

1. Trusted predecessor readback.
2. Pre-outcome receipt capture.
3. Own delayed outcome binding.
4. Independent fresh proposals.
5. Proposal validation.
6. Compare-and-swap commit by one storage transaction authority.
7. Durable readback.
8. Read-only recovery classification.

One publication unit contains all nine routes' learning fields, allocation metadata, pending removal, receiving-ID consumption, successor revision and commit receipt. Diagnostic state remains unchanged in its separate namespace. Partial publication is prohibited. Duplicate application rejects against the durable consumed ledger. Validation/commit mismatch stops without automatic retry. Unknown or mixed recovery states stop and quarantine. Read-only recovery classifies but neither writes nor authorizes retry.

These are required behaviors, not an implemented transaction: no storage backend, writer, lock, journal, crash injection or concurrent test exists in this work. Existing immutable metadata return is not promoted to real atomicity. Storage of audit JSON artifacts is distinct from a learning-state transaction.

## Static verification and decision

The checker compares canonical JSON exactly against the hash-pinned preregistered contract, then checks field inclusion/uniqueness, stage and publication-unit structure and exact false Boolean gates. Eighteen mutations reject: untrusted predecessor, post-outcome capture, collapsed IDs, source clock, recomputed probability, missing allocation, shared producer, producing oracle, partial/split commit, unconditional commit, unknown-state retry, recovery writes, diagnostic mutation, role mismatch, requested-only update budget, adaptive promotion and launch authorization. This is a deliberate specification drift guard, not independent semantic validation of future implementations.

Seven tests cover the baseline without runtime imports, every drift without input mutation, numeric-zero versus Boolean-false, stage order, altered report, changed roster and input budget. Primary and fresh-process replay agree exactly; registered sources and 593 preserved hashes verify. Consumed execute/replay refuse before computation. No learner/calculator execution, task scoring, acquisition, resource measurement, condition adjustment or sealed access occurs.

Fixed candidate/permanent gain remains structurally zero under the original gain gate; shadow route selection remains diagnostic and the original adaptive system remains untested. Original reproduction failure, negative recovery/retention/resource results and all seals remain. SNN-specific accuracy/efficiency or causal benefit is not established; the event-driven scalar is mandatory, not an inferior assumed baseline.

🟠 [Next] Preregister a nonlearning adapter-envelope interface with hand-authored authority references and receipt/scalar/commit records. Statically validate cross-record identity, explicit receipt/delivery revisions, allocation distinction, frozen-error binding and nine-route completeness without model/calculator/storage execution. Hand references remain untrusted for real runtime provenance.

⭕️ [Pending] Real authority capture, independent live after-state producers, durable atomic commit/recovery, independent task qualification and SNN-specific causal evidence. No live launch follows from specification success.

## 日本語

信頼する前状態と受信時記録、割り当て・固定誤差を明示した独立スカラー入力、9経路の状態・消費の一括確定を宣言的仕様として事前登録しました。8段階、同一役割の3比較、18種類の仕様逸脱を固定し、7テストと別プロセス再検証が成功、593件の既存証跡を維持しました。実モデル・計算器・保存装置は実行しておらず、原子性や独立性は未立証です。次は手作業のレコード間の結び付けを検証します。旧失敗・負の結果・封印は維持します。

## 简体中文

预注册可信前状态与接收记录、显式分配和固定误差的独立标量输入，以及九条路径状态与消耗的一次提交规范。固定8阶段、3组同角色比较和18项规范偏差，7项测试与独立进程复验通过，保留593项历史证据。不执行实际模型、计算器或存储写入器，未证明原子性或实现独立性。下一步仅验证手工记录之间的绑定。旧失败、负面结果与封存保持不变。
