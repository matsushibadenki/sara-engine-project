# Active-pause admission composition — 2026-11-08

🟢 [Done] Separately preregistered model-free composition of the unchanged owner-host and admission references. Active host-identity tickets, owner/epoch and stable pre/receipt generations bind one acknowledgement and delivery. Six namespace cases, 26 rejected faults, eight tests and exact fresh-process replay pass; 641 historical artifact hashes remain unchanged. Only control metadata and hand input/result bits are processed; no model, numeric snapshot, calculator, real producer or persistence adapter is attached.

Selection SHA-256: `d5e67545fa71de725dfd4af391e3a652e623a977680f1c4c702342074258dd8f`. Pre/receipt revisions0/2, same-ticket delivery lifetime, private predecessor seal, rejection roster and limits were fixed before implementation. Composition kernel/harness/runner/tests, both unchanged reference kernels, path utility and CPython 3.10.22 were source-registered before primary execution. The only parsed historical report is `route_outcome_admission_result_v1.json`. Six create-only artifacts use `route_pause_admission_*_v1.json`; bounds are 64 KiB report input, 256 KiB/artifact and 2 MiB aggregate. One bounded worker thread checks cross-thread refusal; no performance measurement occurs.

## Binding and lifetime

`PauseAdmission(host, owner, expected_ack)` accepts exact model-free reference types, verifies the host's stable initial metadata and owner capability, and requires receipt revision = pre revision +2. Its own admission gate capabilities remain private to the composition. Host/owner references are caller-supplied cooperative capabilities, not authenticated external origin. The composition cannot import or accept a model or callback.

1. Admit the immutable input-only envelope under the prior admission contract.
2. During a genuine active pre-revision host pause, seal the predecessor View privately. Host revalidation checks exact owner/ticket identity and same-thread lifetime. The returned View is metadata only; it is not a substitute authority capability.
3. Leave that pause; perform exactly one model-free host control mutation interval. It advances stable generation0→2. No model update is performed or inferred.
4. During the active receipt-revision pause, acknowledge the complete planned owner/epoch/receiver/source/revisions/digests/nine routes. The composition records the **actual ticket object**, not a reconstructed equal View.
5. Release the own one-tick delayed result once while that **same ticket remains active on the same thread**. Revalidate host before inspecting payload or calling admission release. A new pause at the same revision cannot reuse an old acknowledgement.

The pre token intentionally expires after private predecessor sealing; the seal is a host-validated control observation, not numeric state or a reusable ticket. Ack/release use the current receipt pause. Exiting that pause invalidates future delivery, even with unchanged revision. Poisoning or closing the host afterwards cannot resurrect it. A copied dataclass ticket, foreign host with identical labels, expired token or wrong owner rejects. Failures stop the composition with no retry; admission may still display ready internally after a host-lifetime failure, but the private release capability remains inaccessible through the stopped composition. A status observation never authorizes application.

The strict same-pause release rule is the frozen scope of this reference. It does not authorize dropping the pause and later releasing on a renewed ticket; a future design needing such behavior requires its own lifetime/revision contract. Result tick65 remains a hand delayed address, not elapsed wall time or proof that an external outcome became available only then. There is no numeric capture, digest calculation, allocation readback or learning-field inspection. Hand digests remain trusted plan labels; active control tickets do not authenticate those labels or any actual model values.

## Failures and verification

All 26 faults reject: no predecessor seal; copied/foreign pre ticket; wrong owner capability; wrong epoch/pre revision/revision gap; wrong receipt revision; copied/foreign receipt ticket; expired ack ticket; pre ticket used as receipt; duplicate preseal/ack; delivery without ack; expired delivery token; new pause after acknowledgement; poison/close after acknowledgement; duplicate release; wrong ack owner/source; partial route acknowledgement; early/late delivery; and cross-thread ack. Invalid acknowledgement constructors reject before submission; other rejected operations stop the composition. Host mutation exceptions are injected only in model-free control contexts and leave the host poisoned, not a numeric state rollback.

Eight unit tests cover all cases without importing new runtime/model modules, equal metadata versus ticket identity, same-revision renewed-pause rejection, expired token refusal before payload inspection and no retry, successful unchanged host generation during acknowledgement/release, foreign-host same-label refusal, missing preseal, and report drift/budget checks. Primary and separate-process replay match; consumed phases refuse before computation. The selected roadmap gate checker passes.

This composition runs the model-free host reference, which was explicitly preregistered here; it does not attach a host to a real model. It tests a real reference lifetime dependency rather than merely comparing equal hand metadata. It still cannot cover arbitrary Python private-attribute access, a malicious supplied host/caller, model aliases/writers, real owner origin, external outcome secrecy, process crashes or durable atomicity. The prior kernel's type/identity/lifecycle constraints remain cooperative process guarantees only.

## Decision and remaining holds

⭕️ [Pending] Actual host/capture integration remains blocked on source-backed model origin and all-writer/alias confinement, an input-only real producer and outcome-release authority, complete independent numeric readback and independently computed scalar after-states. Specify and source-pin those concrete adapters under a separate protocol before any model execution; do not interpret this reference composition as a live launch or add more hand fixtures merely to increase passing counts. Any new fixture expansion must identify a specific uncovered invariant first.

All seven missing runtime/task/SNN evidence categories remain missing. Learner/calculator/model-storage/live-numeric-capture executions, task scores, acquisition and resource measurement are zero. Original reproduction failure, negative recovery/retention/resource results, thresholds/current-state/outcome conditions and test/source seals are unchanged. Event-driven scalar comparisons remain mandatory; generic lifetime/exclusivity checks are not evidence of an SNN-specific advantage or production readiness.

## 日本語

単独所有ホストの有効な停止トークンと受付確認を結合するモデルなし契約を事前登録・実装しました。前状態の内部確認、別状態版での受信確認、同じ停止トークンが有効な間だけの一度配送を扱います。6ケース、故障26件、8テスト、別プロセス再検証が成功し、641件の既存証跡を維持しました。停止解除後に同じ版で停止し直しても、古い確認では配送できません。実モデル・実入力元・数値採取には接続せず、権限・全更新経路・数値ダイジェストの真正性は未立証です。学習器・計算器・モデル保存処理を実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册并实现将所有者主机的有效暂停令牌与接收确认连接的无模型契约。处理私有前状态确认、不同版本下的接收确认，以及同一暂停令牌有效期间的单次配送。6个案例、26项故障、8项测试与独立进程复验通过，保留641项历史证据。解除暂停后，即使在同版本重新暂停，也不能复用旧确认配送。未连接真实模型、输入源或数值采集，未证明权限、全部写入覆盖或数值摘要真实性。不执行学习器、计算器或模型存储，保留旧失败、负面结果与封存。
