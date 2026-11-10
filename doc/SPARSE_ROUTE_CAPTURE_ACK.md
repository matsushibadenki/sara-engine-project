# Capture-generated acknowledgement bridge — 2026-11-10

🟢 [Done] Separately preregistered bridge derives acknowledgement owner/epoch, receiving/source IDs, pre/receipt revisions, digests and nine ordered routes from validated encoded pairs instead of an expected hand acknowledgement argument. Two hand pairs, 26 rejected faults, nine corrected V2 unit tests and exact fresh-process V2 replay pass. V2 preserves 666 artifact hashes. No real model, candidate encoder, producer or numerical update is executed.

V1 selection SHA-256: `402372e46d3c84392ba8a372601f26ce8de0fb63314f6d00ed0d25b557a76ece`. V2 selection SHA-256: `a0f31f68f833b91a4eecbf3ee376c233e983cc6e345f631b4d930eee0d5ca527`. V1 fixes the contract/hand recipes/faults before implementation. V2 separately registers a unit-assertion repair, without changing kernel, harness, input, gates or conditions. Kernel/harness/runner/tests and relevant unchanged reference sources/path utility and CPython 3.10.22 are registered before their primary phases. Only `route_capture_layout_result_v1.json` is parsed as historical input. Core artifacts are `route_capture_ack_*_v1.json` and `*_v2.json`; bounds are 64 KiB historical report input, 256 KiB/artifact and 2 MiB aggregate. No old experiment is resumed.

## Generated binding and private lifetime

`CaptureAckBridge(host, owner, input)` accepts model-free host/owner identity capabilities and an immutable input-only packet. It does not accept `expected_ack`, expected digest labels, callback, model or producer. The host must initially be stable at the packet's owner/epoch. During a live predecessor pause, seal the supplied immutable PhaseCapture after checking owner/revision identity, digest, typed logical projection, absent pending receipt and own input boundary. The private seal is not a public reusable capability.

After one host control interval advances to the distinct receipt revision, `acknowledge(ticket, pair)` requires a current same-host/same-thread active receipt ticket and validates the whole pair with the unchanged capture-layout reader. The pair must match the admitted input and the exact privately sealed predecessor. Only then construct CaptureAck from the validated pair: digests and revisions from its phases, ordered routes from validated pending credits. A private AdmissionGate is created/admitted/acknowledged under that pause; producer/collector capabilities are not exported.

Release requires the **same still-active receipt ticket**, own one-tick delay and a single valid result payload. Copied, foreign, expired or renewed tickets reject. Binding/layout failures stop the bridge without retry. The underlying gate may still display ready after a lifetime error, but the stopped bridge cannot use its private release capability. Ack returned to a caller is immutable binding metadata, not a pause authority token or a live model receipt.

Two existing hand-layout pairs differ in first-key absence versus allocated-zero. Derived digests differ accordingly without preset digest strings. All 435 learning fields and 150 allocation flags are validated through the layout reader; neither probabilities nor outcome errors are recalculated. A structurally valid pair with a new digest cannot replace the privately sealed predecessor. Hash binding and host control identity still do **not** authenticate caller-supplied numeric state or prove that a real model produced it.

## Fixed-plan compatibility

The earlier PauseAdmission reference expects a complete acknowledgement at construction while the host is at pre revision. A capture-generated acknowledgement becomes known at receipt revision, so that constructor is not reused or relabeled. The new bridge defers private gate construction until the pair is validated. A test confirms the old constructor rejects when given the post-phase host and generated acknowledgement. Original fixed-plan sources/artifacts remain unchanged.

The dormant RouteCaptureCandidate currently exposes immutable pairs, not the private pre-phase host capability required by this bridge. It is **not attached**. A future new dormant version should broker preseal/ack internally rather than exporting raw model/host capabilities or asking a caller to fabricate the predecessor proof. Runtime launch remains false and requires its own separate source-pinned protocol.

## Verification and retained unit failure

Twenty-six faults reject: missing preseal; copied/foreign/expired pre token; wrong pre owner/revision/digest; copied/foreign/expired receipt token; pre token used for receipt; bad receipt digest; changed sealed pre; changed input source; partial routes; wrong receipt revision; duplicate preseal/ack; release before ack; expired/renewed delivery token; poison/close after ack; duplicate release; early delivery; and cross-thread acknowledgement. A single worker thread is bounded by five seconds; this is an exclusion check, not resource measurement. Invalid layout constructors reject before submission; live-control/binding rejection stops the bridge.

The initial V1 unit run passed eight tests and failed one assertion: `test_ack_copy_is_not_authority` expected HostError, while the bridge correctly refused an acknowledgement used as a ticket with AdmissionError. V1 primary/replay hand-report artifacts were created by the original command sequence after that unit failure and are retained; they do not make the V1 unit suite successful. No V1 rerun or artifact overwrite occurred.

The erroneous test bytes are archived at `data/processed/benchmark_fixtures/route_capture_ack_failed_unit_source_v1.py.txt`; their SHA matches the original V1 registered test path. The old test path is no longer active, so V1 source verification at its original path is unavailable, not claimed restored. `route_capture_ack_unit_failure_v1.json` and `route_capture_ack_unit_repair_provenance_v1.json` preserve the failure and location change. V2 corrects only the expected exception and test/runner identity. V1/V2 kernel and harness hashes match; the saved hand reports match exactly except registration digest and preservation count. The repair is routine test correction, not scientific tuning or recovery of any old numerical reproduction failure.

Nine V2 tests cover frozen pair/fault roster, pair-derived values, fixed-plan incompatibility, private-preseal replacement refusal, absence of expected-ack argument, copied-ticket rejection, expired delivery before payload inspection, ack metadata used as capability, source-ID binding and input/report budget refusal. All pass without importing learner modules. V2 primary/fresh-process replay and source verification pass. Consumed V2 execute/replay refuse before computation. The selected roadmap gate checker passes.

## Next and holds

🟠 [Next] Preregister a new **launch-disabled** capture-candidate version that brokers the generated acknowledgement inside its private owner: preseal at predecessor pause, validate/ack at receipt pause, and export immutable binding records only. Preserve V1 candidate and both fixed-plan sources; keep runtime gate false, check source-only call/alias coverage, and execute no model, candidate encoder, real producer or storage backend. Do not expose host capabilities just to make the bridge fit.

⭕️ [Pending] Genuine model/run authority, full writer/alias confinement, independent live numeric readback, input-only real source/outcome authority, matched independent scalar after-states, durable nine-route publication/consumption, independent task usefulness and SNN-specific causal/resource evidence. All seven missing categories remain missing. Learner/calculator/model-storage/candidate-encoder/live-numeric-capture executions, task scores, acquisition and resource measurement remain zero. Original reproduction failure, negative recovery/retention/resource results, conditions and test/source seals are unchanged. Generic acknowledgement binding does not establish an SNN-exclusive benefit.

## 日本語

検証済み採取ペアからダイジェスト・状態版・経路IDを取り出す確認ブリッジを事前登録・実装しました。有効な前状態の停止中に内部記録し、受信後の有効な停止中にペアを検証して確認を生成します。手作業2組、拒否26件、V2の9テスト、別プロセス再検証が成功し、666件の証跡を維持しました。V1の例外型期待の誤りは失敗とテスト原文を保存し、実装・条件を変えずV2で訂正しました。実モデル・候補エンコーダー・実入力元には接続せず、旧失敗・負の結果・封印を維持しています。

## 简体中文

预注册并实现从已验证采集对提取摘要、版本及路径ID的确认桥。在有效前状态暂停中私有封存，再在有效接收暂停中验证并生成确认。2组手工记录、26项拒绝、V2的9项测试和独立进程复验通过，保留666项证据。保留V1异常类型断言错误及原始测试，不改变实现或条件，在V2修正。不连接真实模型、候选编码器或输入源，旧失败、负面结果与封存保持不变。
