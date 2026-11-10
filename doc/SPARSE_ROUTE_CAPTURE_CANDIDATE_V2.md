# Launch-disabled private capture candidate V2 — 2026-11-10

🟢 [Done] Separately preregistered source candidate V2 brokers the capture-generated acknowledgement inside its private owner. Sixteen static/refusal checks, thirteen unit tests and exact fresh-process replay pass; 674 historical artifact hashes are preserved. V1 candidate, fixed-plan composition and generated-ack bridge sources/artifacts are unchanged. Runtime launch is fixed false; no model, encoder, host, bridge capture body or model persistence is executed.

Selection SHA-256: `611802ffecfe5b3a89ae31451b8d900f1403d87a5f55c91601a76137c297d6f2`. Private brokerage, phase order, immutable exports, no-result-delivery policy, check roster and budgets were saved before implementation. V2 kernel/harness/runner/tests, ten pinned existing source dependencies, path utility and CPython 3.10.22 were registered before primary verification. Historical parsed reports are `route_capture_ack_result_v2.json` and `route_capture_candidate_result_v1.json`. Six create-only artifacts use `route_capture_candidate_v2_*_v1.json` (V2 candidate, first verification protocol). Limits: 192 KiB source/report/candidate input, 256 KiB/artifact and 2 MiB aggregate.

## Internal brokerage source path

`RouteCaptureCandidateV2.__init__` and `capture` both begin with a fixed-false launch gate. The constructor raises before lazy learner import, host creation or model construction. It takes only owner/epoch labels and frozen-model configuration, never a model, callback, override/permit or host capability. An uninitialized V2 object's capture method also refuses before accessing fields or encoders. This tests refusal, not runtime initialization.

The unreachable source path defines private exact FactorialMemory construction, host/owner storage and a one-use immutable Input. Before entering a predecessor pause, construct a private CaptureAckBridge with the owner capabilities and Input. Then:

1. Under the predecessor pause, validate input identity/clock, perform two independent primitive traversals and internally seal that PhaseCapture in the bridge.
2. Under the owner mutation interval, invoke begin once. This would calculate the existing prediction/receipt if separately authorized in another version; it is not invoked here.
3. Under the distinct receipt pause, perform two more independent traversals, construct/bound the immutable CapturePair, and internally ask the unchanged bridge to validate it and derive the acknowledgement.
4. Revalidate the receipt pause after acknowledgement, leave the pause and close the stable control host in finally. A poisoned host remains terminal without a retry path.
5. Only after cleanup return frozen/slotted AcknowledgedCapture containing exactly CapturePair and CaptureAck metadata.

V2 is a separate implementation rather than a subclass whose operation depends on enabling the V1 gate. It reuses the pinned V1 primitive encoding helpers by reference, not by calling or modifying them during verification. The source checker does not instantiate a model or call those helpers. All capture state is intended to be detached typed immutable values; logical projection and generated acknowledgement are delegated to the unchanged layout/bridge contracts.

## No capability or result lease export

The public instance interface is capture only; helpers are private. There is no generator/yield, callback, borrowed model, outward host/model/owner/bridge reference, pause ticket or outcome parameter. The frozen output fields are exactly pair and ack. Static checks restrict return shapes and reject direct control-object export or handing a raw model to an external callable instead of the allowed primitive `vars` traversal. Private Python attributes remain a cooperative convention, not a security sandbox against arbitrary introspection.

Unlike V1's context-manager yield, V2 returns finalized binding metadata **after** leaving the receipt pause and closing the stable host. The returned acknowledgement is not a live delivery token. The private bridge is not asked to release an outcome, and its old receipt ticket cannot authorize later release after cleanup. No result-delivery lease or producer is created. Any future pipeline requiring result delivery must define a separate lifecycle/ownership contract rather than reopening this pause or treating metadata as authority.

This addresses the concrete compatibility gap: bridge construction occurs at pre revision, preseal has the genuine reference ticket, and pair-derived acknowledgement occurs at receipt revision without exposing capabilities to a caller. It does not show that a real model produced the copied numbers or that every external writer is excluded. Actual runner origin, epoch authenticity, numeric encoding completeness and live independent readback remain absent.

## Static and refusal verification

Sixteen checks require literal false flag; constructor guard before lazy learner import; capture guard first; no eager learner import; exact public/signature scope; private bridge before pre pause; two pauses/two read traversals; preseal under pre pause; begin under mutation; ack under receipt pause; post-ack revalidation; stable cleanup before public return; frozen exact output fields; no yield/control export; no selected learner/result/storage calls; and refusal before encoder/host/model creation.

Thirteen tests include source mutations for enabling the flag, eager import, model/callback/permit parameters, host export, an extra owner output field, yielding a continuation, wrong pre/receipt tokens, aliasing the first copy, observe/result-release calls, raw model handoff, file opening and skipped cleanup. Refusal tests replace the encoder and host constructor with traps and confirm neither is reached, with no additional runtime modules imported. The capture refusal uses an uninitialized object only; no model or host is attached. No fake-model execution or patched-enabled launch is used.

The static checker records source conformance to these exact patterns, not a full Python points-to/call-graph, transitive purity, concurrency or security proof. Helpers, dynamic aliases, exceptions during setup/teardown and untrusted code are not certified safe by passing AST tests. The candidate body, encoding helpers, bridge and numeric state traversal have not executed. Primary and fresh-process verification match exactly; consumed phases refuse before computation. The selected roadmap gate consistency check passes.

## Remaining holds

⭕️ [Pending] Runtime launch is not authorized. A separate source-pinned bounded protocol must resolve genuine owner/run identity and input-only source, all writer/alias coverage, actual numeric encoding/readback, failure/cleanup behavior and outcome-order evidence before any model/encoder invocation. Model configuration arguments are not a selected fresh trial or proof of seed/source eligibility. Do not turn on a flag, monkeypatch globals, replay consumed attempts or infer permission from source success.

Seven runtime/task/SNN evidence categories remain missing. Independent scalar after-states and matched actual learning updates, durable nine-route publication/consumption, independent task usefulness and SNN-specific causal/resource evidence remain separate holds. No extra hand numeric fixtures or live scores are generated. Learner/calculator/model-storage/candidate-encoder/host/live-capture executions, task scoring, acquisition and resource measurements are zero. Original reproduction failure, negative recovery/retention/resource outcomes, scientific conditions and test/source seals remain unchanged. The prior generated-ack V1 unit assertion failure and archived test bytes remain preserved; this study does not restore or rerun it.

Generic ownership and acknowledgement binding also apply to scalar systems. No SNN-exclusive capability, accuracy, energy benefit or production readiness follows from this source candidate.

## 日本語

採取確認ブリッジをprivateな所有者処理の内部で扱う、起動禁止の採取候補V2を別途事前登録・実装しました。前状態の停止中に内部封印し、受信後の停止中に確認を生成した後、停止を終了・ホストを閉じてから不変のペアと確認メタデータだけを返します。ホスト・所有者・停止トークンを外へ渡さず、結果配送も行いません。16静的チェック、13テスト、別プロセス再検証が成功し、674件の既存証跡を維持しました。モデル・エンコーダー・ホスト・モデル保存処理は実行せず、旧失敗・負の結果・封印を維持しました。実権限・数値採取・全更新経路は未検証で、起動は保留です。

## 简体中文

单独预注册并实现禁止启动的采集候选V2，在私有所有者流程内使用采集确认桥。于前状态暂停中封存、接收暂停中生成确认，退出暂停并关闭主机后仅返回不可变采集对和确认元数据。不导出主机、所有者或暂停令牌，也不配送结果。16项静态检查、13项测试与独立进程复验通过，保留674项历史证据。不执行模型、编码器、主机或模型存储，旧失败、负面结果与封存保持不变。真实权限、数值采集和全写入覆盖仍未验证，继续禁止启动。
