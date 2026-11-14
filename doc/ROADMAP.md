# SARA Engine Roadmap

This page tracks current progress, priorities and the conditions for advancing. The historical roadmap contains the full experiment records, earlier decisions and detailed phase plans. [Roadmap history](ROADMAP_HISTORY.md)

## Current progress

| Status | Area | Evidence and remaining work |
| --- | --- | --- |
| 🟢 [Done] | Local route learning and capture | Actual local updates, delayed outcomes, retained-key probabilities and receipt acknowledgements verified in bounded synthetic runs. [Factorization evidence](SPARSE_PREDICTIVE_FACTORIZATION.md) |
| 🟢 [Done] | Independent scalar comparison | 700 receipts and 6,291 local updates matched through two reactivation cycles and their cooldowns. [Comparison](SPARSE_ROUTE_MULTICYCLE_COMPARISON.md) |
| 🟢 [Done] | Durable receipt processing | Nine-route updates and receipt consumption commit together. Sequential processing, duplicate refusal and recovery were verified with local SQLite. [Durability](SPARSE_ROUTE_DURABLE_COMMIT.md) |
| 🟢 [Done] | Authenticated delivery | Separate test producers send authenticated JSON to the stdin CLI. Key rotation, tamper rejection and idempotent replay across restarts pass. [Ingress](SPARSE_ROUTE_DURABLE_COMMIT.md#separate-process-delivery-ingress) |
| ⭕️ [Pending] | Independent task and SNN effect | Qualify an independent source and test the causal contribution of the SNN mechanism against matched event-driven scalar controls. [Decision gate](R2_MECHANISM_DECISION_GATE.md) |
| ⭕️ [Pending] | Operational deployment | Independent producers, reviewed key distribution and broader writer coverage are still required. Candidate V2 launch remains disabled. [Capture candidate](SPARSE_ROUTE_CAPTURE_CANDIDATE_V2.md) |

## Priorities

1. 🟠 [Next] Identify a fresh independently observed delayed-outcome task. Require repeated current states, distinguishable histories, both classes and the existing support and source gates before preregistering a comparison. Use the strongest current-state control and match actual local writes. [Four-country coverage follow-up](ROADMAP_NEXT_READINESS.md#four-country-coverage-follow-up) · [Alternative-source screen](ROADMAP_NEXT_READINESS.md#refit-alternatives--2026-11-14) · [REFIT metadata review](ROADMAP_NEXT_READINESS.md#independent-delayed-outcome-task) · [Independent-task history and gates](ROADMAP_HISTORY.md#existing-research-queue)
2. 🟠 [Next] Prepare the independent producer pilot with reviewed trust material, revision handling and key distribution. Measure authority review, revision completeness, availability, latency and complete-path cost under the existing pilot conditions. [Publisher pilot readiness](ROADMAP_NEXT_READINESS.md#independent-https-publisher-pilot) · [Publisher pilot context](ROADMAP_HISTORY.md#existing-research-queue)
3. ⭕️ [Pending] Promote a mechanism or begin new resource comparisons only after the independent task and causal gates pass. Existing event-work and allocation diagnostics retain their recorded results. [Research focus and promotion gates](ROADMAP_NEXT_LEVEL.md)
4. 🔴 [Later] Compare bounded local representations at equal total byte budgets, including indexes, decoding, residuals and rare-route retention. [LittleBit design reference](SPARSE_PREDICTIVE_FACTORIZATION.md#littlebit-design-reference)
5. 🔴 [Later] Evaluate exact reuse of local readouts when their dependencies are unchanged, after the independent-task and causal gates pass. [Evaluation contract](#local-readout-reuse)

6. 🔴 [Later] Evaluate bounded inference escalation after the same independent-task and causal gates pass. [Evaluation contract](#bounded-inference-escalation)

### Local readout reuse

The referenced computation-reduction proposal motivates reusing a derived local readout while its key, contributing state versions and time-dependent inputs remain equal. Start with a bounded per-route cache for a read-only probability. Recompute on a missing entry, dependency change or uncertain validity. An unchanged input can still coincide with trace decay, a delayed result, expiry or a restored state, so these transitions must be covered by the reuse check. Capture the probability at receipt time and preserve it through delivery.

Require exact output and state parity with the uncached implementation, including delayed results, expiry, recovery and rare-route cases. Apply the same reuse opportunity to the event-driven scalar control and match actual local writes. Account for dependency checks, cache memory, invalidation, fallback and shared event processing using the existing [work accounting contract](R2_EVENT_WORK_ACCOUNTING.md). Attribute a benefit to SNN dynamics only if an advantage survives the matched scalar comparison and the existing causal gate.

Treat error-based omission of local writes as a separate learning intervention. Small errors alone provide insufficient evidence to skip updates that may preserve rare dependencies or aid recovery. Semantic planning and text rendering separation remains outside this route-level comparison.

日本語：依存状態が一致する局所読み出しの再利用を将来候補に追加。遅延結果・期限切れ・復旧を含む完全一致と、同じ再利用を備えたスカラー対照を条件とする。

简体中文：将依赖状态一致时复用局部读出列为后续候选，要求在延迟结果、过期和恢复时完全一致，并采用具备相同复用机制的标量对照。

### Bounded inference escalation

Start with the existing current-state readout. Use development-only evidence of support and uncertainty to decide whether to consult the temporal readout. Escalate on unknown keys or insufficient support. Fix the policy and quality margins before evaluation. Inference routing leaves the receipt lifecycle and local write schedule intact.

Compare always-current, always-temporal and conditional readouts, and give the event-driven scalar control the same routing opportunity. Check aggregate quality and the quality of predictions that stayed on the cheap path, including rare histories and delayed outcomes. Use the existing work-accounting contract to include routing, both readouts on fallback, shared event processing and memory. Report fallback frequency and tail latency alongside total work. Evaluate this policy separately from exact reuse before testing their combination.

日本語：現在状態から読み出し、根拠が不足する場合に時間履歴を参照する段階推論を将来候補に追加。

简体中文：将分级推理列为后续候选，先读取当前状态，证据不足时再读取时间历史。

## Selected execution gates

- 🟢 [Done] Fixed CartPole comparison completed with a negative V2 gate; no retry. <!-- gate:cartpole-fixed-comparison -->
- ⭕️ [Pending] New task/hypothesis and fresh development identities before another local-credit trial. <!-- gate:cartpole-fresh-task -->
- ⭕️ [Pending] Trusted owner/revision/pause and outcome-free access before actual receipt capture. <!-- gate:authoritative-receipt-capture -->
- ⭕️ [Pending] SNN-specific causal benefit over matched event-driven scalar controls. <!-- gate:snn-specific-advantage -->

## Evidence and phase plans

- [Roadmap history through the delivery ingress milestone](ROADMAP_HISTORY.md) preserves the previous roadmap, including negative and failed attempts, execution decisions and Phases 6–20.
- [Implemented features](IMPLEMENTED_FEATURES.md) lists completed capabilities. [Pre-v1.1 archive](old/ROADMAP_v1_1_completion_archive.md) records the release baseline.
- [Next-level roadmap](ROADMAP_NEXT_LEVEL.md) tracks promotion conditions and links to the detailed plans for Phases 21–46.
- Consumed experiments and sealed splits retain their recorded status. The selected execution gates are checked by the [roadmap gate checker](ROADMAP_GATE_CHECK.md).

日本語：合成入力での局所学習、独立スカラーとの照合、復旧可能な永続化、認証付き受信まで完了。次は独立課題の選定と外部送信元の試験準備。SNN固有の効果と資源上の優位性は検証待ち。

简体中文：已完成合成输入下的局部学习、独立标量对照、可恢复持久化与认证接收。下一步是筛选独立任务和准备外部发布方试点。SNN 特有的因果效果和资源优势仍待验证。

## Release baseline audit

* **ROADMAP closure audit:**
  - DONE: release-critical path is complete for the v1.1 baseline.
  - DONE: observed-only evidence remains labeled until stronger external or physical measurements pass.
  - DONE: long-term research backlog is separated into Phase 6 and later roadmap work.
  - DONE: roadmap completion audit is present for the current baseline.
  - DONE: research product completion gate is part of the active release evidence surface.
